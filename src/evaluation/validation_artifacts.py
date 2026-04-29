"""Validation artifact registration and deterministic file writing.

The validation pipeline produces many small CSV and JSON diagnostics.  Keeping
the registry in this module gives task runners a narrow interface: declare a
named section, provide the payload, and let the writer enforce deterministic
paths plus manifest output.  That separation matters for the MLB rebuild because
dashboard payloads, staging snapshots, and governance checks all consume the
manifest rather than inferring files from the filesystem.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.common.utils import ensure_dir


@dataclass(frozen=True, slots=True)
class ValidationSectionSpec:
    """Describe one materialized validation output.

    Attributes:
        section: Stable manifest key used by downstream contracts.
        file_name: Path relative to the validation output root.
        kind: Payload format.  Only ``csv`` and ``json`` are intentionally
            supported so artifact readers do not need a looser dispatch layer.
        tail_rows: Optional display hint for consumers that render a compact
            preview from a larger CSV file.
    """

    section: str
    file_name: str
    kind: str
    tail_rows: int | None = None


@dataclass(slots=True)
class ValidationOutputs:
    """Collect validation task outputs before writing them to disk.

    The object stores defensive copies of payloads.  Validation tasks often
    continue mutating local dataframes while assembling multiple diagnostics, so
    copying at registration time prevents later caller-side mutations from
    silently changing the artifact that will be written.
    """

    sections: list[ValidationSectionSpec] = field(default_factory=list)
    csv_payloads: dict[str, pd.DataFrame] = field(default_factory=dict)
    json_payloads: dict[str, dict[str, Any]] = field(default_factory=dict)
    task_records: list[dict[str, Any]] = field(default_factory=list)

    def add_csv(
        self,
        *,
        section: str,
        file_name: str,
        rows: pd.DataFrame,
        tail_rows: int | None = None,
    ) -> None:
        """Register a dataframe artifact and its manifest section.

        Duplicate section names or artifact paths are rejected immediately.  That
        fail-closed behavior protects the immutable validation contract from
        accidentally overwriting a prior task's output.
        """

        self._register(ValidationSectionSpec(section=section, file_name=file_name, kind="csv", tail_rows=tail_rows))
        self.csv_payloads[section] = rows.copy()

    def add_json(self, *, section: str, file_name: str, payload: dict[str, Any]) -> None:
        """Register a JSON artifact and its manifest section."""

        self._register(ValidationSectionSpec(section=section, file_name=file_name, kind="json"))
        self.json_payloads[section] = dict(payload)

    def merge(self, other: "ValidationOutputs") -> None:
        """Merge another task's outputs into this registry.

        Merging uses the same registration checks as direct additions so a task
        cannot collide with an already registered section or file path.
        """

        for spec in other.sections:
            self._register(spec)
        self.csv_payloads.update({key: value.copy() for key, value in other.csv_payloads.items()})
        self.json_payloads.update({key: dict(value) for key, value in other.json_payloads.items()})
        self.task_records.extend([dict(record) for record in other.task_records])

    def write(
        self,
        out_dir: Path,
        *,
        league: str,
        manifest_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Write all registered artifacts plus the validation manifests.

        Args:
            out_dir: Validation output root for the current league.
            league: Canonical league code to store in the manifest.
            manifest_metadata: Optional run-level metadata embedded under the
                manifest's ``metadata`` key.
        """

        root = ensure_dir(out_dir)
        for spec in self.sections:
            path = root / spec.file_name
            ensure_dir(path.parent)
            if spec.kind == "csv":
                self.csv_payloads[spec.section].to_csv(path, index=False)
            elif spec.kind == "json":
                path.write_text(json.dumps(self.json_payloads[spec.section], indent=2, sort_keys=True))
            else:
                raise ValueError(f"Unsupported validation artifact kind '{spec.kind}' for section '{spec.section}'")

        manifest: dict[str, Any] = {
            "league": league,
            "sections": [asdict(spec) for spec in self.sections],
        }
        if manifest_metadata:
            manifest["metadata"] = dict(manifest_metadata)
        (root / "validation_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
        (root / "validation_outputs_contract.json").write_text(
            json.dumps({"validation_outputs": list(self.task_records)}, indent=2, sort_keys=True)
        )

    def _register(self, spec: ValidationSectionSpec) -> None:
        """Add a manifest section after checking both section and path uniqueness."""

        if spec.section in self.csv_payloads or spec.section in self.json_payloads:
            raise ValueError(f"Duplicate validation section '{spec.section}'")
        if any(existing.file_name == spec.file_name for existing in self.sections):
            raise ValueError(f"Duplicate validation artifact path '{spec.file_name}'")
        self.sections.append(spec)


def validation_path(*parts: Any) -> str:
    """Build a normalized manifest-relative validation artifact path.

    Callers pass semantic path parts rather than pre-joined strings so task code
    can stay readable while still producing slash-separated paths across
    platforms.
    """

    cleaned: list[str] = []
    for part in parts:
        token = str(part).strip("/").replace("\\", "/")
        if token:
            cleaned.append(token)
    return "/".join(cleaned)
