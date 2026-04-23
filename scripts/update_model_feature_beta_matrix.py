#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from src.common.config import load_config
from src.reporting.model_feature_matrix import write_model_feature_matrix_markdown


def _parse_csv_arg(value: str | None) -> list[str] | None:
    token = str(value or "").strip()
    if not token:
        return None
    return [part.strip() for part in token.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the MLB model feature/beta Markdown matrix.")
    parser.add_argument("--config", default="configs/mlb.yaml", help="Config path.")
    parser.add_argument(
        "--output",
        default="docs/mlb/model_feature_beta_matrix.md",
        help="Output Markdown path.",
    )
    parser.add_argument(
        "--production-models",
        default="",
        help="Optional comma-separated production models to include.",
    )
    parser.add_argument(
        "--comparison-models",
        default="",
        help="Optional comma-separated candidate comparison models to include.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_path = write_model_feature_matrix_markdown(
        cfg,
        output_path=Path(args.output),
        production_models=_parse_csv_arg(args.production_models),
        comparison_models=_parse_csv_arg(args.comparison_models),
    )
    print(output_path)


if __name__ == "__main__":
    main()
