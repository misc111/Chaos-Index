from __future__ import annotations

from pathlib import Path

import yaml


def test_pages_staging_workflow_verifies_the_mlb_first_staging_contract() -> None:
    workflow_path = Path(".github/workflows/pages-staging.yml")
    workflow = yaml.safe_load(workflow_path.read_text())

    publish_job = workflow["jobs"]["publish"]
    steps = publish_job["steps"]

    verify_step = next(step for step in steps if step.get("name") == "Verify committed staging snapshot")
    assert verify_step["working-directory"] == "web"
    assert "verify-staging-contract.ts" in verify_step["run"]

    build_step = next(step for step in steps if step.get("name") == "Build static staging site")
    assert build_step["working-directory"] == "web"
    assert build_step["run"] == "npm run build:pages"
