from __future__ import annotations

from pathlib import Path

from src.training.artifact_writer import save_model_artifacts


class _SaveOnlyModel:
    def __init__(self, payload: str) -> None:
        self.payload = payload

    def save(self, path: Path) -> None:
        path.write_text(self.payload)


def test_save_model_artifacts_is_stable_across_model_dict_insertion_order(tmp_path: Path) -> None:
    first = {
        "rf": _SaveOnlyModel("rf"),
        "glm_lasso": _SaveOnlyModel("lasso"),
        "glm_ridge": _SaveOnlyModel("ridge"),
    }
    second = {
        "glm_ridge": _SaveOnlyModel("ridge"),
        "glm_lasso": _SaveOnlyModel("lasso"),
        "rf": _SaveOnlyModel("rf"),
    }

    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()

    first_payload = save_model_artifacts(first, first_dir)
    second_payload = save_model_artifacts(second, second_dir)

    assert list(first_payload) == ["glm_lasso", "glm_ridge", "rf"]
    assert list(second_payload) == list(first_payload)
    assert first_payload == second_payload
