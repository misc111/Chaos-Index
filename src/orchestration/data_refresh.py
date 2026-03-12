"""CLI entry point for the deterministic repo-wide data refresh pipeline."""

from __future__ import annotations

import argparse

from src.orchestration.refresh_pipeline import build_data_refresh_steps, run_steps


def main() -> None:
    """Run or preview the deterministic repo-wide data refresh pipeline."""

    parser = argparse.ArgumentParser(description="Run the deterministic data-refresh pipeline across all supported leagues.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the data-refresh step plan without executing it.",
    )
    args = parser.parse_args()

    steps = build_data_refresh_steps()

    if args.dry_run:
        for index, step in enumerate(steps, start=1):
            print(f"[{index}/{len(steps)}] {step.name}")
            print(f"  cwd={step.cwd}")
            print(f"  cmd={step.display_command}")
        return

    run_steps(steps, pipeline_name="Data refresh")


if __name__ == "__main__":
    main()
