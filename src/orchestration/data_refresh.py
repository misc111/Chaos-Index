"""CLI entry point for deterministic MLB-first data and scoring refreshes."""

from __future__ import annotations

import argparse

from src.orchestration.refresh_pipeline import (
    build_current_season_predictiveness_steps,
    build_daily_score_steps,
    build_data_refresh_steps,
    run_steps,
)


def main() -> None:
    """Run or preview deterministic MLB-first data-only or daily-scoring pipelines."""

    parser = argparse.ArgumentParser(description="Run deterministic MLB-first data-refresh and daily-scoring pipelines.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the data-refresh step plan without executing it.",
    )
    parser.add_argument(
        "--include-current-season-scoring",
        action="store_true",
        help="Append fresh current-season predictiveness scoring without rebuilding features or retraining models.",
    )
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="Run only current-season predictiveness scoring from existing frozen ledgers.",
    )
    args = parser.parse_args()

    if args.score_only and not args.include_current_season_scoring:
        parser.error("--score-only requires --include-current-season-scoring")

    if args.score_only:
        steps = build_current_season_predictiveness_steps()
    elif args.include_current_season_scoring:
        steps = build_daily_score_steps()
    else:
        steps = build_data_refresh_steps()

    if args.dry_run:
        for index, step in enumerate(steps, start=1):
            print(f"[{index}/{len(steps)}] {step.name}")
            print(f"  cwd={step.cwd}")
            print(f"  cmd={step.display_command}")
        return

    if args.score_only:
        pipeline_name = "Daily score-only evidence"
    elif args.include_current_season_scoring:
        pipeline_name = "Daily scoring"
    else:
        pipeline_name = "Data refresh"
    run_steps(steps, pipeline_name=pipeline_name)


if __name__ == "__main__":
    main()
