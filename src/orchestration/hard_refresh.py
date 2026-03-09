import argparse
from src.orchestration.refresh_pipeline import ROOT_DIR, build_hard_refresh_steps, run_steps


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the deterministic NHL/NBA hard-refresh pipeline without rebuilding features."
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Optional comma-separated model list to train for both leagues. Defaults to the full model suite.",
    )
    parser.add_argument(
        "--approve-feature-changes",
        action="store_true",
        help="Explicitly allow feature-contract updates during the train step.",
    )
    parser.add_argument(
        "--skip-pages-build",
        action="store_true",
        help="Skip `npm run build:pages` after generating staging data.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the hard-refresh step plan without executing it.",
    )
    args = parser.parse_args()

    steps = build_hard_refresh_steps(
        models_arg=args.models,
        approve_feature_changes=bool(args.approve_feature_changes),
        include_pages_build=not bool(args.skip_pages_build),
    )

    if args.dry_run:
        for index, step in enumerate(steps, start=1):
            print(f"[{index}/{len(steps)}] {step.name}")
            print(f"  cwd={step.cwd}")
            print(f"  cmd={step.display_command}")
        return

    run_steps(steps, pipeline_name="Hard refresh")


if __name__ == "__main__":
    main()
