"""Command-line entry point for the network experiments."""
from __future__ import annotations

import argparse
from contextlib import chdir
from pathlib import Path
from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="toniot-ids",
        description="LightGBM training and XAI feature selection on TON_IoT network data.",
    )
    parser.add_argument(
        "--workspace", type=Path, default=Path.cwd(),
        help="Directory containing data_network/; models/ and reports/ are written here (default: current directory).",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("preprocess", help="Download missing raw data and generate the encoded data splits.")
    commands.add_parser("analyze", help="Generate descriptive statistics for the network CSV.")

    baseline = commands.add_parser("baseline", help="Train and evaluate the baseline LightGBM model.")
    baseline.add_argument("--xai", action=argparse.BooleanOptionalAction, default=True,
                          help="Run XAI feature selection after training (default: enabled).")
    baseline.add_argument("--save-results", action=argparse.BooleanOptionalAction, default=True,
                          help="Write model and reports (default: enabled).")
    baseline.add_argument("--features", type=Path,
                          help="CSV with a feature column; paths are relative to the workspace.")

    reduction = commands.add_parser("reduce", help="Run one reduction stage, or all stages in order.")
    reduction.add_argument("stage", choices=("1", "2", "3", "all"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    workspace = args.workspace.expanduser().resolve()
    if not workspace.is_dir():
        parser.error(f"Workspace directory not found: {workspace}")

    if args.command == "baseline" and args.features is not None:
        feature_file = args.features.expanduser()
        if not feature_file.is_absolute():
            feature_file = workspace / feature_file
        if not feature_file.is_file():
            parser.error(f"Feature list not found: {feature_file}")
        args.features = feature_file.resolve()

    with chdir(workspace):
        if args.command in ("preprocess", "analyze"):
            from . import data

            data.ensure_raw_dataset()
            if args.command == "preprocess":
                data.preprocess_dataset()
            else:
                data.analyze_toniot()
            return 0

        from . import experiments

        if args.command == "baseline":
            experiments.run_lightgbm_pipeline(
                save_results=args.save_results,
                feature_selection=args.features is not None,
                fs_config=args.features,
                run_xai=args.xai,
            )
            return 0

        stages = {
            "1": experiments.run_reduction_1_experiment,
            "2": experiments.run_reduction_2_experiment,
            "3": experiments.run_reduction_3_experiment,
        }
        selected = ("1", "2", "3") if args.stage == "all" else (args.stage,)
        for stage in selected:
            result = stages[stage]()
            if not result["accepted"]:
                print(f"[INFO] Reduction-{stage} was not accepted; reduction sequence finished.")
                break
    return 0
