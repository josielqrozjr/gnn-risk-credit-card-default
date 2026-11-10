"""Run the complete real-data baseline, GNN matrix, and OOF hybrid."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


CUSTOMERS = 458_913


def main() -> int:
    args = _parse_args()
    root = Path.cwd()
    if not (root / "configs" / "baseline.json").exists():
        raise SystemExit("Run this command from the repository root.")
    commands = [
        [
            sys.executable,
            "-m",
            "src.baseline.run",
            "--config",
            "configs/baseline.json",
            "--sample-customers",
            str(args.sample_customers),
            "--artifacts-dir",
            str(args.output_dir / "baseline"),
        ],
        [
            sys.executable,
            "-m",
            "scripts.run_real_experiments",
            "--features-glob",
            "data/raw/parquet/data_*.parquet",
            "--labels-path",
            "data/raw/parquet/train_labels",
            "--sample-customers",
            str(args.sample_customers),
            "--output",
            str(args.output_dir / "metrics.json"),
            "--device",
            args.device,
        ],
        [
            sys.executable,
            "-m",
            "scripts.run_hybrid_oof",
            "--features-glob",
            "data/raw/parquet/data_*.parquet",
            "--labels-path",
            "data/raw/parquet/train_labels",
            "--sample-customers",
            str(args.sample_customers),
            "--output",
            str(args.output_dir / "hybrid_oof.json"),
            "--device",
            args.device,
        ],
    ]
    if args.dry_run:
        for command in commands:
            print(" ".join(command))
        return 0

    Path("logs").mkdir(exist_ok=True)
    for index, command in enumerate(commands, start=1):
        log_path = Path("logs") / f"{args.log_prefix}-{index}.log"
        print(f"[{index}/{len(commands)}] {' '.join(command)}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        if completed.returncode != 0:
            print(f"Phase {index} failed; inspect {log_path}", file=sys.stderr)
            return completed.returncode
    print("Full experiment pipeline completed.", flush=True)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sample-customers", type=int, default=CUSTOMERS)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/experiments/full")
    )
    parser.add_argument("--log-prefix", default="full")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
