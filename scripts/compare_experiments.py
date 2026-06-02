# -*- coding: utf-8 -*-
"""Build comparison CSV/Markdown/plot from experiment run directories."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from experiment_utils import build_comparison_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, help="Directory containing experiment run folders")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_comparison_report(args.output_dir)
    print(f"wrote comparison for {len(rows)} runs under {args.output_dir}")


if __name__ == "__main__":
    main()
