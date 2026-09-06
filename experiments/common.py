"""Shared plumbing for the experiment scripts: seeds, CSV output, timing."""

from __future__ import annotations

import argparse
import csv
import pathlib
from typing import Dict, Iterable, List

RESULTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "results"


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--seed", type=int, default=0, help="base seed")
    parser.add_argument(
        "--instances", type=int, default=20, help="instances per configuration"
    )
    parser.add_argument("--out", type=str, default=None, help="output CSV name")
    return parser


def write_csv(name: str, rows: List[Dict], fieldnames: Iterable[str]) -> pathlib.Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / name
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {path}")
    return path
