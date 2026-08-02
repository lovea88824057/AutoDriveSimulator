from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from worldsimflow.core.scenario_data_manager import ScenarioDataManager


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a WorldSimFlow scenario dataset index.")
    parser.add_argument("--roots", nargs="+", default=["data"], help="Scenario files or directories to scan.")
    parser.add_argument("--pattern", default="*.json")
    parser.add_argument("--output", default="outputs/scenario_index.json")
    args = parser.parse_args()

    roots = [resolve_path(item) for item in args.roots]
    output = resolve_path(args.output)
    manager = ScenarioDataManager()
    index = manager.scan(roots, pattern=args.pattern)
    manager.write_index(index, output)

    print(f"index={output}")
    print(f"count={len(index.records)}")
    print(f"sources={index.summary['sources']}")
    print(f"interventions={index.summary['interventions']}")
    print(f"map_aware_count={index.summary['map_aware_count']}")
    if index.summary["errors"]:
        print(f"errors={len(index.summary['errors'])}")


def resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


if __name__ == "__main__":
    main()
