from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from worldsimflow.core.scenario import ScenarioLoader
from worldsimflow.core.scenario_diff import ScenarioDiff


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare an original scenario with an intervention variant.")
    parser.add_argument("--base", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    base_path = resolve(args.base)
    variant_path = resolve(args.variant)
    base = ScenarioLoader().load(base_path)
    variant = ScenarioLoader().load(variant_path)
    report = ScenarioDiff().compare(base, variant).to_dict()
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        output = resolve(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")


def resolve(path: str) -> Path:
    item = Path(path)
    return item if item.is_absolute() else (ROOT / item).resolve()


if __name__ == "__main__":
    main()