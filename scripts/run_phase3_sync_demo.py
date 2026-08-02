from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from worldsimflow.core.message_sync import run_sync_demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 3 message synchronization diagnostics demo.")
    parser.add_argument("--mode", choices=["success", "missing", "stale", "out_of_order", "all"], default="all")
    parser.add_argument("--output", default=str(ROOT / "outputs" / "phase3_sync_report.json"))
    args = parser.parse_args()

    modes = ["success", "missing", "stale", "out_of_order"] if args.mode == "all" else [args.mode]
    reports = [run_sync_demo(mode) for mode in modes]
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"reports": reports}, ensure_ascii=False, indent=2), encoding="utf-8")

    print("phase3_sync_demo=ok")
    print(f"output={output}")
    for report in reports:
        summary = report["summary"]
        print(f"mode={report['mode']} ok={summary['ok']} frames={summary['frame_count']} issues={summary['issue_counts']} hash={summary['trace_hash'][:12]}")


def resolve(path: str) -> Path:
    item = Path(path)
    return item if item.is_absolute() else (ROOT / item).resolve()


if __name__ == "__main__":
    main()