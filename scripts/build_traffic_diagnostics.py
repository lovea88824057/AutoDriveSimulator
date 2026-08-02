from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from worldsimflow.core.traffic_diagnostics import TrafficDiagnosticsDashboard, TrafficDiagnosticsConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Build C3 Traffic Diagnostics Dashboard from WorldSimFlow transition JSONL.")
    parser.add_argument("--observations", default="outputs/all_results/rl_eval/observations.jsonl", help="Input transition_v1 JSONL file.")
    parser.add_argument("--observation-report", default="outputs/all_results/rl_eval/observations_report.json", help="Optional observations report JSON.")
    parser.add_argument("--lane-diagnostics", default="", help="Optional lane diagnostics JSON.")
    parser.add_argument("--run-report", default="", help="Optional run report JSON.")
    parser.add_argument("--output", default="outputs/all_results/diagnostics/traffic_diagnostics.html", help="Output dashboard HTML.")
    parser.add_argument("--summary", default="outputs/all_results/diagnostics/traffic_diagnostics.summary.json", help="Output summary JSON.")
    args = parser.parse_args()

    dashboard = TrafficDiagnosticsDashboard(TrafficDiagnosticsConfig())
    summary = dashboard.build(
        observations_jsonl=resolve(args.observations),
        output_html=resolve(args.output),
        summary_json=resolve(args.summary) if args.summary else None,
        observation_report=resolve(args.observation_report) if args.observation_report else None,
        lane_diagnostics=resolve(args.lane_diagnostics) if args.lane_diagnostics else None,
        run_report=resolve(args.run_report) if args.run_report else None,
    )

    print("traffic_diagnostics=ok")
    print(f"scenario_ids={','.join(summary.get('scenario_ids', []))}")
    print(f"transition_count={summary.get('transition_count')}")
    print(f"total_reward={summary.get('total_reward')}")
    print(f"min_front_gap={summary.get('min_front_gap')}")
    print(f"max_abs_ego_lane_l={summary.get('max_abs_ego_lane_l')}")
    print(f"event_count={summary.get('event_count')}")
    print(f"output={resolve(args.output)}")
    if args.summary:
        print(f"summary={resolve(args.summary)}")


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


if __name__ == "__main__":
    main()
