from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from worldsimflow.core.coverage import ScenarioCoverageAnalyzer
from worldsimflow.core.scenario_data_manager import ScenarioDataManager


def main() -> None:
    parser = argparse.ArgumentParser(description="Build WorldSimFlow scenario coverage report.")
    parser.add_argument("--roots", nargs="+", default=["data"])
    parser.add_argument("--batch-reports", nargs="*", default=default_reports())
    parser.add_argument("--output", default="outputs/all_results/coverage/scenario_coverage.json")
    parser.add_argument("--html", default="outputs/all_results/coverage/scenario_coverage.html")
    args = parser.parse_args()

    roots = [resolve(item) for item in args.roots]
    reports = [resolve(item) for item in args.batch_reports]
    manager = ScenarioDataManager()
    index = manager.scan(roots)
    analyzer = ScenarioCoverageAnalyzer()
    run_results = analyzer.load_run_results(reports)
    coverage = analyzer.from_index_and_runs(index, run_results)
    json_path = analyzer.write_json(coverage, resolve(args.output))
    html_path = analyzer.write_html(coverage, resolve(args.html))

    print("coverage=ok")
    print(f"scenarios={coverage.scenario_count}")
    print(f"map_aware_count={coverage.map_aware_count}")
    print(f"traffic_policy_count={coverage.traffic_policy_count}")
    print(f"run_count={coverage.run_count}")
    print(f"failure_count={coverage.failure_count}")
    print(f"output={json_path}")
    print(f"html={html_path}")


def default_reports() -> list[str]:
    return [
        "outputs/all_results/phase2_sample_report.json",
        "outputs/all_results/phase2_openlog_waymo_report.json",
        "outputs/all_results/targeted/nuscenes_target_hard_brake.run.json",
        "outputs/all_results/targeted/nuscenes_target_speed_change.run.json",
        "outputs/all_results/targeted/nuscenes_target_cut_in.run.json",
        "outputs/all_results/targeted/nuscenes_target_lateral_shift.run.json",
        "outputs/all_results/map_aware_idm/mini_curve_map_aware_idm.run.json",
        "outputs/all_results/map_aware_idm/nuscenes_map_aware_idm.run.json",
        "outputs/all_results/map_aware_idm/waymo_map_aware_idm.run.json",
    ]


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


if __name__ == "__main__":
    main()


