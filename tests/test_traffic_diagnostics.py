from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from worldsimflow import ScenarioLoader, TransitionDatasetExporter, TransitionExportConfig
from worldsimflow.core.traffic_diagnostics import TrafficDiagnosticsDashboard


def simple_policy(obs):
    lane_l = float(obs.get("ego_lane_l") or 0.0)
    speed = float(obs.get("ego_speed") or 0.0)
    return {"acceleration": max(-1.0, min(1.0, 7.0 - speed)), "steering": max(-0.2, min(0.2, -lane_l * 0.1))}


def test_traffic_diagnostics_dashboard_builds_html_and_summary(tmp_path):
    scenario = ScenarioLoader().load(ROOT / "data" / "sample_scenario.json")
    observations = tmp_path / "observations.jsonl"
    TransitionDatasetExporter(
        scenario,
        TransitionExportConfig(episodes=1, max_steps=8, seed=42, include_full_observation=False),
    ).export(observations, simple_policy)

    html_path = tmp_path / "traffic_diagnostics.html"
    summary_path = tmp_path / "traffic_diagnostics.summary.json"
    summary = TrafficDiagnosticsDashboard().build(
        observations_jsonl=observations,
        output_html=html_path,
        summary_json=summary_path,
    )

    html = html_path.read_text(encoding="utf-8")
    saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["dashboard_schema_version"] == "traffic_diagnostics_v1"
    assert summary["transition_count"] == 8
    assert summary["feature_count"] == 17
    assert summary["scenario_ids"] == ["straight_close_follow_001"]
    assert "WorldSimFlow Traffic Diagnostics Dashboard" in html
    assert "ego_speed" in html
    assert "front_gap" in html
    assert saved_summary["html"] == str(html_path)


def test_build_traffic_diagnostics_script_runs(tmp_path):
    observations = tmp_path / "observations.jsonl"
    html_path = tmp_path / "dashboard.html"
    summary_path = tmp_path / "summary.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "export_observations_jsonl.py"),
            "--scenario",
            "data/sample_scenario.json",
            "--episodes",
            "1",
            "--steps",
            "5",
            "--compact",
            "--output",
            str(observations),
            "--report",
            str(tmp_path / "observations_report.json"),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_traffic_diagnostics.py"),
            "--observations",
            str(observations),
            "--output",
            str(html_path),
            "--summary",
            str(summary_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert "traffic_diagnostics=ok" in result.stdout
    assert html_path.exists()
    assert summary["transition_count"] == 5
    assert summary["dashboard_schema_version"] == "traffic_diagnostics_v1"
