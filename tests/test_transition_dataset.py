from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from worldsimflow import ScenarioLoader, TransitionDatasetExporter, TransitionExportConfig


def simple_policy(obs):
    lane_l = float(obs.get("ego_lane_l") or 0.0)
    return {"acceleration": 0.0, "steering": max(-0.2, min(0.2, -lane_l * 0.1))}


def test_transition_dataset_exporter_writes_obs_action_next_obs_jsonl(tmp_path):
    scenario = ScenarioLoader().load(ROOT / "data" / "sample_scenario.json")
    output = tmp_path / "observations.jsonl"
    report = TransitionDatasetExporter(
        scenario,
        TransitionExportConfig(episodes=1, max_steps=6, seed=123, include_full_observation=False),
    ).export(output, simple_policy)

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    first = rows[0]

    assert report["transition_count"] == 6
    assert report["observation_schema_version"] == "state_v1"
    assert report["feature_count"] == 17
    assert first["record_type"] == "transition"
    assert first["dataset_schema_version"] == "transition_v1"
    assert set(["obs", "action", "reward", "next_obs", "terminated", "truncated", "event_codes"]).issubset(first)
    assert "done_reason" in first
    assert "success_info" in first
    assert first["done_reason"]["reason"] in {"running", "max_steps", "collision", "offroad", "stale_replay", "success"}
    assert first["obs"]["schema_version"] == "state_v1"
    assert len(first["obs"]["state_vector"]) == 17
    assert first["next_obs"]["schema_version"] == "state_v1"


def test_export_observations_jsonl_script_runs(tmp_path):
    output = tmp_path / "observations.jsonl"
    report = tmp_path / "observations.report.json"
    result = subprocess.run(
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
            str(output),
            "--report",
            str(report),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    summary = json.loads(report.read_text(encoding="utf-8"))

    assert "observations_jsonl=ok" in result.stdout
    assert output.exists()
    assert summary["transition_count"] == 5
    assert summary["observation_schema_version"] == "state_v1"
    assert summary["feature_count"] == 17
