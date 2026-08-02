from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from worldsimflow import ScenarioLoader, TransitionDatasetExporter, TransitionExportConfig


def steady_policy(_obs):
    return {"acceleration": 0.0, "steering": 0.0}


def test_transition_exporter_writes_compact_bev_history(tmp_path):
    scenario = ScenarioLoader().load(ROOT / "data" / "sample_scenario.json")
    output = tmp_path / "bev_observations.jsonl"
    report = TransitionDatasetExporter(
        scenario,
        TransitionExportConfig(
            episodes=1,
            max_steps=3,
            seed=31,
            include_full_observation=False,
            include_bev=True,
            bev_history_length=4,
            bev_width=16,
            bev_height=12,
            bev_meters_per_pixel=2.0,
        ),
    ).export(output, steady_policy)
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    first = rows[0]

    assert report["include_bev"] is True
    assert report["bev_history_length"] == 4
    assert report["bev_observation_space_spec"]["shape"] == [5, 12, 16]
    assert first["obs"]["bev"]["shape"] == [5, 12, 16]
    assert first["obs"]["bev_history"]["shape"] == [4, 5, 12, 16]
    assert first["next_obs"]["bev_history"]["shape"] == [4, 5, 12, 16]
    assert first["obs"]["normalized_vector"]


def test_export_observations_jsonl_script_supports_bev_history(tmp_path):
    output = tmp_path / "observations.jsonl"
    report_path = tmp_path / "report.json"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "export_observations_jsonl.py"),
            "--scenario",
            "data/sample_scenario.json",
            "--episodes",
            "1",
            "--steps",
            "2",
            "--compact",
            "--include-bev",
            "--bev-history-length",
            "3",
            "--bev-width",
            "12",
            "--bev-height",
            "12",
            "--output",
            str(output),
            "--report",
            str(report_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    row = json.loads(output.read_text(encoding="utf-8").splitlines()[0])

    assert "observations_jsonl=ok" in result.stdout
    assert "include_bev=True" in result.stdout
    assert report["include_bev"] is True
    assert report["bev_history_length"] == 3
    assert row["obs"]["bev_history"]["shape"] == [3, 5, 12, 12]


def test_batch_transition_exporter_reports_bev_history_coverage(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "export_batch_transitions.py"),
            "--roots",
            "data/sample_scenario.json",
            "data/route_goal_success_demo.json",
            "--episodes",
            "1",
            "--steps",
            "2",
            "--compact",
            "--include-bev",
            "--bev-history-length",
            "2",
            "--bev-width",
            "10",
            "--bev-height",
            "10",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    coverage = json.loads((tmp_path / "coverage_report.json").read_text(encoding="utf-8"))
    manifest = json.loads((tmp_path / "dataset_manifest.json").read_text(encoding="utf-8"))

    assert "batch_transition_export=ok" in result.stdout
    assert "include_bev=True" in result.stdout
    assert manifest["include_bev"] is True
    assert manifest["bev_history_length"] == 2
    assert coverage["bev_complete"] is True
    assert coverage["bev_history_complete"] is True
    assert coverage["bev_history_transition_count"] == coverage["transition_count"]
