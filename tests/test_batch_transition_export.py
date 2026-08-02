from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from worldsimflow import BatchTransitionExportConfig, BatchTransitionExporter


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_batch_transition_exporter_writes_dataset_manifest_and_coverage(tmp_path):
    output_dir = tmp_path / "dataset"
    report = BatchTransitionExporter(
        BatchTransitionExportConfig(
            roots=[ROOT / "data" / "sample_scenario.json", ROOT / "data" / "worldsimflow_mini_straight.json"],
            output_dir=output_dir,
            episodes=1,
            max_steps=4,
            seed=123,
            policy="rule",
            include_full_observation=False,
        )
    ).export()

    observations = output_dir / "observations.jsonl"
    manifest = json.loads((output_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    coverage = json.loads((output_dir / "coverage_report.json").read_text(encoding="utf-8"))
    rows = read_jsonl(observations)
    first = rows[0]

    assert report["batch_transition_export"] == "ok"
    assert report["scenario_count"] == 2
    assert report["variant_count"] == 2
    assert report["transition_count"] == 8
    assert report["feature_count"] == 17
    assert observations.exists()
    assert manifest["variant_count"] == 2
    assert manifest["transition_count"] == 8
    assert manifest["observation_space_spec"]["shape"] == [17]
    assert manifest["action_space_spec"]["shape"] == [2]
    assert coverage["transition_count"] == 8
    assert coverage["normalized_vector_complete"] is True
    assert "variant_id" in first
    assert first["policy_name"] == "rule"
    assert first["batch_dataset_schema_version"] == "batch_transition_v1"
    assert len(first["obs"]["normalized_vector"]) == 17
    assert len(first["next_obs"]["normalized_vector"]) == 17


def test_export_batch_transitions_script_runs(tmp_path):
    output_dir = tmp_path / "script_dataset"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "export_batch_transitions.py"),
            "--roots",
            "data/sample_scenario.json",
            "data/worldsimflow_mini_straight.json",
            "--episodes",
            "1",
            "--steps",
            "3",
            "--compact",
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    report = json.loads((output_dir / "export_report.json").read_text(encoding="utf-8"))
    coverage = json.loads((output_dir / "coverage_report.json").read_text(encoding="utf-8"))

    assert "batch_transition_export=ok" in result.stdout
    assert report["transition_count"] == 6
    assert report["variant_count"] == 2
    assert coverage["normalized_vector_complete"] is True
    assert (output_dir / "observations.jsonl").exists()
