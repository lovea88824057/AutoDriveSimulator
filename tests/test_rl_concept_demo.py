from pathlib import Path
import importlib.util
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_rl_concept_demo.py"


def load_demo_module():
    spec = importlib.util.spec_from_file_location("run_rl_concept_demo", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_discretize_observation_uses_speed_gap_lane_buckets():
    demo = load_demo_module()
    state = demo.discretize_observation({"ego_speed": 7.0, "front_gap": 10.0, "ego_lane_l": 0.1})
    assert state == "speed=target|gap=close|lane=center"


def test_rl_concept_demo_script_runs_and_writes_report(tmp_path):
    output = tmp_path / "rl_concept_report.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--scenario",
            "data/sample_scenario.json",
            "--episodes",
            "3",
            "--steps",
            "8",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    report = json.loads(output.read_text(encoding="utf-8"))

    assert "rl_concept_demo=ok" in result.stdout
    assert report["demo"] == "q_learning_concept"
    assert report["observation_schema_version"] == "state_v1"
    assert report["feature_count"] == 17
    assert len(report["episodes_detail"]) == 3
    assert report["learned_policy"]
