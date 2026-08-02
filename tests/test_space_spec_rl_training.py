from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from worldsimflow import BoxSpaceSpec, ScenarioLoader, WorldSimFlowEnv


def test_box_space_spec_sample_contains_clip_and_normalize():
    spec = BoxSpaceSpec(name="toy", shape=(2,), low=[-2.0, 0.0], high=[2.0, 10.0], labels=["x", "y"])
    sample = spec.sample(seed=123)

    assert spec.contains(sample)
    assert spec.contains(spec.clip([-5.0, 20.0]))
    assert spec.clip([-5.0, 20.0]) == [-2.0, 10.0]
    assert spec.normalize([-2.0, 10.0]) == [-1.0, 1.0]
    assert spec.denormalize([-1.0, 1.0]) == [-2.0, 10.0]
    assert spec.to_dict()["shape"] == [2]


def test_env_exposes_space_specs_and_normalized_vector():
    scenario = ScenarioLoader().load(ROOT / "data" / "sample_scenario.json")
    env = WorldSimFlowEnv(scenario, max_steps=4)
    obs, reset_info = env.reset(seed=7)
    action = env.sample_action(seed=7)
    next_obs, reward, terminated, truncated, info = env.step(action)

    assert reset_info["observation_space_spec"]["shape"] == [17]
    assert reset_info["action_space_spec"]["shape"] == [2]
    assert len(obs["normalized_vector"]) == 17
    assert len(next_obs["normalized_vector"]) == 17
    assert all(-1.000001 <= value <= 1.000001 for value in next_obs["normalized_vector"])
    assert env.contains_action(action)
    assert env.clip_action({"acceleration": -99.0, "steering": 9.0}) == {"acceleration": -6.0, "steering": 1.0}
    assert info["action_within_space"] is True
    assert isinstance(reward, float)
    assert terminated is False
    assert truncated is False
    env.close()


def test_minimal_rl_training_script_runs_with_space_specs(tmp_path):
    output = tmp_path / "minimal_rl_training_report.json"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_minimal_rl_training.py"),
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

    assert "minimal_rl_training=ok" in result.stdout
    assert report["demo"] == "minimal_normalized_q_learning"
    assert report["observation_space_spec"]["shape"] == [17]
    assert report["action_space_spec"]["shape"] == [2]
    assert report["all_actions_valid"] is True
    assert report["all_normalized_values_in_range"] is True
    assert len(report["episode_rewards"]) == 3
