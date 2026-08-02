from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from worldsimflow import PolicyEvaluationConfig, PolicyEvaluator, ScenarioLoader, WorldSimFlowEnv, make_evaluation_policy


def test_evaluation_policies_act_within_env_action_space():
    scenario = ScenarioLoader().load(ROOT / "data" / "sample_scenario.json")
    env = WorldSimFlowEnv(scenario, max_steps=3)
    obs, _info = env.reset(seed=1)
    for name in ["rule", "random", "minimal-q"]:
        policy = make_evaluation_policy(name, seed=123)
        action = policy.act(obs)
        assert env.contains_action(action.action)
        assert action.name
    env.close()


def test_policy_evaluator_compares_multiple_policies(tmp_path):
    report = PolicyEvaluator(
        PolicyEvaluationConfig(
            roots=[ROOT / "data" / "sample_scenario.json", ROOT / "data" / "worldsimflow_mini_straight.json"],
            output_dir=tmp_path,
            policies=["rule", "random", "minimal-q"],
            episodes=1,
            max_steps=5,
            seed=42,
        )
    ).evaluate()
    saved = json.loads((tmp_path / "policy_eval_report.json").read_text(encoding="utf-8"))

    assert report["policy_evaluation"] == "ok"
    assert report["scenario_count"] == 2
    assert report["policy_count"] == 3
    assert report["episode_count"] == 6
    assert len(report["scenario_results"]) == 6
    assert set(report["summary"]["by_policy"]) == {"rule", "random", "minimal-q"}
    assert len(report["summary"]["ranking"]) == 3
    assert (tmp_path / "policy_eval_dashboard.html").exists()
    assert saved["evaluation_schema_version"] == "policy_eval_v2"
    first = saved["scenario_results"][0]
    assert "done_reason_counts" in first
    assert "success_count" in first
    assert "success_rate" in first
    assert "done_reason_counts" in saved["summary"]["by_policy"]["rule"]
    html = (tmp_path / "policy_eval_dashboard.html").read_text(encoding="utf-8")
    assert "policyFilter" in html
    assert "Done Reason" in html
    assert "????" not in html
    assert "\ufffd" not in html


def test_evaluate_policies_script_runs(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "evaluate_policies.py"),
            "--roots",
            "data/sample_scenario.json",
            "data/worldsimflow_mini_straight.json",
            "--policies",
            "rule",
            "random",
            "minimal-q",
            "--episodes",
            "1",
            "--steps",
            "4",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    report = json.loads((tmp_path / "policy_eval_report.json").read_text(encoding="utf-8"))

    assert "policy_evaluation=ok" in result.stdout
    assert report["policy_count"] == 3
    assert report["scenario_count"] == 2
    assert len(report["summary"]["ranking"]) == 3


def test_policy_evaluator_counts_route_goal_success(tmp_path):
    report = PolicyEvaluator(
        PolicyEvaluationConfig(
            roots=[ROOT / "data" / "route_goal_success_demo.json"],
            output_dir=tmp_path,
            policies=["rule"],
            episodes=2,
            max_steps=30,
            seed=12,
        )
    ).evaluate()
    result = report["scenario_results"][0]
    summary = report["summary"]["by_policy"]["rule"]

    assert result["done_reason_counts"] == {"success": 2}
    assert result["success_count"] == 2
    assert result["success_rate"] == 1.0
    assert summary["success_count"] == 2
    assert summary["success_rate"] == 1.0
