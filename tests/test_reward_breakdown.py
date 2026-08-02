from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from worldsimflow import PolicyEvaluationConfig, PolicyEvaluator, ScenarioLoader, TransitionDatasetExporter, TransitionExportConfig, WorldSimFlowEnv
from worldsimflow.core.reward import compute_reward_breakdown
from worldsimflow.backends import ReplayBackend
from worldsimflow.core.types import Action


def simple_policy(obs):
    return {"acceleration": 0.0, "steering": 0.0}


def test_env_step_exposes_reward_breakdown_and_cost_info():
    scenario = ScenarioLoader().load(ROOT / "data" / "sample_scenario.json")
    env = WorldSimFlowEnv(scenario, max_steps=3)
    obs, _ = env.reset(seed=1)
    next_obs, reward, terminated, truncated, info = env.step({"acceleration": 0.0, "steering": 0.0})

    breakdown = info["reward_breakdown"]
    cost = info["cost_info"]
    expected_reward = round(sum(value for key, value in breakdown.items() if key != "total"), 8)

    assert breakdown["total"] == expected_reward
    assert reward == breakdown["total"]
    assert set(["speed_reward", "lane_penalty", "gap_penalty", "collision_penalty", "offroad_penalty"]).issubset(breakdown)
    assert set(["collision_cost", "offroad_cost", "close_gap_cost", "total_cost"]).issubset(cost)
    assert cost["total_cost"] >= 0.0
    assert terminated is False
    assert truncated is False
    env.close()


def test_transition_dataset_records_reward_breakdown_and_cost_info(tmp_path):
    scenario = ScenarioLoader().load(ROOT / "data" / "sample_scenario.json")
    output = tmp_path / "observations.jsonl"
    TransitionDatasetExporter(
        scenario,
        TransitionExportConfig(episodes=1, max_steps=3, seed=5, include_full_observation=False),
    ).export(output, simple_policy)
    first = json.loads(output.read_text(encoding="utf-8").splitlines()[0])

    assert "reward_breakdown" in first
    assert "cost_info" in first
    assert first["reward"] == first["reward_breakdown"]["total"]
    assert first["cost_info"]["total_cost"] >= 0.0


def test_policy_evaluator_summarizes_reward_breakdown_and_cost_info(tmp_path):
    report = PolicyEvaluator(
        PolicyEvaluationConfig(
            roots=[ROOT / "data" / "sample_scenario.json", ROOT / "data" / "worldsimflow_mini_straight.json"],
            output_dir=tmp_path,
            policies=["rule", "random"],
            episodes=1,
            max_steps=4,
            seed=9,
        )
    ).evaluate()
    rule_summary = report["summary"]["by_policy"]["rule"]
    first_result = report["scenario_results"][0]
    html = (tmp_path / "policy_eval_dashboard.html").read_text(encoding="utf-8")

    assert "reward_breakdown_totals" in rule_summary
    assert "cost_totals" in rule_summary
    assert "speed_reward" in rule_summary["reward_breakdown_totals"]
    assert "total_cost" not in rule_summary["cost_totals"]
    assert "reward_breakdown_totals" in first_result
    assert "cost_totals" in first_result
    assert "reward breakdown" in html
    assert "cost info" in html


def test_compute_reward_breakdown_matches_existing_reward_formula():
    scenario = ScenarioLoader().load(ROOT / "data" / "sample_scenario.json")
    backend = ReplayBackend(scenario)
    backend.reset()
    result = backend.step(Action(acceleration=0.0, steering=0.0))
    breakdown, cost = compute_reward_breakdown(result.observation, result.events)

    assert round(result.reward, 8) == breakdown.total
    assert result.info["cost_info"]["total_cost"] == cost.total_cost
