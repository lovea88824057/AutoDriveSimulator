from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from worldsimflow import ScenarioLoader, WorldSimFlowEnv, infer_done_reason


def test_infer_done_reason_prioritizes_failure_events():
    done_reason, success_info = infer_done_reason(
        terminated=True,
        truncated=False,
        event_codes=["stale_replay", "collision"],
        step_id=12,
        max_steps=20,
        scenario_max_steps=120,
    )

    assert done_reason.reason == "collision"
    assert done_reason.success is False
    assert success_info.failure is True
    assert success_info.failure_code == "collision"


def test_env_reports_max_steps_done_reason_and_success_info():
    scenario = ScenarioLoader().load(ROOT / "data" / "worldsimflow_mini_straight.json")
    env = WorldSimFlowEnv(scenario, max_steps=1)
    obs, _reset_info = env.reset(seed=3)
    _next_obs, _reward, terminated, truncated, info = env.step({"acceleration": 0.0, "steering": 0.0})

    assert terminated is False
    assert truncated is True
    assert info["done_reason"]["reason"] == "max_steps"
    assert info["success_info"]["success"] is False
    assert info["success_info"]["clean_end"] is True
    assert info["success_info"]["reached_eval_horizon"] is True
    env.close()


def test_env_reports_already_done_after_episode_end():
    scenario = ScenarioLoader().load(ROOT / "data" / "worldsimflow_mini_straight.json")
    env = WorldSimFlowEnv(scenario, max_steps=1)
    env.reset(seed=4)
    env.step({"acceleration": 0.0, "steering": 0.0})
    _obs, _reward, terminated, truncated, info = env.step({"acceleration": 0.0, "steering": 0.0})

    assert terminated is True
    assert truncated is False
    assert info["already_done"] is True
    assert info["done_reason"]["reason"] == "already_done"
    env.close()


def test_env_reports_route_goal_success():
    scenario = ScenarioLoader().load(ROOT / "data" / "route_goal_success_demo.json")
    env = WorldSimFlowEnv(scenario, max_steps=30)
    _obs, reset_info = env.reset(seed=5)

    assert reset_info["route_goal_spec"]["enabled"] is True
    seen_success = False
    for _ in range(30):
        _obs, _reward, terminated, truncated, info = env.step({"acceleration": 0.0, "steering": 0.0})
        if terminated or truncated:
            seen_success = True
            break

    assert seen_success is True
    assert terminated is True
    assert truncated is False
    assert info["done_reason"]["reason"] == "success"
    assert info["done_reason"]["success"] is True
    assert info["success_info"]["success"] is True
    assert info["success_info"]["route_goal_reached"] is True
    assert info["success_info"]["route_goal_info"]["reason"] == "route_goal_reached"
    assert info["route_goal"]["reached"] is True
    env.close()


def test_route_goal_does_not_turn_clean_horizon_into_success_when_disabled():
    scenario = ScenarioLoader().load(ROOT / "data" / "worldsimflow_mini_straight.json")
    env = WorldSimFlowEnv(scenario, max_steps=1)
    _obs, reset_info = env.reset(seed=6)
    _obs, _reward, terminated, truncated, info = env.step({"acceleration": 0.0, "steering": 0.0})

    assert reset_info["route_goal_spec"]["enabled"] is False
    assert terminated is False
    assert truncated is True
    assert info["done_reason"]["reason"] == "max_steps"
    assert info["success_info"]["success"] is False
    assert info["route_goal"]["enabled"] is False
    env.close()
