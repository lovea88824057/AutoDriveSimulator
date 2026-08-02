from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from worldsimflow import ObservationBuilder, ScenarioLoader, WorldSimFlowEnv
from worldsimflow.backends import ReplayBackend
from worldsimflow.core.types import Action


def test_observation_builder_outputs_stable_state_schema():
    scenario = ScenarioLoader().load(ROOT / "data" / "sample_scenario.json")
    backend = ReplayBackend(scenario)
    builder = ObservationBuilder(scenario)

    raw0 = backend.reset()
    step_result = backend.step(Action(acceleration=0.0, steering=0.0))
    obs = builder.build(step_result.observation, previous_observation=raw0, last_action=Action(0.0, 0.0), events=step_result.events)

    assert obs["schema_version"] == "state_v1"
    assert obs["scenario_id"] == scenario.scenario_id
    assert obs["ego"] is not None
    assert obs["ego_lane_id"] is not None
    assert obs["front_gap"] is not None
    assert obs["front_actor_id"] == "lead_vehicle"
    assert obs["feature_names"] == builder.feature_names
    assert len(obs["state_vector"]) == len(builder.feature_names)
    assert obs["state_vector"][builder.feature_names.index("last_acceleration")] == 0.0


def test_worldsimflow_env_uses_observation_builder_contract():
    scenario = ScenarioLoader().load(ROOT / "data" / "sample_scenario.json")
    env = WorldSimFlowEnv(scenario, max_steps=5)

    obs, info = env.reset()
    next_obs, reward, terminated, truncated, step_info = env.step({"acceleration": 0.0, "steering": 0.0})

    assert info["observation_schema_version"] == "state_v1"
    assert step_info["observation_schema_version"] == "state_v1"
    assert obs["schema_version"] == "state_v1"
    assert next_obs["schema_version"] == "state_v1"
    assert next_obs["feature_names"] == env.feature_names
    assert len(next_obs["state_vector"]) == len(env.feature_names)
    assert next_obs["ego"] is not None
    assert "lane_center_offset" in next_obs
    assert isinstance(reward, float)
    assert terminated is False
    assert truncated is False
    env.close()
