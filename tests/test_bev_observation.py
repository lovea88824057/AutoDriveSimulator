from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from worldsimflow import BEVRasterConfig, BEVRasterObservationBuilder, ScenarioLoader, WorldSimFlowEnv


def _count_channel(bev, name):
    idx = bev["channels"].index(name)
    return sum(sum(row) for row in bev["raster"][idx])


def test_bev_raster_builder_outputs_stable_shape_and_channels():
    scenario = ScenarioLoader().load(ROOT / "data" / "sample_scenario.json")
    builder = BEVRasterObservationBuilder(scenario, BEVRasterConfig(width=48, height=40, meters_per_pixel=1.0))
    raw = {"step": 0, "ego": scenario.ego, "actors": [actor.states[0] for actor in scenario.actors]}
    bev = builder.build(raw)

    assert bev["schema_version"] == "bev_raster_v1"
    assert bev["shape"] == [5, 40, 48]
    assert bev["channels"] == ["drivable", "lane_center", "ego", "actor_vehicle", "actor_vru"]
    assert _count_channel(bev, "drivable") > 0
    assert _count_channel(bev, "lane_center") > 0
    assert _count_channel(bev, "ego") > 0
    assert _count_channel(bev, "actor_vehicle") > 0


def test_env_can_include_bev_observation_without_breaking_state_vector():
    scenario = ScenarioLoader().load(ROOT / "data" / "sample_scenario.json")
    env = WorldSimFlowEnv(scenario, max_steps=2, include_bev=True, bev_config=BEVRasterConfig(width=32, height=32, meters_per_pixel=1.0))
    obs, reset_info = env.reset(seed=11)
    next_obs, _reward, _terminated, _truncated, info = env.step({"acceleration": 0.0, "steering": 0.0})

    assert len(obs["normalized_vector"]) == 17
    assert obs["bev"]["shape"] == [5, 32, 32]
    assert next_obs["bev"]["shape"] == [5, 32, 32]
    assert reset_info["bev_observation_space_spec"]["shape"] == [5, 32, 32]
    assert info["bev_observation_space_spec"]["shape"] == [5, 32, 32]
    env.close()


def test_env_bev_history_stack_rolls_forward():
    scenario = ScenarioLoader().load(ROOT / "data" / "sample_scenario.json")
    env = WorldSimFlowEnv(
        scenario,
        max_steps=4,
        include_bev=True,
        bev_config=BEVRasterConfig(width=16, height=16, meters_per_pixel=2.0),
        bev_history_length=3,
    )
    obs, _reset_info = env.reset(seed=21)
    assert obs["bev_history"]["shape"] == [3, 5, 16, 16]
    assert obs["bev_history"]["frame_steps"] == [0, 0, 0]

    obs, _reward, _terminated, _truncated, _info = env.step({"acceleration": 0.0, "steering": 0.0})
    assert obs["bev_history"]["shape"] == [3, 5, 16, 16]
    assert obs["bev_history"]["frame_steps"] == [0, 0, 0]

    obs, _reward, _terminated, _truncated, _info = env.step({"acceleration": 0.0, "steering": 0.0})
    assert obs["bev_history"]["frame_steps"] == [0, 0, 1]
    env.close()
