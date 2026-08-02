from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from worldsimflow import DeterministicFlowController, LaneKeepPolicy, ReplayBackend, ScenarioLoader
from worldsimflow.core.metrics import summarize_trace
from worldsimflow.core.scenario_data_manager import ScenarioDataManager
from worldsimflow.core.scenario_generation import write_scenario_json
from worldsimflow.core.traffic_policy import IDMConfig, TrafficPolicyRunner


def test_scenario_data_manager_indexes_core_fields(tmp_path):
    index = ScenarioDataManager().scan([ROOT / "data" / "sample_scenario.json"])

    assert len(index.records) == 1
    record = index.records[0]
    assert record.scenario_id == "straight_close_follow_001"
    assert record.source == "synthetic_log"
    assert record.actor_count == 2
    assert record.actor_types == {"VEHICLE": 2}
    assert index.summary["sources"] == {"synthetic_log": 1}


def test_loader_accepts_top_level_map_schema(tmp_path):
    path = tmp_path / "map_schema.json"
    data = json.loads((ROOT / "data" / "sample_scenario.json").read_text(encoding="utf-8"))
    data["map_features"] = [
        {"feature_id": "left_edge", "type": "ROAD_EDGE_BOUNDARY", "polyline": [[0.0, -2.0], [30.0, -2.0]]},
        {"feature_id": "right_edge", "type": "ROAD_EDGE_BOUNDARY", "polyline": [[0.0, 2.0], [30.0, 2.0]]},
    ]
    data["drivable_area"] = {"min_x": -1.0, "max_x": 30.0, "min_y": -2.0, "max_y": 2.0, "polygons": []}
    path.write_text(json.dumps(data), encoding="utf-8")

    scenario = ScenarioLoader().load(path)

    assert len(scenario.map_features) == 2
    assert scenario.drivable_area is not None
    assert scenario.drivable_area.contains(0.0, 0.0)


def test_map_aware_offroad_uses_drivable_area(tmp_path):
    path = tmp_path / "offroad.json"
    data = json.loads((ROOT / "data" / "sample_scenario.json").read_text(encoding="utf-8"))
    data["ego"] = {"x": 0.0, "y": 4.0, "yaw": 0.0, "speed": 0.0}
    data["actors"] = []
    data["max_steps"] = 5
    data["drivable_area"] = {"min_x": -5.0, "max_x": 50.0, "min_y": -2.0, "max_y": 2.0, "polygons": []}
    path.write_text(json.dumps(data), encoding="utf-8")

    scenario = ScenarioLoader().load(path)
    flow = DeterministicFlowController(scenario, LaneKeepPolicy(), backend=ReplayBackend(scenario))
    trace = flow.run(5)
    metrics = summarize_trace(trace)

    assert metrics.done is True
    assert metrics.event_counts == {"offroad": 1}


def test_idm_lite_adds_replayable_reactive_actor():
    base = ScenarioLoader().load(ROOT / "data" / "sample_scenario.json")
    scenario = TrafficPolicyRunner().add_idm_follower(
        base,
        actor_id="test_idm_follower",
        config=IDMConfig(initial_gap=18.0, desired_speed=8.0, min_gap=3.0),
    )

    follower = next(actor for actor in scenario.actors if actor.actor_id == "test_idm_follower")
    assert len(follower.states) == scenario.max_steps
    assert all(state.speed >= 0.0 for state in follower.states)
    assert scenario.metadata["traffic_policies"][0]["kind"] == "idm_lite_follower"

    flow = DeterministicFlowController(scenario, LaneKeepPolicy(), backend=ReplayBackend(scenario))
    trace = flow.run(30)

    assert len(trace) == 30
    assert flow.final_trace_hash()
