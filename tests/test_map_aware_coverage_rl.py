from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from worldsimflow import DeterministicFlowController, LaneGraph, LaneKeepPolicy, ReplayBackend, ScenarioLoader, TrafficManagerConfig, TrafficManagerLite, WorldSimFlowEnv
from worldsimflow.core.coverage import ScenarioCoverageAnalyzer
from worldsimflow.core.metrics import summarize_trace
from worldsimflow.core.scenario_data_manager import ScenarioDataManager
from worldsimflow.core.traffic_policy import MapAwareTrafficConfig, TrafficPolicyRunner


def test_map_aware_idm_rebuilds_vehicle_traffic_and_runs():
    base = ScenarioLoader().load(ROOT / "data" / "converted" / "openlog_waymo_2a1e44d405a6833f.json")
    scenario = TrafficPolicyRunner().rebuild_map_aware_idm_traffic(
        base,
        MapAwareTrafficConfig(max_actor_count=3, desired_speed=7.0, lane_snap_ratio=0.35),
    )

    assert scenario.metadata["traffic_mode"] == "lane_aware_idm"
    assert "map_aware_idm" in scenario.metadata["tags"]
    assert "lane_aware_idm" in scenario.metadata["tags"]
    assert scenario.metadata["traffic_policies"][-1]["kind"] == "lane_aware_idm"
    assert scenario.metadata["traffic_policies"][-1]["lane_graph"]["lane_count"] > 0
    assert len(scenario.actors) == len(base.actors)

    flow = DeterministicFlowController(scenario, LaneKeepPolicy(), backend=ReplayBackend(scenario))
    trace = flow.run(30)
    metrics = summarize_trace(trace)

    assert metrics.steps == 30
    assert flow.final_trace_hash()




def test_traffic_manager_lite_hybrid_controls_selected_vehicle_subset():
    base = ScenarioLoader().load(ROOT / "data" / "converted" / "openlog_waymo_2a1e44d405a6833f.json")
    vehicle_ids = tuple(actor.actor_id for actor in base.actors if actor.states[0].object_type == "VEHICLE")[:2]
    scenario = TrafficManagerLite().build_scenario(
        base,
        TrafficManagerConfig(
            mode="hybrid",
            reactive_actor_ids=vehicle_ids,
            desired_speed=7.5,
            lane_snap_ratio=0.35,
        ),
    )

    assert scenario.metadata["traffic_manager_mode"] == "hybrid"
    assert tuple(scenario.metadata["traffic_manager"]["reactive_actor_ids"]) == vehicle_ids
    assert scenario.metadata["traffic_mode"] == "lane_aware_idm"
    assert len(scenario.actors) == len(base.actors)

    graph = LaneGraph.from_scenario(scenario)
    controlled = next(actor for actor in scenario.actors if actor.actor_id == vehicle_ids[0])
    assert graph.project_state(controlled.states[10]) is not None

    flow = DeterministicFlowController(scenario, LaneKeepPolicy(), backend=ReplayBackend(scenario))
    trace = flow.run(20)
    assert len(trace) == 20
    assert flow.final_trace_hash()

def test_coverage_report_combines_index_and_run_results():
    index = ScenarioDataManager().scan([ROOT / "data" / "sample_scenario.json"])
    report = ScenarioCoverageAnalyzer().from_index_and_runs(
        index,
        [{"scenario_id": "straight_close_follow_001", "done": True, "events": {"collision": 2}}],
    )

    assert report.scenario_count == 1
    assert report.source_coverage == {"synthetic_log": 1}
    assert report.actor_type_coverage == {"VEHICLE": 2}
    assert report.event_coverage == {"collision": 2}
    assert report.failure_count == 1
    assert report.recommendations


def test_worldsimflow_env_reset_step_contract():
    scenario = ScenarioLoader().load(ROOT / "data" / "sample_scenario.json")
    env = WorldSimFlowEnv(scenario, max_steps=5)

    obs, info = env.reset()
    next_obs, reward, terminated, truncated, step_info = env.step({"acceleration": 0.0, "steering": 0.0})

    assert info["scenario_id"] == scenario.scenario_id
    assert obs["ego"] is not None
    assert isinstance(reward, float)
    assert terminated is False
    assert truncated is False
    assert "trace_hash" in step_info
    assert next_obs["step"] == 0
    env.close()
