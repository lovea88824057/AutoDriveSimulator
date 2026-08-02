from pathlib import Path
import math
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from worldsimflow.core.lane_graph import LaneGraph
from worldsimflow.core.scenario import ScenarioLoader


def test_lane_graph_fallback_projects_sample_scenario():
    scenario = ScenarioLoader().load(ROOT / "data" / "sample_scenario.json")
    graph = LaneGraph.from_scenario(scenario)

    assert graph.lanes
    projection = graph.project_state(scenario.ego)

    assert projection is not None
    assert projection.distance < scenario.road.lane_width


def test_lane_graph_derives_nuscenes_lanes_from_boundaries():
    scenario = ScenarioLoader().load(ROOT / "data" / "converted" / "openlog_nuscenes_scene-0061.json")
    graph = LaneGraph.from_scenario(scenario)
    actor = next(actor for actor in scenario.actors if actor.actor_id == "bc38961ca0ac4b14ab90e547ba79fbb6")
    projection = graph.project_state(actor.states[0])

    assert graph.to_summary()["sources"].get("derived_from_boundaries", 0) > 0
    assert projection is not None
    assert projection.distance < 3.5


def test_lane_graph_sample_project_roundtrip():
    scenario = ScenarioLoader().load(ROOT / "data" / "converted" / "openlog_nuscenes_scene-0061.json")
    graph = LaneGraph.from_scenario(scenario)
    lane = max(graph.driving_lanes, key=lambda item: item.length)
    pose = graph.sample_pose(lane.lane_id, lane.length * 0.4, l=0.7)
    projection = graph.project(pose.x, pose.y)

    assert projection is not None
    assert projection.lane_id == lane.lane_id
    assert abs(projection.s - pose.s) < 1.0
    assert abs(projection.l - pose.l) < 0.2
    assert abs((projection.heading - pose.heading + math.pi) % (2.0 * math.pi) - math.pi) < 1e-6
