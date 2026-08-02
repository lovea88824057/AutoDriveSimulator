import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from worldsimflow.core.lane_graph import LaneGraph
from worldsimflow.core.scenario import ScenarioLoader
from worldsimflow.core.scenario_diff import ScenarioDiff
from worldsimflow.core.target_intervention import TargetActorInterventionEngine, TargetInterventionSpec


def test_target_hard_brake_changes_only_selected_actor():
    scenario = ScenarioLoader().load(ROOT / "data" / "sample_scenario.json")
    variant = TargetActorInterventionEngine().apply(
        scenario,
        TargetInterventionSpec(
            kind="hard_brake",
            actor_id="lead_vehicle",
            start_step=20,
            seed=1,
            params={"deceleration": -5.0},
        ),
    )

    diff = ScenarioDiff().compare(scenario, variant)
    changed = {item.actor_id for item in diff.changed_actors}

    assert changed == {"lead_vehicle"}
    assert variant.metadata["mutation"]["target_actor_id"] == "lead_vehicle"
    before = next(actor for actor in scenario.actors if actor.actor_id == "lead_vehicle")
    after = next(actor for actor in variant.actors if actor.actor_id == "lead_vehicle")
    assert after.states[30].speed < before.states[30].speed


def test_target_speed_change_reaches_target_speed():
    scenario = ScenarioLoader().load(ROOT / "data" / "sample_scenario.json")
    variant = TargetActorInterventionEngine().apply(
        scenario,
        TargetInterventionSpec(
            kind="speed_change",
            actor_id="lead_vehicle",
            start_step=10,
            duration=10,
            seed=2,
            params={"target_speed": 2.5},
        ),
    )

    actor = next(actor for actor in variant.actors if actor.actor_id == "lead_vehicle")

    assert abs(actor.states[25].speed - 2.5) < 1e-6
    assert "speed_change" in variant.metadata["tags"]


def test_target_lateral_shift_is_deterministic():
    scenario = ScenarioLoader().load(ROOT / "data" / "sample_scenario.json")
    spec = TargetInterventionSpec(
        kind="lateral_shift",
        actor_id="cut_in_vehicle",
        start_step=8,
        duration=12,
        seed=3,
        params={"shift": -1.2},
    )

    first = TargetActorInterventionEngine().apply(scenario, spec)
    second = TargetActorInterventionEngine().apply(scenario, spec)
    first_actor = next(actor for actor in first.actors if actor.actor_id == "cut_in_vehicle")
    second_actor = next(actor for actor in second.actors if actor.actor_id == "cut_in_vehicle")

    assert first_actor.states == second_actor.states
    assert first_actor.states[25].y < scenario.actors[0].states[25].y


def test_intervention_lab_html_contains_canvas_actor_picker():
    sys.path.insert(0, str(ROOT / "scripts"))
    from build_intervention_lab import render
    from worldsimflow.core.scenario_generation import scenario_to_dict

    scenario = ScenarioLoader().load(ROOT / "data" / "sample_scenario.json")
    data = scenario_to_dict(scenario)
    data["source_path"] = str(ROOT / "data" / "sample_scenario.json")
    data["display_name"] = "sample synthetic log"

    html = render([data])

    assert "canvas.addEventListener('dblclick'" in html
    assert "function hitActorAt" in html
    assert "function selectActorById" in html
    assert "actorSelect.value = actorId" in html
    assert "前方" in html and "后方" in html


def test_nuscenes_cut_in_resyncs_heading_with_motion():
    scenario = ScenarioLoader().load(ROOT / "data" / "converted" / "openlog_nuscenes_scene-0061.json")
    actor_id = "bc38961ca0ac4b14ab90e547ba79fbb6"
    variant = TargetActorInterventionEngine().apply(
        scenario,
        TargetInterventionSpec(
            kind="cut_in",
            actor_id=actor_id,
            start_step=12,
            duration=24,
            seed=4,
            params={"target_lateral": 0.0},
        ),
    )

    assert variant.metadata["mutation"]["route_aware"] is True
    assert variant.metadata["mutation"]["route_aware_status"] == "ok"
    assert variant.metadata["mutation"]["target_lane_id"]

    graph = LaneGraph.from_scenario(scenario)
    actor = next(actor for actor in variant.actors if actor.actor_id == actor_id)
    post_cut_projection = graph.project_state(actor.states[40])
    assert post_cut_projection is not None
    assert abs(post_cut_projection.l) < 1.5

    max_error = 0.0
    for step in range(12, len(actor.states) - 1):
        current = actor.states[step]
        following = actor.states[step + 1]
        dx = following.x - current.x
        dy = following.y - current.y
        if math.hypot(dx, dy) < 1e-4:
            continue
        motion_yaw = math.atan2(dy, dx)
        error = abs((motion_yaw - current.yaw + math.pi) % (2.0 * math.pi) - math.pi)
        max_error = max(max_error, error)

    assert math.degrees(max_error) < 0.1