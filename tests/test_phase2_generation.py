from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from worldsimflow import DeterministicFlowController, LaneKeepPolicy, ScenarioLoader
from worldsimflow.core.metrics import summarize_trace
from worldsimflow.core.scenario_generation import InterventionSpec, ScenarioInterventionEngine, scenario_to_dict


def test_phase2_suite_generates_expected_variants():
    scenario = ScenarioLoader().load(ROOT / "data" / "sample_scenario.json")
    engine = ScenarioInterventionEngine()

    variants = engine.generate_suite(scenario, seed=1234)
    kinds = [item.metadata["mutation"]["kind"] for item in variants]

    assert kinds == ["hard_brake", "cut_in", "close_follow", "pedestrian_crossing"]
    assert all(item.metadata["phase2"] is True for item in variants)
    assert all(len(item.actors) >= len(scenario.actors) for item in variants)


def test_phase2_generation_is_deterministic():
    scenario = ScenarioLoader().load(ROOT / "data" / "sample_scenario.json")
    engine = ScenarioInterventionEngine()
    spec = InterventionSpec("pedestrian_crossing", seed=77)

    first = scenario_to_dict(engine.apply(scenario, spec))
    second = scenario_to_dict(engine.apply(scenario, spec))

    assert first == second


def test_phase2_variants_run_in_replay_backend():
    scenario = ScenarioLoader().load(ROOT / "data" / "sample_scenario.json")
    engine = ScenarioInterventionEngine()

    for variant in engine.generate_suite(scenario, seed=222):
        flow = DeterministicFlowController(variant, LaneKeepPolicy())
        trace = flow.run(80)
        metrics = summarize_trace(trace)
        assert metrics.steps > 0
        assert flow.final_trace_hash()


def test_pedestrian_crossing_adds_pedestrian_actor():
    scenario = ScenarioLoader().load(ROOT / "data" / "sample_scenario.json")
    engine = ScenarioInterventionEngine()
    variant = engine.apply(scenario, InterventionSpec("pedestrian_crossing", seed=9))

    object_types = [actor.states[0].object_type for actor in variant.actors]

    assert "PEDESTRIAN" in object_types
    assert variant.metadata["mutation"]["kind"] == "pedestrian_crossing"