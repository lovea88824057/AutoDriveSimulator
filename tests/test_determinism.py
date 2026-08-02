from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from worldsimflow import DeterministicFlowController, LaneKeepPolicy, ScenarioLoader
from worldsimflow.core.metrics import summarize_trace


def test_same_scenario_has_same_trace_hash():
    scenario = ScenarioLoader().load(ROOT / "data" / "sample_scenario.json")
    first = DeterministicFlowController(scenario, LaneKeepPolicy())
    second = DeterministicFlowController(scenario, LaneKeepPolicy())

    first.run(120)
    second.run(120)

    assert first.final_trace_hash() == second.final_trace_hash()


def test_default_scenario_runs_full_phase1_length_without_events():
    scenario = ScenarioLoader().load(ROOT / "data" / "sample_scenario.json")
    flow = DeterministicFlowController(scenario, LaneKeepPolicy())

    trace = flow.run(120)
    metrics = summarize_trace(trace)

    assert metrics.steps == 120
    assert metrics.done is False
    assert metrics.event_counts == {}
    assert metrics.min_front_gap is not None
