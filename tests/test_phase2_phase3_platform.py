from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from worldsimflow.core.batch_runner import BatchRunner
from worldsimflow.core.message_sync import run_sync_demo
from worldsimflow.core.procedural_scenario_generator import ProceduralScenarioConfig, ProceduralScenarioGenerator
from worldsimflow.core.scenario import ScenarioLoader
from worldsimflow.core.scenario_diff import ScenarioDiff
from worldsimflow.core.scenario_generation import InterventionSpec, ScenarioInterventionEngine, write_scenario_json


def test_scenario_diff_detects_added_pedestrian():
    base = ScenarioLoader().load(ROOT / "data" / "sample_scenario.json")
    variant = ScenarioInterventionEngine().apply(base, InterventionSpec("pedestrian_crossing", seed=42))
    report = ScenarioDiff().compare(base, variant)

    assert report.added_actors == ["phase2_pedestrian_42"]
    assert report.removed_actors == []
    assert report.changed_actor_count == 0


def test_batch_runner_runs_generated_variant(tmp_path):
    base = ScenarioLoader().load(ROOT / "data" / "sample_scenario.json")
    variant = ScenarioInterventionEngine().apply(base, InterventionSpec("close_follow", seed=43))
    scenario_path = write_scenario_json(variant, tmp_path / "variant.json")

    results = BatchRunner().run_paths([scenario_path], steps=20, html_dir=tmp_path / "html")

    assert len(results) == 1
    assert results[0].steps == 20
    assert results[0].html is not None


def test_procedural_generator_produces_replayable_scenario():
    scenario = ProceduralScenarioGenerator().generate(ProceduralScenarioConfig(map_name="C", seed=7, vehicle_count=3, max_steps=30))

    assert scenario.scenario_id == "procedural_C_7"
    assert len(scenario.actors) == 3
    assert scenario.metadata["procedural_generator"]["map"] == "C"


def test_phase3_sync_success_and_failure_modes():
    success = run_sync_demo("success")
    missing = run_sync_demo("missing")
    stale = run_sync_demo("stale")
    out_of_order = run_sync_demo("out_of_order")

    assert success["summary"]["ok"] is True
    assert success["summary"]["issue_counts"] == {}
    assert missing["summary"]["ok"] is True
    assert missing["summary"]["issue_counts"] == {"reused_previous": 1}
    assert stale["summary"]["ok"] is False
    assert stale["summary"]["issue_counts"]["stale_message"] == 1
    assert out_of_order["summary"]["ok"] is False
    assert out_of_order["summary"]["issue_counts"]["out_of_order"] == 1