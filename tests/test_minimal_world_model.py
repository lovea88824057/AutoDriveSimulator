from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from worldsimflow import MinimalWorldModelTrainer, ScenarioLoader, TransitionDatasetExporter, TransitionExportConfig, WorldModelConfig


def steady_policy(_obs):
    return {"acceleration": 0.0, "steering": 0.0}


def make_dataset(tmp_path):
    scenario = ScenarioLoader().load(ROOT / "data" / "sample_scenario.json")
    output = tmp_path / "bev_world_model_samples.jsonl"
    TransitionDatasetExporter(
        scenario,
        TransitionExportConfig(
            episodes=1,
            max_steps=5,
            seed=44,
            include_full_observation=False,
            include_bev=True,
            bev_history_length=3,
            bev_width=12,
            bev_height=12,
        ),
    ).export(output, steady_policy)
    return output


def test_minimal_world_model_trainer_loads_and_trains(tmp_path):
    dataset = make_dataset(tmp_path)
    trainer = MinimalWorldModelTrainer(WorldModelConfig(epochs=4, learning_rate=0.02, seed=1))
    samples = trainer.load_samples(dataset)
    report = trainer.train(samples)

    assert report["world_model_demo"] == "ok"
    assert report["sample_count"] == 5
    assert report["history_shape"] == [3, 5, 12, 12]
    assert report["input_contract"] == "bev_history_t + action_t -> next_bev_history_t_plus_1"
    assert report["final_eval"]["mse"] >= 0.0
    assert 0.0 <= report["final_eval"]["occupancy_iou"] <= 1.0
    assert report["examples"]
    assert report["examples"][0]["predicted_next_frame"]


def test_minimal_world_model_script_runs(tmp_path):
    dataset = make_dataset(tmp_path)
    output = tmp_path / "world_model_report.json"
    html = tmp_path / "world_model.html"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_minimal_world_model_demo.py"),
            "--dataset",
            str(dataset),
            "--epochs",
            "4",
            "--learning-rate",
            "0.02",
            "--output",
            str(output),
            "--html",
            str(html),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    page = html.read_text(encoding="utf-8")

    assert "minimal_world_model=ok" in result.stdout
    assert report["world_model_demo"] == "ok"
    assert report["input_contract"] == "bev_history_t + action_t -> next_bev_history_t_plus_1"
    assert "Last Input BEV" in page
    assert "Predicted Next BEV" in page
    assert "????" not in page
