from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from worldsimflow import MinimalWorldModelTrainer, WorldModelConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tiny BEV world-model smoke demo: bev_history_t + action_t -> bev_history_t+1.")
    parser.add_argument("--dataset", default="outputs/all_results/datasets/worldsimflow_bev_history_v1/observations.jsonl")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output", default="outputs/all_results/world_model/minimal_world_model_report.json")
    parser.add_argument("--html", default="outputs/all_results/world_model/minimal_world_model_dashboard.html")
    args = parser.parse_args()

    dataset = resolve(args.dataset)
    trainer = MinimalWorldModelTrainer(
        WorldModelConfig(epochs=args.epochs, learning_rate=args.learning_rate, seed=args.seed)
    )
    samples = trainer.load_samples(dataset, max_samples=args.max_samples or None)
    report = trainer.train(samples)
    report["dataset"] = str(dataset)
    report["purpose"] = "Smoke test the world-model loop: BEV history plus action predicts the next BEV history."
    report["worldsimflow_design_note"] = {
        "similarity": "This learns a transition model from WorldSimFlow obs/action/next_obs samples and demonstrates the world-model data loop.",
        "difference": "This demo is dependency-free and tiny; it predicts lightweight BEV occupancy, not photorealistic RGB or full physics.",
    }

    output = resolve(args.output)
    html_output = resolve(args.html)
    output.parent.mkdir(parents=True, exist_ok=True)
    html_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    html_output.write_text(render_html(report), encoding="utf-8", newline="\n")

    final_eval = report["final_eval"]
    baseline = report["baseline_persistence_eval"]
    print("minimal_world_model=ok")
    print(f"dataset={dataset}")
    print(f"sample_count={report['sample_count']}")
    print(f"history_shape={report['history_shape']}")
    print(f"baseline_eval_mse={baseline['mse']}")
    print(f"final_eval_mse={final_eval['mse']}")
    print(f"final_eval_iou={final_eval['occupancy_iou']}")
    print(f"output={output}")
    print(f"html={html_output}")


def render_html(report: dict) -> str:
    payload = json.dumps(report, ensure_ascii=False, separators=(",", ":"))
    title = "WorldSimFlow Minimal World Model Demo"
    return """<!doctype html>
<html lang=\"zh-CN\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>__TITLE__</title>
  <style>
    body { margin:0; background:#f5f7fa; color:#172033; font:14px/1.5 Arial,'Microsoft YaHei',sans-serif; }
    header { background:#172033; color:white; padding:22px 28px; }
    h1 { margin:0; font-size:24px; }
    header p { margin:6px 0 0; color:#c7d2e2; }
    main { max-width:1280px; margin:0 auto; padding:18px; }
    .cards { display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:12px; margin-bottom:14px; }
    .panel { background:white; border:1px solid #dce2ea; border-radius:8px; padding:14px; box-shadow:0 8px 22px rgba(20,31,48,.07); }
    .label { color:#667085; font-size:12px; }
    .value { font-size:22px; font-weight:700; margin-top:4px; }
    .layout { display:grid; grid-template-columns:260px 1fr; gap:14px; }
    .viewer { display:grid; grid-template-columns:repeat(3, minmax(0,1fr)); gap:12px; }
    canvas { width:100%; image-rendering:pixelated; background:#111827; border-radius:6px; }
    select { width:100%; padding:7px 8px; border:1px solid #dce2ea; border-radius:6px; background:white; }
    table { width:100%; border-collapse:collapse; }
    td,th { border-bottom:1px solid #edf0f5; padding:7px 6px; text-align:left; }
    th { color:#667085; }
    .bar-row { display:grid; grid-template-columns:86px 1fr 80px; align-items:center; gap:8px; margin:7px 0; }
    .track { height:10px; background:#edf1f6; border-radius:999px; overflow:hidden; }
    .bar { height:100%; background:#2563eb; }
    @media (max-width: 900px) { .cards,.layout,.viewer { grid-template-columns:1fr; } }
  </style>
</head>
<body>
<header><h1>__TITLE__</h1><p>Smoke demo: train a tiny forward predictor from bev_history_t + action_t to next BEV occupancy.</p></header>
<main>
  <section class=\"cards\" id=\"cards\"></section>
  <section class=\"layout\">
    <aside class=\"panel\">
      <label>Example</label><select id=\"exampleSelect\"></select>
      <label style=\"margin-top:12px;display:block\">Channel</label><select id=\"channelSelect\"></select>
      <h2>Metrics</h2><table id=\"exampleMetrics\"></table>
    </aside>
    <section class=\"panel\">
      <div class=\"viewer\">
        <div><h2>Last Input BEV</h2><canvas id=\"lastCanvas\"></canvas></div>
        <div><h2>Target Next BEV</h2><canvas id=\"targetCanvas\"></canvas></div>
        <div><h2>Predicted Next BEV</h2><canvas id=\"predCanvas\"></canvas></div>
      </div>
    </section>
  </section>
  <section class=\"panel\" style=\"margin-top:14px\"><h2>Loss Curve</h2><div id=\"lossCurve\"></div></section>
  <section class=\"panel\" style=\"margin-top:14px\"><h2>Learned Parameters</h2><table id=\"modelTable\"></table></section>
</main>
<script id=\"payload\" type=\"application/json\">__PAYLOAD__</script>
<script>
const report = JSON.parse(document.getElementById('payload').textContent);
const examples = report.examples || [];
const channels = report.channels || [];
const exampleSelect = document.getElementById('exampleSelect');
const channelSelect = document.getElementById('channelSelect');
exampleSelect.innerHTML = examples.map((ex, i) => `<option value="${i}">${ex.scenario_id} / step=${ex.step}</option>`).join('');
channelSelect.innerHTML = channels.map((c, i) => `<option value="${i}">${c}</option>`).join('');
exampleSelect.addEventListener('change', draw);
channelSelect.addEventListener('change', draw);
document.getElementById('cards').innerHTML = [
  ['Samples', report.sample_count],
  ['Shape', (report.history_shape || []).join(' x ')],
  ['Baseline MSE', fmt(report.baseline_persistence_eval.mse)],
  ['Model MSE', fmt(report.final_eval.mse)],
  ['Eval IoU', fmt(report.final_eval.occupancy_iou)]
].map(([k,v]) => `<div class="panel"><div class="label">${k}</div><div class="value">${v}</div></div>`).join('');
function draw() {
  if (!examples.length) return;
  const ex = examples[Number(exampleSelect.value || 0)];
  const ci = Number(channelSelect.value || 0);
  drawGrid('lastCanvas', ex.last_observed_frame[ci]);
  drawGrid('targetCanvas', ex.target_next_frame[ci]);
  drawGrid('predCanvas', ex.predicted_next_frame[ci]);
  document.getElementById('exampleMetrics').innerHTML = Object.entries(ex.metrics || {}).map(([k,v]) => `<tr><th>${k}</th><td>${fmt(v)}</td></tr>`).join('') + `<tr><th>action</th><td>${JSON.stringify(ex.action)}</td></tr>`;
}
function drawGrid(id, grid) {
  const canvas = document.getElementById(id);
  const ctx = canvas.getContext('2d');
  const h = grid.length, w = grid[0].length;
  canvas.width = w; canvas.height = h;
  const image = ctx.createImageData(w, h);
  for (let y=0; y<h; y++) for (let x=0; x<w; x++) {
    const v = Math.max(0, Math.min(1, Number(grid[y][x] || 0)));
    const p = (y*w+x)*4;
    image.data[p] = Math.round(30 + 225*v);
    image.data[p+1] = Math.round(40 + 160*v);
    image.data[p+2] = Math.round(55 + 70*v);
    image.data[p+3] = 255;
  }
  ctx.putImageData(image, 0, 0);
}
function renderLoss() {
  const curve = report.loss_curve || [];
  const max = Math.max(...curve.map(r => r.eval_mse), 1e-9);
  document.getElementById('lossCurve').innerHTML = curve.map(r => `<div class="bar-row"><div>epoch ${r.epoch}</div><div class="track"><div class="bar" style="width:${Math.max(2, r.eval_mse/max*100)}%"></div></div><div>${fmt(r.eval_mse)}</div></div>`).join('');
}
function renderModel() {
  const m = report.model || {};
  document.getElementById('modelTable').innerHTML = '<tr><th>channel</th><th>w_last</th><th>w_acc</th><th>w_steer</th><th>bias</th></tr>' + channels.map((c,i) => `<tr><td>${c}</td><td>${fmt(m.weights_last[i])}</td><td>${fmt(m.weights_acceleration[i])}</td><td>${fmt(m.weights_steering[i])}</td><td>${fmt(m.bias[i])}</td></tr>`).join('');
}
function fmt(v) { return Number(v).toFixed(6); }
renderLoss(); renderModel(); draw();
</script>
</body>
</html>""".replace("__TITLE__", html.escape(title)).replace("__PAYLOAD__", payload)


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


if __name__ == "__main__":
    main()

