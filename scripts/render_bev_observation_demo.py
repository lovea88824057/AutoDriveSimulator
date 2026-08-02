from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from worldsimflow import BEVRasterConfig, ScenarioLoader, WorldSimFlowEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a lightweight BEV observation demo page.")
    parser.add_argument("--scenario", default="data/sample_scenario.json")
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--meters-per-pixel", type=float, default=1.0)
    parser.add_argument("--output", default="outputs/all_results/bev/bev_observation_demo.html")
    parser.add_argument("--json", default="outputs/all_results/bev/bev_observation_demo.json")
    args = parser.parse_args()

    scenario_path = resolve(args.scenario)
    scenario = ScenarioLoader().load(scenario_path)
    config = BEVRasterConfig(width=args.width, height=args.height, meters_per_pixel=args.meters_per_pixel)
    env = WorldSimFlowEnv(scenario, max_steps=args.steps, include_bev=True, bev_config=config)
    obs, reset_info = env.reset(seed=scenario.seed)
    frames = [frame_payload(obs, reset_info, step=0)]
    try:
        for step in range(1, args.steps + 1):
            obs, reward, terminated, truncated, info = env.step({"acceleration": 0.0, "steering": 0.0})
            frames.append(frame_payload(obs, info, step=step, reward=reward, terminated=terminated, truncated=truncated))
            if terminated or truncated:
                break
    finally:
        env.close()

    report = {
        "bev_demo": "ok",
        "scenario_id": scenario.scenario_id,
        "scenario": str(scenario_path),
        "frame_count": len(frames),
        "bev_config": config.to_dict(),
        "observation_schema_version": reset_info.get("observation_schema_version"),
        "feature_names": reset_info.get("feature_names", []),
        "frames": frames,
    }
    output = resolve(args.output)
    json_output = resolve(args.json)
    output.parent.mkdir(parents=True, exist_ok=True)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    output.write_text(render_html(report), encoding="utf-8", newline="\n")
    print("bev_demo=ok")
    print(f"scenario_id={scenario.scenario_id}")
    print(f"frame_count={len(frames)}")
    print(f"html={output}")
    print(f"json={json_output}")


def frame_payload(obs, info, *, step, reward=None, terminated=False, truncated=False):
    return {
        "step": step,
        "reward": reward,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "done_reason": info.get("done_reason", {"reason": "reset"}),
        "ego_speed": obs.get("ego_speed"),
        "front_gap": obs.get("front_gap"),
        "lane_l": obs.get("ego_lane_l", obs.get("lane_center_offset")),
        "state_vector": obs.get("state_vector", []),
        "normalized_vector": obs.get("normalized_vector", []),
        "bev": obs["bev"],
    }


def render_html(report):
    payload = json.dumps(report, ensure_ascii=False, separators=(",", ":"))
    title = "WorldSimFlow BEV Observation Demo"
    return """<!doctype html>
<html lang=\"zh-CN\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>__TITLE__</title>
  <style>
    body { margin:0; background:#f5f7fa; color:#1f2937; font:14px/1.5 Arial,'Microsoft YaHei',sans-serif; }
    header { background:#182033; color:white; padding:22px 28px; }
    h1 { margin:0; font-size:24px; }
    header p { margin:6px 0 0; color:#c7d2e2; }
    main { max-width:1180px; margin:0 auto; padding:18px; }
    .toolbar,.cards { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin-bottom:14px; }
    .panel { background:white; border:1px solid #dce2ea; border-radius:8px; padding:14px; box-shadow:0 8px 22px rgba(20,31,48,.07); }
    label { display:block; color:#667085; font-size:12px; margin-bottom:5px; }
    input,select { width:100%; }
    canvas { width:100%; max-width:420px; image-rendering:pixelated; background:#111827; border-radius:6px; }
    .layout { display:grid; grid-template-columns:430px 1fr; gap:14px; }
    .value { font-size:22px; font-weight:700; }
    .legend span { display:inline-block; margin:4px 8px 4px 0; }
    .swatch { width:12px; height:12px; border-radius:3px; vertical-align:-1px; margin-right:4px; }
    pre { margin:0; white-space:pre-wrap; overflow:auto; }
    @media (max-width: 900px) { .toolbar,.cards,.layout { grid-template-columns:1fr; } }
  </style>
</head>
<body>
<header><h1>__TITLE__</h1><p>C4 lightweight BEV raster: ego-centric channels for world-model/VLA/RL input inspection.</p></header>
<main>
  <section class=\"panel toolbar\"><div><label>Frame</label><input id=\"frame\" type=\"range\" min=\"0\" value=\"0\"></div><div><label>Channel</label><select id=\"channel\"></select></div><div><label>Scenario</label><div id=\"scenario\"></div></div><div><label>Shape</label><div id=\"shape\"></div></div></section>
  <section class=\"cards\" id=\"cards\"></section>
  <section class=\"layout\"><div class=\"panel\"><canvas id=\"canvas\"></canvas><div class=\"legend\" id=\"legend\"></div></div><div class=\"panel\"><h2>Observation Payload</h2><pre id=\"payloadView\"></pre></div></section>
</main>
<script id=\"data\" type=\"application/json\">__PAYLOAD__</script>
<script>
const report = JSON.parse(document.getElementById('data').textContent);
const frames = report.frames;
const frameInput = document.getElementById('frame');
const channelSelect = document.getElementById('channel');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
frameInput.max = frames.length - 1;
document.getElementById('scenario').textContent = report.scenario_id;
document.getElementById('shape').textContent = report.bev_config.shape.join(' x ');
const channels = report.bev_config.channels;
channelSelect.innerHTML = ['all', ...channels].map(c => `<option value="${c}">${c}</option>`).join('');
frameInput.addEventListener('input', draw);
channelSelect.addEventListener('change', draw);
const colors = {drivable:[48,79,125], lane_center:[245,200,66], ego:[50,220,130], actor_vehicle:[245,90,80], actor_vru:[210,120,245]};
function draw() {
  const frame = frames[Number(frameInput.value)];
  const bev = frame.bev;
  const h = bev.shape[1], w = bev.shape[2];
  canvas.width = w; canvas.height = h;
  const image = ctx.createImageData(w, h);
  const selected = channelSelect.value;
  for (let y=0; y<h; y++) {
    for (let x=0; x<w; x++) {
      const p = (y*w + x)*4;
      image.data[p] = 17; image.data[p+1] = 24; image.data[p+2] = 39; image.data[p+3] = 255;
      channels.forEach((name, ci) => {
        if (selected !== 'all' && selected !== name) return;
        if ((bev.raster[ci][y][x] || 0) > 0) {
          const c = colors[name] || [255,255,255];
          image.data[p] = c[0]; image.data[p+1] = c[1]; image.data[p+2] = c[2];
        }
      });
    }
  }
  ctx.putImageData(image, 0, 0);
  document.getElementById('cards').innerHTML = [
    ['step', frame.step], ['speed', fmt(frame.ego_speed)], ['front_gap', fmt(frame.front_gap)], ['done', frame.done_reason.reason || 'running']
  ].map(([k,v]) => `<div class="panel"><div>${k}</div><div class="value">${v}</div></div>`).join('');
  document.getElementById('legend').innerHTML = channels.map(name => `<span><i class="swatch" style="display:inline-block;background:rgb(${colors[name] || [255,255,255]})"></i>${name}</span>`).join('');
  document.getElementById('payloadView').textContent = JSON.stringify({step:frame.step, reward:frame.reward, done_reason:frame.done_reason, state_vector:frame.state_vector, normalized_vector:frame.normalized_vector}, null, 2);
}
function fmt(v) { return v === null || v === undefined ? '-' : Number(v).toFixed(3); }
draw();
</script>
</body>
</html>""".replace("__TITLE__", html.escape(title)).replace("__PAYLOAD__", payload)


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


if __name__ == "__main__":
    main()
