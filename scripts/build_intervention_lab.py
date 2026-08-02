from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from worldsimflow.core.scenario import ScenarioLoader
from worldsimflow.core.scenario_generation import scenario_to_dict

DEFAULT_SCENARIOS = [
    "data/converted/openlog_nuscenes_scene-0061.json",
    "data/converted/openlog_waymo_2a1e44d405a6833f.json",
    "data/sample_scenario.json",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an interactive log intervention lab HTML.")
    parser.add_argument("--scenarios", nargs="+", default=DEFAULT_SCENARIOS)
    parser.add_argument("--output", default="outputs/all_results/labs/log_intervention_lab.html")
    args = parser.parse_args()

    scenarios = []
    loader = ScenarioLoader()
    for item in args.scenarios:
        path = resolve(item)
        scenario = loader.load(path)
        data = scenario_to_dict(scenario)
        data["source_path"] = str(path)
        data["display_name"] = display_name(path, data)
        scenarios.append(data)

    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render(scenarios), encoding="utf-8")
    print("intervention_lab=ok")
    print(f"output={output}")
    print(f"scenarios={len(scenarios)}")
    for scenario in scenarios:
        print(f"{scenario['scenario_id']} actors={len(scenario.get('actors', []))} map_features={len(scenario.get('map_features', []))}")


def display_name(path: Path, data: dict[str, Any]) -> str:
    scenario_id = data.get("scenario_id", path.stem)
    if "nuscenes" in scenario_id:
        return "nuScenes mini log"
    if "waymo" in scenario_id:
        return "Waymo mini log"
    if "sample" in path.name or "straight_close" in scenario_id:
        return "sample synthetic log"
    return str(scenario_id)


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


def render(scenarios: list[dict[str, Any]]) -> str:
    payload = json.dumps({"scenarios": scenarios}, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    template = r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>WorldSimFlow Log Intervention Lab</title>
  <style>
    :root {
      color-scheme: light dark;
      --bg: #f5f7fb;
      --fg: #1f2933;
      --muted: #667085;
      --panel: #ffffff;
      --line: #d0d5dd;
      --road: #314155;
      --map: #e5e7eb;
      --ego: #2563eb;
      --actor: #f97316;
      --selected: #facc15;
      --pedestrian: #22c55e;
      --danger: #d92d20;
      --ok: #118244;
      --accent: #246bfe;
      --soft: #e8f0ff;
    }
    @media (prefers-color-scheme: dark) {
      :root {
        --bg: #101418;
        --fg: #edf2f7;
        --muted: #a0aec0;
        --panel: #171b21;
        --line: #344054;
        --road: #222b35;
        --map: #cbd5e1;
        --soft: #172847;
      }
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--bg); color: var(--fg); font-family: Arial, Helvetica, sans-serif; }
    main { max-width: 1380px; margin: 0 auto; padding: 22px; }
    h1 { margin: 0 0 6px; font-size: 22px; font-weight: 600; }
    .intro { margin: 0 0 14px; color: var(--muted); line-height: 1.55; }
    .layout { display: grid; grid-template-columns: minmax(280px, 360px) minmax(0, 1fr); gap: 14px; align-items: start; }
    .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px; }
    .controls { display: grid; gap: 12px; }
    label { display: grid; gap: 5px; color: var(--muted); font-size: 12px; }
    select, input { width: 100%; border: 1px solid var(--line); border-radius: 6px; padding: 8px 9px; background: var(--panel); color: var(--fg); font-size: 14px; }
    .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .buttons { display: flex; flex-wrap: wrap; gap: 8px; }
    button { border: 1px solid var(--line); background: var(--panel); color: var(--fg); border-radius: 6px; padding: 8px 10px; cursor: pointer; }
    button.primary { border-color: var(--accent); background: var(--accent); color: #fff; font-weight: 600; }
    button.active { border-color: var(--accent); color: var(--accent); background: var(--soft); }
    .status { display: grid; gap: 6px; padding: 10px; border-radius: 8px; background: var(--soft); color: var(--fg); font-size: 13px; line-height: 1.45; }
    .status strong { font-weight: 600; }
    .command { white-space: pre-wrap; overflow-wrap: anywhere; color: var(--muted); font-family: Consolas, monospace; font-size: 12px; line-height: 1.45; }
    canvas { width: 100%; height: auto; display: block; border: 1px solid var(--line); border-radius: 8px; background: #2f6f47; cursor: crosshair; }
    .timeline { display: grid; grid-template-columns: auto 1fr auto auto auto; gap: 8px; align-items: center; margin-top: 10px; }
    input[type=range] { padding: 0; }
    .hud { display: grid; grid-template-columns: repeat(auto-fit, minmax(138px, 1fr)); gap: 8px; margin-top: 10px; }
    .metric { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 9px 10px; min-width: 0; }
    .metric span { display: block; color: var(--muted); font-size: 12px; margin-bottom: 4px; }
    .metric strong { display: block; overflow-wrap: anywhere; font-size: 14px; }
    .events { color: var(--danger); }
    .ok { color: var(--ok); }
    @media (max-width: 900px) {
      main { padding: 14px; }
      .layout { grid-template-columns: 1fr; }
      .timeline { grid-template-columns: 1fr 1fr; }
      .timeline input { grid-column: 1 / -1; }
    }
  </style>
</head>
<body>
  <main>
    <h1>WorldSimFlow Log Intervention Lab</h1>
    <p class="intro">选择一条 log、一个 actor 和干预类型。你可以从下拉菜单选 actor_id，也可以在 birdview 里双击车辆/行人自动选中目标。干预类型选“原始 log”时完全不改轨迹；填写参数并应用后，页面会生成反事实 replay，用来快速观察目标车急刹、切入、速度改变或横向偏移的效果。</p>
    <section class="layout">
      <aside class="panel controls">
        <label>Log 场景
          <select id="scenarioSelect"></select>
        </label>
        <label>目标 actor_id
          <select id="actorSelect"></select>
        </label>
        <label>干预类型
          <select id="kindSelect">
            <option value="none">原始 log，不做干预</option>
            <option value="hard_brake">hard_brake 急刹</option>
            <option value="cut_in">cut_in 切入</option>
            <option value="speed_change">speed_change 速度改变</option>
            <option value="lateral_shift">lateral_shift 横向偏移</option>
          </select>
        </label>
        <div class="grid2">
          <label>start_step
            <input id="startStep" type="number" min="0" step="1" value="12">
          </label>
          <label>duration
            <input id="duration" type="number" min="1" step="1" value="24">
          </label>
        </div>
        <div class="grid2">
          <label>deceleration
            <input id="deceleration" type="number" step="0.1" value="-4.5">
          </label>
          <label>target_lateral
            <input id="targetLateral" type="number" step="0.1" value="0.0">
          </label>
        </div>
        <div class="grid2">
          <label>target_speed
            <input id="targetSpeed" type="number" step="0.1" value="1.0">
          </label>
          <label>lateral shift
            <input id="shift" type="number" step="0.1" value="2.0">
          </label>
        </div>
        <div class="buttons">
          <button id="applyBtn" class="primary" type="button">应用干预</button>
          <button id="resetBtn" type="button">恢复原始 log</button>
        </div>
        <div class="status" id="statusBox"></div>
        <div class="command" id="commandBox"></div>
      </aside>
      <section>
        <canvas id="scene" width="1080" height="620" aria-label="interactive log intervention birdview"></canvas>
        <div class="timeline">
          <button id="playBtn" type="button">Play</button>
          <input id="scrub" type="range" min="0" value="0" step="1">
          <button id="prevBtn" type="button">Prev</button>
          <button id="nextBtn" type="button">Next</button>
          <button id="followBtn" class="active" type="button">Follow Ego</button>
        </div>
        <div class="hud">
          <div class="metric"><span>Frame</span><strong id="frameText"></strong></div>
          <div class="metric"><span>Mode</span><strong id="modeText"></strong></div>
          <div class="metric"><span>Ego</span><strong id="egoText"></strong></div>
          <div class="metric"><span>Selected Actor</span><strong id="actorText"></strong></div>
          <div class="metric"><span>Front Gap</span><strong id="gapText"></strong></div>
          <div class="metric"><span>Events</span><strong id="eventText"></strong></div>
        </div>
      </section>
    </section>
  </main>
  <script>
    const payload = __LAB_DATA__;
    const scenarios = payload.scenarios;
    const scenarioSelect = document.getElementById('scenarioSelect');
    const actorSelect = document.getElementById('actorSelect');
    const kindSelect = document.getElementById('kindSelect');
    const startStep = document.getElementById('startStep');
    const durationInput = document.getElementById('duration');
    const decelerationInput = document.getElementById('deceleration');
    const targetLateralInput = document.getElementById('targetLateral');
    const targetSpeedInput = document.getElementById('targetSpeed');
    const shiftInput = document.getElementById('shift');
    const applyBtn = document.getElementById('applyBtn');
    const resetBtn = document.getElementById('resetBtn');
    const statusBox = document.getElementById('statusBox');
    const commandBox = document.getElementById('commandBox');
    const canvas = document.getElementById('scene');
    const ctx = canvas.getContext('2d');
    const scrub = document.getElementById('scrub');
    const playBtn = document.getElementById('playBtn');
    const prevBtn = document.getElementById('prevBtn');
    const nextBtn = document.getElementById('nextBtn');
    const followBtn = document.getElementById('followBtn');
    let scenario = scenarios[0];
    let activeActors = scenario.actors;
    let mode = '原始 log';
    let selectedActorId = '';
    let index = 0;
    let timer = null;
    let follow = true;

    for (let i = 0; i < scenarios.length; i++) {
      const option = document.createElement('option');
      option.value = String(i);
      option.textContent = `${scenarios[i].display_name} - ${scenarios[i].scenario_id}`;
      scenarioSelect.appendChild(option);
    }

    function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
    function clone(value) { return JSON.parse(JSON.stringify(value)); }
    function clamp(value, min, max) { return Math.max(min, Math.min(max, value)); }
    function num(input, fallback) { const v = Number(input.value); return Number.isFinite(v) ? v : fallback; }
    function deg(rad) { return rad * 180 / Math.PI; }
    function fmt(value, digits = 2) { return value == null || !Number.isFinite(Number(value)) ? 'n/a' : Number(value).toFixed(digits); }
    function egoStates() {
      const raw = scenario.metadata?.ego_replay_states;
      if (raw && raw.length) return raw.map(s => ({actor_id: 'ego', length: 4.6, width: 1.9, object_type: 'EGO', ...s}));
      const states = [];
      const ego = scenario.ego;
      for (let step = 0; step < scenario.max_steps; step++) {
        states.push({actor_id: 'ego', x: ego.x + ego.speed * scenario.dt * step * Math.cos(ego.yaw), y: ego.y + ego.speed * scenario.dt * step * Math.sin(ego.yaw), yaw: ego.yaw, speed: ego.speed, length: ego.length || 4.6, width: ego.width || 1.9, object_type: ego.object_type || 'EGO'});
      }
      return states;
    }
    function egoAt(step) { const states = egoStates(); return states[Math.min(step, states.length - 1)]; }
    function toLocal(x, y, ego) {
      const dx = x - ego.x, dy = y - ego.y;
      const c = Math.cos(ego.yaw), s = Math.sin(ego.yaw);
      return {forward: c * dx + s * dy, lateral: -s * dx + c * dy};
    }
    function fromLocal(forward, lateral, ego) {
      const c = Math.cos(ego.yaw), s = Math.sin(ego.yaw);
      return {x: ego.x + c * forward - s * lateral, y: ego.y + s * forward + c * lateral};
    }
    function populateActors() {
      actorSelect.innerHTML = '';
      const ordered = actorOptionsForFrame(index);
      ordered.forEach(({actor, state, local}) => {
        const option = document.createElement('option');
        option.value = actor.actor_id;
        option.textContent = actorOptionText(actor.actor_id, state, local);
        actorSelect.appendChild(option);
      });
      selectedActorId = actorSelect.value || '';
    }
    function actorOptionsForFrame(step) {
      const ego = egoAt(step);
      return scenario.actors.map(actor => {
        const state = actor.states[Math.min(step, actor.states.length - 1)];
        return {actor, state, local: toLocal(state.x, state.y, ego)};
      }).sort((a, b) => {
        const aAhead = a.local.forward >= 0 ? 0 : 1;
        const bAhead = b.local.forward >= 0 ? 0 : 1;
        return aAhead - bAhead || Math.abs(a.local.lateral) - Math.abs(b.local.lateral) || a.local.forward - b.local.forward;
      });
    }
    function actorOptionText(actorId, state, local) {
      const direction = local.forward >= 0 ? '前方' : '后方';
      return `${actorId} (${state.object_type}, ${direction}${fmt(Math.abs(local.forward), 1)}m, 横向${fmt(local.lateral, 1)}m, v=${fmt(state.speed)}m/s)`;
    }
    function refreshActorLabels() {
      const selected = actorSelect.value || selectedActorId;
      const labels = new Map(actorOptionsForFrame(index).map(({actor, state, local}) => [actor.actor_id, actorOptionText(actor.actor_id, state, local)]));
      Array.from(actorSelect.options).forEach(option => {
        option.textContent = labels.get(option.value) || option.textContent;
      });
      if (selected && labels.has(selected)) actorSelect.value = selected;
    }
    function resetScenario() {
      activeActors = clone(scenario.actors);
      mode = '原始 log';
      kindSelect.value = 'none';
      selectedActorId = actorSelect.value || '';
      updateStatus([]);
      draw();
    }
    function applyIntervention() {
      selectedActorId = actorSelect.value;
      const kind = kindSelect.value;
      activeActors = clone(scenario.actors);
      if (kind === 'none') {
        mode = '原始 log';
        updateStatus([]);
        draw();
        return;
      }
      const actor = activeActors.find(a => a.actor_id === selectedActorId);
      if (!actor) return;
      const start = clamp(Math.trunc(num(startStep, 0)), 0, scenario.max_steps - 1);
      const duration = Math.max(1, Math.trunc(num(durationInput, 24)));
      if (kind === 'hard_brake') applyHardBrake(actor, start, num(decelerationInput, -4.5));
      if (kind === 'cut_in') applyCutIn(actor, start, duration, num(targetLateralInput, 0));
      if (kind === 'speed_change') applySpeedChange(actor, start, duration, num(targetSpeedInput, 1));
      if (kind === 'lateral_shift') applyLateralShift(actor, start, duration, num(shiftInput, 2));
      mode = `${kind} / actor=${selectedActorId}`;
      updateStatus(eventsAt(index));
      draw();
    }
    function applyHardBrake(actor, start, decel) {
      for (let step = start + 1; step < actor.states.length; step++) {
        const prev = actor.states[step - 1];
        const raw = actor.states[step];
        const speed = Math.max(0, prev.speed + decel * scenario.dt);
        raw.speed = speed;
        raw.x = prev.x + speed * scenario.dt * Math.cos(raw.yaw);
        raw.y = prev.y + speed * scenario.dt * Math.sin(raw.yaw);
      }
    }
    function applySpeedChange(actor, start, duration, targetSpeed) {
      const startSpeed = actor.states[start].speed;
      for (let step = start + 1; step < actor.states.length; step++) {
        const raw = actor.states[step];
        const prev = actor.states[step - 1];
        const progress = Math.min(1, (step - start) / duration);
        const smooth = progress * progress * (3 - 2 * progress);
        const speed = Math.max(0, startSpeed + (targetSpeed - startSpeed) * smooth);
        raw.speed = speed;
        raw.x = prev.x + speed * scenario.dt * Math.cos(raw.yaw);
        raw.y = prev.y + speed * scenario.dt * Math.sin(raw.yaw);
      }
    }
    function applyCutIn(actor, start, duration, targetLateral) {
      const ego = egoAt(start);
      const startLocal = toLocal(actor.states[start].x, actor.states[start].y, ego);
      rewriteRouteRelativeLateral(actor, start, duration, startLocal, targetLateral);
    }
    function applyLateralShift(actor, start, duration, shift) {
      const ego = egoAt(start);
      const source = toLocal(actor.states[start].x, actor.states[start].y, ego).lateral;
      rewriteLateral(actor, start, duration, source, source + shift);
      syncKinematics(actor, start);
    }
    function rewriteRouteRelativeLateral(actor, start, duration, startLocal, targetLateral) {
      let egoDistance = 0;
      let actorDistance = 0;
      let previousEgo = egoAt(start);
      let previousActor = actor.states[start];
      for (let step = start; step < actor.states.length; step++) {
        const raw = actor.states[step];
        const ego = egoAt(step);
        if (step > start) {
          egoDistance += Math.hypot(ego.x - previousEgo.x, ego.y - previousEgo.y);
          actorDistance += Math.max(0, previousActor.speed || 0) * scenario.dt;
        }
        const progress = Math.min(1, (step - start) / duration);
        const smooth = progress * progress * (3 - 2 * progress);
        const forward = startLocal.forward + actorDistance - egoDistance;
        const lateral = startLocal.lateral + (targetLateral - startLocal.lateral) * smooth;
        const p = fromLocal(forward, lateral, ego);
        raw.x = p.x; raw.y = p.y;
        previousEgo = ego;
        previousActor = raw;
      }
      syncKinematics(actor, start);
    }
    function rewriteLateral(actor, start, duration, source, target) {
      for (let step = start; step < actor.states.length; step++) {
        const raw = actor.states[step];
        const ego = egoAt(step);
        const local = toLocal(raw.x, raw.y, ego);
        const progress = Math.min(1, (step - start) / duration);
        const smooth = progress * progress * (3 - 2 * progress);
        const p = fromLocal(local.forward, source + (target - source) * smooth, ego);
        raw.x = p.x; raw.y = p.y;
      }
    }
    function syncKinematics(actor, start) {
      for (let step = start; step < actor.states.length; step++) {
        const raw = actor.states[step];
        const reference = step < actor.states.length - 1 ? actor.states[step + 1] : actor.states[step - 1];
        const dx = step < actor.states.length - 1 ? reference.x - raw.x : raw.x - reference.x;
        const dy = step < actor.states.length - 1 ? reference.y - raw.y : raw.y - reference.y;
        const distance = Math.hypot(dx, dy);
        if (distance > 1e-6) {
          raw.yaw = Math.atan2(dy, dx);
          raw.speed = distance / Math.max(1e-6, scenario.dt);
        } else {
          const prev = step > 0 ? actor.states[step - 1] : raw;
          raw.yaw = prev.yaw;
          raw.speed = 0;
        }
      }
    }
    function eventsAt(step) {
      const ego = egoAt(step);
      const events = [];
      for (const actor of activeActors) {
        const state = actor.states[Math.min(step, actor.states.length - 1)];
        const overlapX = Math.abs(ego.x - state.x) <= ((ego.length || 4.6) + (state.length || 4.6)) / 2 + 0.3;
        const overlapY = Math.abs(ego.y - state.y) <= ((ego.width || 1.9) + (state.width || 1.9)) / 2 + 0.3;
        if (overlapX && overlapY) events.push(`collision:${actor.actor_id}`);
      }
      return events;
    }
    function frontGap(step) {
      const ego = egoAt(step);
      let best = null;
      for (const actor of activeActors) {
        const state = actor.states[Math.min(step, actor.states.length - 1)];
        const local = toLocal(state.x, state.y, ego);
        if (local.forward >= 0 && Math.abs(local.lateral) < scenario.road.lane_width / 2) best = best == null ? local.forward : Math.min(best, local.forward);
      }
      return best;
    }
    function bounds() {
      const xs = [], ys = [];
      for (const actor of activeActors) for (const state of actor.states) { xs.push(state.x); ys.push(state.y); }
      for (const state of egoStates()) { xs.push(state.x); ys.push(state.y); }
      for (const feature of scenario.map_features || []) for (const p of feature.polyline || []) { xs.push(p[0]); ys.push(p[1]); }
      if (!xs.length) return {minX: -20, maxX: 80, minY: -30, maxY: 30};
      const pad = 25;
      return {minX: Math.min(...xs) - pad, maxX: Math.max(...xs) + pad, minY: Math.min(...ys) - pad, maxY: Math.max(...ys) + pad};
    }
    function transform(step) {
      const ego = egoAt(step);
      if (follow) return {mode: 'follow', ego, scale: Math.min(canvas.width / 90, canvas.height / 95), cx: canvas.width / 2, cy: canvas.height * 0.72};
      const b = bounds();
      const pad = 38;
      const scale = Math.min((canvas.width - pad * 2) / Math.max(1, b.maxX - b.minX), (canvas.height - pad * 2) / Math.max(1, b.maxY - b.minY));
      return {mode: 'global', scale, ox: (canvas.width - (b.maxX - b.minX) * scale) / 2 - b.minX * scale, oy: (canvas.height + (b.maxY - b.minY) * scale) / 2 + b.minY * scale};
    }
    function worldToCanvas(x, y, t) {
      if (t.mode === 'follow') {
        const p = toLocal(x, y, t.ego);
        return {x: t.cx + p.lateral * t.scale, y: t.cy - p.forward * t.scale};
      }
      return {x: t.ox + x * t.scale, y: t.oy - y * t.scale};
    }
    function drawMap(t) {
      ctx.save();
      ctx.lineCap = 'round';
      for (const feature of scenario.map_features || []) {
        const points = feature.polyline || [];
        if (points.length < 2) continue;
        ctx.beginPath();
        points.forEach((p, i) => { const q = worldToCanvas(p[0], p[1], t); if (i === 0) ctx.moveTo(q.x, q.y); else ctx.lineTo(q.x, q.y); });
        ctx.strokeStyle = feature.type?.includes('ROAD_EDGE') ? '#9aa6b2' : '#f0f4f8';
        ctx.globalAlpha = feature.type?.includes('ROAD_EDGE') ? 0.72 : 0.88;
        ctx.lineWidth = feature.type?.includes('BROKEN') ? 1.3 : 1.8;
        ctx.setLineDash(feature.type?.includes('BROKEN') ? [10, 12] : []);
        ctx.stroke();
      }
      ctx.setLineDash([]); ctx.restore();
      if (!(scenario.map_features || []).length) drawFallbackRoad(t);
    }
    function drawFallbackRoad(t) {
      ctx.save(); ctx.strokeStyle = 'rgba(255,255,255,0.35)'; ctx.lineWidth = 1;
      const half = scenario.road.lane_width * scenario.road.lane_count / 2;
      for (let y = -half; y <= half + 0.01; y += scenario.road.lane_width) {
        const a = worldToCanvas(-20, y, t), b = worldToCanvas(scenario.road.length, y, t);
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
      }
      ctx.restore();
    }
    function vehicleCanvasPose(v, t) {
      return {
        p: worldToCanvas(v.x, v.y, t),
        angle: t.mode === 'follow' ? v.yaw - t.ego.yaw - Math.PI / 2 : -v.yaw,
        len: Math.max(6, (v.length || 4.6) * t.scale),
        wid: Math.max(4, (v.width || 1.9) * t.scale),
      };
    }
    function drawVehicle(v, color, t, selected=false) {
      const pose = vehicleCanvasPose(v, t);
      ctx.save(); ctx.translate(pose.p.x, pose.p.y); ctx.rotate(pose.angle);
      ctx.fillStyle = color; ctx.strokeStyle = selected ? css('--selected') : 'rgba(0,0,0,0.35)'; ctx.lineWidth = selected ? 3 : 1;
      if (v.object_type === 'PEDESTRIAN') { ctx.beginPath(); ctx.arc(0, 0, Math.max(4, pose.wid / 2), 0, Math.PI * 2); ctx.fill(); ctx.stroke(); }
      else { ctx.fillRect(-pose.len / 2, -pose.wid / 2, pose.len, pose.wid); ctx.strokeRect(-pose.len / 2, -pose.wid / 2, pose.len, pose.wid); }
      ctx.restore();
    }
    function canvasPointFromEvent(event) {
      const rect = canvas.getBoundingClientRect();
      return {
        x: (event.clientX - rect.left) * canvas.width / rect.width,
        y: (event.clientY - rect.top) * canvas.height / rect.height,
      };
    }
    function hitActorAt(point) {
      const t = transform(index);
      const hits = [];
      for (const actor of activeActors) {
        const state = actor.states[Math.min(index, actor.states.length - 1)];
        const pose = vehicleCanvasPose(state, t);
        const dx = point.x - pose.p.x;
        const dy = point.y - pose.p.y;
        const c = Math.cos(pose.angle);
        const s = Math.sin(pose.angle);
        const localX = c * dx + s * dy;
        const localY = -s * dx + c * dy;
        const pad = state.object_type === 'PEDESTRIAN' ? 8 : 5;
        const halfLen = pose.len / 2 + pad;
        const halfWid = pose.wid / 2 + pad;
        const hit = state.object_type === 'PEDESTRIAN'
          ? Math.hypot(localX, localY) <= Math.max(8, pose.wid / 2 + pad)
          : Math.abs(localX) <= halfLen && Math.abs(localY) <= halfWid;
        if (hit) hits.push({actorId: actor.actor_id, distance: Math.hypot(dx, dy)});
      }
      hits.sort((a, b) => a.distance - b.distance);
      return hits[0]?.actorId || null;
    }
    function selectActorById(actorId, source = 'select') {
      if (!actorId || !Array.from(actorSelect.options).some(option => option.value === actorId)) return;
      const previous = selectedActorId;
      selectedActorId = actorId;
      actorSelect.value = actorId;
      if (source === 'canvas' && previous !== actorId && kindSelect.value !== 'none') {
        activeActors = clone(scenario.actors);
        mode = `已选择 actor=${actorId}，点击应用 ${kindSelect.value}`;
      } else if (kindSelect.value === 'none') {
        mode = '原始 log';
      }
      updateStatus(eventsAt(index));
      draw();
    }
    function draw() {
      scrub.max = Math.max(0, scenario.max_steps - 1);
      index = clamp(index, 0, Number(scrub.max));
      scrub.value = index;
      const t = transform(index);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#2f6f47'; ctx.fillRect(0, 0, canvas.width, canvas.height);
      drawMap(t);
      for (const actor of activeActors) {
        const v = actor.states[Math.min(index, actor.states.length - 1)];
        const color = v.object_type === 'PEDESTRIAN' ? css('--pedestrian') : css('--actor');
        drawVehicle(v, color, t, actor.actor_id === selectedActorId);
      }
      drawVehicle(egoAt(index), css('--ego'), t, false);
      refreshActorLabels();
      updateHud();
    }
    function updateHud() {
      const ego = egoAt(index);
      const actor = activeActors.find(a => a.actor_id === selectedActorId);
      const state = actor?.states[Math.min(index, actor.states.length - 1)];
      const events = eventsAt(index);
      document.getElementById('frameText').textContent = `${index + 1}/${scenario.max_steps}`;
      document.getElementById('modeText').textContent = mode;
      document.getElementById('egoText').textContent = `x=${fmt(ego.x,1)}, y=${fmt(ego.y,1)}, v=${fmt(ego.speed)} m/s`;
      document.getElementById('actorText').textContent = state ? `${selectedActorId}, v=${fmt(state.speed)} m/s, y=${fmt(state.y,1)}` : 'none';
      document.getElementById('gapText').textContent = frontGap(index) == null ? 'n/a' : `${fmt(frontGap(index))} m`;
      const eventText = document.getElementById('eventText');
      eventText.textContent = events.length ? events.join(', ') : 'none';
      eventText.className = events.length ? 'events' : 'ok';
      updateStatus(events);
    }
    function updateStatus(events) {
      const kind = kindSelect.value;
      const actor = actorSelect.value || 'none';
      statusBox.innerHTML = `<strong>${mode}</strong><span>scenario=${scenario.scenario_id}</span><span>actor=${actor}</span><span>events=${events.length ? events.join(', ') : 'none'}</span>`;
      commandBox.textContent = commandPreview(kind, actor);
    }
    function commandPreview(kind, actor) {
      if (kind === 'none') return 'CLI: 原始 log replay，无需干预命令。';
      const base = `python scripts/intervene_target_actor.py --scenario ${scenario.source_path} --actor-id ${actor} --kind ${kind} --start-step ${Math.trunc(num(startStep, 0))} --duration ${Math.trunc(num(durationInput, 24))}`;
      if (kind === 'hard_brake') return `${base} --deceleration ${num(decelerationInput, -4.5)}`;
      if (kind === 'cut_in') return `${base} --target-lateral ${num(targetLateralInput, 0)}`;
      if (kind === 'speed_change') return `${base} --target-speed ${num(targetSpeedInput, 1)}`;
      if (kind === 'lateral_shift') return `${base} --shift ${num(shiftInput, 2)}`;
      return base;
    }
    function loadScenario() {
      scenario = scenarios[Number(scenarioSelect.value)];
      index = 0;
      populateActors();
      resetScenario();
    }
    function tick() { index = index >= Number(scrub.max) ? 0 : index + 1; draw(); }
    scenarioSelect.addEventListener('change', loadScenario);
    actorSelect.addEventListener('change', () => selectActorById(actorSelect.value, 'select'));
    canvas.addEventListener('dblclick', event => {
      const actorId = hitActorAt(canvasPointFromEvent(event));
      if (actorId) selectActorById(actorId, 'canvas');
    });
    kindSelect.addEventListener('change', () => { updateStatus(eventsAt(index)); });
    [startStep, durationInput, decelerationInput, targetLateralInput, targetSpeedInput, shiftInput].forEach(el => el.addEventListener('input', () => updateStatus(eventsAt(index))));
    applyBtn.addEventListener('click', applyIntervention);
    resetBtn.addEventListener('click', resetScenario);
    scrub.addEventListener('input', () => { index = Number(scrub.value); draw(); });
    prevBtn.addEventListener('click', () => { index = Math.max(0, index - 1); draw(); });
    nextBtn.addEventListener('click', () => { index = Math.min(Number(scrub.max), index + 1); draw(); });
    followBtn.addEventListener('click', () => { follow = !follow; followBtn.textContent = follow ? 'Follow Ego' : 'Global'; followBtn.classList.toggle('active', follow); draw(); });
    playBtn.addEventListener('click', () => { if (timer) { clearInterval(timer); timer = null; playBtn.textContent = 'Play'; } else { timer = setInterval(tick, Math.max(40, scenario.dt * 1000)); playBtn.textContent = 'Pause'; } });
    loadScenario();
  </script>
</body>
</html>
'''
    return template.replace("__LAB_DATA__", payload)


if __name__ == "__main__":
    main()



