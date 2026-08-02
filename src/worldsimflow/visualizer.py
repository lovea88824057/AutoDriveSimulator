from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from .core.types import Action, Scenario, StepResult, VehicleState


def render_trace_html(
    scenario: Scenario,
    trace: Iterable[StepResult],
    output_path: str | Path,
    real_video: dict | None = None,
) -> Path:
    frames = [_frame_to_dict(frame) for frame in trace]
    payload = {
        "scenario": {
            "scenario_id": scenario.scenario_id,
            "dt": scenario.dt,
            "road": asdict(scenario.road),
            "seed": scenario.seed,
            "metadata": scenario.metadata,
            "map_features": [_map_feature_to_dict(feature) for feature in scenario.map_features],
            "drivable_area": asdict(scenario.drivable_area) if scenario.drivable_area else None,
        },
        "frames": frames,
        "real_video": real_video,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    output.write_text(_html(data_json), encoding="utf-8")
    return output


def _map_feature_to_dict(feature) -> dict:
    return {
        "feature_id": feature.feature_id,
        "type": feature.feature_type,
        "polyline": [[round(float(x), 4), round(float(y), 4)] for x, y in feature.polyline],
    }


def _frame_to_dict(frame: StepResult) -> dict:
    obs = frame.observation
    return {
        "step": frame.step,
        "ego": _vehicle_to_dict(obs["ego"]),
        "actors": [_vehicle_to_dict(actor) for actor in obs["actors"]],
        "front_gap": obs["front_gap"],
        "lane_center_offset": obs["lane_center_offset"],
        "closest_actor_distance": obs.get("closest_actor_distance"),
        "nearby_actor_count": obs.get("nearby_actor_count"),
        "map_feature_count": obs.get("map_feature_count"),
        "ego_mode": obs.get("ego_mode"),
        "reward": round(frame.reward, 6),
        "done": frame.done,
        "events": [asdict(event) for event in frame.events],
        "trace_hash": frame.trace_hash,
        "action": _action_to_dict(frame.action),
    }


def _vehicle_to_dict(vehicle: VehicleState) -> dict:
    return {
        "actor_id": vehicle.actor_id,
        "x": round(vehicle.x, 4),
        "y": round(vehicle.y, 4),
        "yaw": round(vehicle.yaw, 4),
        "speed": round(vehicle.speed, 4),
        "length": vehicle.length,
        "width": vehicle.width,
        "object_type": vehicle.object_type,
    }


def _action_to_dict(action: Action | None) -> dict | None:
    if action is None:
        return None
    return {"acceleration": round(action.acceleration, 6), "steering": round(action.steering, 6)}


def _html(data_json: str) -> str:
    template = """<!doctype html>
<html lang=\"zh-CN\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>WorldSimFlow Trace</title>
  <style>
    :root {
      color-scheme: light dark;
      --bg: #f6f7f9;
      --fg: #1f2933;
      --muted: #667085;
      --line: #d0d5dd;
      --road: #343a40;
      --lane: #f8fafc;
      --edge: #98a2b3;
      --ego: #2563eb;
      --actor: #f97316;
      --truck: #a855f7;
      --pedestrian: #22c55e;
      --cyclist: #06b6d4;
      --danger: #d92d20;
      --panel: #ffffff;
      --video-bg: #0d1117;
    }
    @media (prefers-color-scheme: dark) {
      :root {
        --bg: #101418;
        --fg: #edf2f7;
        --muted: #a0aec0;
        --line: #344054;
        --road: #242a31;
        --lane: #e5e7eb;
        --edge: #8a94a6;
        --panel: #171b21;
        --video-bg: #06080c;
      }
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--bg); color: var(--fg); font-family: Arial, Helvetica, sans-serif; }
    main { max-width: 1320px; margin: 0 auto; padding: 24px; }
    h1 { margin: 0 0 12px; font-size: 22px; font-weight: 600; }
    .summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin-bottom: 16px; }
    .metric { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px; min-width: 0; }
    .metric span { display: block; color: var(--muted); font-size: 12px; margin-bottom: 6px; }
    .metric strong { font-size: 16px; font-weight: 600; overflow-wrap: anywhere; }
    .viewer { display: grid; grid-template-columns: 1fr; gap: 14px; align-items: start; }
    body.has-video .viewer { grid-template-columns: minmax(0, 1fr) minmax(320px, 0.85fr); }
    .pane { min-width: 0; }
    .pane-head { display: flex; align-items: center; justify-content: space-between; gap: 10px; margin: 0 0 8px; }
    .pane h2 { margin: 0; font-size: 14px; font-weight: 600; color: var(--muted); }
    .mode { display: flex; gap: 6px; }
    canvas { width: 100%; height: auto; background: #2f6f47; border: 1px solid var(--line); border-radius: 8px; display: block; cursor: crosshair; }
    .video-pane { display: none; }
    body.has-video .video-pane { display: block; }
    video { width: 100%; aspect-ratio: 16 / 9; background: var(--video-bg); border: 1px solid var(--line); border-radius: 8px; display: block; }
    .controls { display: flex; flex-wrap: wrap; align-items: center; gap: 10px; margin: 14px 0; }
    button { border: 1px solid var(--line); background: var(--panel); color: var(--fg); border-radius: 6px; padding: 8px 12px; cursor: pointer; min-width: 64px; }
    button.active { border-color: var(--ego); color: var(--ego); font-weight: 600; }
    input[type=range] { flex: 1 1 260px; }
    .detail { color: var(--muted); font-size: 14px; overflow-wrap: anywhere; line-height: 1.55; }
    .event { color: var(--danger); font-weight: 600; }
    @media (max-width: 860px) { body.has-video .viewer { grid-template-columns: 1fr; } main { padding: 16px; } }
  </style>
</head>
<body>
  <main>
    <h1>WorldSimFlow Trace</h1>
    <section class=\"summary\" aria-label=\"trace summary\">
      <div class=\"metric\"><span>Scenario</span><strong id=\"scenarioId\"></strong></div>
      <div class=\"metric\"><span>Frame</span><strong id=\"frameText\"></strong></div>
      <div class=\"metric\"><span>Ego Speed</span><strong id=\"speedText\"></strong></div>
      <div class=\"metric\"><span>Action</span><strong id=\"actionText\"></strong></div>
      <div class=\"metric\"><span>Event</span><strong id=\"eventText\"></strong></div>
      <div class=\"metric\"><span>Trace Hash</span><strong id=\"hashText\"></strong></div>
      <div class=\"metric\"><span>Selected Actor</span><strong id=\"selectedActorText\">none</strong></div>
      <div class=\"metric\" id=\"videoMetric\"><span>Video Time</span><strong id=\"videoTimeText\">none</strong></div>
    </section>
    <section class=\"viewer\" aria-label=\"trace viewer\">
      <div class=\"pane sim-pane\">
        <div class=\"pane-head\">
          <h2>Simulation Birdview</h2>
          <div class=\"mode\">
            <button id=\"followBtn\" type=\"button\" class=\"active\">Follow Ego</button>
            <button id=\"globalBtn\" type=\"button\">Global</button>
          </div>
        </div>
        <canvas id=\"scene\" width=\"1080\" height=\"620\" aria-label=\"2D driving replay\"></canvas>
      </div>
      <div class=\"pane video-pane\" id=\"videoPane\">
        <h2>Real Camera Video</h2>
        <video id=\"realVideo\" controls preload=\"metadata\"></video>
      </div>
    </section>
    <div class=\"controls\">
      <button id=\"playBtn\" type=\"button\">Play</button>
      <input id=\"scrub\" type=\"range\" min=\"0\" value=\"0\" step=\"1\" aria-label=\"frame\">
      <button id=\"prevBtn\" type=\"button\">Prev</button>
      <button id=\"nextBtn\" type=\"button\">Next</button>
    </div>
    <p id=\"detail\" class=\"detail\"></p>
  </main>
  <script>
    const data = __TRACE_DATA__;
    const frames = data.frames;
    const mapFeatures = data.scenario.map_features || data.scenario.metadata.map_features || [];
    const realVideoConfig = data.real_video;
    const canvas = document.getElementById('scene');
    const ctx = canvas.getContext('2d');
    const playBtn = document.getElementById('playBtn');
    const prevBtn = document.getElementById('prevBtn');
    const nextBtn = document.getElementById('nextBtn');
    const followBtn = document.getElementById('followBtn');
    const globalBtn = document.getElementById('globalBtn');
    const scrub = document.getElementById('scrub');
    const scenarioId = document.getElementById('scenarioId');
    const frameText = document.getElementById('frameText');
    const speedText = document.getElementById('speedText');
    const actionText = document.getElementById('actionText');
    const eventText = document.getElementById('eventText');
    const hashText = document.getElementById('hashText');
    const detail = document.getElementById('detail');
    const realVideo = document.getElementById('realVideo');
    const videoTimeText = document.getElementById('videoTimeText');
    const selectedActorText = document.getElementById('selectedActorText');
    let index = 0;
    let timer = null;
    let syncingVideo = false;
    let viewMode = 'follow';
    let selectedActorId = null;
    let hitTargets = [];

    scenarioId.textContent = data.scenario.scenario_id;
    scrub.max = Math.max(0, frames.length - 1);
    if (realVideoConfig && realVideoConfig.src) {
      document.body.classList.add('has-video');
      realVideo.src = realVideoConfig.src;
    }

    function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
    function deg(rad) { return rad * 180 / Math.PI; }
    function clamp(value, min, max) { return Math.max(min, Math.min(max, value)); }

    function computeBounds() {
      const xs = [];
      const ys = [];
      for (const frame of frames) {
        for (const v of [frame.ego, ...frame.actors]) {
          xs.push(v.x - v.length / 2, v.x + v.length / 2);
          ys.push(v.y - v.width / 2, v.y + v.width / 2);
        }
      }
      for (const feature of mapFeatures) {
        for (const p of feature.polyline || []) { xs.push(p[0]); ys.push(p[1]); }
      }
      if (!xs.length) return {minX: -30, maxX: 80, minY: -40, maxY: 40};
      let minX = Math.min(...xs), maxX = Math.max(...xs);
      let minY = Math.min(...ys), maxY = Math.max(...ys);
      const padX = Math.max(20, (maxX - minX) * 0.08);
      const padY = Math.max(20, (maxY - minY) * 0.08);
      return {minX: minX - padX, maxX: maxX + padX, minY: minY - padY, maxY: maxY + padY};
    }
    const globalBounds = computeBounds();

    function globalTransform() {
      const w = canvas.width, h = canvas.height;
      const pad = 42;
      const sx = (w - pad * 2) / Math.max(1, globalBounds.maxX - globalBounds.minX);
      const sy = (h - pad * 2) / Math.max(1, globalBounds.maxY - globalBounds.minY);
      const scale = Math.max(1, Math.min(sx, sy));
      const usedW = (globalBounds.maxX - globalBounds.minX) * scale;
      const usedH = (globalBounds.maxY - globalBounds.minY) * scale;
      const ox = (w - usedW) / 2 - globalBounds.minX * scale;
      const oy = (h + usedH) / 2 + globalBounds.minY * scale;
      return {mode: 'global', scale, ox, oy};
    }

    function followTransform(frame) {
      const w = canvas.width, h = canvas.height;
      return {
        mode: 'follow',
        ego: frame.ego,
        scale: Math.min(w / 84, h / 92),
        cx: w / 2,
        cy: h * 0.72,
        forwardLimit: 70,
        backLimit: 25,
        lateralLimit: 42,
      };
    }

    function toLocal(x, y, ego) {
      const dx = x - ego.x;
      const dy = y - ego.y;
      const c = Math.cos(ego.yaw);
      const s = Math.sin(ego.yaw);
      return {forward: c * dx + s * dy, lateral: -s * dx + c * dy};
    }

    function worldToCanvas(x, y, t) {
      if (t.mode === 'follow') {
        const p = toLocal(x, y, t.ego);
        return {x: t.cx + p.lateral * t.scale, y: t.cy - p.forward * t.scale, local: p};
      }
      return {x: t.ox + x * t.scale, y: t.oy - y * t.scale};
    }

    function vehicleAngle(vehicle, t) {
      if (t.mode === 'follow') return vehicle.yaw - t.ego.yaw - Math.PI / 2;
      return -vehicle.yaw;
    }

    function videoTimeForFrame(frameIndex) {
      const offset = Number(realVideoConfig?.frame_offset || 0);
      const fps = Number(realVideoConfig?.fps || 0);
      if (fps > 0) return Math.max(0, (frameIndex + offset) / fps);
      const frame = frames[Math.max(0, Math.min(frameIndex, frames.length - 1))];
      return Math.max(0, frame.step * Number(data.scenario.dt || 0));
    }
    function frameFromVideoTime(time) {
      const offset = Number(realVideoConfig?.frame_offset || 0);
      const fps = Number(realVideoConfig?.fps || 0);
      if (fps > 0) return Math.round(time * fps - offset);
      return Math.round(time / Number(data.scenario.dt || 0.1));
    }
    function syncVideoToFrame(frameIndex) {
      if (!realVideoConfig || !realVideo.src) return;
      const target = videoTimeForFrame(frameIndex);
      videoTimeText.textContent = `${target.toFixed(2)} s`;
      if (!Number.isFinite(target) || realVideo.readyState === 0) return;
      if (Math.abs(realVideo.currentTime - target) > 0.08) {
        syncingVideo = true;
        realVideo.currentTime = target;
        window.setTimeout(() => { syncingVideo = false; }, 120);
      }
    }

    function featureColor(type) {
      if (type.includes('ROAD_EDGE')) return css('--edge');
      if (type.includes('YELLOW')) return '#facc15';
      if (type.includes('BROKEN')) return css('--lane');
      if (type.includes('SOLID')) return '#ffffff';
      if (type.includes('LANE')) return '#cbd5e1';
      return 'rgba(255,255,255,0.45)';
    }

    function drawMapFeatures(t) {
      if (!mapFeatures.length) return false;
      ctx.save();
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      for (const feature of mapFeatures) {
        const points = feature.polyline || [];
        if (points.length < 2) continue;
        ctx.beginPath();
        let visible = false;
        for (let i = 0; i < points.length; i++) {
          const p = worldToCanvas(points[i][0], points[i][1], t);
          if (p.x >= -80 && p.x <= canvas.width + 80 && p.y >= -80 && p.y <= canvas.height + 80) visible = true;
          if (i === 0) ctx.moveTo(p.x, p.y); else ctx.lineTo(p.x, p.y);
        }
        if (!visible) continue;
        ctx.strokeStyle = featureColor(feature.type || '');
        ctx.globalAlpha = feature.type?.includes('ROAD_EDGE') ? 0.75 : 0.92;
        ctx.lineWidth = feature.type?.includes('BROKEN') ? 1.4 : 1.8;
        ctx.setLineDash(feature.type?.includes('BROKEN') ? [12, 12] : []);
        ctx.stroke();
      }
      ctx.setLineDash([]);
      ctx.restore();
      return true;
    }

    function drawFallbackRoad(t) {
      if (t.mode === 'follow') {
        ctx.save();
        ctx.strokeStyle = 'rgba(255,255,255,0.20)';
        ctx.lineWidth = 1;
        for (let y = -36; y <= 36; y += 12) {
          const a = {x: t.cx + y * t.scale, y: t.cy + t.backLimit * t.scale};
          const b = {x: t.cx + y * t.scale, y: t.cy - t.forwardLimit * t.scale};
          ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
        }
        ctx.restore();
        return;
      }
      const road = data.scenario.road;
      const laneWidth = Number(road.lane_width);
      const laneCount = Number(road.lane_count);
      const halfWidth = laneWidth * laneCount / 2;
      const left = globalBounds.minX;
      const right = globalBounds.maxX;
      const top = worldToCanvas(left, halfWidth, t);
      const bottom = worldToCanvas(right, -halfWidth, t);
      ctx.fillStyle = css('--road');
      ctx.fillRect(top.x, top.y, bottom.x - top.x, bottom.y - top.y);
      ctx.strokeStyle = css('--lane');
      ctx.lineWidth = 1.2;
      for (let i = 1; i < laneCount; i++) {
        const y = -halfWidth + laneWidth * i;
        const a = worldToCanvas(left, y, t);
        const b = worldToCanvas(right, y, t);
        ctx.setLineDash([14, 14]);
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
      }
      ctx.setLineDash([]);
    }

    function fmt(value, digits = 2, suffix = '') {
      if (value === null || value === undefined || !Number.isFinite(Number(value))) return 'n/a';
      return `${Number(value).toFixed(digits)}${suffix}`;
    }

    function drawHud(frame, visibleActors) {
      const action = frame.action || {acceleration: 0, steering: 0};
      const typeCounts = actorTypeCounts(frame);
      const typeText = `car=${typeCounts.car || 0} truck=${typeCounts.truck || 0} ped=${typeCounts.pedestrian || 0} cyc=${typeCounts.cyclist || 0}`;
      const lines = [
        ['Frame', `${index + 1}/${frames.length}`],
        ['Time', fmt(frame.step * Number(data.scenario.dt || 0), 2, ' s')],
        ['Mode', `${frame.ego_mode || data.scenario.metadata.ego_mode || 'closed_loop'} / ${viewMode}`],
        ['Speed', fmt(frame.ego.speed, 2, ' m/s')],
        ['Yaw', fmt(deg(frame.ego.yaw), 1, ' deg')],
        ['Pos', `x=${fmt(frame.ego.x, 1)} y=${fmt(frame.ego.y, 1)}`],
        ['Action', `a=${fmt(action.acceleration, 2)} steer=${fmt(action.steering, 2)}`],
        ['Front Gap', fmt(frame.front_gap, 2, ' m')],
        ['Nearest', fmt(frame.closest_actor_distance, 2, ' m')],
        ['Actors', `${visibleActors}/${frame.actors.length} nearby=${frame.nearby_actor_count ?? 'n/a'}`],
        ['Types', typeText],
        ['Map Lines', `${frame.map_feature_count ?? mapFeatures.length}`],
        ['Reward', fmt(frame.reward, 2)],
      ];
      const eventText = frame.events.length ? frame.events.map(e => e.code).join(',') : 'none';
      lines.push(['Event', eventText]);

      ctx.save();
      ctx.font = '12px Arial, Helvetica, sans-serif';
      const labelWidth = 72;
      const valueWidth = Math.max(...lines.map(([_, value]) => ctx.measureText(String(value)).width));
      const width = Math.min(canvas.width - 24, Math.max(248, labelWidth + valueWidth + 30));
      const lineHeight = 18;
      const height = lines.length * lineHeight + 18;
      const x = 14;
      const y = 14;
      ctx.fillStyle = 'rgba(15, 23, 42, 0.78)';
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.22)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.roundRect(x, y, width, height, 8);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = '#ffffff';
      ctx.font = '600 13px Arial, Helvetica, sans-serif';
      ctx.fillText('Ego Diagnostics', x + 12, y + 19);
      ctx.font = '12px Arial, Helvetica, sans-serif';
      for (let i = 0; i < lines.length; i++) {
        const [label, value] = lines[i];
        const yy = y + 40 + i * lineHeight;
        ctx.fillStyle = 'rgba(226, 232, 240, 0.72)';
        ctx.fillText(label, x + 12, yy);
        ctx.fillStyle = label === 'Event' && value !== 'none' ? '#fecaca' : '#ffffff';
        ctx.fillText(String(value), x + labelWidth + 12, yy);
      }
      ctx.restore();
    }

    function actorKind(vehicle, isEgo = false) {
      if (isEgo || vehicle.object_type === 'EGO') return 'ego';
      const type = String(vehicle.object_type || 'VEHICLE').toUpperCase();
      if (type.includes('PEDESTRIAN')) return 'pedestrian';
      if (type.includes('CYCLIST') || type.includes('BICYCLE')) return 'cyclist';
      if (type.includes('CONE') || type.includes('BARRIER')) return 'obstacle';
      if (Number(vehicle.length) >= 7.0 || Number(vehicle.width) >= 2.6) return 'truck';
      return 'car';
    }

    function actorColor(vehicle, isEgo = false) {
      const kind = actorKind(vehicle, isEgo);
      if (kind === 'ego') return css('--ego');
      if (kind === 'truck') return css('--truck');
      if (kind === 'pedestrian') return css('--pedestrian');
      if (kind === 'cyclist') return css('--cyclist');
      if (kind === 'obstacle') return '#facc15';
      return css('--actor');
    }

    function actorTypeCounts(frame) {
      const counts = {car: 0, truck: 0, pedestrian: 0, cyclist: 0, obstacle: 0};
      for (const actor of frame.actors) {
        const kind = actorKind(actor, false);
        counts[kind] = (counts[kind] || 0) + 1;
      }
      return counts;
    }

    function selectedActorInfo(frame) {
      if (!selectedActorId) return null;
      const actor = frame.actors.find(item => item.actor_id === selectedActorId);
      if (!actor) return null;
      const loc = toLocal(actor.x, actor.y, frame.ego);
      return {
        actor,
        kind: actorKind(actor),
        distance: Math.hypot(actor.x - frame.ego.x, actor.y - frame.ego.y),
        forward: loc.forward,
        lateral: loc.lateral,
      };
    }

    function selectedActorTextFor(frame) {
      const info = selectedActorInfo(frame);
      if (!info) return 'none';
      return `${info.actor.actor_id} ${info.kind} ${fmt(info.actor.speed, 1, ' m/s')}`;
    }

    function selectedActorDetail(frame) {
      const info = selectedActorInfo(frame);
      if (!info) return 'selected_actor=none';
      const a = info.actor;
      return `selected_actor=${a.actor_id}, type=${a.object_type || 'UNKNOWN'}, kind=${info.kind}, speed=${fmt(a.speed, 2, ' m/s')}, yaw=${fmt(deg(a.yaw), 1, ' deg')}, distance=${fmt(info.distance, 2, ' m')}, forward=${fmt(info.forward, 2, ' m')}, lateral=${fmt(info.lateral, 2, ' m')}`;
    }
    function drawVehicle(vehicle, color, t, isEgo = false) {
      const p = worldToCanvas(vehicle.x, vehicle.y, t);
      if (t.mode === 'follow') {
        const loc = p.local;
        if (loc.forward < -t.backLimit - 8 || loc.forward > t.forwardLimit + 8 || Math.abs(loc.lateral) > t.lateralLimit + 8) return;
      }
      const hitRadius = Math.max(9, Math.max(vehicle.length, vehicle.width) * t.scale * 0.65);
      if (!isEgo) hitTargets.push({actor_id: vehicle.actor_id, x: p.x, y: p.y, radius: hitRadius});
      ctx.save();
      ctx.translate(p.x, p.y);
      ctx.rotate(vehicleAngle(vehicle, t));
      const kind = actorKind(vehicle, isEgo);
      color = actorColor(vehicle, isEgo);
      const length = Math.max(isEgo ? 16 : 10, vehicle.length * t.scale);
      const width = Math.max(isEgo ? 7 : 5, vehicle.width * t.scale);
      ctx.fillStyle = color;
      ctx.strokeStyle = 'rgba(0,0,0,0.38)';
      ctx.lineWidth = 1;
      if (kind === 'pedestrian') {
        ctx.beginPath();
        ctx.arc(0, 0, Math.max(4, width * 0.55), 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      } else if (kind === 'cyclist') {
        ctx.beginPath();
        ctx.moveTo(0, -Math.max(5, width * 0.75));
        ctx.lineTo(Math.max(8, length * 0.45), 0);
        ctx.lineTo(0, Math.max(5, width * 0.75));
        ctx.lineTo(-Math.max(8, length * 0.45), 0);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
      } else {
        ctx.beginPath();
        ctx.roundRect(-length / 2, -width / 2, length, width, kind === 'truck' ? 2 : 4);
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = '#ffffff';
        ctx.globalAlpha = 0.88;
        if (kind === 'truck') {
          ctx.fillRect(length * 0.20, -width * 0.34, length * 0.18, width * 0.68);
          ctx.globalAlpha = 0.25;
          ctx.fillRect(-length * 0.42, -width * 0.36, length * 0.42, width * 0.72);
        } else {
          ctx.fillRect(length * 0.16, -width * 0.28, length * 0.18, width * 0.56);
        }
      }
      if (!isEgo && vehicle.actor_id === selectedActorId) {
        ctx.globalAlpha = 1;
        ctx.strokeStyle = '#fef08a';
        ctx.lineWidth = 3;
        ctx.setLineDash([8, 5]);
        if (kind === 'pedestrian') {
          ctx.beginPath();
          ctx.arc(0, 0, Math.max(8, width * 0.85), 0, Math.PI * 2);
          ctx.stroke();
        } else {
          ctx.strokeRect(-length / 2 - 4, -width / 2 - 4, length + 8, width + 8);
        }
        ctx.setLineDash([]);
      }
      ctx.restore();
    }

    function draw(i, options = {}) {
      if (!frames.length) return;
      index = clamp(i, 0, frames.length - 1);
      scrub.value = index;
      const frame = frames[index];
      const t = viewMode === 'follow' ? followTransform(frame) : globalTransform();
      hitTargets = [];
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#2f6f47';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      const hasMap = drawMapFeatures(t);
      if (!hasMap) drawFallbackRoad(t);
      for (const actor of frame.actors) drawVehicle(actor, actorColor(actor), t, false);
      drawVehicle(frame.ego, actorColor(frame.ego, true), t, true);
      const visibleActors = t.mode === 'follow'
        ? frame.actors.filter(a => {
            const loc = toLocal(a.x, a.y, frame.ego);
            return loc.forward >= -t.backLimit && loc.forward <= t.forwardLimit && Math.abs(loc.lateral) <= t.lateralLimit;
          }).length
        : frame.actors.length;
      drawHud(frame, visibleActors);

      frameText.textContent = `${index + 1} / ${frames.length}`;
      speedText.textContent = `${frame.ego.speed.toFixed(2)} m/s`;
      const action = frame.action || {acceleration: 0, steering: 0};
      actionText.textContent = `a=${action.acceleration.toFixed(2)}, steer=${action.steering.toFixed(2)}`;
      const events = frame.events.map(e => e.code).join(', ');
      eventText.innerHTML = events ? `<span class=\"event\">${events}</span>` : 'none';
      hashText.textContent = frame.trace_hash.slice(0, 12) + '...';
      selectedActorText.textContent = selectedActorTextFor(frame);
      const gap = frame.front_gap === null ? 'none' : `${frame.front_gap.toFixed(2)} m`;
      detail.textContent = `mode=${viewMode}, map_features=${mapFeatures.length}, ego x=${frame.ego.x.toFixed(2)} m, y=${frame.ego.y.toFixed(2)} m, yaw=${deg(frame.ego.yaw).toFixed(1)} deg, speed=${frame.ego.speed.toFixed(2)} m/s, front_gap=${gap}, nearest=${fmt(frame.closest_actor_distance, 2, ' m')}, visible_actors=${visibleActors}/${frame.actors.length}, reward=${frame.reward.toFixed(3)}`;
      if (!options.fromVideo) syncVideoToFrame(index);
    }

    function stop() {
      if (timer) clearInterval(timer);
      timer = null;
      playBtn.textContent = 'Play';
      if (realVideoConfig && !realVideo.paused) realVideo.pause();
    }
    function setMode(mode) {
      viewMode = mode;
      followBtn.classList.toggle('active', mode === 'follow');
      globalBtn.classList.toggle('active', mode === 'global');
      draw(index);
    }
    playBtn.addEventListener('click', () => {
      if (timer) { stop(); return; }
      playBtn.textContent = 'Pause';
      if (realVideoConfig) { syncVideoToFrame(index); realVideo.play().catch(() => {}); }
      timer = setInterval(() => {
        if (index >= frames.length - 1) { stop(); return; }
        draw(index + 1);
      }, Math.max(40, Number(data.scenario.dt || 0.1) * 1000));
    });
    prevBtn.addEventListener('click', () => { stop(); draw(index - 1); });
    nextBtn.addEventListener('click', () => { stop(); draw(index + 1); });
    followBtn.addEventListener('click', () => setMode('follow'));
    globalBtn.addEventListener('click', () => setMode('global'));
    scrub.addEventListener('input', () => { stop(); draw(Number(scrub.value)); });
    realVideo.addEventListener('timeupdate', () => {
      if (!realVideoConfig || syncingVideo || timer) return;
      draw(frameFromVideoTime(realVideo.currentTime), {fromVideo: true});
      videoTimeText.textContent = `${realVideo.currentTime.toFixed(2)} s`;
    });
    realVideo.addEventListener('pause', () => { if (timer) stop(); });
    canvas.addEventListener('click', event => {
      const rect = canvas.getBoundingClientRect();
      const x = (event.clientX - rect.left) * (canvas.width / rect.width);
      const y = (event.clientY - rect.top) * (canvas.height / rect.height);
      let picked = null;
      for (let i = hitTargets.length - 1; i >= 0; i--) {
        const target = hitTargets[i];
        if (Math.hypot(x - target.x, y - target.y) <= target.radius) {
          picked = target.actor_id;
          break;
        }
      }
      selectedActorId = picked === selectedActorId ? null : picked;
      draw(index);
    });    window.addEventListener('resize', () => draw(index));
    draw(0);
  </script>
</body>
</html>
"""
    return template.replace('__TRACE_DATA__', data_json)
