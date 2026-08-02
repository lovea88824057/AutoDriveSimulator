from __future__ import annotations

import html
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrafficDiagnosticsConfig:
    """Configuration for C3 traffic diagnostics dashboard generation."""

    title: str = "WorldSimFlow Traffic Diagnostics Dashboard"
    no_front_gap_threshold: float = 999.0
    float_digits: int = 6


class TrafficDiagnosticsDashboard:
    """Build an explainable traffic diagnostics report from transition JSONL data.

    The dashboard is intentionally dependency-free. It reads the same transition_v1
    JSONL used by RL/world-model exports and turns the key signals into a static HTML
    page: speed, lane deviation, front gap, actions, rewards and events.
    """

    def __init__(self, config: TrafficDiagnosticsConfig | None = None):
        self.config = config or TrafficDiagnosticsConfig()

    def build(
        self,
        *,
        observations_jsonl: str | Path,
        output_html: str | Path,
        summary_json: str | Path | None = None,
        observation_report: str | Path | None = None,
        lane_diagnostics: str | Path | None = None,
        run_report: str | Path | None = None,
    ) -> dict[str, Any]:
        observations_path = Path(observations_jsonl)
        output_path = Path(output_html)
        records = self._read_jsonl(observations_path)
        report = self._read_optional_json(observation_report)
        lane_report = self._read_optional_json(lane_diagnostics)
        run = self._read_optional_json(run_report)
        payload = self._payload(records, observations_path, report, lane_report, run)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self._render_html(payload), encoding="utf-8", newline="\n")

        summary = payload["summary"].copy()
        summary.update(
            {
                "dashboard_schema_version": "traffic_diagnostics_v1",
                "html": str(output_path),
                "observations_jsonl": str(observations_path),
            }
        )
        if summary_json:
            summary_path = Path(summary_json)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
        return summary

    def _read_jsonl(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(f"observations_jsonl not found: {path}")
        rows: list[dict[str, Any]] = []
        for lineno, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{lineno}: {exc}") from exc
            if row.get("record_type") == "transition" or {"obs", "action", "reward", "next_obs"}.issubset(row):
                rows.append(row)
        if not rows:
            raise ValueError(f"No transition records found in {path}")
        return rows

    def _read_optional_json(self, path: str | Path | None) -> dict[str, Any]:
        if not path:
            return {}
        target = Path(path)
        if not target.exists():
            return {"missing_path": str(target)}
        try:
            value = json.loads(target.read_text(encoding="utf-8-sig"))
        except Exception as exc:
            return {"error": str(exc), "path": str(target)}
        return value if isinstance(value, dict) else {"value": value}

    def _payload(
        self,
        records: list[dict[str, Any]],
        observations_path: Path,
        observation_report: dict[str, Any],
        lane_report: dict[str, Any],
        run_report: dict[str, Any],
    ) -> dict[str, Any]:
        episodes = sorted({int(row.get("episode", 0) or 0) for row in records})
        scenario_ids = sorted({str(row.get("scenario_id", "unknown")) for row in records})
        feature_names = self._feature_names(records)
        series = self._series(records)
        summary = self._summary(records, series, scenario_ids, episodes, feature_names, observations_path)
        return {
            "summary": summary,
            "series": series,
            "feature_names": feature_names,
            "event_counts": dict(Counter(code for row in records for code in row.get("event_codes", []))),
            "episode_rewards": self._episode_rewards(records),
            "sources": {
                "observations_jsonl": str(observations_path),
                "observation_report": observation_report,
                "lane_diagnostics": self._lane_summary(lane_report),
                "run_report": self._run_summary(run_report),
            },
        }

    def _feature_names(self, records: list[dict[str, Any]]) -> list[str]:
        for row in records:
            obs = row.get("obs") or {}
            names = obs.get("feature_names")
            if isinstance(names, list) and names:
                return [str(name) for name in names]
        return []

    def _series(self, records: list[dict[str, Any]]) -> dict[str, list[Any]]:
        fields = [
            "episode",
            "step",
            "ego_speed",
            "ego_accel",
            "ego_yaw_rate",
            "ego_lane_l",
            "ego_heading_error",
            "route_progress",
            "front_gap",
            "front_relative_speed",
            "nearest_actor_distance",
            "nearby_actor_count",
            "reactive_actor_count",
            "acceleration",
            "steering",
            "reward",
            "collision_flag",
            "offroad_flag",
            "stale_replay_flag",
            "event_label",
        ]
        series: dict[str, list[Any]] = {field: [] for field in fields}
        for index, row in enumerate(records):
            obs = row.get("obs") or {}
            action = row.get("action") or {}
            step = int(row.get("step", obs.get("step", index)) or 0)
            event_codes = [str(code) for code in row.get("event_codes", [])]
            series["episode"].append(int(row.get("episode", 0) or 0))
            series["step"].append(step)
            for key in fields:
                if key in {"episode", "step", "acceleration", "steering", "reward", "event_label"}:
                    continue
                series[key].append(self._round(self._obs_value(obs, key)))
            series["acceleration"].append(self._round(self._number(action.get("acceleration"))))
            series["steering"].append(self._round(self._number(action.get("steering"))))
            series["reward"].append(self._round(self._number(row.get("reward"))))
            series["event_label"].append(",".join(event_codes) if event_codes else "")
        return series

    def _summary(
        self,
        records: list[dict[str, Any]],
        series: dict[str, list[Any]],
        scenario_ids: list[str],
        episodes: list[int],
        feature_names: list[str],
        observations_path: Path,
    ) -> dict[str, Any]:
        front_values = [value for value in series["front_gap"] if isinstance(value, (int, float)) and value < self.config.no_front_gap_threshold]
        rewards = [float(value) for value in series["reward"] if isinstance(value, (int, float))]
        event_counts = Counter(code for row in records for code in row.get("event_codes", []))
        max_abs_lane_l = self._max_abs(series["ego_lane_l"])
        max_abs_heading_error = self._max_abs(series["ego_heading_error"])
        min_front_gap = min(front_values) if front_values else None
        risk_notes = self._risk_notes(max_abs_lane_l, max_abs_heading_error, min_front_gap, event_counts)
        return {
            "scenario_ids": scenario_ids,
            "episodes": len(episodes),
            "episode_ids": episodes,
            "transition_count": len(records),
            "feature_count": len(feature_names),
            "observation_schema_version": self._first(records, ["obs", "schema_version"]),
            "dataset_schema_version": str(records[0].get("dataset_schema_version", "unknown")),
            "total_reward": self._round(sum(rewards)),
            "mean_reward": self._round(sum(rewards) / len(rewards)) if rewards else 0.0,
            "min_reward": self._round(min(rewards)) if rewards else 0.0,
            "max_reward": self._round(max(rewards)) if rewards else 0.0,
            "min_front_gap": self._round(min_front_gap),
            "max_abs_ego_lane_l": self._round(max_abs_lane_l),
            "max_abs_heading_error": self._round(max_abs_heading_error),
            "event_count": sum(event_counts.values()),
            "event_codes": dict(event_counts),
            "trace_hash_tail": str(records[-1].get("final_trace_hash_so_far") or records[-1].get("trace_hash") or ""),
            "observations_jsonl": str(observations_path),
            "risk_notes": risk_notes,
        }

    def _episode_rewards(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rewards: dict[int, float] = defaultdict(float)
        counts: dict[int, int] = defaultdict(int)
        for row in records:
            episode = int(row.get("episode", 0) or 0)
            rewards[episode] += self._number(row.get("reward"))
            counts[episode] += 1
        return [
            {"episode": episode, "transition_count": counts[episode], "total_reward": self._round(rewards[episode])}
            for episode in sorted(rewards)
        ]

    def _lane_summary(self, lane_report: dict[str, Any]) -> dict[str, Any]:
        if not lane_report:
            return {}
        actor = lane_report.get("actor") if isinstance(lane_report.get("actor"), dict) else {}
        actor_summary = actor.get("summary", {}) if isinstance(actor, dict) else {}
        return {
            "base_scenario_id": lane_report.get("base_scenario_id"),
            "target_scenario_id": lane_report.get("target_scenario_id"),
            "lane_graph": lane_report.get("lane_graph", {}),
            "ego": lane_report.get("ego", {}),
            "actor_id": actor.get("actor_id") if isinstance(actor, dict) else None,
            "actor_summary": actor_summary,
            "missing_path": lane_report.get("missing_path"),
            "error": lane_report.get("error"),
        }

    def _run_summary(self, run_report: dict[str, Any]) -> dict[str, Any]:
        if not run_report:
            return {}
        keys = ["scenario_id", "steps", "done", "events", "trace_hash", "reward", "html", "metrics"]
        return {key: run_report.get(key) for key in keys if key in run_report}

    def _risk_notes(
        self,
        max_abs_lane_l: float | None,
        max_abs_heading_error: float | None,
        min_front_gap: float | None,
        event_counts: Counter,
    ) -> list[str]:
        notes: list[str] = []
        if event_counts:
            notes.append("存在仿真事件，需要结合 event timeline 定位发生帧。")
        if max_abs_lane_l is not None and max_abs_lane_l > 1.75:
            notes.append("ego_lane_l 绝对值超过半个常见车道宽，建议检查是否跨线或 lane projection 是否合理。")
        if max_abs_heading_error is not None and max_abs_heading_error > 0.35:
            notes.append("heading_error 较大，可能存在车头方向与参考车道方向不一致。")
        if min_front_gap is not None and min_front_gap < 5.0:
            notes.append("front_gap 小于 5m，属于近距离跟驰或碰撞风险场景。")
        if not notes:
            notes.append("未发现明显风险信号，可继续观察曲线平滑性和 trace_hash 稳定性。")
        return notes

    def _obs_value(self, obs: dict[str, Any], name: str) -> float:
        if name in obs:
            return self._number(obs.get(name))
        names = obs.get("feature_names") or []
        values = obs.get("state_vector") or []
        if isinstance(names, list) and isinstance(values, list) and name in names:
            index = names.index(name)
            if index < len(values):
                return self._number(values[index])
        return 0.0

    def _number(self, value: Any) -> float:
        if value is None or value == "":
            return 0.0
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.0
        return number if math.isfinite(number) else 0.0

    def _round(self, value: Any) -> float | None:
        if value is None:
            return None
        return round(float(value), self.config.float_digits)

    def _max_abs(self, values: list[Any]) -> float | None:
        numeric = [abs(float(value)) for value in values if isinstance(value, (int, float)) and math.isfinite(float(value))]
        return max(numeric) if numeric else None

    def _first(self, records: list[dict[str, Any]], path: list[str]) -> Any:
        value: Any = records[0]
        for key in path:
            if not isinstance(value, dict):
                return None
            value = value.get(key)
        return value

    def _render_html(self, payload: dict[str, Any]) -> str:
        payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        escaped_title = html.escape(self.config.title)
        return f"""<!doctype html>
<html lang=\"zh-CN\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{escaped_title}</title>
  <style>
    :root {{
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #18212f;
      --muted: #687386;
      --line: #d9dee8;
      --blue: #2364d2;
      --green: #17895d;
      --red: #c83f46;
      --amber: #af6a11;
      --violet: #6f4cc3;
      --shadow: 0 8px 24px rgba(25, 35, 50, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--ink); font: 14px/1.5 Arial, 'Microsoft YaHei', sans-serif; }}
    header {{ background: #17202d; color: white; padding: 22px 28px; }}
    h1 {{ margin: 0; font-size: 24px; letter-spacing: 0; }}
    header p {{ margin: 6px 0 0; color: #c6ced9; max-width: 980px; }}
    main {{ max-width: 1320px; margin: 0 auto; padding: 22px; }}
    .grid {{ display: grid; gap: 16px; }}
    .summary {{ grid-template-columns: repeat(6, minmax(0, 1fr)); }}
    .charts {{ grid-template-columns: repeat(2, minmax(0, 1fr)); margin-top: 16px; }}
    .panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; box-shadow: var(--shadow); padding: 16px; min-width: 0; }}
    .metric {{ min-height: 104px; }}
    .label {{ color: var(--muted); font-size: 12px; }}
    .value {{ font-size: 24px; font-weight: 700; margin-top: 8px; word-break: break-word; }}
    .sub {{ color: var(--muted); margin-top: 4px; font-size: 12px; }}
    h2 {{ margin: 0 0 12px; font-size: 16px; }}
    canvas {{ width: 100%; height: 230px; display: block; border: 1px solid #edf0f5; border-radius: 6px; background: #fbfcfe; }}
    .wide {{ grid-column: 1 / -1; }}
    .controls {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 6px 0 14px; }}
    button {{ border: 1px solid var(--line); background: #fff; color: var(--ink); border-radius: 6px; padding: 7px 10px; cursor: pointer; }}
    button.active {{ background: var(--blue); color: white; border-color: var(--blue); }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ border-bottom: 1px solid #edf0f5; padding: 8px 6px; text-align: left; vertical-align: top; }}
    th {{ color: var(--muted); font-weight: 600; }}
    code {{ background: #eef2f7; border-radius: 4px; padding: 2px 5px; }}
    .notes {{ margin: 0; padding-left: 18px; }}
    .notes li {{ margin: 6px 0; }}
    .pill {{ display: inline-block; padding: 2px 8px; border-radius: 999px; background: #eef2f7; color: #314155; margin: 2px 4px 2px 0; }}
    .risk {{ color: var(--amber); font-weight: 700; }}
    @media (max-width: 980px) {{ .summary, .charts {{ grid-template-columns: 1fr 1fr; }} }}
    @media (max-width: 640px) {{ main {{ padding: 14px; }} .summary, .charts {{ grid-template-columns: 1fr; }} h1 {{ font-size: 21px; }} }}
  </style>
</head>
<body>
  <header>
    <h1>{escaped_title}</h1>
    <p>把 transition 数据中的速度、车道偏移、前车距离、动作、reward 和事件放到同一个时间轴上，用于定位仿真异常和解释干预效果。</p>
  </header>
  <main>
    <section class=\"grid summary\" id=\"summary\"></section>

    <section class=\"panel wide\" style=\"margin-top:16px\">
      <h2>诊断结论</h2>
      <ul class=\"notes\" id=\"riskNotes\"></ul>
    </section>

    <section class=\"grid charts\">
      <div class=\"panel\"><h2>速度 / 加速度</h2><canvas id=\"speedChart\"></canvas></div>
      <div class=\"panel\"><h2>车道横向偏移 / 航向误差</h2><canvas id=\"laneChart\"></canvas></div>
      <div class=\"panel\"><h2>前车距离 / 最近目标距离</h2><canvas id=\"gapChart\"></canvas></div>
      <div class=\"panel\"><h2>动作时间线</h2><canvas id=\"actionChart\"></canvas></div>
      <div class=\"panel\"><h2>Reward 时间线</h2><canvas id=\"rewardChart\"></canvas></div>
      <div class=\"panel\"><h2>事件时间线</h2><canvas id=\"eventChart\"></canvas></div>
    </section>

    <section class=\"panel wide\" style=\"margin-top:16px\">
      <h2>输入与辅助诊断</h2>
      <div id=\"sourceInfo\"></div>
    </section>

    <section class=\"panel wide\" style=\"margin-top:16px\">
      <h2>Episode Reward</h2>
      <table id=\"episodeTable\"></table>
    </section>
  </main>
  <script id=\"payload\" type=\"application/json\">{payload_json}</script>
  <script>
    const payload = JSON.parse(document.getElementById('payload').textContent);
    const s = payload.summary;
    const series = payload.series;

    function fmt(v) {{
      if (v === null || v === undefined || v === '') return '-';
      if (typeof v === 'number') return Number.isInteger(v) ? String(v) : v.toFixed(3);
      return String(v);
    }}

    function addSummary() {{
      const items = [
        ['场景', (s.scenario_ids || []).join(', ') || 'unknown', `episodes=${{s.episodes}}`],
        ['Transitions', s.transition_count, `schema=${{s.observation_schema_version || 'unknown'}}`],
        ['Total Reward', fmt(s.total_reward), `mean=${{fmt(s.mean_reward)}}`],
        ['Min Front Gap', fmt(s.min_front_gap), '排除无前车的 1000m 占位值'],
        ['Max |lane_l|', fmt(s.max_abs_ego_lane_l), '横向偏移越大，越需要检查跨线'],
        ['Events', s.event_count, Object.keys(s.event_codes || {{}}).join(', ') || 'none'],
      ];
      document.getElementById('summary').innerHTML = items.map(([label, value, sub]) => `
        <div class=\"panel metric\"><div class=\"label\">${{label}}</div><div class=\"value\">${{value}}</div><div class=\"sub\">${{sub}}</div></div>
      `).join('');
      document.getElementById('riskNotes').innerHTML = (s.risk_notes || []).map(note => `<li>${{note}}</li>`).join('');
    }}

    function drawChart(canvasId, specs, opts={{}}) {{
      const canvas = document.getElementById(canvasId);
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.max(320, Math.floor(rect.width * dpr));
      canvas.height = Math.max(210, Math.floor(rect.height * dpr));
      const ctx = canvas.getContext('2d');
      ctx.scale(dpr, dpr);
      const w = canvas.width / dpr, h = canvas.height / dpr;
      const pad = {{left: 46, right: 16, top: 18, bottom: 34}};
      ctx.clearRect(0, 0, w, h);
      ctx.strokeStyle = '#d9dee8';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(pad.left, pad.top);
      ctx.lineTo(pad.left, h - pad.bottom);
      ctx.lineTo(w - pad.right, h - pad.bottom);
      ctx.stroke();

      const xValues = series.step || [];
      const points = [];
      specs.forEach(spec => {{
        (series[spec.key] || []).forEach((y, i) => {{
          if (typeof y === 'number' && Number.isFinite(y)) points.push(y);
        }});
      }});
      if (!xValues.length || !points.length) return;
      let minY = opts.minY ?? Math.min(...points);
      let maxY = opts.maxY ?? Math.max(...points);
      if (Math.abs(maxY - minY) < 1e-9) {{ maxY += 1; minY -= 1; }}
      const minX = Math.min(...xValues), maxX = Math.max(...xValues);
      const xScale = x => pad.left + ((x - minX) / Math.max(1, maxX - minX)) * (w - pad.left - pad.right);
      const yScale = y => h - pad.bottom - ((y - minY) / (maxY - minY)) * (h - pad.top - pad.bottom);

      ctx.fillStyle = '#687386';
      ctx.font = '12px Arial';
      ctx.fillText(fmt(maxY), 6, pad.top + 4);
      ctx.fillText(fmt(minY), 6, h - pad.bottom + 4);
      ctx.fillText(String(minX), pad.left, h - 10);
      ctx.fillText(String(maxX), w - pad.right - 28, h - 10);

      specs.forEach(spec => {{
        const values = series[spec.key] || [];
        ctx.strokeStyle = spec.color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        let started = false;
        values.forEach((y, i) => {{
          if (typeof y !== 'number' || !Number.isFinite(y)) return;
          const x = xScale(xValues[i]);
          const yy = yScale(y);
          if (!started) {{ ctx.moveTo(x, yy); started = true; }} else {{ ctx.lineTo(x, yy); }}
        }});
        ctx.stroke();
      }});

      let lx = pad.left, ly = 12;
      specs.forEach(spec => {{
        ctx.fillStyle = spec.color;
        ctx.fillRect(lx, ly - 8, 10, 10);
        ctx.fillStyle = '#18212f';
        ctx.fillText(spec.label, lx + 14, ly + 1);
        lx += spec.label.length * 7 + 34;
      }});
    }}

    function addSources() {{
      const src = payload.sources || {{}};
      const eventPills = Object.entries(payload.event_counts || {{}}).map(([k, v]) => `<span class=\"pill\">${{k}}=${{v}}</span>`).join('') || '<span class=\"pill\">none</span>';
      const lane = src.lane_diagnostics || {{}};
      const run = src.run_report || {{}};
      document.getElementById('sourceInfo').innerHTML = `
        <table>
          <tr><th>observations.jsonl</th><td><code>${{src.observations_jsonl || ''}}</code></td></tr>
          <tr><th>event counts</th><td>${{eventPills}}</td></tr>
          <tr><th>trace hash tail</th><td><code>${{s.trace_hash_tail || ''}}</code></td></tr>
          <tr><th>lane diagnostics</th><td>lane_count=${{fmt(lane.lane_graph?.lane_count)}}，ego max lane deviation=${{fmt(lane.ego?.max_lane_deviation)}}，actor=${{lane.actor_id || '-'}}</td></tr>
          <tr><th>run report</th><td>done=${{fmt(run.done)}}，steps=${{fmt(run.steps)}}，events=${{JSON.stringify(run.events || {{}})}}</td></tr>
        </table>
      `;
      document.getElementById('episodeTable').innerHTML = '<tr><th>episode</th><th>transition_count</th><th>total_reward</th></tr>' +
        (payload.episode_rewards || []).map(row => `<tr><td>${{row.episode}}</td><td>${{row.transition_count}}</td><td>${{fmt(row.total_reward)}}</td></tr>`).join('');
    }}

    function drawAll() {{
      drawChart('speedChart', [
        {{key:'ego_speed', label:'ego_speed', color:'#2364d2'}},
        {{key:'ego_accel', label:'ego_accel', color:'#17895d'}},
      ]);
      drawChart('laneChart', [
        {{key:'ego_lane_l', label:'ego_lane_l', color:'#c83f46'}},
        {{key:'ego_heading_error', label:'heading_error', color:'#6f4cc3'}},
      ]);
      drawChart('gapChart', [
        {{key:'front_gap', label:'front_gap', color:'#af6a11'}},
        {{key:'nearest_actor_distance', label:'nearest_actor_distance', color:'#2364d2'}},
      ]);
      drawChart('actionChart', [
        {{key:'acceleration', label:'acceleration', color:'#17895d'}},
        {{key:'steering', label:'steering', color:'#c83f46'}},
      ]);
      drawChart('rewardChart', [{{key:'reward', label:'reward', color:'#2364d2'}}]);
      const eventBinary = (series.event_label || []).map(v => v ? 1 : 0);
      series.event_binary = eventBinary;
      drawChart('eventChart', [
        {{key:'event_binary', label:'event present', color:'#c83f46'}},
        {{key:'collision_flag', label:'collision_flag', color:'#6f4cc3'}},
        {{key:'offroad_flag', label:'offroad_flag', color:'#af6a11'}},
      ], {{minY:0, maxY:1}});
    }}

    addSummary();
    addSources();
    drawAll();
    window.addEventListener('resize', drawAll);
  </script>
</body>
</html>
"""
