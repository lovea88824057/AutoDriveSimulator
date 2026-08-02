from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from worldsimflow.core.lane_graph import LaneGraph
from worldsimflow.core.scenario import ScenarioLoader
from worldsimflow.core.scenario_generation import vehicle_to_dict
from worldsimflow.core.types import ReplayActor, Scenario, VehicleState


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose WorldSimFlow LaneGraph/Frenet projection quality.")
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--variant", default=None, help="Optional intervention scenario to compare with the base map.")
    parser.add_argument("--actor-id", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--output", default="outputs/all_results/lane_graph/lane_graph_diagnostics.json")
    parser.add_argument("--html", default="outputs/all_results/lane_graph/lane_graph_diagnostics.html")
    args = parser.parse_args()

    base = ScenarioLoader().load(resolve(args.scenario))
    target = ScenarioLoader().load(resolve(args.variant)) if args.variant else base
    graph = LaneGraph.from_scenario(base)
    steps = min(args.steps or target.max_steps, target.max_steps)
    report = build_report(base, target, graph, args.actor_id, steps)
    output = resolve(args.output)
    html = resolve(args.html)
    output.parent.mkdir(parents=True, exist_ok=True)
    html.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    html.write_text(render_html(report), encoding="utf-8")

    print("lane_graph_diagnostics=ok")
    print(f"scenario={base.scenario_id}")
    print(f"target={target.scenario_id}")
    print(f"lane_count={report['lane_graph']['lane_count']}")
    if report.get("actor"):
        summary = report["actor"]["summary"]
        print(f"actor_id={report['actor']['actor_id']}")
        print(f"max_lane_deviation={summary['max_lane_deviation']}")
        print(f"max_heading_mismatch_deg={summary['max_heading_mismatch_deg']}")
        print(f"lane_switch_count={summary['lane_switch_count']}")
    print(f"output={output}")
    print(f"html={html}")


def build_report(base: Scenario, target: Scenario, graph: LaneGraph, actor_id: str | None, steps: int) -> dict[str, Any]:
    report: dict[str, Any] = {
        "base_scenario_id": base.scenario_id,
        "target_scenario_id": target.scenario_id,
        "steps": steps,
        "lane_graph": graph.to_summary(),
        "ego": summarize_states(graph, ego_states(target)[:steps]),
        "actor": None,
        "front_back_sample": front_back_sample(graph, target, min(steps - 1, 20)),
    }
    if actor_id:
        actor = find_actor(target, actor_id)
        report["actor"] = {
            "actor_id": actor_id,
            "summary": summarize_states(graph, actor.states[:steps]),
            "frames": frame_diagnostics(graph, actor.states[:steps]),
        }
    return report


def summarize_states(graph: LaneGraph, states: list[VehicleState]) -> dict[str, Any]:
    deviations: list[float] = []
    heading_errors: list[float] = []
    projection_distances: list[float] = []
    lane_ids: list[str] = []
    unprojected = 0
    for state in states:
        projection = graph.project_state(state)
        if projection is None:
            unprojected += 1
            continue
        deviations.append(abs(projection.l))
        projection_distances.append(projection.distance)
        lane_ids.append(projection.lane_id)
        mismatch = graph.heading_mismatch(state, projection)
        if mismatch is not None:
            heading_errors.append(math.degrees(mismatch))
    return {
        "frame_count": len(states),
        "projected_count": len(projection_distances),
        "unprojected_count": unprojected,
        "unique_lane_count": len(set(lane_ids)),
        "lane_switch_count": sum(1 for prev, cur in zip(lane_ids, lane_ids[1:]) if prev != cur),
        "max_lane_deviation": round(max(deviations), 4) if deviations else None,
        "mean_lane_deviation": round(sum(deviations) / len(deviations), 4) if deviations else None,
        "max_projection_distance": round(max(projection_distances), 4) if projection_distances else None,
        "max_heading_mismatch_deg": round(max(heading_errors), 4) if heading_errors else None,
        "mean_heading_mismatch_deg": round(sum(heading_errors) / len(heading_errors), 4) if heading_errors else None,
    }


def frame_diagnostics(graph: LaneGraph, states: list[VehicleState]) -> list[dict[str, Any]]:
    rows = []
    for step, state in enumerate(states):
        projection = graph.project_state(state)
        if projection is None:
            rows.append({"step": step, "projected": False, "state": vehicle_to_dict(state)})
            continue
        mismatch = graph.heading_mismatch(state, projection)
        rows.append(
            {
                "step": step,
                "projected": True,
                "lane_id": projection.lane_id,
                "s": round(projection.s, 4),
                "l": round(projection.l, 4),
                "projection_distance": round(projection.distance, 4),
                "lane_heading_deg": round(math.degrees(projection.heading), 4),
                "state_heading_deg": round(math.degrees(state.yaw), 4),
                "heading_mismatch_deg": round(math.degrees(mismatch), 4) if mismatch is not None else None,
                "state": vehicle_to_dict(state),
            }
        )
    return rows


def front_back_sample(graph: LaneGraph, scenario: Scenario, step: int) -> dict[str, Any]:
    ego = ego_states(scenario)[step]
    actors = [actor.states[min(step, len(actor.states) - 1)] for actor in scenario.actors]
    front, back = graph.find_front_back_actors(ego, actors)
    return {
        "step": step,
        "front": front_back_item(front),
        "back": front_back_item(back),
    }


def front_back_item(item) -> dict[str, Any] | None:
    if item is None:
        return None
    actor, projection, distance = item
    return {"actor_id": actor.actor_id, "lane_id": projection.lane_id, "s_gap": round(distance, 4), "l": round(projection.l, 4)}


def ego_states(scenario: Scenario) -> list[VehicleState]:
    raw = scenario.metadata.get("ego_replay_states")
    if raw:
        states = [VehicleState(actor_id="ego", **item) for item in raw]
        if len(states) >= scenario.max_steps:
            return states[: scenario.max_steps]
        return states + [states[-1]] * (scenario.max_steps - len(states))
    return [
        VehicleState(
            actor_id="ego",
            x=scenario.ego.x + scenario.ego.speed * scenario.dt * step * math.cos(scenario.ego.yaw),
            y=scenario.ego.y + scenario.ego.speed * scenario.dt * step * math.sin(scenario.ego.yaw),
            yaw=scenario.ego.yaw,
            speed=scenario.ego.speed,
            length=scenario.ego.length,
            width=scenario.ego.width,
            object_type=scenario.ego.object_type,
        )
        for step in range(scenario.max_steps)
    ]


def find_actor(scenario: Scenario, actor_id: str) -> ReplayActor:
    for actor in scenario.actors:
        if actor.actor_id == actor_id:
            return actor
    raise KeyError(f"actor_id={actor_id!r} not found")


def render_html(report: dict[str, Any]) -> str:
    payload = json.dumps(report, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return f'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>WorldSimFlow LaneGraph Diagnostics</title>
  <style>
    body {{ margin: 0; font-family: Arial, Helvetica, sans-serif; background: #f6f7f9; color: #1f2933; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0 0 8px; font-size: 24px; }}
    p {{ color: #667085; line-height: 1.55; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 10px; margin: 16px 0; }}
    .card {{ background: #fff; border: 1px solid #d0d5dd; border-radius: 8px; padding: 12px; }}
    .card span {{ display: block; color: #667085; font-size: 12px; margin-bottom: 6px; }}
    .card strong {{ font-size: 20px; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #d0d5dd; }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid #eaecf0; text-align: left; font-size: 13px; vertical-align: top; }}
    th {{ color: #667085; }}
    code {{ font-size: 12px; }}
  </style>
</head>
<body>
  <main>
    <h1>WorldSimFlow LaneGraph Diagnostics</h1>
    <p>基于 LaneGraph + Frenet projection 统计 ego 或目标 actor 的车道偏移、航向误差和 lane switch。</p>
    <div class="grid" id="stats"></div>
    <section><h2>Frame Diagnostics</h2><table><thead><tr><th>step</th><th>lane</th><th>s</th><th>l</th><th>distance</th><th>heading mismatch</th></tr></thead><tbody id="frames"></tbody></table></section>
  </main>
  <script>
    const report = {payload};
    const actor = report.actor;
    const summary = actor ? actor.summary : report.ego;
    const stats = [
      ['Lane count', report.lane_graph.lane_count],
      ['Projected frames', summary.projected_count],
      ['Max lane deviation', summary.max_lane_deviation],
      ['Max heading mismatch', summary.max_heading_mismatch_deg + ' deg'],
      ['Lane switches', summary.lane_switch_count],
    ];
    document.getElementById('stats').innerHTML = stats.map(([k,v]) => `<div class="card"><span>${{k}}</span><strong>${{v}}</strong></div>`).join('');
    const rows = actor ? actor.frames : [];
    document.getElementById('frames').innerHTML = rows.map(row => `<tr><td>${{row.step}}</td><td><code>${{row.lane_id || 'none'}}</code></td><td>${{row.s ?? ''}}</td><td>${{row.l ?? ''}}</td><td>${{row.projection_distance ?? ''}}</td><td>${{row.heading_mismatch_deg ?? ''}}</td></tr>`).join('') || '<tr><td colspan="6">No actor selected.</td></tr>';
  </script>
</body>
</html>'''


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


if __name__ == "__main__":
    main()
