
from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import Any

from worldsimflow.backends import ReplayBackend
from worldsimflow.core.flow import DeterministicFlowController
from worldsimflow.core.lane_graph import LaneGraph
from worldsimflow.core.metrics import summarize_trace
from worldsimflow.core.scenario import ScenarioLoader
from worldsimflow.core.scenario_diff import ScenarioDiff
from worldsimflow.core.scenario_generation import write_scenario_json
from worldsimflow.core.target_intervention import TargetActorInterventionEngine, TargetInterventionSpec
from worldsimflow.core.traffic_manager import TrafficManagerConfig, TrafficManagerLite
from worldsimflow.core.traffic_policy import MapAwareTrafficConfig, TrafficPolicyRunner
from worldsimflow.core.types import ReplayActor, Scenario, VehicleState
from worldsimflow.policies import LaneKeepPolicy
from worldsimflow.visualizer import render_trace_html

DEFAULT_LAB_SCENARIOS = [
    "data/converted/openlog_nuscenes_scene-0061.json",
    "data/converted/openlog_waymo_2a1e44d405a6833f.json",
    "data/sample_scenario.json",
]


class ExperimentManager:
    """Create replayable WorldSimFlow experiments for Log Lab 2.0."""

    def __init__(self, root: str | Path, scenario_paths: list[str | Path] | None = None):
        self.root = Path(root).resolve()
        self.loader = ScenarioLoader()
        self.scenario_paths = [self._resolve(path) for path in (scenario_paths or DEFAULT_LAB_SCENARIOS)]
        self.output_root = self.root / "outputs" / "experiments"

    def list_scenarios(self) -> list[dict[str, Any]]:
        records = []
        for path in self.scenario_paths:
            if path.exists():
                scenario = self.loader.load(path)
                records.append(self._scenario_record(path, scenario))
        return records

    def get_scenario(self, key_or_path: str) -> dict[str, Any]:
        path = self.resolve_scenario_path(key_or_path)
        scenario = self.loader.load(path)
        graph = LaneGraph.from_scenario(scenario)
        return {
            **self._scenario_record(path, scenario),
            "actors": [self._actor_summary(actor, graph) for actor in scenario.actors],
            "scenario": self._scenario_payload(scenario),
        }

    def resolve_scenario_path(self, key_or_path: str) -> Path:
        candidate = self._resolve(key_or_path)
        if candidate.exists():
            return candidate
        for path in self.scenario_paths:
            if not path.exists():
                continue
            scenario = self.loader.load(path)
            record = self._scenario_record(path, scenario)
            if key_or_path in {record["key"], record["scenario_id"], Path(record["path"]).name}:
                return path
        raise FileNotFoundError(f"Scenario not found: {key_or_path}")

    def run(self, request: dict[str, Any]) -> dict[str, Any]:
        mode = str(request.get("mode", "replay")).strip().lower()
        scenario_path = self.resolve_scenario_path(str(request["scenario"]))
        base = self.loader.load(scenario_path)
        steps = int(request.get("steps") or min(base.max_steps, 80))
        seed = int(request.get("seed", 20260723))
        actor_id = str(request.get("actor_id", "")).strip() or None
        label = mode
        if mode == "replay":
            variant = base
        elif mode == "b2":
            actor_id = str(request["actor_id"])
            kind = str(request.get("kind", "cut_in"))
            spec = TargetInterventionSpec(kind=kind, actor_id=actor_id, start_step=int(request.get("start_step", 12)), duration=int(request.get("duration", 24)), seed=seed, params=self._b2_params(request))
            variant = TargetActorInterventionEngine().apply(base, spec)
            label = f"b2_{kind}_{self._safe(actor_id)}"
        elif mode == "b3":
            ids = self._actor_ids(request)
            actor_id = ids[0] if ids else actor_id
            config = MapAwareTrafficConfig(desired_speed=float(request.get("desired_speed", 8.0)), min_gap=float(request.get("min_gap", 4.0)), time_headway=float(request.get("time_headway", 1.3)), lane_snap_ratio=float(request.get("lane_snap_ratio", 0.35)), max_actor_count=None if ids else self._optional_int(request.get("max_reactive_actors")), actor_ids=tuple(ids) if ids else None)
            variant = TrafficPolicyRunner().rebuild_map_aware_idm_traffic(base, config)
            label = "b3_lane_aware_idm"
        elif mode == "b4":
            ids = self._actor_ids(request)
            actor_id = ids[0] if ids else actor_id
            config = TrafficManagerConfig(mode=str(request.get("traffic_mode", "hybrid")), desired_speed=float(request.get("desired_speed", 8.0)), min_gap=float(request.get("min_gap", 4.0)), time_headway=float(request.get("time_headway", 1.3)), lane_snap_ratio=float(request.get("lane_snap_ratio", 0.35)), max_reactive_actors=self._optional_int(request.get("max_reactive_actors")), reactive_actor_ids=tuple(ids) if ids else None)
            variant = TrafficManagerLite().build_scenario(base, config)
            label = f"b4_{variant.metadata.get('traffic_manager_mode', 'traffic')}"
        else:
            raise ValueError(f"Unsupported experiment mode: {mode}")
        return self._write_outputs(base, variant, mode, label, actor_id, steps, seed)

    def list_experiments(self, limit: int = 50) -> list[dict[str, Any]]:
        if not self.output_root.exists():
            return []
        reports = []
        for path in sorted(self.output_root.glob("*/run.json"), key=lambda item: item.stat().st_mtime, reverse=True)[:limit]:
            try:
                report = json.loads(path.read_text(encoding="utf-8-sig"))
                if isinstance(report, dict) and "paths" in report and "urls" not in report:
                    report["urls"] = self._artifact_urls(report["paths"])
                reports.append(report)
            except Exception:
                continue
        return reports

    def _write_outputs(self, base: Scenario, variant: Scenario, mode: str, label: str, actor_id: str | None, steps: int, seed: int) -> dict[str, Any]:
        experiment_id = self._experiment_id(base.scenario_id, label, seed)
        out_dir = self.output_root / experiment_id
        out_dir.mkdir(parents=True, exist_ok=True)
        scenario_out = write_scenario_json(variant, out_dir / "scenario.json")
        diff = ScenarioDiff().compare(base, variant).to_dict() if variant is not base else self._empty_diff(base)
        diff_out = out_dir / "diff.json"
        diff_out.write_text(json.dumps(diff, ensure_ascii=False, indent=2), encoding="utf-8")
        flow = DeterministicFlowController(variant, LaneKeepPolicy(), backend=ReplayBackend(variant))
        try:
            trace = flow.run(steps)
            metrics = summarize_trace(trace)
            html_out = render_trace_html(variant, trace, out_dir / "replay.html")
            trace_hash = flow.final_trace_hash()
        finally:
            flow.close()
        lane_report = self._lane_diagnostics(base, variant, actor_id, min(steps, variant.max_steps))
        lane_json = out_dir / "lane_diagnostics.json"
        lane_html = out_dir / "lane_diagnostics.html"
        lane_json.write_text(json.dumps(lane_report, ensure_ascii=False, indent=2), encoding="utf-8")
        lane_html.write_text(self._render_lane_html(lane_report), encoding="utf-8")
        paths = {"scenario": str(scenario_out), "replay_html": str(html_out), "run_report": str(out_dir / "run.json"), "diff": str(diff_out), "lane_diagnostics_json": str(lane_json), "lane_diagnostics_html": str(lane_html)}
        report = {"experiment_id": experiment_id, "mode": mode, "scenario_id": variant.scenario_id, "base_scenario_id": base.scenario_id, "actor_id": actor_id, "steps": metrics.steps, "done": metrics.done, "events": metrics.event_counts, "total_reward": metrics.total_reward, "avg_speed": metrics.avg_speed, "min_front_gap": metrics.min_front_gap, "trace_hash": trace_hash, "traffic_manager": variant.metadata.get("traffic_manager"), "traffic_mode": variant.metadata.get("traffic_mode"), "mutation": variant.metadata.get("mutation"), "paths": paths}
        (out_dir / "run.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return {**report, "urls": self._artifact_urls(paths)}

    def _b2_params(self, request: dict[str, Any]) -> dict[str, Any]:
        params = {}
        for key in ["deceleration", "target_lateral", "target_lane_l", "min_route_speed", "target_speed", "speed_delta", "speed_scale", "shift"]:
            if key in request and request[key] not in {None, ""}:
                params[key] = float(request[key])
        return params

    def _actor_ids(self, request: dict[str, Any]) -> list[str]:
        raw = request.get("actor_ids", request.get("actor_id", ""))
        if isinstance(raw, list):
            return [str(item).strip() for item in raw if str(item).strip()]
        return [item.strip() for item in str(raw).split(",") if item.strip()]

    def _optional_int(self, value: Any) -> int | None:
        return None if value in {None, ""} else int(value)

    def _scenario_record(self, path: Path, scenario: Scenario) -> dict[str, Any]:
        vehicle_count = sum(1 for actor in scenario.actors if actor.states[0].object_type == "VEHICLE")
        pedestrian_count = sum(1 for actor in scenario.actors if actor.states[0].object_type == "PEDESTRIAN")
        return {"key": self._safe(scenario.scenario_id), "scenario_id": scenario.scenario_id, "display_name": self._display_name(path, scenario), "path": str(path), "actor_count": len(scenario.actors), "vehicle_count": vehicle_count, "pedestrian_count": pedestrian_count, "map_feature_count": len(scenario.map_features), "max_steps": scenario.max_steps, "dt": scenario.dt, "tags": list(scenario.metadata.get("tags", []))}

    def _actor_summary(self, actor: ReplayActor, graph: LaneGraph) -> dict[str, Any]:
        state = actor.states[0]
        projection = graph.project_state(state)
        return {"actor_id": actor.actor_id, "object_type": state.object_type, "length": state.length, "width": state.width, "initial_speed": state.speed, "state_count": len(actor.states), "projected": projection is not None, "lane_id": projection.lane_id if projection else None, "initial_s": round(projection.s, 4) if projection else None, "initial_l": round(projection.l, 4) if projection else None}

    def _scenario_payload(self, scenario: Scenario) -> dict[str, Any]:
        return {"scenario_id": scenario.scenario_id, "dt": scenario.dt, "max_steps": scenario.max_steps, "road": scenario.road.__dict__, "ego": self._state(scenario.ego), "ego_replay_states": scenario.metadata.get("ego_replay_states", []), "actors": [{"actor_id": actor.actor_id, "states": [self._state(state) for state in actor.states]} for actor in scenario.actors], "map_features": [{"feature_id": f.feature_id, "feature_type": f.feature_type, "polyline": f.polyline} for f in scenario.map_features], "metadata": scenario.metadata}

    def _state(self, state: VehicleState) -> dict[str, Any]:
        return {"actor_id": state.actor_id, "x": state.x, "y": state.y, "yaw": state.yaw, "speed": state.speed, "length": state.length, "width": state.width, "object_type": state.object_type}

    def _ego_state_at(self, scenario: Scenario, step: int) -> VehicleState:
        raw = scenario.metadata.get("ego_replay_states")
        if raw:
            return VehicleState(actor_id="ego", **raw[min(step, len(raw) - 1)])
        return VehicleState(actor_id="ego", x=scenario.ego.x + scenario.ego.speed * scenario.dt * step * math.cos(scenario.ego.yaw), y=scenario.ego.y + scenario.ego.speed * scenario.dt * step * math.sin(scenario.ego.yaw), yaw=scenario.ego.yaw, speed=scenario.ego.speed, length=scenario.ego.length, width=scenario.ego.width, object_type=scenario.ego.object_type)

    def _empty_diff(self, scenario: Scenario) -> dict[str, Any]:
        return {"base_scenario_id": scenario.scenario_id, "variant_scenario_id": scenario.scenario_id, "added_actors": [], "removed_actors": [], "changed_actors": [], "changed_actor_count": 0, "metadata": {"base_actor_count": len(scenario.actors), "variant_actor_count": len(scenario.actors)}}

    def _lane_diagnostics(self, base: Scenario, target: Scenario, actor_id: str | None, steps: int) -> dict[str, Any]:
        graph = LaneGraph.from_scenario(base)
        report = {"base_scenario_id": base.scenario_id, "target_scenario_id": target.scenario_id, "steps": steps, "lane_graph": graph.to_summary(), "ego": self._summarize_states(graph, [self._ego_state_at(target, step) for step in range(steps)]), "actor": None}
        if actor_id:
            actor = next((item for item in target.actors if item.actor_id == actor_id), None)
            if actor:
                states = actor.states[:steps]
                report["actor"] = {"actor_id": actor_id, "summary": self._summarize_states(graph, states), "frames": self._frame_diagnostics(graph, states[: min(120, steps)])}
        return report

    def _summarize_states(self, graph: LaneGraph, states: list[VehicleState]) -> dict[str, Any]:
        deviations, heading_errors, lane_ids = [], [], []
        unprojected = 0
        for state in states:
            projection = graph.project_state(state)
            if projection is None:
                unprojected += 1
                continue
            deviations.append(abs(projection.l)); lane_ids.append(projection.lane_id)
            mismatch = graph.heading_mismatch(state, projection)
            if mismatch is not None:
                heading_errors.append(math.degrees(mismatch))
        return {"frame_count": len(states), "projected_count": len(states) - unprojected, "unprojected_count": unprojected, "unique_lane_count": len(set(lane_ids)), "lane_switch_count": sum(1 for a, b in zip(lane_ids, lane_ids[1:]) if a != b), "max_lane_deviation": round(max(deviations), 4) if deviations else None, "mean_lane_deviation": round(sum(deviations) / len(deviations), 4) if deviations else None, "max_heading_mismatch_deg": round(max(heading_errors), 4) if heading_errors else None, "mean_heading_mismatch_deg": round(sum(heading_errors) / len(heading_errors), 4) if heading_errors else None}

    def _frame_diagnostics(self, graph: LaneGraph, states: list[VehicleState]) -> list[dict[str, Any]]:
        rows = []
        for step, state in enumerate(states):
            projection = graph.project_state(state)
            rows.append({"step": step, "projected": projection is not None, "lane_id": projection.lane_id if projection else None, "s": round(projection.s, 4) if projection else None, "l": round(projection.l, 4) if projection else None, "heading_mismatch_deg": round(math.degrees(graph.heading_mismatch(state, projection)), 4) if projection else None, "speed": round(state.speed, 4)})
        return rows

    def _render_lane_html(self, report: dict[str, Any]) -> str:
        payload = json.dumps(report, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
        return f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>WorldSimFlow Lane Diagnostics</title><style>body{{margin:0;background:#f6f7f9;color:#1f2933;font-family:Arial,Helvetica,sans-serif}}main{{max-width:1180px;margin:0 auto;padding:24px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin:16px 0}}.card{{background:#fff;border:1px solid #d0d5dd;border-radius:8px;padding:12px}}.card span{{display:block;color:#667085;font-size:12px;margin-bottom:6px}}table{{width:100%;border-collapse:collapse;background:#fff;border:1px solid #d0d5dd}}th,td{{border-bottom:1px solid #eaecf0;padding:8px 10px;text-align:left;font-size:13px}}code{{font-size:12px}}</style></head><body><main><h1>WorldSimFlow Lane Diagnostics</h1><p>Log Lab 2.0 ??? LaneGraph/Frenet ??????????????????????????????</p><div class='grid' id='stats'></div><table><thead><tr><th>step</th><th>lane</th><th>s</th><th>l</th><th>heading mismatch</th><th>speed</th></tr></thead><tbody id='rows'></tbody></table></main><script>const report={payload};const actor=report.actor;const summary=actor?actor.summary:report.ego;const stats=[['Lane count',report.lane_graph.lane_count],['Projected frames',summary.projected_count],['Max lane deviation',summary.max_lane_deviation],['Max heading mismatch',summary.max_heading_mismatch_deg],['Lane switches',summary.lane_switch_count]];document.getElementById('stats').innerHTML=stats.map(([k,v])=>`<div class='card'><span>${{k}}</span><strong>${{v??'n/a'}}</strong></div>`).join('');document.getElementById('rows').innerHTML=(actor?actor.frames:[]).map(r=>`<tr><td>${{r.step}}</td><td><code>${{r.lane_id||'none'}}</code></td><td>${{r.s??''}}</td><td>${{r.l??''}}</td><td>${{r.heading_mismatch_deg??''}}</td><td>${{r.speed??''}}</td></tr>`).join('')||'<tr><td colspan=6>No actor selected.</td></tr>';</script></body></html>"""

    def _artifact_urls(self, paths: dict[str, str]) -> dict[str, str]:
        urls = {}
        for key, value in paths.items():
            rel = Path(value).resolve().relative_to(self.root).as_posix()
            urls[key] = f"/artifact/{rel}"
        return urls

    def _experiment_id(self, scenario_id: str, label: str, seed: int) -> str:
        return self._safe(f"{time.strftime('%Y%m%d_%H%M%S')}_{scenario_id}_{label}_{seed}")[:150]

    def _display_name(self, path: Path, scenario: Scenario) -> str:
        sid = scenario.scenario_id.lower()
        if "nuscenes" in sid:
            return "nuScenes mini log"
        if "waymo" in sid:
            return "Waymo mini log"
        if "sample" in path.name or "straight_close" in sid:
            return "sample synthetic log"
        return scenario.scenario_id

    def _safe(self, value: str) -> str:
        return re.sub(r"[^0-9A-Za-z_\-]+", "_", str(value)).strip("_") or "experiment"

    def _resolve(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (self.root / path).resolve()
