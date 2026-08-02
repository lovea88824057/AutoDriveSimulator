from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .batch_runner import BatchRunResult
from .scenario_data_manager import ScenarioDataManager, ScenarioIndex, ScenarioRecord


@dataclass(frozen=True)
class ScenarioCoverageReport:
    scenario_count: int
    source_coverage: dict[str, int]
    difficulty_coverage: dict[str, int]
    tag_coverage: dict[str, int]
    actor_type_coverage: dict[str, int]
    intervention_coverage: dict[str, int]
    map_aware_count: int
    drivable_area_count: int
    traffic_policy_count: int
    event_coverage: dict[str, int] = field(default_factory=dict)
    failure_count: int = 0
    run_count: int = 0
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class ScenarioCoverageAnalyzer:
    """Build dataset and run-result coverage summaries from scenario indexes and batch reports."""

    def from_index_and_runs(
        self,
        index: ScenarioIndex,
        run_results: Iterable[BatchRunResult | dict[str, Any]] | None = None,
    ) -> ScenarioCoverageReport:
        records = index.records
        event_coverage: dict[str, int] = {}
        failure_count = 0
        run_count = 0
        for result in run_results or []:
            item = result.to_dict() if hasattr(result, "to_dict") else dict(result)
            run_count += 1
            events = item.get("events", {}) or {}
            if item.get("done") or events:
                failure_count += 1
            for code, count in events.items():
                event_coverage[str(code)] = event_coverage.get(str(code), 0) + int(count)
        report = ScenarioCoverageReport(
            scenario_count=len(records),
            source_coverage=self._count(records, lambda record: record.source),
            difficulty_coverage=self._count(records, lambda record: record.difficulty),
            tag_coverage=self._count_many(records, lambda record: record.tags),
            actor_type_coverage=self._sum_dicts(record.actor_types for record in records),
            intervention_coverage=self._count(records, lambda record: record.intervention_kind or "none"),
            map_aware_count=sum(1 for record in records if record.map_feature_count > 0),
            drivable_area_count=sum(1 for record in records if record.has_drivable_area),
            traffic_policy_count=sum(1 for record in records if "traffic_policy" in record.tags or "map_aware_idm" in record.tags),
            event_coverage=dict(sorted(event_coverage.items())),
            failure_count=failure_count,
            run_count=run_count,
            recommendations=[],
        )
        return ScenarioCoverageReport(**{**report.to_dict(), "recommendations": self._recommendations(report)})

    def from_paths(
        self,
        roots: Iterable[str | Path],
        batch_reports: Iterable[str | Path] | None = None,
    ) -> ScenarioCoverageReport:
        index = ScenarioDataManager().scan(roots)
        return self.from_index_and_runs(index, self.load_run_results(batch_reports or []))

    def load_run_results(self, paths: Iterable[str | Path]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for path in paths:
            item = Path(path)
            if not item.exists():
                continue
            data = json.loads(item.read_text(encoding="utf-8-sig"))
            if isinstance(data, dict) and isinstance(data.get("results"), list):
                results.extend(data["results"])
            elif isinstance(data, dict) and data.get("scenario_id"):
                results.append(data)
        return results

    def write_json(self, report: ScenarioCoverageReport, output_path: str | Path) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def write_html(self, report: ScenarioCoverageReport, output_path: str | Path) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(report.to_dict(), ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
        path.write_text(self._html(payload), encoding="utf-8")
        return path

    def _count(self, records: list[ScenarioRecord], key_fn) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in records:
            key = str(key_fn(record))
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items()))

    def _count_many(self, records: list[ScenarioRecord], key_fn) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in records:
            for key in key_fn(record):
                counts[str(key)] = counts.get(str(key), 0) + 1
        return dict(sorted(counts.items()))

    def _sum_dicts(self, dicts: Iterable[dict[str, int]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in dicts:
            for key, value in item.items():
                counts[key] = counts.get(key, 0) + int(value)
        return dict(sorted(counts.items()))

    def _recommendations(self, report: ScenarioCoverageReport) -> list[str]:
        recs = []
        if report.map_aware_count < max(1, report.scenario_count // 3):
            recs.append("地图语义覆盖偏少：建议增加带 map_features / drivable_area 的场景。")
        if report.traffic_policy_count == 0:
            recs.append("交通 actor policy 覆盖为空：建议加入 map_aware_idm 或 IDM-lite 场景。")
        if report.intervention_coverage.get("none", 0) == report.scenario_count:
            recs.append("缺少干预场景：建议加入 hard_brake / cut_in / pedestrian_crossing / speed_change。")
        if not report.event_coverage and report.run_count > 0:
            recs.append("当前运行结果没有异常事件：可以保留少量故意失败样例用于验证诊断链路。")
        if not recs:
            recs.append("当前数据集已覆盖 log replay、干预、地图语义、交通策略和异常诊断的基础维度。")
        return recs

    def _html(self, payload: str) -> str:
        return f'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>WorldSimFlow Scenario Coverage</title>
  <style>
    body {{ margin: 0; font-family: Arial, Helvetica, sans-serif; background: #f5f7fb; color: #1f2933; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0 0 8px; font-size: 24px; }}
    p {{ color: #667085; line-height: 1.55; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; margin: 16px 0; }}
    .card {{ background: #fff; border: 1px solid #d0d5dd; border-radius: 8px; padding: 12px; }}
    .card span {{ display: block; color: #667085; font-size: 12px; margin-bottom: 6px; }}
    .card strong {{ font-size: 22px; }}
    .tables {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #d0d5dd; }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid #eaecf0; text-align: left; font-size: 13px; }}
    th {{ color: #667085; }}
    .bar {{ height: 8px; background: #e8f0ff; border-radius: 999px; overflow: hidden; }}
    .bar i {{ display: block; height: 100%; background: #246bfe; }}
    li {{ margin: 7px 0; }}
  </style>
</head>
<body>
  <main>
    <h1>WorldSimFlow Scenario Coverage</h1>
    <p>基于 ScenarioDataManager 和 BatchRunner 汇总数据集覆盖率，帮助判断当前场景是否覆盖了来源、干预、地图语义、交通策略和异常诊断。</p>
    <div class="grid" id="stats"></div>
    <section class="tables" id="tables"></section>
    <section><h2>Recommendations</h2><ul id="recommendations"></ul></section>
  </main>
  <script>
    const report = {payload};
    const stats = [
      ['Scenario', report.scenario_count],
      ['Map-aware', report.map_aware_count],
      ['Traffic Policy', report.traffic_policy_count],
      ['Runs', report.run_count],
      ['Failures', report.failure_count],
    ];
    document.getElementById('stats').innerHTML = stats.map(([k,v]) => `<div class="card"><span>${{k}}</span><strong>${{v}}</strong></div>`).join('');
    const tables = [
      ['Source', report.source_coverage],
      ['Difficulty', report.difficulty_coverage],
      ['Intervention', report.intervention_coverage],
      ['Actor Type', report.actor_type_coverage],
      ['Events', report.event_coverage],
      ['Tags', report.tag_coverage],
    ];
    function table(name, obj) {{
      const values = Object.values(obj || {{}});
      const max = Math.max(1, ...values);
      const rows = Object.entries(obj || {{}}).map(([key, value]) => `<tr><td>${{key}}</td><td>${{value}}</td><td><div class="bar"><i style="width:${{Math.max(4, value / max * 100)}}%"></i></div></td></tr>`).join('');
      return `<div><h2>${{name}}</h2><table><thead><tr><th>Item</th><th>Count</th><th></th></tr></thead><tbody>${{rows || '<tr><td colspan="3">none</td></tr>'}}</tbody></table></div>`;
    }}
    document.getElementById('tables').innerHTML = tables.map(([name, obj]) => table(name, obj)).join('');
    document.getElementById('recommendations').innerHTML = report.recommendations.map(item => `<li>${{item}}</li>`).join('');
  </script>
</body>
</html>'''
