from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from worldsimflow.backends import ReplayBackend
from worldsimflow.core.flow import DeterministicFlowController
from worldsimflow.core.metrics import summarize_trace
from worldsimflow.core.scenario import ScenarioLoader
from worldsimflow.policies import LaneKeepPolicy
from worldsimflow.visualizer import render_trace_html


@dataclass(frozen=True)
class BatchRunResult:
    scenario_id: str
    scenario_path: str
    steps: int
    done: bool
    events: dict[str, int]
    total_reward: float
    avg_speed: float
    min_front_gap: float | None
    trace_hash: str
    html: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class BatchRunner:
    """Run many Scenario JSON files through the deterministic replay pipeline."""

    def __init__(self, policy: LaneKeepPolicy | None = None):
        self.policy = policy or LaneKeepPolicy()
        self.loader = ScenarioLoader()

    def run_paths(
        self,
        scenario_paths: Iterable[str | Path],
        steps: int | None = None,
        html_dir: str | Path | None = None,
    ) -> list[BatchRunResult]:
        results = []
        html_root = Path(html_dir) if html_dir is not None else None
        if html_root is not None:
            html_root.mkdir(parents=True, exist_ok=True)
        for path in scenario_paths:
            scenario_path = Path(path)
            scenario = self.loader.load(scenario_path)
            flow = DeterministicFlowController(scenario, self.policy, backend=ReplayBackend(scenario))
            try:
                trace = flow.run(steps)
                metrics = summarize_trace(trace)
                html_path = None
                if html_root is not None:
                    html_path = render_trace_html(scenario, trace, html_root / f"{scenario.scenario_id}.html")
                results.append(
                    BatchRunResult(
                        scenario_id=scenario.scenario_id,
                        scenario_path=str(scenario_path),
                        steps=metrics.steps,
                        done=metrics.done,
                        events=metrics.event_counts,
                        total_reward=metrics.total_reward,
                        avg_speed=metrics.avg_speed,
                        min_front_gap=metrics.min_front_gap,
                        trace_hash=flow.final_trace_hash(),
                        html=str(html_path) if html_path else None,
                    )
                )
            finally:
                flow.close()
        return results

    def write_report(self, results: list[BatchRunResult], output_path: str | Path) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "count": len(results),
            "failures": [item.scenario_id for item in results if item.done or item.events],
            "results": [item.to_dict() for item in results],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path