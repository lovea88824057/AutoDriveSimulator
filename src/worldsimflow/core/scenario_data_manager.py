from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .scenario import ScenarioLoader
from .types import Scenario


@dataclass(frozen=True)
class ScenarioRecord:
    scenario_id: str
    path: str
    source: str
    backend: str
    tags: list[str]
    difficulty: str
    max_steps: int
    dt: float
    actor_count: int
    actor_types: dict[str, int]
    map_feature_count: int
    has_drivable_area: bool
    intervention_kind: str | None = None
    expected_event: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScenarioIndex:
    records: list[ScenarioRecord]
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": len(self.records),
            "summary": self.summary,
            "records": [record.to_dict() for record in self.records],
        }


class ScenarioDataManager:
    """Index and filter WorldSimFlow Scenario JSON datasets."""

    def __init__(self, loader: ScenarioLoader | None = None):
        self.loader = loader or ScenarioLoader()

    def scan(self, roots: Iterable[str | Path], pattern: str = "*.json") -> ScenarioIndex:
        paths = self._collect_paths(roots, pattern)
        records = []
        errors = []
        for path in paths:
            try:
                scenario = self.loader.load(path)
                records.append(self.record_from_scenario(scenario, path))
            except Exception as exc:
                errors.append({"path": str(path), "error": str(exc)})
        return ScenarioIndex(records=records, summary=self._summary(records, errors))

    def filter_records(
        self,
        index: ScenarioIndex,
        tags: Iterable[str] | None = None,
        source: str | None = None,
        intervention_kind: str | None = None,
        difficulty: str | None = None,
    ) -> list[ScenarioRecord]:
        required_tags = set(tags or [])
        results = []
        for record in index.records:
            if required_tags and not required_tags.issubset(record.tags):
                continue
            if source is not None and record.source != source:
                continue
            if intervention_kind is not None and record.intervention_kind != intervention_kind:
                continue
            if difficulty is not None and record.difficulty != difficulty:
                continue
            results.append(record)
        return results

    def write_index(self, index: ScenarioIndex, output_path: str | Path) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(index.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def record_from_scenario(self, scenario: Scenario, path: str | Path) -> ScenarioRecord:
        metadata = scenario.metadata
        actor_types = self._actor_type_counts(scenario)
        intervention = metadata.get("mutation") or {}
        tags = sorted(set(str(tag) for tag in metadata.get("tags", [])))
        if scenario.map_features and "map_aware" not in tags:
            tags.append("map_aware")
        return ScenarioRecord(
            scenario_id=scenario.scenario_id,
            path=str(Path(path)),
            source=str(metadata.get("source", "unknown")),
            backend=str(metadata.get("backend", "replay")),
            tags=sorted(tags),
            difficulty=str(metadata.get("difficulty", self._infer_difficulty(scenario))),
            max_steps=scenario.max_steps,
            dt=scenario.dt,
            actor_count=len(scenario.actors),
            actor_types=actor_types,
            map_feature_count=len(scenario.map_features),
            has_drivable_area=scenario.drivable_area is not None,
            intervention_kind=intervention.get("kind") or intervention.get("type"),
            expected_event=metadata.get("expected_event"),
        )

    def _collect_paths(self, roots: Iterable[str | Path], pattern: str) -> list[Path]:
        paths: list[Path] = []
        for root in roots:
            item = Path(root)
            if item.is_file() and item.suffix.lower() == ".json":
                paths.append(item)
            elif item.is_dir():
                paths.extend(sorted(item.rglob(pattern)))
        return sorted(set(paths), key=lambda path: str(path).lower())

    def _actor_type_counts(self, scenario: Scenario) -> dict[str, int]:
        counts: dict[str, int] = {}
        for actor in scenario.actors:
            object_type = actor.states[0].object_type if actor.states else "UNKNOWN"
            counts[object_type] = counts.get(object_type, 0) + 1
        return dict(sorted(counts.items()))

    def _infer_difficulty(self, scenario: Scenario) -> str:
        actor_count = len(scenario.actors)
        if actor_count >= 20 or scenario.metadata.get("phase2"):
            return "hard"
        if actor_count >= 6 or len(scenario.map_features) >= 20:
            return "medium"
        return "easy"

    def _summary(self, records: list[ScenarioRecord], errors: list[dict[str, str]]) -> dict[str, Any]:
        sources: dict[str, int] = {}
        tags: dict[str, int] = {}
        interventions: dict[str, int] = {}
        difficulties: dict[str, int] = {}
        for record in records:
            sources[record.source] = sources.get(record.source, 0) + 1
            difficulties[record.difficulty] = difficulties.get(record.difficulty, 0) + 1
            if record.intervention_kind:
                interventions[record.intervention_kind] = interventions.get(record.intervention_kind, 0) + 1
            for tag in record.tags:
                tags[tag] = tags.get(tag, 0) + 1
        return {
            "sources": dict(sorted(sources.items())),
            "tags": dict(sorted(tags.items())),
            "interventions": dict(sorted(interventions.items())),
            "difficulties": dict(sorted(difficulties.items())),
            "map_aware_count": sum(1 for record in records if record.map_feature_count > 0),
            "errors": errors,
        }
