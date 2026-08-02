from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from .scenario import ScenarioLoader
from .scenario_data_manager import ScenarioDataManager, ScenarioRecord
from .transition_dataset import PolicyFn, TransitionDatasetExporter, TransitionExportConfig
from .types import Action, Scenario


@dataclass(frozen=True)
class BatchTransitionExportConfig:
    """Configuration for C2.8 batch transition dataset export."""

    roots: list[str | Path]
    output_dir: str | Path
    episodes: int = 1
    max_steps: int | None = 20
    seed: int = 20260725
    policy: str = "rule"
    include_full_observation: bool = False
    include_bev: bool = False
    bev_history_length: int = 1
    bev_width: int = 64
    bev_height: int = 64
    bev_meters_per_pixel: float = 1.0
    max_scenarios: int | None = None
    dataset_name: str = "worldsimflow_state_v1"
    dataset_schema_version: str = "batch_transition_v1"
    pattern: str = "*.json"


@dataclass(frozen=True)
class VariantExportRecord:
    variant_id: str
    scenario_id: str
    scenario_path: str
    source: str
    traffic_mode: str
    intervention_kind: str | None
    policy_name: str
    episodes: int
    max_steps: int
    transition_count: int
    reward_curve: list[float]
    final_trace_hashes: list[str]
    part_path: str
    tags: list[str] = field(default_factory=list)
    difficulty: str = "unknown"
    actor_count: int = 0
    map_feature_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BatchTransitionExporter:
    """Export a small multi-scenario transition dataset for RL/world-model work.

    It scans Scenario JSON files, runs each scenario through TransitionDatasetExporter,
    annotates every transition with variant metadata, and writes a merged JSONL plus
    manifest/coverage reports.
    """

    def __init__(self, config: BatchTransitionExportConfig, loader: ScenarioLoader | None = None):
        self.config = config
        self.loader = loader or ScenarioLoader()
        self.manager = ScenarioDataManager(self.loader)

    def export(self) -> dict[str, Any]:
        output_dir = Path(self.config.output_dir)
        variants_dir = output_dir / "variants"
        output_dir.mkdir(parents=True, exist_ok=True)
        variants_dir.mkdir(parents=True, exist_ok=True)

        index = self.manager.scan(self.config.roots, pattern=self.config.pattern)
        records = list(index.records)
        if self.config.max_scenarios is not None:
            records = records[: self.config.max_scenarios]
        if not records:
            raise ValueError("No valid scenario JSON files found for batch transition export")

        observations_path = output_dir / "observations.jsonl"
        manifest_path = output_dir / "dataset_manifest.json"
        coverage_path = output_dir / "coverage_report.json"
        export_report_path = output_dir / "export_report.json"

        variants: list[VariantExportRecord] = []
        dataset_hash = hashlib.sha256()
        total_transitions = 0
        first_space_spec = None
        first_action_spec = None
        first_bev_spec = None
        feature_names: list[str] = []

        with observations_path.open("w", encoding="utf-8", newline="\n") as merged:
            for variant_index, record in enumerate(records):
                scenario = self.loader.load(record.path)
                variant_seed = self.config.seed + variant_index * 1009
                policy_fn = self._policy(self.config.policy, variant_seed)
                variant_id = self._variant_id(record, variant_index)
                part_path = variants_dir / f"{self._safe_name(variant_id)}.jsonl"
                report = TransitionDatasetExporter(
                    scenario,
                    TransitionExportConfig(
                        dataset_schema_version="transition_v1",
                        episodes=self.config.episodes,
                        max_steps=self.config.max_steps,
                        seed=variant_seed,
                        include_full_observation=self.config.include_full_observation,
                        include_bev=self.config.include_bev,
                        bev_history_length=self.config.bev_history_length,
                        bev_width=self.config.bev_width,
                        bev_height=self.config.bev_height,
                        bev_meters_per_pixel=self.config.bev_meters_per_pixel,
                    ),
                ).export(part_path, policy_fn)

                first_space_spec = first_space_spec or report.get("observation_space_spec")
                first_action_spec = first_action_spec or report.get("action_space_spec")
                first_bev_spec = first_bev_spec or report.get("bev_observation_space_spec")
                if not feature_names:
                    feature_names = list(report.get("feature_names", []))

                transition_count = self._merge_part(
                    part_path=part_path,
                    merged=merged,
                    dataset_hash=dataset_hash,
                    variant_id=variant_id,
                    record=record,
                    policy_name=self.config.policy,
                    traffic_mode=self._traffic_mode(scenario, record),
                )
                total_transitions += transition_count
                variants.append(
                    VariantExportRecord(
                        variant_id=variant_id,
                        scenario_id=record.scenario_id,
                        scenario_path=record.path,
                        source=record.source,
                        traffic_mode=self._traffic_mode(scenario, record),
                        intervention_kind=record.intervention_kind,
                        policy_name=self.config.policy,
                        episodes=self.config.episodes,
                        max_steps=self.config.max_steps or scenario.max_steps,
                        transition_count=transition_count,
                        reward_curve=list(report.get("reward_curve", [])),
                        final_trace_hashes=list(report.get("final_trace_hashes", [])),
                        part_path=str(part_path),
                        tags=list(record.tags),
                        difficulty=record.difficulty,
                        actor_count=record.actor_count,
                        map_feature_count=record.map_feature_count,
                    )
                )

        manifest = self._manifest(
            variants=variants,
            index_summary=index.summary,
            observations_path=observations_path,
            dataset_hash=dataset_hash.hexdigest(),
            feature_names=feature_names,
            observation_space_spec=first_space_spec,
            action_space_spec=first_action_spec,
            bev_observation_space_spec=first_bev_spec,
        )
        coverage = self._coverage(variants, observations_path, dataset_hash.hexdigest())
        export_report = {
            "batch_transition_export": "ok",
            "dataset_name": self.config.dataset_name,
            "dataset_schema_version": self.config.dataset_schema_version,
            "scenario_count": len({item.scenario_id for item in variants}),
            "variant_count": len(variants),
            "transition_count": total_transitions,
            "feature_count": len(feature_names),
            "normalized_vector": True,
            "include_bev": self.config.include_bev,
            "bev_history_length": self.config.bev_history_length if self.config.include_bev else 0,
            "policy": self.config.policy,
            "observations": str(observations_path),
            "manifest": str(manifest_path),
            "coverage": str(coverage_path),
            "dataset_hash": dataset_hash.hexdigest(),
        }

        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
        coverage_path.write_text(json.dumps(coverage, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
        export_report_path.write_text(json.dumps(export_report, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
        return export_report

    def _merge_part(
        self,
        *,
        part_path: Path,
        merged: Any,
        dataset_hash: hashlib._Hash,
        variant_id: str,
        record: ScenarioRecord,
        policy_name: str,
        traffic_mode: str,
    ) -> int:
        count = 0
        for line in part_path.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            row.update(
                {
                    "batch_dataset_schema_version": self.config.dataset_schema_version,
                    "dataset_name": self.config.dataset_name,
                    "variant_id": variant_id,
                    "policy_name": policy_name,
                    "traffic_mode": traffic_mode,
                    "intervention_kind": record.intervention_kind,
                    "scenario_source": record.source,
                    "scenario_tags": list(record.tags),
                    "scenario_difficulty": record.difficulty,
                    "scenario_path": record.path,
                }
            )
            text = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            merged.write(text)
            merged.write("\n")
            dataset_hash.update(text.encode("utf-8"))
            dataset_hash.update(b"\n")
            count += 1
        return count

    def _manifest(
        self,
        *,
        variants: list[VariantExportRecord],
        index_summary: dict[str, Any],
        observations_path: Path,
        dataset_hash: str,
        feature_names: list[str],
        observation_space_spec: dict[str, Any] | None,
        action_space_spec: dict[str, Any] | None,
        bev_observation_space_spec: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return {
            "dataset_name": self.config.dataset_name,
            "dataset_schema_version": self.config.dataset_schema_version,
            "purpose": "Multi-scenario transition dataset for lightweight RL/world-model training and evaluation.",
            "observations": str(observations_path),
            "dataset_hash": dataset_hash,
            "seed": self.config.seed,
            "policy": self.config.policy,
            "episodes": self.config.episodes,
            "max_steps": self.config.max_steps,
            "include_full_observation": self.config.include_full_observation,
            "include_bev": self.config.include_bev,
            "bev_history_length": self.config.bev_history_length if self.config.include_bev else 0,
            "bev_observation_space_spec": bev_observation_space_spec,
            "feature_names": feature_names,
            "feature_count": len(feature_names),
            "observation_space_spec": observation_space_spec,
            "action_space_spec": action_space_spec,
            "scenario_index_summary": index_summary,
            "variant_count": len(variants),
            "transition_count": sum(item.transition_count for item in variants),
            "variants": [item.to_dict() for item in variants],
        }

    def _coverage(self, variants: list[VariantExportRecord], observations_path: Path, dataset_hash: str) -> dict[str, Any]:
        sources = Counter(item.source for item in variants)
        traffic_modes = Counter(item.traffic_mode for item in variants)
        interventions = Counter(item.intervention_kind or "none" for item in variants)
        policies = Counter(item.policy_name for item in variants)
        difficulties = Counter(item.difficulty for item in variants)
        tag_counts = Counter(tag for item in variants for tag in item.tags)
        event_counts: Counter[str] = Counter()
        transition_count = 0
        normalized_count = 0
        bev_count = 0
        bev_history_count = 0
        for line in observations_path.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            transition_count += 1
            if row.get("obs", {}).get("normalized_vector") and row.get("next_obs", {}).get("normalized_vector"):
                normalized_count += 1
            if row.get("obs", {}).get("bev") and row.get("next_obs", {}).get("bev"):
                bev_count += 1
            if row.get("obs", {}).get("bev_history") and row.get("next_obs", {}).get("bev_history"):
                bev_history_count += 1
            event_counts.update(str(code) for code in row.get("event_codes", []))
        return {
            "coverage_schema_version": "transition_coverage_v1",
            "dataset_hash": dataset_hash,
            "scenario_count": len({item.scenario_id for item in variants}),
            "variant_count": len(variants),
            "transition_count": transition_count,
            "normalized_transition_count": normalized_count,
            "normalized_vector_complete": transition_count == normalized_count,
            "bev_transition_count": bev_count,
            "bev_history_transition_count": bev_history_count,
            "bev_complete": (not self.config.include_bev) or transition_count == bev_count,
            "bev_history_complete": (not self.config.include_bev) or transition_count == bev_history_count,
            "sources": dict(sorted(sources.items())),
            "traffic_modes": dict(sorted(traffic_modes.items())),
            "intervention_kinds": dict(sorted(interventions.items())),
            "policies": dict(sorted(policies.items())),
            "difficulties": dict(sorted(difficulties.items())),
            "tags": dict(sorted(tag_counts.items())),
            "event_counts": dict(sorted(event_counts.items())),
            "total_actor_count_across_variants": sum(item.actor_count for item in variants),
            "map_aware_variant_count": sum(1 for item in variants if item.map_feature_count > 0),
        }

    def _policy(self, name: str, seed: int) -> PolicyFn:
        if name == "rule":
            return rule_policy
        if name == "random":
            return random_policy(random.Random(seed))
        raise ValueError(f"Unsupported policy: {name}")

    def _variant_id(self, record: ScenarioRecord, index: int) -> str:
        traffic_mode = record.backend or "replay"
        intervention = record.intervention_kind or "none"
        return f"{index:04d}_{record.scenario_id}_{self.config.policy}_{traffic_mode}_{intervention}"

    def _traffic_mode(self, scenario: Scenario, record: ScenarioRecord) -> str:
        metadata = scenario.metadata or {}
        return str(metadata.get("traffic_manager_mode") or metadata.get("traffic_mode") or record.backend or "replay")

    def _safe_name(self, value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:180]


def rule_policy(obs: dict[str, Any]) -> dict[str, float]:
    speed = float(obs.get("ego_speed") or 0.0)
    front_gap = obs.get("front_gap")
    lane_l = float(obs.get("ego_lane_l") or obs.get("lane_center_offset") or 0.0)
    target_speed = 8.0
    acceleration = max(-2.5, min(2.0, (target_speed - speed) * 0.45))
    if front_gap is not None and float(front_gap) < 10.0:
        acceleration = min(acceleration, -1.5)
    steering = max(-0.35, min(0.35, -lane_l * 0.16))
    return {"acceleration": acceleration, "steering": steering}


def random_policy(rng: random.Random) -> Callable[[dict[str, Any]], dict[str, float]]:
    actions = [
        {"acceleration": -2.0, "steering": 0.0},
        {"acceleration": 0.0, "steering": 0.0},
        {"acceleration": 1.5, "steering": 0.0},
        {"acceleration": 0.0, "steering": 0.25},
        {"acceleration": 0.0, "steering": -0.25},
    ]

    def choose(_obs: dict[str, Any]) -> dict[str, float]:
        return dict(rng.choice(actions))

    return choose
