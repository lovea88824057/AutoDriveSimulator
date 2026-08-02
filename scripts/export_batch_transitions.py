from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from worldsimflow.core.batch_transition_export import BatchTransitionExportConfig, BatchTransitionExporter


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a multi-scenario WorldSimFlow transition dataset for RL/world-model work.")
    parser.add_argument(
        "--roots",
        nargs="+",
        default=["data/sample_scenario.json", "data/worldsimflow_mini_straight.json", "data/generated/phase2"],
        help="Scenario JSON files or directories to scan.",
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--policy", choices=["rule", "random"], default="rule")
    parser.add_argument("--compact", action="store_true", help="Write compact obs/next_obs payload with vectors only.")
    parser.add_argument("--include-bev", action="store_true", help="Include BEV raster and BEV history in obs/next_obs.")
    parser.add_argument("--bev-history-length", type=int, default=4)
    parser.add_argument("--bev-width", type=int, default=32)
    parser.add_argument("--bev-height", type=int, default=32)
    parser.add_argument("--bev-meters-per-pixel", type=float, default=1.0)
    parser.add_argument("--max-scenarios", type=int, default=0, help="Optional cap for quick smoke tests. 0 means no cap.")
    parser.add_argument("--dataset-name", default="worldsimflow_state_v1")
    parser.add_argument("--output-dir", default="outputs/all_results/datasets/worldsimflow_state_v1")
    args = parser.parse_args()

    config = BatchTransitionExportConfig(
        roots=[resolve(root) for root in args.roots],
        output_dir=resolve(args.output_dir),
        episodes=args.episodes,
        max_steps=args.steps,
        seed=args.seed,
        policy=args.policy,
        include_full_observation=not args.compact,
        include_bev=args.include_bev,
        bev_history_length=args.bev_history_length,
        bev_width=args.bev_width,
        bev_height=args.bev_height,
        bev_meters_per_pixel=args.bev_meters_per_pixel,
        max_scenarios=args.max_scenarios or None,
        dataset_name=args.dataset_name,
    )
    report = BatchTransitionExporter(config).export()

    print("batch_transition_export=ok")
    print(f"dataset_name={report['dataset_name']}")
    print(f"scenario_count={report['scenario_count']}")
    print(f"variant_count={report['variant_count']}")
    print(f"transition_count={report['transition_count']}")
    print(f"feature_count={report['feature_count']}")
    print(f"normalized_vector={report['normalized_vector']}")
    print(f"include_bev={report['include_bev']}")
    print(f"bev_history_length={report['bev_history_length']}")
    print(f"policy={report['policy']}")
    print(f"dataset_hash={report['dataset_hash']}")
    print(f"observations={report['observations']}")
    print(f"manifest={report['manifest']}")
    print(f"coverage={report['coverage']}")
    print(f"export_report={Path(report['manifest']).with_name('export_report.json')}")


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


if __name__ == "__main__":
    main()
