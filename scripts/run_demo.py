from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from worldsimflow import DeterministicFlowController, LaneKeepPolicy, ReplayBackend, ScenarioLoader
from worldsimflow.core.config import load_config, resolve_project_path
from worldsimflow.core.metrics import summarize_trace
from worldsimflow.visualizer import render_trace_html


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a deterministic WorldSimFlow replay simulation.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.json"))
    parser.add_argument("--scenario", default=str(ROOT / "data" / "sample_scenario.json"))
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--render-html", default=None)
    parser.add_argument("--real-video", default=None)
    parser.add_argument("--real-video-fps", type=float, default=None)
    parser.add_argument("--real-video-frame-offset", type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    runtime_config = config["runtime"]
    policy_config = config["policy"]

    steps = args.steps if args.steps is not None else int(runtime_config["steps"])
    render_html = args.render_html if args.render_html is not None else runtime_config.get("render_html")

    scenario = ScenarioLoader().load(args.scenario)
    policy = LaneKeepPolicy(target_speed=float(policy_config["target_speed"]))
    flow = DeterministicFlowController(scenario, policy, backend=ReplayBackend(scenario))
    try:
        trace = flow.run(steps)
        metrics = summarize_trace(trace)

        print(f"scenario_id={scenario.scenario_id}")
        print("backend=replay")
        print(f"steps={metrics.steps}")
        print(f"done={metrics.done}")
        print(f"events={metrics.event_counts}")
        print(f"total_reward={metrics.total_reward}")
        print(f"avg_speed={metrics.avg_speed}")
        print(f"min_front_gap={metrics.min_front_gap}")
        print(f"trace_hash={flow.final_trace_hash()}")
        if render_html:
            video = build_video_config(scenario, args.real_video, args.real_video_fps, args.real_video_frame_offset)
            html_path = render_trace_html(scenario, trace, resolve_project_path(ROOT, render_html), real_video=video)
            print(f"visualization={html_path}")
            if video:
                print(f"real_video={video['path']}")
    finally:
        flow.close()


def build_video_config(scenario, video_path: str | None, fps: float | None, frame_offset: int) -> dict | None:
    metadata = dict(scenario.metadata.get("real_video", {}))
    if video_path is not None:
        metadata["path"] = video_path
    if fps is not None:
        metadata["fps"] = fps
    if frame_offset:
        metadata["frame_offset"] = frame_offset
    if "path" not in metadata:
        return None
    path = Path(metadata["path"])
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return {
        "path": str(path),
        "src": path.as_uri(),
        "fps": metadata.get("fps"),
        "frame_offset": int(metadata.get("frame_offset", 0)),
    }


if __name__ == "__main__":
    main()
