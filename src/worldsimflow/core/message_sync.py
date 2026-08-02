from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable


@dataclass(frozen=True)
class TopicMessage:
    topic: str
    timestamp: int
    sequence: int
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SyncIssue:
    severity: str
    code: str
    message: str
    topic: str | None = None
    timestamp: int | None = None


@dataclass(frozen=True)
class SyncFrame:
    timestamp: int
    messages: dict[str, TopicMessage]
    issues: list[SyncIssue]


class TopicBuffer:
    """A deterministic per-topic message buffer with stale/out-of-order diagnostics."""

    def __init__(self, topic: str):
        self.topic = topic
        self._messages: dict[int, TopicMessage] = {}
        self._last_sequence: int | None = None
        self.issues: list[SyncIssue] = []

    def push(self, message: TopicMessage) -> None:
        if message.topic != self.topic:
            self.issues.append(SyncIssue("error", "wrong_topic", f"message topic {message.topic} pushed to {self.topic}", message.topic, message.timestamp))
            return
        if self._last_sequence is not None and message.sequence <= self._last_sequence:
            self.issues.append(SyncIssue("error", "out_of_order", "message sequence is not strictly increasing", message.topic, message.timestamp))
        self._last_sequence = message.sequence
        self._messages[message.timestamp] = message

    def get(self, timestamp: int) -> TopicMessage | None:
        return self._messages.get(timestamp)

    def timestamps(self) -> set[int]:
        return set(self._messages)


class TimestampBarrier:
    """Align required topics by exact timestamp and report missing/stale messages."""

    def __init__(self, required_topics: Iterable[str], stale_after_frames: int = 1):
        self.required_topics = list(required_topics)
        self.stale_after_frames = stale_after_frames
        self.buffers = {topic: TopicBuffer(topic) for topic in self.required_topics}

    def push(self, message: TopicMessage) -> None:
        if message.topic not in self.buffers:
            self.buffers[message.topic] = TopicBuffer(message.topic)
        self.buffers[message.topic].push(message)

    def build_frames(self) -> list[SyncFrame]:
        timestamps = sorted(set().union(*(buffer.timestamps() for buffer in self.buffers.values())))
        frames: list[SyncFrame] = []
        last_seen: dict[str, TopicMessage] = {}
        for timestamp in timestamps:
            messages: dict[str, TopicMessage] = {}
            issues: list[SyncIssue] = []
            for topic in self.required_topics:
                current = self.buffers[topic].get(timestamp)
                if current is not None:
                    messages[topic] = current
                    last_seen[topic] = current
                    continue
                previous = last_seen.get(topic)
                if previous is None:
                    issues.append(SyncIssue("fatal", "missing_message", "required topic has no message at timestamp", topic, timestamp))
                    continue
                age = timestamp - previous.timestamp
                if age > self.stale_after_frames:
                    issues.append(SyncIssue("error", "stale_message", f"last message is {age} frames old", topic, timestamp))
                else:
                    issues.append(SyncIssue("warn", "reused_previous", "using previous message for this timestamp", topic, timestamp))
                    messages[topic] = previous
            frames.append(SyncFrame(timestamp, messages, [*issues, *self._buffer_issues_at(timestamp)]))
        return frames

    def _buffer_issues_at(self, timestamp: int) -> list[SyncIssue]:
        return [issue for buffer in self.buffers.values() for issue in buffer.issues if issue.timestamp == timestamp]


class DeterministicScheduler:
    """Execute module callbacks in a fixed order for every synchronized frame."""

    def __init__(self, module_order: Iterable[str]):
        self.module_order = list(module_order)

    def run(self, frames: Iterable[SyncFrame], callbacks: dict[str, Callable[[SyncFrame], Any]]) -> list[dict[str, Any]]:
        outputs = []
        for frame in frames:
            module_outputs = {}
            for module in self.module_order:
                if module in callbacks:
                    module_outputs[module] = callbacks[module](frame)
            outputs.append({"timestamp": frame.timestamp, "outputs": module_outputs, "issues": [issue.__dict__ for issue in frame.issues]})
        return outputs


class SyncDiagnostics:
    """Summarize synchronization health and deterministic replay fingerprint."""

    def summarize(self, frames: list[SyncFrame], scheduled_outputs: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        issue_counts: dict[str, int] = {}
        for frame in frames:
            for issue in frame.issues:
                issue_counts[issue.code] = issue_counts.get(issue.code, 0) + 1
        payload = {
            "frame_count": len(frames),
            "issue_counts": dict(sorted(issue_counts.items())),
            "ok": not any(issue.severity in {"error", "fatal"} for frame in frames for issue in frame.issues),
            "trace_hash": self.trace_hash(frames, scheduled_outputs or []),
        }
        return payload

    def trace_hash(self, frames: list[SyncFrame], scheduled_outputs: list[dict[str, Any]]) -> str:
        payload = {
            "frames": [
                {
                    "timestamp": frame.timestamp,
                    "messages": {topic: self._message_to_json(message) for topic, message in sorted(frame.messages.items())},
                    "issues": [issue.__dict__ for issue in frame.issues],
                }
                for frame in frames
            ],
            "scheduled_outputs": scheduled_outputs,
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _message_to_json(self, message: TopicMessage) -> dict[str, Any]:
        return {"topic": message.topic, "timestamp": message.timestamp, "sequence": message.sequence, "payload": message.payload}


def make_demo_messages(mode: str = "success", frames: int = 8) -> list[TopicMessage]:
    messages: list[TopicMessage] = []
    topics = ["localization", "perception", "planning"]
    sequence = 0
    for timestamp in range(frames):
        for topic in topics:
            if mode == "missing" and topic == "perception" and timestamp == 3:
                continue
            if mode == "stale" and topic == "planning" and timestamp in {4, 5}:
                continue
            sequence += 1
            messages.append(TopicMessage(topic=topic, timestamp=timestamp, sequence=sequence, payload={"value": f"{topic}_{timestamp}"}))
    if mode == "out_of_order":
        messages.append(TopicMessage(topic="perception", timestamp=2, sequence=1, payload={"value": "late_old_sequence"}))
    return messages


def run_sync_demo(mode: str = "success") -> dict[str, Any]:
    barrier = TimestampBarrier(["localization", "perception", "planning"], stale_after_frames=1)
    for message in make_demo_messages(mode):
        barrier.push(message)
    frames = barrier.build_frames()
    scheduler = DeterministicScheduler(["perception", "planning", "control"])
    outputs = scheduler.run(
        frames,
        {
            "perception": lambda frame: sorted(frame.messages),
            "planning": lambda frame: frame.timestamp,
            "control": lambda frame: len(frame.issues),
        },
    )
    summary = SyncDiagnostics().summarize(frames, outputs)
    return {"mode": mode, "summary": summary, "frames": [frame_to_dict(frame) for frame in frames], "scheduled_outputs": outputs}


def frame_to_dict(frame: SyncFrame) -> dict[str, Any]:
    return {
        "timestamp": frame.timestamp,
        "topics": sorted(frame.messages),
        "issues": [issue.__dict__ for issue in frame.issues],
    }