from __future__ import annotations

import math

from .monitor import HealthMonitor
from .reward import compute_reward_breakdown
from .types import Action, Scenario, VehicleState


class MiniDrivingSimulator:
    """A deterministic 2D driving simulator with log-replayed actors."""

    def __init__(self, scenario: Scenario):
        self.scenario = scenario
        self.monitor = HealthMonitor(
            scenario.road,
            scenario.max_steps,
            non_terminal_codes=self._non_terminal_monitor_codes(),
            drivable_area=scenario.drivable_area,
        )
        self.ego_replay_states = self._load_ego_replay_states()
        self.last_reward_breakdown = None
        self.last_cost_info = None
        self.reset()

    def _non_terminal_monitor_codes(self) -> set[str]:
        if self.scenario.metadata.get("monitor_mode") == "log_replay_inspection":
            return {"collision", "offroad", "stale_replay"}
        return set()

    def reset(self) -> dict:
        self.step_id = 0
        self.ego = self.ego_replay_states[0] if self.ego_replay_states else self.scenario.ego
        self.done = False
        return self._observe([])

    def step(self, action: Action) -> tuple[dict, float, bool, list]:
        if self.done:
            return self._observe([]), 0.0, True, []

        actors, replay_exhausted = self._replay_actors(self.step_id)
        if self.ego_replay_states:
            if self.step_id < len(self.ego_replay_states):
                self.ego = self.ego_replay_states[self.step_id]
            else:
                self.ego = self.ego_replay_states[-1]
                replay_exhausted = True
        else:
            self.ego = self._advance_ego(self.ego, action)
        events = self.monitor.check(self.step_id, self.ego, actors, replay_exhausted)
        self.done = self.monitor.is_terminal(events)
        obs = self._observe(actors)
        reward = self._reward(obs, events)
        self.step_id += 1
        return obs, reward, self.done, events

    def _load_ego_replay_states(self) -> list[VehicleState]:
        raw_states = self.scenario.metadata.get("ego_replay_states")
        if not raw_states:
            return []
        states = [VehicleState(actor_id="ego", **state) for state in raw_states]
        if len(states) >= self.scenario.max_steps:
            return states[: self.scenario.max_steps]
        return states + [states[-1]] * (self.scenario.max_steps - len(states))

    def _advance_ego(self, ego: VehicleState, action: Action) -> VehicleState:
        dt = self.scenario.dt
        speed = max(0.0, ego.speed + action.acceleration * dt)
        yaw = ego.yaw + action.steering * dt
        return VehicleState(
            actor_id=ego.actor_id,
            x=ego.x + speed * dt,
            y=ego.y + yaw * dt,
            yaw=yaw,
            speed=speed,
            length=ego.length,
            width=ego.width,
            object_type=ego.object_type,
        )

    def _replay_actors(self, step: int) -> tuple[list[VehicleState], bool]:
        actors = []
        replay_exhausted = False
        for actor in self.scenario.actors:
            if step < len(actor.states):
                actors.append(actor.states[step])
            else:
                actors.append(actor.states[-1])
                replay_exhausted = True
        return actors, replay_exhausted

    def _observe(self, actors: list[VehicleState]) -> dict:
        front_gap = None
        closest_actor_distance = None
        nearby_actor_count = 0
        for actor in actors:
            forward, lateral = self._to_ego_local(actor)
            distance = math.hypot(actor.x - self.ego.x, actor.y - self.ego.y)
            closest_actor_distance = distance if closest_actor_distance is None else min(closest_actor_distance, distance)
            if distance <= 50.0:
                nearby_actor_count += 1
            same_lane = abs(lateral) < self.scenario.road.lane_width / 2.0
            if forward >= 0 and same_lane:
                front_gap = forward if front_gap is None else min(front_gap, forward)
        return {
            "step": self.step_id,
            "ego": self.ego,
            "actors": actors,
            "front_gap": front_gap,
            "lane_center_offset": self.ego.y,
            "closest_actor_distance": closest_actor_distance,
            "nearby_actor_count": nearby_actor_count,
            "map_feature_count": len(self.scenario.map_features),
            "ego_mode": self.scenario.metadata.get("ego_mode", "closed_loop"),
        }

    def _to_ego_local(self, actor: VehicleState) -> tuple[float, float]:
        dx = actor.x - self.ego.x
        dy = actor.y - self.ego.y
        cos_yaw = math.cos(self.ego.yaw)
        sin_yaw = math.sin(self.ego.yaw)
        forward = cos_yaw * dx + sin_yaw * dy
        lateral = -sin_yaw * dx + cos_yaw * dy
        return forward, lateral

    def _reward(self, obs: dict, events: list) -> float:
        breakdown, cost_info = compute_reward_breakdown(obs, events)
        self.last_reward_breakdown = breakdown
        self.last_cost_info = cost_info
        return breakdown.total

    def reward_info(self) -> dict:
        return {
            "reward_breakdown": self.last_reward_breakdown.to_dict() if self.last_reward_breakdown else {},
            "cost_info": self.last_cost_info.to_dict() if self.last_cost_info else {},
        }

    def snapshot(self) -> dict:
        return {
            "step": self.step_id,
            "ego": self.ego,
            "scenario_id": self.scenario.scenario_id,
        }
