from __future__ import annotations

import html
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from worldsimflow.policies import PolicyAction, make_evaluation_policy

from .rl_env import WorldSimFlowEnv
from .scenario import ScenarioLoader
from .scenario_data_manager import ScenarioDataManager, ScenarioRecord


@dataclass(frozen=True)
class PolicyEvaluationConfig:
    """Configuration for C5 multi-policy evaluation."""

    roots: list[str | Path]
    output_dir: str | Path
    policies: list[str] = field(default_factory=lambda: ["rule", "random", "minimal-q"])
    episodes: int = 2
    max_steps: int = 30
    seed: int = 20260725
    max_scenarios: int | None = None
    pattern: str = "*.json"


@dataclass(frozen=True)
class PolicyScenarioResult:
    policy_name: str
    scenario_id: str
    scenario_path: str
    source: str
    traffic_mode: str
    intervention_kind: str | None
    difficulty: str
    episodes: int
    steps: int
    total_reward: float
    mean_reward: float
    reward_breakdown_totals: dict[str, float]
    cost_totals: dict[str, float]
    event_counts: dict[str, int]
    done_reason_counts: dict[str, int]
    action_counts: dict[str, int]
    final_trace_hashes: list[str]
    terminated_count: int
    truncated_count: int
    success_count: int
    success_rate: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PolicyEvaluator:
    """Evaluate multiple lightweight policies on the same scenario set."""

    def __init__(self, config: PolicyEvaluationConfig, loader: ScenarioLoader | None = None):
        self.config = config
        self.loader = loader or ScenarioLoader()
        self.manager = ScenarioDataManager(self.loader)

    def evaluate(self) -> dict[str, Any]:
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        index = self.manager.scan(self.config.roots, pattern=self.config.pattern)
        records = list(index.records)
        if self.config.max_scenarios is not None:
            records = records[: self.config.max_scenarios]
        if not records:
            raise ValueError("No valid scenario JSON files found for policy evaluation")

        scenario_results: list[PolicyScenarioResult] = []
        policy_summaries: dict[str, dict[str, Any]] = {}
        for policy_index, policy_name in enumerate(self.config.policies):
            policy = make_evaluation_policy(policy_name, seed=self.config.seed + policy_index * 100003)
            for scenario_index, record in enumerate(records):
                scenario = self.loader.load(record.path)
                result = self._evaluate_policy_on_scenario(policy_name, policy, scenario, record, scenario_index)
                scenario_results.append(result)
            policy_summaries[policy_name] = policy.summary()

        summary = self._summary(scenario_results, policy_summaries, index.summary)
        report = {
            "policy_evaluation": "ok",
            "evaluation_schema_version": "policy_eval_v2",
            "scenario_count": len({item.scenario_id for item in scenario_results}),
            "policy_count": len(self.config.policies),
            "episode_count": len(scenario_results) * self.config.episodes,
            "steps_per_episode": self.config.max_steps,
            "seed": self.config.seed,
            "policies": list(self.config.policies),
            "summary": summary,
            "scenario_results": [item.to_dict() for item in scenario_results],
            "policy_summaries": policy_summaries,
        }
        report_path = output_dir / "policy_eval_report.json"
        html_path = output_dir / "policy_eval_dashboard.html"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
        html_path.write_text(self._render_html(report), encoding="utf-8", newline="\n")
        report["report"] = str(report_path)
        report["html"] = str(html_path)
        return report

    def _evaluate_policy_on_scenario(self, policy_name: str, policy: Any, scenario: Any, record: ScenarioRecord, scenario_index: int) -> PolicyScenarioResult:
        total_reward = 0.0
        total_steps = 0
        event_counts: Counter[str] = Counter()
        done_reason_counts: Counter[str] = Counter()
        action_counts: Counter[str] = Counter()
        reward_breakdown_totals: Counter[str] = Counter()
        cost_totals: Counter[str] = Counter()
        final_trace_hashes: list[str] = []
        terminated_count = 0
        truncated_count = 0
        success_count = 0

        for episode in range(self.config.episodes):
            env = WorldSimFlowEnv(scenario, max_steps=self.config.max_steps)
            episode_seed = self.config.seed + scenario_index * 1009 + episode
            obs, _reset_info = env.reset(seed=episode_seed)
            policy.reset(seed=episode_seed)
            episode_done_reason = "running"
            episode_success = False
            try:
                for _step in range(self.config.max_steps):
                    policy_action = policy.act(obs)
                    action_name = policy_action.name if isinstance(policy_action, PolicyAction) else "action"
                    action = env.clip_action(policy_action.action if isinstance(policy_action, PolicyAction) else policy_action)
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    policy.observe(obs, policy_action, reward, next_obs, terminated, truncated, info)
                    total_reward += float(reward)
                    total_steps += 1
                    action_counts[action_name] += 1
                    event_counts.update(str(code) for code in info.get("event_codes", []))
                    reward_breakdown_totals.update({key: float(value) for key, value in info.get("reward_breakdown", {}).items() if key != "total"})
                    cost_totals.update({key: float(value) for key, value in info.get("cost_info", {}).items() if key != "total_cost"})
                    done_info = info.get("done_reason", {})
                    success_info = info.get("success_info", {})
                    episode_done_reason = str(done_info.get("reason") or episode_done_reason)
                    episode_success = bool(success_info.get("success", False))
                    obs = next_obs
                    if terminated or truncated:
                        terminated_count += int(terminated)
                        truncated_count += int(truncated)
                        break
                done_reason_counts[episode_done_reason] += 1
                success_count += int(episode_success)
                final_trace_hashes.append(env.final_trace_hash())
            finally:
                env.close()

        mean_reward = total_reward / max(1, self.config.episodes)
        return PolicyScenarioResult(
            policy_name=policy_name,
            scenario_id=record.scenario_id,
            scenario_path=record.path,
            source=record.source,
            traffic_mode=self._traffic_mode(scenario, record),
            intervention_kind=record.intervention_kind,
            difficulty=record.difficulty,
            episodes=self.config.episodes,
            steps=total_steps,
            total_reward=round(total_reward, 6),
            mean_reward=round(mean_reward, 6),
            reward_breakdown_totals=dict(sorted((key, round(value, 6)) for key, value in reward_breakdown_totals.items())),
            cost_totals=dict(sorted((key, round(value, 6)) for key, value in cost_totals.items())),
            event_counts=dict(sorted(event_counts.items())),
            done_reason_counts=dict(sorted(done_reason_counts.items())),
            action_counts=dict(sorted(action_counts.items())),
            final_trace_hashes=final_trace_hashes,
            terminated_count=terminated_count,
            truncated_count=truncated_count,
            success_count=success_count,
            success_rate=round(success_count / max(1, self.config.episodes), 6),
        )

    def _summary(self, results: list[PolicyScenarioResult], policy_summaries: dict[str, dict[str, Any]], index_summary: dict[str, Any]) -> dict[str, Any]:
        by_policy: dict[str, dict[str, Any]] = {}
        grouped: dict[str, list[PolicyScenarioResult]] = defaultdict(list)
        for result in results:
            grouped[result.policy_name].append(result)
        for policy_name, items in sorted(grouped.items()):
            event_counts: Counter[str] = Counter()
            done_reason_counts: Counter[str] = Counter()
            action_counts: Counter[str] = Counter()
            reward_breakdown_totals: Counter[str] = Counter()
            cost_totals: Counter[str] = Counter()
            for item in items:
                event_counts.update(item.event_counts)
                done_reason_counts.update(item.done_reason_counts)
                action_counts.update(item.action_counts)
                reward_breakdown_totals.update(item.reward_breakdown_totals)
                cost_totals.update(item.cost_totals)
            episode_count = sum(item.episodes for item in items)
            success_count = sum(item.success_count for item in items)
            by_policy[policy_name] = {
                "scenario_count": len(items),
                "episode_count": episode_count,
                "total_steps": sum(item.steps for item in items),
                "total_reward": round(sum(item.total_reward for item in items), 6),
                "mean_reward_per_scenario": round(sum(item.mean_reward for item in items) / max(1, len(items)), 6),
                "success_count": success_count,
                "success_rate": round(success_count / max(1, episode_count), 6),
                "reward_breakdown_totals": dict(sorted((key, round(value, 6)) for key, value in reward_breakdown_totals.items())),
                "cost_totals": dict(sorted((key, round(value, 6)) for key, value in cost_totals.items())),
                "event_counts": dict(sorted(event_counts.items())),
                "done_reason_counts": dict(sorted(done_reason_counts.items())),
                "action_counts": dict(sorted(action_counts.items())),
                "terminated_count": sum(item.terminated_count for item in items),
                "truncated_count": sum(item.truncated_count for item in items),
                "policy_summary": policy_summaries.get(policy_name, {}),
            }
        ranking = sorted(
            (
                {
                    "policy_name": name,
                    "mean_reward_per_scenario": data["mean_reward_per_scenario"],
                    "success_rate": data["success_rate"],
                }
                for name, data in by_policy.items()
            ),
            key=lambda item: (item["success_rate"], item["mean_reward_per_scenario"]),
            reverse=True,
        )
        return {
            "by_policy": by_policy,
            "ranking": ranking,
            "scenario_index_summary": index_summary,
        }

    def _traffic_mode(self, scenario: Any, record: ScenarioRecord) -> str:
        metadata = scenario.metadata or {}
        return str(metadata.get("traffic_manager_mode") or metadata.get("traffic_mode") or record.backend or "replay")

    def _render_html(self, report: dict[str, Any]) -> str:
        payload = json.dumps(report, ensure_ascii=False, separators=(",", ":"))
        title = "WorldSimFlow Policy Diagnostics Dashboard"
        template = """<!doctype html>
<html lang=\"zh-CN\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>__TITLE__</title>
  <style>
    :root { color-scheme: light; --ink:#18212f; --muted:#647085; --line:#dce2ea; --bg:#f5f7fa; --panel:#ffffff; --blue:#2563eb; --green:#0f9f6e; --orange:#d97706; --red:#dc2626; --purple:#7c3aed; }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--bg); color:var(--ink); font:14px/1.5 Arial,'Microsoft YaHei',sans-serif; }
    header { background:#152033; color:white; padding:22px 28px; }
    h1 { margin:0; font-size:24px; letter-spacing:0; }
    header p { margin:6px 0 0; color:#c8d2df; max-width:980px; }
    main { max-width:1420px; margin:0 auto; padding:20px; }
    .toolbar { display:grid; grid-template-columns:repeat(4,minmax(160px,1fr)); gap:12px; margin-bottom:14px; }
    label { display:block; color:var(--muted); font-size:12px; margin-bottom:5px; }
    select, input { width:100%; border:1px solid var(--line); background:white; color:var(--ink); border-radius:6px; padding:9px 10px; font:14px Arial,'Microsoft YaHei',sans-serif; }
    .cards { display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:12px; }
    .panel { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px; box-shadow:0 8px 22px rgba(20,31,48,.07); }
    .label { color:var(--muted); font-size:12px; }
    .value { font-size:24px; font-weight:700; margin-top:4px; }
    .hint { color:var(--muted); font-size:12px; margin-top:4px; }
    .layout { display:grid; grid-template-columns:1fr 1fr; gap:14px; margin-top:14px; }
    h2 { margin:0 0 10px; font-size:17px; }
    .bar-row { display:grid; grid-template-columns:130px 1fr 80px; align-items:center; gap:10px; margin:8px 0; }
    .bar-label { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .bar-track { height:12px; background:#edf1f6; border-radius:999px; overflow:hidden; }
    .bar { height:100%; border-radius:999px; background:var(--blue); min-width:2px; }
    .bar.cost { background:var(--red); }
    .bar.success { background:var(--green); }
    .bar.reason { background:var(--purple); }
    .bar.action { background:var(--orange); }
    table { width:100%; border-collapse:collapse; margin-top:6px; }
    th,td { border-bottom:1px solid #edf0f5; padding:8px 7px; text-align:left; vertical-align:top; }
    th { color:var(--muted); font-weight:600; position:sticky; top:0; background:white; }
    .table-wrap { max-height:440px; overflow:auto; }
    .pill { display:inline-block; padding:2px 7px; border-radius:999px; background:#eef2f7; margin:2px; font-size:12px; white-space:nowrap; }
    .empty { color:var(--muted); padding:18px 0; }
    .wide { margin-top:14px; }
    @media (max-width: 980px) { .toolbar { grid-template-columns:1fr 1fr; } .cards { grid-template-columns:1fr 1fr; } .layout { grid-template-columns:1fr; } }
    @media (max-width: 620px) { main { padding:12px; } .toolbar,.cards { grid-template-columns:1fr; } .bar-row { grid-template-columns:100px 1fr 64px; } }
  </style>
</head>
<body>
<header>
  <h1>__TITLE__</h1>
  <p>Interactive C5 policy diagnostics: filter by policy / scenario / traffic / intervention, then inspect reward, reward breakdown, cost info, done reason, and action distribution.</p>
</header>
<main>
  <section class=\"panel\">
    <div class=\"toolbar\">
      <div><label for=\"policyFilter\">Policy</label><select id=\"policyFilter\"></select></div>
      <div><label for=\"scenarioFilter\">Scenario</label><select id=\"scenarioFilter\"></select></div>
      <div><label for=\"trafficFilter\">Traffic Mode</label><select id=\"trafficFilter\"></select></div>
      <div><label for=\"interventionFilter\">Intervention</label><select id=\"interventionFilter\"></select></div>
    </div>
  </section>
  <section class=\"cards\" id=\"summaryCards\"></section>
  <section class=\"layout\">
    <div class=\"panel\"><h2>Policy Reward</h2><div id=\"rewardChart\"></div></div>
    <div class=\"panel\"><h2>Policy Cost</h2><div id=\"costChart\"></div></div>
    <div class=\"panel\"><h2>Done Reason</h2><div id=\"reasonChart\"></div></div>
    <div class=\"panel\"><h2>Action Distribution</h2><div id=\"actionChart\"></div></div>
  </section>
  <section class=\"panel wide\"><h2>AllAll</h2><div class=\"table-wrap\"><table id=\"rankingTable\"></table></div></section>
  <section class=\"panel wide\"><h2>AllAll</h2><div class=\"table-wrap\"><table id=\"scenarioTable\"></table></div></section>
</main>
<script id=\"payload\" type=\"application/json\">__PAYLOAD__</script>
<script>
const report = JSON.parse(document.getElementById('payload').textContent);
const rows = report.scenario_results || [];
const filters = {
  policy: document.getElementById('policyFilter'),
  scenario: document.getElementById('scenarioFilter'),
  traffic: document.getElementById('trafficFilter'),
  intervention: document.getElementById('interventionFilter')
};
function uniq(values) { return Array.from(new Set(values.map(v => v || 'none'))).sort(); }
function fillSelect(select, values) {
  select.innerHTML = ['all', ...uniq(values)].map(v => `<option value="${escapeAttr(v)}">${v === 'all' ? 'All' : escapeHtml(v)}</option>`).join('');
}
fillSelect(filters.policy, rows.map(r => r.policy_name));
fillSelect(filters.scenario, rows.map(r => r.scenario_id));
fillSelect(filters.traffic, rows.map(r => r.traffic_mode));
fillSelect(filters.intervention, rows.map(r => r.intervention_kind || 'none'));
Object.values(filters).forEach(select => select.addEventListener('change', render));
function selectedRows() {
  return rows.filter(r =>
    (filters.policy.value === 'all' || r.policy_name === filters.policy.value) &&
    (filters.scenario.value === 'all' || r.scenario_id === filters.scenario.value) &&
    (filters.traffic.value === 'all' || r.traffic_mode === filters.traffic.value) &&
    (filters.intervention.value === 'all' || (r.intervention_kind || 'none') === filters.intervention.value)
  );
}
function sumObj(items, key) {
  const out = {};
  for (const row of items) for (const [k,v] of Object.entries(row[key] || {})) out[k] = (out[k] || 0) + Number(v || 0);
  return out;
}
function groupBy(items, keyFn, valueFn) {
  const out = {};
  for (const row of items) {
    const key = keyFn(row);
    out[key] = (out[key] || 0) + valueFn(row);
  }
  return out;
}
function fmt(v, digits=3) { return Number.isFinite(Number(v)) ? Number(v).toFixed(digits) : String(v All '-'); }
function pct(v) { return `${fmt(Number(v || 0) * 100, 1)}%`; }
function renderCards(items) {
  const episodes = items.reduce((s,r) => s + Number(r.episodes || 0), 0);
  const steps = items.reduce((s,r) => s + Number(r.steps || 0), 0);
  const reward = items.reduce((s,r) => s + Number(r.total_reward || 0), 0);
  const success = items.reduce((s,r) => s + Number(r.success_count || 0), 0);
  const costs = sumObj(items, 'cost_totals');
  const totalCost = Object.values(costs).reduce((s,v) => s + Number(v || 0), 0);
  const cards = [
    ['AllAll', items.length, `scenario-policy All`],
    ['Episodes', episodes, `steps=${steps}`],
    ['All Reward', episodes ? fmt(reward / episodes) : '0.000', `total=${fmt(reward)}`],
    ['Success Rate', episodes ? pct(success / episodes) : '0.0%', `success=${success}`],
    ['Total Cost', fmt(totalCost), 'collision/offroad All?']
  ];
  document.getElementById('summaryCards').innerHTML = cards.map(([a,b,c]) => `<div class="panel"><div class="label">${a}</div><div class="value">${b}</div><div class="hint">${c}</div></div>`).join('');
}
function renderBars(el, values, cls='') {
  const entries = Object.entries(values).sort((a,b) => b[1] - a[1]);
  if (!entries.length) { el.innerHTML = '<div class="empty">AllAllAllAll</div>'; return; }
  const maxAbs = Math.max(...entries.map(([,v]) => Math.abs(Number(v))), 1e-9);
  el.innerHTML = entries.map(([k,v]) => {
    const width = Math.max(2, Math.abs(Number(v)) / maxAbs * 100);
    return `<div class="bar-row"><div class="bar-label" title="${escapeAttr(k)}">${escapeHtml(k)}</div><div class="bar-track"><div class="bar ${cls}" style="width:${width}%"></div></div><div>${fmt(v)}</div></div>`;
  }).join('');
}
function renderTables(items) {
  const ranking = Object.entries(groupBy(items, r => r.policy_name, r => Number(r.mean_reward || 0)))
    .map(([policy,total]) => {
      const policyRows = items.filter(r => r.policy_name === policy);
      const episodes = policyRows.reduce((s,r) => s + Number(r.episodes || 0), 0);
      const success = policyRows.reduce((s,r) => s + Number(r.success_count || 0), 0);
      return {policy, mean: total / Math.max(1, policyRows.length), successRate: success / Math.max(1, episodes), episodes};
    })
    .sort((a,b) => (b.successRate - a.successRate) || (b.mean - a.mean));
  document.getElementById('rankingTable').innerHTML = '<tr><th>#</th><th>policy</th><th>mean reward</th><th>success rate</th><th>episodes</th></tr>' +
    ranking.map((r,i) => `<tr><td>${i+1}</td><td>${escapeHtml(r.policy)}</td><td>${fmt(r.mean)}</td><td>${pct(r.successRate)}</td><td>${r.episodes}</td></tr>`).join('');
  document.getElementById('scenarioTable').innerHTML = '<tr><th>policy</th><th>scenario</th><th>source</th><th>traffic</th><th>intervention</th><th>mean reward</th><th>success</th><th>done reason</th><th>cost</th><th>events</th></tr>' +
    items.map(r => `<tr><td>${escapeHtml(r.policy_name)}</td><td>${escapeHtml(r.scenario_id)}</td><td>${escapeHtml(r.source)}</td><td>${escapeHtml(r.traffic_mode)}</td><td>${escapeHtml(r.intervention_kind || 'none')}</td><td>${fmt(r.mean_reward)}</td><td>${pct(r.success_rate || 0)}</td><td>${pills(r.done_reason_counts)}</td><td>${pills(r.cost_totals)}</td><td>${pills(r.event_counts)}</td></tr>`).join('');
}
function pills(obj) { const entries = Object.entries(obj || {}); return entries.length ? entries.map(([k,v]) => `<span class="pill">${escapeHtml(k)}=${fmt(v)}</span>`).join('') : '<span class="pill">none</span>'; }
function escapeHtml(value) { return String(value All '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch])); }
function escapeAttr(value) { return escapeHtml(value); }
function render() {
  const items = selectedRows();
  renderCards(items);
  renderBars(document.getElementById('rewardChart'), groupBy(items, r => r.policy_name, r => Number(r.mean_reward || 0)), 'success');
  renderBars(document.getElementById('costChart'), groupBy(items, r => r.policy_name, r => Object.values(r.cost_totals || {}).reduce((s,v) => s + Number(v || 0), 0)), 'cost');
  renderBars(document.getElementById('reasonChart'), sumObj(items, 'done_reason_counts'), 'reason');
  renderBars(document.getElementById('actionChart'), sumObj(items, 'action_counts'), 'action');
  renderTables(items);
}
render();
</script>
</body>
</html>
"""
        return template.replace("__TITLE__", html.escape(title)).replace("__PAYLOAD__", payload)
