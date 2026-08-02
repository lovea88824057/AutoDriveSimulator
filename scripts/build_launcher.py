from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
RESULT_ROOT = OUTPUTS / "all_results"


@dataclass(frozen=True)
class LauncherItem:
    title: str
    phase: str
    kind: str
    description: str
    status: str
    status_detail: str
    path: str
    url: str
    recommended: bool = False
    action_label: str = "打开页面"

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def main() -> None:
    html_root = RESULT_ROOT if RESULT_ROOT.exists() else OUTPUTS
    html_files = [p for p in sorted(html_root.rglob("*.html")) if should_show(p)]
    report_files = [p for p in sorted(html_root.rglob("*.json")) if should_show(p)]
    sync_report = OUTPUTS / "phase3_sync_report.json"
    if sync_report.exists() and should_show(sync_report):
        report_files.append(sync_report)

    status = load_report_status(report_files)
    items = sorted([describe_html(path, status) for path in html_files], key=item_sort_key)
    reports = sorted([describe_report(path) for path in report_files], key=lambda item: (item["kind"], item["path"]))
    workflows = build_workflows(items)
    highlights = build_highlights(items)

    output = OUTPUTS / "worldsimflow_launcher.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render(items, reports, workflows, highlights), encoding="utf-8")
    print("launcher=ok")
    print(f"output={output}")
    print(f"html_count={len(items)}")
    print(f"report_count={len(reports)}")


def should_show(path: Path) -> bool:
    rel = str(path.relative_to(ROOT)).lower().replace("\\", "/")
    return path.name != "worldsimflow_launcher.html" and "__pycache__" not in rel


def load_report_status(report_files: list[Path]) -> dict[str, dict[str, Any]]:
    status: dict[str, dict[str, Any]] = {}
    for path in report_files:
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:
            continue
        if isinstance(data, dict) and data.get("scenario_id") and {"events", "trace_hash"}.issubset(data):
            status[str(data["scenario_id"])] = data
        if isinstance(data, dict):
            for result in data.get("results", []):
                scenario_id = result.get("scenario_id")
                if scenario_id:
                    status[str(scenario_id)] = result
    return status


def describe_html(path: Path, report_status: dict[str, dict[str, Any]]) -> LauncherItem:
    rel = str(path.relative_to(ROOT))
    lower = rel.lower().replace("\\", "/")
    stem = path.stem
    title = readable_stem(stem)
    phase = "结果"
    kind = "HTML"
    description = "可视化或分析页面。"
    action_label = "打开页面"
    recommended = False

    if "log_lab_v2" in lower or "log_intervention_lab" in lower:
        title = "Log Lab 2.0"
        phase = "交互实验"
        kind = "Actor intervention lab"
        description = "选择场景、点选 actor，并运行 replay、干预或 traffic mode。"
        action_label = "进入实验室"
        recommended = True
    elif "world_model" in lower:
        title = "Minimal World Model"
        phase = "世界模型"
        kind = "BEV predictor"
        description = "用 BEV history 和 action 预测下一帧，验证世界模型数据闭环。"
        action_label = "查看训练结果"
        recommended = True
    elif "bev" in lower:
        title = "BEV Observation"
        phase = "Observation"
        kind = "BEV raster"
        description = "检查 ego-centric BEV 栅格输入和 history stack。"
        action_label = "查看 BEV"
        recommended = True
    elif "policy_eval" in lower:
        title = "Policy Evaluation"
        phase = "策略评测"
        kind = "Reward / cost / success"
        description = "比较 rule、random、minimal-q 的表现，并解释得分原因。"
        action_label = "查看评测"
        recommended = True
    elif "coverage" in lower:
        title = "Scenario Coverage"
        phase = "数据覆盖"
        kind = "Coverage report"
        description = "统计场景、actor、地图、干预、traffic mode 和事件覆盖。"
        action_label = "查看覆盖"
        recommended = True
    elif "traffic_diagnostics" in lower or "lane_diagnostics" in lower:
        title = "Traffic Diagnostics"
        phase = "诊断"
        kind = "Speed / lane / gap"
        description = "查看 speed、lane offset、front gap 和事件时间线。"
        action_label = "查看诊断"
        recommended = True
    elif "traffic_manager" in lower:
        title = "TrafficManagerLite"
        phase = "交通管理"
        kind = "Replay / IDM / Hybrid"
        description = "统一管理背景 actor 的 replay、lane-aware IDM 和 hybrid 模式。"
        action_label = "查看交通"
        recommended = True
    elif "targeted" in lower or "cut_in" in lower or "hard_brake" in lower or "speed_change" in lower or "lateral_shift" in lower:
        title = readable_intervention_title(stem)
        phase = "场景干预"
        kind = "Counterfactual replay"
        description = "指定 actor 的反事实干预结果，可配合 diff 和 lane diagnostics 复盘。"
        action_label = "查看干预"
    elif "rl_eval" in lower or "rl" in lower:
        title = "RL Concept Demo"
        phase = "RL"
        kind = "Env loop"
        description = "验证 reset、step、action、reward、done、info 的最小闭环。"
        action_label = "查看 RL"
        recommended = True
    elif "sample" in lower or "demo_trace" in lower:
        title = "基础 LogSim Replay"
        phase = "Phase 1"
        kind = "Deterministic replay"
        description = "最小确定性回放，包含 metrics、events 和 trace_hash。"
        action_label = "打开回放"
        recommended = True

    result = find_status_for_html(stem, report_status)
    status_text, detail = status_from_result(result, phase)
    return LauncherItem(title, phase, kind, description, status_text, detail, rel, path.as_uri(), recommended, action_label)


def item_sort_key(item: LauncherItem) -> tuple[int, str, str]:
    order = {
        "交互实验": 0,
        "Phase 1": 1,
        "场景干预": 2,
        "交通管理": 3,
        "诊断": 4,
        "策略评测": 5,
        "Observation": 6,
        "世界模型": 7,
        "RL": 8,
        "数据覆盖": 9,
    }
    return (order.get(item.phase, 99), item.title, item.path)


def status_from_result(result: dict[str, Any] | None, phase: str) -> tuple[str, str]:
    if phase in {"数据覆盖", "世界模型", "Observation", "策略评测", "RL"}:
        return "报告", "分析结果已生成"
    if not result:
        return "可查看", "页面可直接打开"
    events = result.get("events", {}) or {}
    done_reason = result.get("done_reason") or result.get("reason")
    if events:
        return "有事件", ", ".join(f"{key}={value}" for key, value in events.items())
    if done_reason:
        return "已结束", str(done_reason)
    return "正常", "无异常事件"


def find_status_for_html(stem: str, report_status: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    for scenario_id, result in report_status.items():
        if stem in scenario_id or scenario_id.endswith(stem) or stem.endswith(scenario_id):
            return result
        html_path = str(result.get("html", "")).replace("\\", "/")
        if html_path and Path(html_path).stem == stem:
            return result
    return None


def describe_report(path: Path) -> dict[str, Any]:
    rel = str(path.relative_to(ROOT))
    lower = rel.lower().replace("\\", "/")
    kind = "结构化报告"
    description = "JSON 报告，用于复盘 metrics、events、trace_hash 和 actor 统计。"
    if "world_model" in lower:
        kind = "世界模型"
        description = "训练样本量、BEV history shape、baseline MSE、final MSE 和 IoU。"
    elif "dataset" in lower or "coverage" in lower:
        kind = "数据集/覆盖率"
        description = "transition dataset manifest、coverage 和多场景统计。"
    elif "policy_eval" in lower:
        kind = "策略评测"
        description = "policy、scenario、traffic、intervention、reward breakdown 和 success info。"
    elif "traffic" in lower or "lane" in lower:
        kind = "交通诊断"
        description = "lane deviation、front gap、speed、events 和交通模式配置。"
    elif lower.endswith(".run.json"):
        kind = "单场景运行"
        description = "steps、events、reward、metrics、done_reason 和 trace_hash。"
    elif "diff" in lower:
        kind = "场景差异"
        description = "干预前后 actor 轨迹与场景字段变化。"
    return {"title": path.name, "kind": kind, "description": description, "path": rel, "url": path.as_uri(), "action_label": "打开 JSON"}


def build_workflows(items: list[LauncherItem]) -> list[dict[str, str]]:
    def find(*needles: str) -> LauncherItem | None:
        for item in items:
            lower = item.path.lower().replace("\\", "/")
            if all(needle in lower for needle in needles):
                return item
        return None

    specs = [
        ("1", "Log Lab", "交互式选择场景和 actor", find("log_lab_v2") or find("log_intervention_lab"), "进入"),
        ("2", "Traffic Diagnostics", "解释速度、车道偏移和跟车距离", find("traffic_diagnostics") or find("lane_diagnostics"), "诊断"),
        ("3", "Policy Evaluation", "比较策略分数和成功率", find("policy_eval"), "评测"),
        ("4", "BEV Observation", "检查视觉化 observation 输入", find("bev"), "查看"),
        ("5", "World Model", "验证 BEV forward predictor", find("world_model"), "训练结果"),
    ]
    workflows = []
    for step, title, description, target, action in specs:
        workflows.append({
            "step": step,
            "title": title,
            "description": description,
            "url": target.url if target else "",
            "path": target.path if target else "尚未生成",
            "action_label": action if target else "待生成",
        })
    return workflows


def build_highlights(items: list[LauncherItem]) -> list[LauncherItem]:
    wanted = ["Log Lab 2.0", "Policy Evaluation", "BEV Observation", "Minimal World Model", "Scenario Coverage", "Traffic Diagnostics"]
    picked: list[LauncherItem] = []
    for title in wanted:
        match = next((item for item in items if item.title == title), None)
        if match and match not in picked:
            picked.append(match)
    for item in items:
        if len(picked) >= 8:
            break
        if item.recommended and item not in picked:
            picked.append(item)
    return picked


def readable_intervention_title(stem: str) -> str:
    labels = {
        "hard_brake": "指定 actor 急刹",
        "cut_in": "指定 actor 切入",
        "speed_change": "指定 actor 速度改变",
        "lateral_shift": "指定 actor 横向偏移",
        "pedestrian_crossing": "行人横穿干预",
        "close_follow": "近距离跟驰干预",
    }
    for key, label in labels.items():
        if key in stem:
            return label
    return "场景干预回放"


def readable_stem(stem: str) -> str:
    return stem.replace("_", " ")


def render(items: list[LauncherItem], reports: list[dict[str, Any]], workflows: list[dict[str, str]], highlights: list[LauncherItem]) -> str:
    phases = sorted({item.phase for item in items})
    phase_counts = {phase: sum(1 for item in items if item.phase == phase) for phase in phases}
    payload = json.dumps(
        {
            "items": [item.to_dict() for item in items],
            "reports": reports,
            "workflows": workflows,
            "highlights": [item.to_dict() for item in highlights],
            "phase_counts": phase_counts,
            "summary": {
                "html_count": len(items),
                "report_count": len(reports),
                "recommended_count": sum(1 for item in items if item.recommended),
                "event_count": sum(1 for item in items if item.status == "有事件"),
            },
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).replace("</", "<\\/")
    template = r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>WorldSimFlow Result Center</title>
  <style>
    :root {
      --bg:#f5f7fb; --panel:#ffffff; --panel-soft:#f9fafb; --fg:#182230; --muted:#667085;
      --line:#d7dde7; --soft:#eef2f7; --accent:#1769e0; --accent-2:#0f8f72;
      --ink:#111827; --ok:#147a46; --warn:#b25e09; --bad:#c13535; --shadow:0 12px 28px rgba(16,24,40,.08);
    }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--bg); color:var(--fg); font-family:Inter, "Segoe UI", Arial, Helvetica, sans-serif; letter-spacing:0; }
    a { color:inherit; }
    .shell { max-width:1320px; margin:0 auto; padding:22px; }
    .topbar { display:flex; align-items:center; justify-content:space-between; gap:16px; margin-bottom:18px; }
    .brand { display:flex; align-items:center; gap:12px; min-width:0; }
    .mark { width:38px; height:38px; border-radius:8px; display:grid; place-items:center; background:#182230; color:#fff; font-weight:800; }
    h1 { margin:0; font-size:25px; line-height:1.15; letter-spacing:0; }
    .subtitle { margin:4px 0 0; color:var(--muted); font-size:13px; line-height:1.45; }
    .top-actions { display:flex; gap:8px; flex-wrap:wrap; justify-content:flex-end; }
    .button { display:inline-flex; align-items:center; justify-content:center; min-height:36px; padding:8px 12px; border:1px solid var(--line); border-radius:8px; background:var(--panel); color:var(--fg); text-decoration:none; font-size:13px; font-weight:700; cursor:pointer; }
    .button.primary { border-color:var(--accent); background:var(--accent); color:#fff; }
    .button.ghost { background:transparent; }
    .hero { display:grid; grid-template-columns:minmax(280px, .92fr) minmax(420px, 1.55fr); gap:16px; align-items:stretch; }
    .panel { background:var(--panel); border:1px solid var(--line); border-radius:8px; box-shadow:var(--shadow); }
    .overview { padding:18px; display:grid; gap:16px; }
    .overview h2, .section-title h2 { margin:0; font-size:18px; letter-spacing:0; }
    .overview p { margin:0; color:var(--muted); font-size:13px; line-height:1.65; }
    .stats { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:10px; }
    .stat { padding:12px; border:1px solid var(--soft); border-radius:8px; background:var(--panel-soft); }
    .stat span { display:block; color:var(--muted); font-size:12px; margin-bottom:5px; }
    .stat strong { display:block; font-size:24px; line-height:1; color:var(--ink); }
    .workflow-panel { padding:18px; }
    .workflow-list { display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:10px; margin-top:12px; }
    .flow-step { min-width:0; padding:12px; border:1px solid var(--soft); border-radius:8px; background:#fff; display:grid; gap:8px; align-content:start; }
    .flow-head { display:flex; align-items:center; gap:8px; min-width:0; }
    .badge-num { width:26px; height:26px; border-radius:999px; display:grid; place-items:center; background:var(--accent); color:#fff; font-weight:800; font-size:13px; flex:0 0 auto; }
    .flow-step h3 { margin:0; font-size:14px; line-height:1.25; overflow-wrap:anywhere; }
    .flow-step p { margin:0; color:var(--muted); font-size:12px; line-height:1.5; min-height:36px; }
    .flow-step .button { justify-self:start; min-height:32px; padding:6px 10px; }
    .flow-step .button.disabled { background:#98a2b3; border-color:#98a2b3; color:#fff; pointer-events:none; }
    .section { margin-top:18px; }
    .section-title { display:flex; justify-content:space-between; align-items:flex-end; gap:12px; margin-bottom:10px; }
    .section-title p { margin:4px 0 0; color:var(--muted); font-size:13px; }
    .highlight-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; }
    .card { padding:14px; background:var(--panel); border:1px solid var(--line); border-radius:8px; display:grid; gap:10px; min-width:0; }
    .card.highlight { box-shadow:0 8px 22px rgba(16,24,40,.06); }
    .card h3 { margin:0; font-size:16px; line-height:1.3; overflow-wrap:anywhere; }
    .row { display:flex; justify-content:space-between; align-items:flex-start; gap:10px; }
    .desc { margin:0; color:var(--muted); font-size:13px; line-height:1.55; }
    .meta { display:flex; flex-wrap:wrap; gap:6px; }
    .chip { display:inline-flex; align-items:center; min-height:22px; padding:3px 8px; border-radius:999px; background:#eef2f7; color:#344054; font-size:12px; white-space:nowrap; }
    .chip.status-ok { background:#e8f6ee; color:var(--ok); }
    .chip.status-report { background:#eaf2ff; color:var(--accent); }
    .chip.status-bad { background:#ffeceb; color:var(--bad); }
    .chip.status-warn { background:#fff4e6; color:var(--warn); }
    .path { color:#7a8494; font-size:12px; overflow-wrap:anywhere; line-height:1.45; }
    .open { justify-self:start; display:inline-flex; align-items:center; justify-content:center; min-height:34px; padding:8px 11px; border-radius:8px; background:var(--accent); color:#fff; font-weight:800; font-size:13px; text-decoration:none; }
    .workspace { display:grid; grid-template-columns:240px minmax(0,1fr); gap:14px; }
    .filters { position:sticky; top:12px; align-self:start; padding:12px; background:var(--panel); border:1px solid var(--line); border-radius:8px; }
    .filters h3 { margin:0 0 10px; font-size:14px; }
    .filter-list { display:grid; gap:7px; }
    .filter { width:100%; display:flex; justify-content:space-between; align-items:center; gap:8px; min-height:34px; padding:7px 9px; border:1px solid transparent; border-radius:8px; background:transparent; color:var(--fg); cursor:pointer; font-size:13px; text-align:left; }
    .filter.active { border-color:var(--accent); background:#eef5ff; color:var(--accent); font-weight:800; }
    .filter small { color:var(--muted); font-size:12px; }
    .searchbar { display:grid; grid-template-columns:minmax(0,1fr) auto; gap:10px; margin-bottom:10px; }
    input[type="search"] { width:100%; min-height:40px; border:1px solid var(--line); border-radius:8px; background:#fff; color:var(--fg); padding:9px 12px; font-size:14px; }
    .result-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(300px,1fr)); gap:12px; }
    .reports-panel { padding:0; overflow:hidden; }
    table { width:100%; border-collapse:collapse; background:#fff; }
    th, td { border-bottom:1px solid var(--soft); padding:11px 12px; text-align:left; vertical-align:top; font-size:13px; line-height:1.45; }
    th { color:#475467; background:#f8fafc; font-weight:800; }
    td a { color:var(--accent); font-weight:800; text-decoration:none; }
    .empty { padding:22px; border:1px dashed var(--line); border-radius:8px; color:var(--muted); background:#fff; }
    @media (max-width:1100px) { .hero { grid-template-columns:1fr; } .workflow-list { grid-template-columns:repeat(3,minmax(0,1fr)); } .highlight-grid { grid-template-columns:repeat(2,minmax(0,1fr)); } .workspace { grid-template-columns:1fr; } .filters { position:static; } .filter-list { grid-template-columns:repeat(2,minmax(0,1fr)); } }
    @media (max-width:720px) { .shell { padding:14px; } .topbar { align-items:flex-start; flex-direction:column; } .stats { grid-template-columns:repeat(2,minmax(0,1fr)); } .workflow-list, .highlight-grid, .result-grid, .filter-list { grid-template-columns:1fr; } .searchbar { grid-template-columns:1fr; } h1 { font-size:23px; } }
  </style>
</head>
<body>
  <main class="shell">
    <header class="topbar">
      <div class="brand">
        <div class="mark">WSF</div>
        <div>
          <h1>WorldSimFlow Result Center</h1>
          <p class="subtitle">仿真、干预、诊断、RL 和世界模型结果入口</p>
        </div>
      </div>
      <div class="top-actions">
        <a class="button primary" id="primaryLink" href="#">进入 Log Lab</a>
        <a class="button ghost" href="../README.md">README</a>
      </div>
    </header>

    <section class="hero">
      <div class="panel overview">
        <div>
          <h2>当前结果概览</h2>
          <p>从左侧推荐路线进入主流程；需要找具体实验时，用下方筛选和搜索。</p>
        </div>
        <div class="stats">
          <div class="stat"><span>页面</span><strong id="htmlCount"></strong></div>
          <div class="stat"><span>报告</span><strong id="reportCount"></strong></div>
          <div class="stat"><span>推荐</span><strong id="recommendedCount"></strong></div>
          <div class="stat"><span>事件</span><strong id="eventCount"></strong></div>
        </div>
      </div>
      <div class="panel workflow-panel">
        <div class="section-title"><div><h2>推荐路线</h2><p>按顺序看，能最快理解平台能力。</p></div></div>
        <div class="workflow-list" id="workflows"></div>
      </div>
    </section>

    <section class="section">
      <div class="section-title"><div><h2>关键入口</h2><p>最值得先打开的页面。</p></div></div>
      <div class="highlight-grid" id="highlights"></div>
    </section>

    <section class="section workspace">
      <aside class="filters">
        <h3>分类筛选</h3>
        <div class="filter-list" id="filters"></div>
      </aside>
      <div>
        <div class="section-title"><div><h2>全部页面</h2><p id="resultCount"></p></div></div>
        <div class="searchbar">
          <input id="search" type="search" placeholder="搜索 Log Lab / traffic / policy / BEV / world model / actor..." aria-label="搜索结果">
          <button class="button" id="clearSearch" type="button">清空</button>
        </div>
        <div class="result-grid" id="items"></div>
      </div>
    </section>

    <section class="section">
      <div class="section-title"><div><h2>结构化报告</h2><p>JSON 报告用于复盘指标、事件、数据集和训练结果。</p></div></div>
      <div class="panel reports-panel"><table><thead><tr><th>类型</th><th>文件</th><th>说明</th><th>打开</th></tr></thead><tbody id="reports"></tbody></table></div>
    </section>
  </main>
  <script>
    const data = __LAUNCHER_DATA__;
    const phaseOrder = ['全部', '推荐', '事件', ...Object.keys(data.phase_counts).sort()];
    let active = '全部';
    const search = document.getElementById('search');
    const filters = document.getElementById('filters');
    const itemsEl = document.getElementById('items');
    const highlightsEl = document.getElementById('highlights');
    const reportsEl = document.getElementById('reports');
    const workflowsEl = document.getElementById('workflows');
    document.getElementById('htmlCount').textContent = data.summary.html_count;
    document.getElementById('reportCount').textContent = data.summary.report_count;
    document.getElementById('recommendedCount').textContent = data.summary.recommended_count;
    document.getElementById('eventCount').textContent = data.summary.event_count;
    const lab = data.items.find(item => item.title === 'Log Lab 2.0') || data.items.find(item => item.recommended);
    if (lab) document.getElementById('primaryLink').href = lab.url;
    function statusClass(status) { if (status === '正常') return 'status-ok'; if (status === '报告') return 'status-report'; if (status === '有事件') return 'status-bad'; return 'status-warn'; }
    function countForPhase(phase) { if (phase === '全部') return data.items.length; if (phase === '推荐') return data.items.filter(item => item.recommended).length; if (phase === '事件') return data.items.filter(item => item.status === '有事件').length; return data.phase_counts[phase] || 0; }
    function cardHtml(item, highlight=false) { return `<article class="card ${highlight ? 'highlight' : ''}"><div class="row"><h3>${escapeHtml(item.title)}</h3><span class="chip ${statusClass(item.status)}">${escapeHtml(item.status)}</span></div><p class="desc">${escapeHtml(item.description)}</p><div class="meta"><span class="chip">${escapeHtml(item.phase)}</span><span class="chip">${escapeHtml(item.kind)}</span></div><div class="path">${escapeHtml(item.path)}</div><a class="open" href="${item.url}">${escapeHtml(item.action_label)}</a></article>`; }
    function workflowHtml(item) { const disabled = item.url ? '' : ' disabled'; const href = item.url || '#'; return `<article class="flow-step"><div class="flow-head"><span class="badge-num">${escapeHtml(item.step)}</span><h3>${escapeHtml(item.title)}</h3></div><p>${escapeHtml(item.description)}</p><a class="button primary${disabled}" href="${href}">${escapeHtml(item.action_label)}</a></article>`; }
    function reportHtml(report) { return `<tr><td>${escapeHtml(report.kind)}</td><td>${escapeHtml(report.path)}</td><td>${escapeHtml(report.description)}</td><td><a href="${report.url}">${escapeHtml(report.action_label)}</a></td></tr>`; }
    function renderFilters() { filters.innerHTML = phaseOrder.map(phase => `<button type="button" class="filter ${phase === active ? 'active' : ''}" data-phase="${escapeAttr(phase)}"><span>${escapeHtml(phase)}</span><small>${countForPhase(phase)}</small></button>`).join(''); filters.querySelectorAll('button').forEach(btn => btn.addEventListener('click', () => { active = btn.dataset.phase; render(); })); }
    function matches(item, q) { const haystack = `${item.title} ${item.phase} ${item.kind} ${item.description} ${item.status} ${item.status_detail} ${item.path}`.toLowerCase(); return haystack.includes(q); }
    function filteredItems() { const q = search.value.trim().toLowerCase(); return data.items.filter(item => { if (active === '推荐' && !item.recommended) return false; if (active === '事件' && item.status !== '有事件') return false; if (!['全部','推荐','事件'].includes(active) && item.phase !== active) return false; return !q || matches(item, q); }); }
    function render() { renderFilters(); const list = filteredItems(); workflowsEl.innerHTML = data.workflows.map(workflowHtml).join('') || '<div class="empty">暂无推荐路线。</div>'; highlightsEl.innerHTML = data.highlights.map(item => cardHtml(item, true)).join('') || '<div class="empty">暂无关键入口。</div>'; itemsEl.innerHTML = list.map(item => cardHtml(item)).join('') || '<div class="empty">没有匹配的页面。</div>'; document.getElementById('resultCount').textContent = `${list.length} / ${data.items.length} 个页面`; reportsEl.innerHTML = data.reports.map(reportHtml).join('') || '<tr><td colspan="4">暂无报告。</td></tr>'; }
    function escapeHtml(value) { return String(value).replace(/[&<>'"]/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[ch])); }
    function escapeAttr(value) { return escapeHtml(value).replace(/`/g, '&#96;'); }
    search.addEventListener('input', render);
    document.getElementById('clearSearch').addEventListener('click', () => { search.value = ''; render(); search.focus(); });
    render();
  </script>
</body>
</html>
'''
    return template.replace("__LAUNCHER_DATA__", payload)


if __name__ == "__main__":
    main()
