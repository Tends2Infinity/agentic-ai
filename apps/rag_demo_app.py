"""
Streamlit demo: AgentRunner + RAG tool (hybrid retrieval + injection hygiene)

How it works:
- Builds a ToolRegistry and registers retrieve_context
- Creates a deterministic planner that produces a 1-step plan:
    call retrieve_context(query, index_path, top_k, mode, backend)
- Runs AgentRunner (Planner → Executor → Verifier)
- Displays:
  - retrieval results with citations + warnings
  - the run trace (JSONL) for observability

Product-grade enhancements:
- Tabs: Results / Trace / Index Stats
- Copy run_id UX (text input + code block copy)
- Download run trace JSONL (best Streamlit equivalent of "link to file")
- Index stats: num chunks, vocab size (idf terms), embedding presence

Run:
  .venv\\Scripts\\python.exe -m streamlit run apps/rag_demo_app.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import streamlit as st
import time
from datetime import datetime

from src.common.agents.runner import AgentRunner
from src.common.agents.simple_planner_verifier import SimpleVerifier
from src.common.agents.types import Plan, PlanStep
from src.common.rag.index import load_index
from src.common.rag.tool import RetrieveContextIn, RetrieveContextOut, retrieve_context_tool
from src.common.tools.registry import ToolRegistry, ToolSpec
from src.common.utils.config import load_config
from src.common.utils.logging import new_run_context


# -----------------------------
# Planner + ToolRegistry wiring
# -----------------------------

class RagPlanner:
    """Deterministic planner: always calls retrieve_context once."""

    def __init__(self, index_path: str, backend: str, mode: str, top_k: int) -> None:
        self.index_path = index_path
        self.backend = backend
        self.mode = mode
        self.top_k = top_k

    def create_plan(self, user_input: str) -> Plan:
        return Plan(
            objective="Retrieve relevant context for the query",
            steps=[
                PlanStep(
                    step_id="step_1",
                    tool_name="retrieve_context",
                    args={
                        "index_path": self.index_path,
                        "query": user_input,
                        "top_k": self.top_k,
                        "mode": self.mode,
                        "backend": self.backend,
                    },
                    rationale="Use RAG to fetch relevant chunks with injection hygiene.",
                )
            ],
        )


def build_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(
        ToolSpec(
            name="retrieve_context",
            description="Retrieve relevant chunks from a RAG index with injection hygiene.",
            input_model=RetrieveContextIn,
            output_model=RetrieveContextOut,
            fn=retrieve_context_tool,
        )
    )
    return reg


# -----------------------------
# Helpers
# -----------------------------

def run_trace_path(runs_dir: str, run_id: str) -> Path:
    return Path(runs_dir) / f"{run_id}.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    return events


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    with open(path, encoding="utf-8") as f:
        return f.read()


def extract_plan_steps(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Extract plan steps from the plan_created event (if present).
    Returns rows suitable for st.dataframe.
    """
    for e in events:
        if e.get("event_type") == "plan_created":
            payload = e.get("payload") or {}
            steps = payload.get("steps") or []
            rows: list[dict[str, Any]] = []
            for s in steps:
                rows.append(
                    {
                        "step_id": s.get("step_id"),
                        "tool_name": s.get("tool_name"),
                        "rationale": s.get("rationale"),
                        "args": json.dumps(s.get("args") or {}, ensure_ascii=False),
                    }
                )
            return rows
    return []


def extract_tool_events(events: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Extract tool call timeline + outputs.

    Returns:
      - timeline_rows: for a summary table (started/succeeded/failed)
      - outputs: list of per-step outputs (from tool_call_succeeded)
    """
    timeline_rows: list[dict[str, Any]] = []
    outputs: list[dict[str, Any]] = []

    for e in events:
        et = e.get("event_type")
        payload = e.get("payload") or {}
        if et in ("tool_call_started", "tool_call_succeeded", "tool_call_failed"):
            timeline_rows.append(
                {
                    "ts_ms": e.get("ts_ms"),
                    "event_type": et,
                    "step_id": payload.get("step_id"),
                    "tool_name": payload.get("tool_name"),
                }
            )

        if et == "tool_call_succeeded":
            outputs.append(
                {
                    "step_id": payload.get("step_id"),
                    "tool_name": payload.get("tool_name"),
                    "output": payload.get("output"),
                }
            )

    return timeline_rows, outputs


def extract_verification(events: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Extract verification details from run_completed event (if present).
    """
    for e in events:
        if e.get("event_type") == "run_completed":
            payload = e.get("payload") or {}
            return {
                "verified": payload.get("verified"),
                "verification_errors": payload.get("verification_errors") or [],
                "num_steps_executed": payload.get("num_steps_executed"),
            }
    return {"verified": None, "verification_errors": [], "num_steps_executed": None}


def compute_index_stats(index_path: str) -> dict[str, Any]:
    """
    Load index and compute a few useful stats.
    """
    idx = load_index(index_path)
    num_chunks = len(idx.chunks)
    vocab_size = len(idx.idf)

    # Embedding presence check (older indexes may have None embeddings)
    with_embeddings = 0
    for c in idx.chunks:
        # attribute exists by now, but may be None
        emb = getattr(c, "embedding_vector", None)
        if emb:
            with_embeddings += 1

    return {
        "index_path": index_path,
        "num_chunks": num_chunks,
        "vocab_size_idf_terms": vocab_size,
        "chunks_with_embedding_vector": with_embeddings,
        "chunks_missing_embedding_vector": num_chunks - with_embeddings,
    }


def format_ts_ms(ts_ms: int | None) -> str:
    """
    Convert epoch milliseconds to readable local time string.
    """
    if ts_ms is None:
        return "-"
    dt = datetime.fromtimestamp(ts_ms / 1000.0)
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # trim to ms


def compute_relative_ms(events: list[dict[str, Any]]) -> dict[int, int]:
    """
    Compute relative time (ms) from first event.
    Returns mapping: ts_ms -> delta_ms
    """
    if not events:
        return {}

    ts_values = [e.get("ts_ms") for e in events if e.get("ts_ms") is not None]
    if not ts_values:
        return {}

    start = min(ts_values)
    return {ts: ts - start for ts in ts_values}


def to_local_dt(ts_ms: int) -> datetime:
    return datetime.fromtimestamp(ts_ms / 1000.0)


def build_step_latencies(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Build per-step latency stats by pairing tool_call_started with tool_call_succeeded/failed.
    Returns rows suitable for tables and charts.
    """
    started: dict[tuple[str, str], int] = {}  # (step_id, tool_name) -> ts_ms
    rows: list[dict[str, Any]] = []

    for e in events:
        et = e.get("event_type")
        payload = e.get("payload") or {}
        step_id = payload.get("step_id")
        tool_name = payload.get("tool_name")
        ts = e.get("ts_ms")

        if not step_id or not tool_name or ts is None:
            continue

        key = (step_id, tool_name)

        if et == "tool_call_started":
            started[key] = ts

        if et in ("tool_call_succeeded", "tool_call_failed"):
            if key in started:
                start_ts = started[key]
                end_ts = ts
                rows.append(
                    {
                        "step_id": step_id,
                        "tool_name": tool_name,
                        "status": "success" if et == "tool_call_succeeded" else "failure",
                        "start_ts_ms": start_ts,
                        "end_ts_ms": end_ts,
                        "latency_ms": end_ts - start_ts,
                        "start_time": to_local_dt(start_ts),
                        "end_time": to_local_dt(end_ts),
                    }
                )

    # Sort by start time
    rows.sort(key=lambda r: r["start_ts_ms"])
    return rows


def build_run_window(events: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Compute overall run start/end from first/last ts_ms.
    """
    ts_values = [e.get("ts_ms") for e in events if e.get("ts_ms") is not None]
    if not ts_values:
        return {"run_start_ts_ms": None, "run_end_ts_ms": None, "run_duration_ms": None}

    start_ts = min(ts_values)
    end_ts = max(ts_values)
    return {
        "run_start_ts_ms": start_ts,
        "run_end_ts_ms": end_ts,
        "run_duration_ms": end_ts - start_ts,
        "run_start_time": to_local_dt(start_ts),
        "run_end_time": to_local_dt(end_ts),
    }


def summarize_run(result: Any, events: list[dict[str, Any]], run_id: str, query: str, backend: str, mode: str) -> dict[str, Any]:
    """
    Create a compact summary dict for run comparison.
    """
    ver = extract_verification(events)
    step_lat = build_step_latencies(events)
    run_win = build_run_window(events)

    # Pull tool outputs from step_results
    num_results = 0
    warnings_count = 0
    top_score = 0.0

    if result and getattr(result, "step_results", None):
        for sr in result.step_results:
            out = sr.output or {}
            res_list = out.get("results", [])
            if isinstance(res_list, list):
                num_results += len(res_list)
                for item in res_list:
                    w = item.get("warnings") or []
                    warnings_count += len(w)
                    s = float(item.get("score", 0.0))
                    top_score = max(top_score, s)

    return {
        "run_id": run_id,
        "query": query,
        "backend": backend,
        "mode": mode,
        "verified": bool(ver.get("verified")),
        "verification_errors": ver.get("verification_errors") or [],
        "num_steps": len(step_lat),
        "run_duration_ms": run_win.get("run_duration_ms"),
        "num_results": num_results,
        "warnings_count": warnings_count,
        "top_score": top_score,
        "step_latencies": step_lat,
        "events_path": str(run_trace_path(load_config().runs_dir, run_id)),
    }


def evaluation_score(summary: dict[str, Any]) -> dict[str, Any]:
    """
    Explainable scorecard (0-100) for demo purposes.
    This is NOT a “truth metric”; it’s a product-facing quality signal.
    """
    # Components (0..1)
    safety = 1.0
    if summary["warnings_count"] > 0:
        # Penalize warnings in warn mode; in trusted_only, warnings should be 0
        safety = max(0.0, 1.0 - 0.25 * summary["warnings_count"])

    reliability = 1.0 if summary["verified"] and not summary["verification_errors"] else 0.0

    relevance = min(1.0, summary["top_score"])  # already roughly [0..1] for hybrid due to normalization

    usefulness = 1.0 if summary["num_results"] > 0 else 0.0

    # Weighted total (tuneable)
    total = (
        0.35 * reliability +
        0.30 * safety +
        0.25 * relevance +
        0.10 * usefulness
    )
    score_100 = int(round(total * 100))

    return {
        "score_100": score_100,
        "reliability": reliability,
        "safety": safety,
        "relevance": relevance,
        "usefulness": usefulness,
    }

# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="Agentic-AI: RAG Tool Demo", layout="wide")

st.title("Agentic-AI — RAG Tool Demo")
st.caption("AgentRunner + ToolRegistry + RAG (TF-IDF / Embeddings / Hybrid) + prompt-injection hygiene + run traces.")

cfg = load_config()

# Sidebar controls
with st.sidebar:
    st.header("Controls")

    index_path = st.text_input("Index path", value="data/indexes/sample.json")
    backend = st.selectbox("Retrieval backend", ["tfidf", "embeddings", "hybrid"], index=2)
    mode = st.selectbox("Hygiene mode", ["warn", "trusted_only"], index=0)
    top_k = st.slider("Top K", min_value=1, max_value=10, value=5)

    st.markdown("---")
    st.subheader("Build / Refresh Index")
    st.code(
        ".venv\\Scripts\\python.exe -m src.common.rag.cli ingest --docs-dir data/sample_docs --out data/indexes/sample.json",
        language="powershell",
    )
    st.caption("Tip: Re-ingest after changing docs so embeddings are present in the index.")

# Main input
query = st.text_input("Query", value="onboarding SOP")

col_run, col_clear = st.columns([1, 1])
with col_run:
    run_clicked = st.button("Run Agent", type="primary")
with col_clear:
    clear_clicked = st.button("Clear Last Run")

# Session state for last run
if clear_clicked:
    st.session_state.pop("last_run_id", None)
    st.session_state.pop("last_result", None)
    st.session_state.pop("last_trace_path", None)
    st.session_state.pop("run_history", None)

if run_clicked:
    if not os.path.exists(index_path):
        st.error(f"Index not found: {index_path}\n\nRun ingest first to create it.")
        st.stop()

    registry = build_registry()
    planner = RagPlanner(index_path=index_path, backend=backend, mode=mode, top_k=top_k)
    verifier = SimpleVerifier()
    ctx = new_run_context()

    runner = AgentRunner(
        planner=planner,
        registry=registry,
        verifier=verifier,
        runs_dir=cfg.runs_dir,
        run_id=ctx.run_id,
        max_steps=3,
    )

    result = runner.run(query)
    trace_file = run_trace_path(cfg.runs_dir, ctx.run_id)

    st.session_state["last_run_id"] = ctx.run_id
    st.session_state["last_result"] = result
    st.session_state["last_trace_path"] = str(trace_file)

    # Store run history for comparison (keep last 10)
    trace_file = run_trace_path(cfg.runs_dir, ctx.run_id)
    events = read_jsonl(trace_file)

    run_summary = summarize_run(
        result=result,
        events=events,
        run_id=ctx.run_id,
        query=query,
        backend=backend,
        mode=mode,
    )

    history = st.session_state.get("run_history", [])
    history.insert(0, run_summary)
    st.session_state["run_history"] = history[:10]

# Header summary (if there is a run)
last_run_id = st.session_state.get("last_run_id")
last_result = st.session_state.get("last_result")
last_trace_path_str = st.session_state.get("last_trace_path")

if last_run_id and last_result:
    st.success("Run completed.")
    c1, c2, c3 = st.columns([1.2, 0.6, 1.2])

    with c1:
        st.write("**run_id**")
        # Copy-friendly text input
        st.text_input("Copy run_id", value=last_run_id, label_visibility="collapsed")
        # Also show code block (often has a copy button in Streamlit UI)
        st.code(last_run_id)

    with c2:
        st.write("**verified**")
        st.metric(label="Verified", value=str(bool(last_result.verified)))

    with c3:
        st.write("**run trace file**")
        st.code(last_trace_path_str or "")
        # Download trace file (acts like "link")
        if last_trace_path_str:
            trace_path = Path(last_trace_path_str)
            content = read_text(trace_path)
            if content:
                st.download_button(
                    label="Download run trace (JSONL)",
                    data=content,
                    file_name=trace_path.name,
                    mime="application/jsonl",
                )

    if getattr(last_result, "verification_errors", None):
        if last_result.verification_errors:
            st.error("Verification errors:")
            st.write(last_result.verification_errors)

# Tabs
tab_results, tab_trace, tab_stats, tab_compare = st.tabs(["Results", "Trace", "Index Stats", "Compare"])

# -----------------------------
# Results tab
# -----------------------------
with tab_results:
    st.subheader("Retrieved Context (Tool Output)")

    if not last_run_id or not last_result:
        st.info("Run the agent to see results.")
    else:
        if not last_result.step_results or not last_result.step_results[0].output:
            st.warning("No tool output returned.")
        else:
            out = last_result.step_results[0].output
            results = out.get("results", [])

            if not results:
                st.info("No results returned.")
            else:
                st.caption("Each result includes: score, source, chunk_id, warnings (if any), and the retrieved chunk text.")
                for i, item in enumerate(results, start=1):
                    title = f"[{i}] score={item['score']:.4f} | source={item['source']} | chunk={item['chunk_id']}"
                    expanded = i == 1
                    with st.expander(title, expanded=expanded):
                        if item.get("warnings"):
                            st.warning(f"Warnings: {item['warnings']}")
                        st.code(item["text"], language="markdown")

# -----------------------------
# Trace tab
# -----------------------------
with tab_trace:
    st.subheader("Run Trace")

    if not last_run_id or not last_trace_path_str:
        st.info("Run the agent to see trace events.")
    else:
        path = Path(last_trace_path_str)
        events = read_jsonl(path)
        if not events:
            st.warning("No trace events found. (Unexpected — check runs_dir and file permissions.)")
        else:
            # -----------------------------
            # Verification alert area
            # -----------------------------
            ver = extract_verification(events)
            if ver["verified"] is True:
                st.success("✅ Verification passed (run_completed.verified=true).")
            elif ver["verified"] is False:
                st.error("❌ Verification failed (run_completed.verified=false).")
            else:
                st.warning("Verification status unknown (run_completed event missing).")

            if ver["verification_errors"]:
                st.error("Verification Errors")
                st.write(ver["verification_errors"])
            else:
                st.caption("No verification errors.")

            # -----------------------------
            # Plan steps table
            # -----------------------------
            st.markdown("### Plan Steps")
            plan_rows = extract_plan_steps(events)
            if not plan_rows:
                st.info("No plan_created event found, so plan steps are unavailable.")
            else:
                st.dataframe(plan_rows, use_container_width=True, hide_index=True)

            # -----------------------------
            # Trace summary table (timeline)
            # -----------------------------
            st.markdown("### Trace Summary")
            timeline_rows, outputs = extract_tool_events(events)

            # Add plan + run events into the timeline for completeness
            enriched_timeline: list[dict[str, Any]] = []
            for e in events:
                et = e.get("event_type")
                payload = e.get("payload") or {}
                if et in ("plan_created", "run_completed"):
                    enriched_timeline.append(
                        {
                            "ts_ms": e.get("ts_ms"),
                            "event_type": et,
                            "step_id": payload.get("step_id"),
                            "tool_name": payload.get("tool_name"),
                        }
                    )
            enriched_timeline.extend(timeline_rows)
            enriched_timeline.sort(key=lambda r: (r["ts_ms"] or 0))

            # Compute relative times
            rel_map = compute_relative_ms(events)

            timeline_display: list[dict[str, Any]] = []
            for r in enriched_timeline:
                ts = r.get("ts_ms")
                timeline_display.append(
                    {
                        "time_local": format_ts_ms(ts),
                        "delta_ms": f"+{rel_map.get(ts, 0)}",
                        "event_type": r.get("event_type"),
                        "step_id": r.get("step_id"),
                        "tool_name": r.get("tool_name"),
                    }
                )

            st.dataframe(timeline_display, use_container_width=True, hide_index=True)
            

            # -----------------------------
            # Tool outputs per step
            # -----------------------------
            st.markdown("### Tool Outputs (per step)")
            if not outputs:
                st.info("No tool_call_succeeded events found, so tool outputs are unavailable.")
            else:
                for out in outputs:
                    step_id = out.get("step_id")
                    tool_name = out.get("tool_name")
                    payload = out.get("output")

                    with st.expander(f"{step_id} — {tool_name}", expanded=True):
                        if payload is None:
                            st.warning("No output payload.")
                        else:
                            # Pretty-print output JSON
                            st.json(payload)

            # -----------------------------
            # Raw events (debug)
            # -----------------------------
            st.markdown("### Raw Events (JSON)")
            st.json(events)

            # -----------------------------
            # Latency metrics per step
            # -----------------------------
            st.markdown("### Step Latency Metrics")

            step_lat = build_step_latencies(events)
            if not step_lat:
                st.info("No step latency data found.")
            else:
                # Metrics row
                latencies = [r["latency_ms"] for r in step_lat]
                st.metric("Total steps", len(step_lat))
                st.metric("Total tool time (ms)", sum(latencies))
                st.metric("Max step latency (ms)", max(latencies))

                st.dataframe(
                    [
                        {
                            "step_id": r["step_id"],
                            "tool_name": r["tool_name"],
                            "status": r["status"],
                            "start_time": r["start_time"].strftime("%H:%M:%S.%f")[:-3],
                            "end_time": r["end_time"].strftime("%H:%M:%S.%f")[:-3],
                            "latency_ms": r["latency_ms"],
                        }
                        for r in step_lat
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

            # -----------------------------
            # Gantt-style timeline (Altair if available)
            # -----------------------------
            st.markdown("### Gantt Timeline")

            run_win = build_run_window(events)
            if not step_lat or run_win["run_duration_ms"] is None:
                st.info("Not enough timing data to draw a timeline.")
            else:
                # We try Altair; if missing, fallback to a simple table.
                try:
                    import pandas as pd
                    import altair as alt

                    df = pd.DataFrame(
                        [
                            {
                                "task": f"{r['step_id']}::{r['tool_name']}",
                                "start": r["start_time"],
                                "end": r["end_time"],
                                "status": r["status"],
                                "latency_ms": r["latency_ms"],
                            }
                            for r in step_lat
                        ]
                    )

                    chart = (
                        alt.Chart(df)
                        .mark_bar()
                        .encode(
                            y=alt.Y("task:N", sort="-x"),
                            x=alt.X("start:T", title="Time"),
                            x2="end:T",
                            tooltip=["task", "status", "latency_ms"],
                        )
                        .properties(height=50 + 28 * len(df))
                    )
                    st.altair_chart(chart, use_container_width=True)

                except Exception:
                    st.warning("Altair not available; showing timeline as a table instead.")
                    st.dataframe(step_lat, use_container_width=True, hide_index=True)

            # -----------------------------
            # Evaluation score summary
            # -----------------------------
            st.markdown("### Evaluation Score Summary")

            # Build summary from current run
            summary = summarize_run(
                result=last_result,
                events=events,
                run_id=last_run_id,
                query=query,
                backend=backend,
                mode=mode,
            )
            score = evaluation_score(summary)

            c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])
            c1.metric("Score (0-100)", score["score_100"])
            c2.metric("Reliability", f"{int(score['reliability']*100)}%")
            c3.metric("Safety", f"{int(score['safety']*100)}%")
            c4.metric("Relevance", f"{int(score['relevance']*100)}%")
            c5.metric("Usefulness", f"{int(score['usefulness']*100)}%")

            st.caption("Scoring is a simple, explainable heuristic for demo purposes (not ground truth).")

# -----------------------------
# Index Stats tab
# -----------------------------
with tab_stats:
    st.subheader("Index Stats")

    if not os.path.exists(index_path):
        st.warning(f"Index not found: {index_path}")
        st.info("Ingest sample docs first to generate an index.")
    else:
        stats = compute_index_stats(index_path)
        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Chunks", stats["num_chunks"])
        c2.metric("Vocab (IDF terms)", stats["vocab_size_idf_terms"])
        c3.metric("Chunks w/ embeddings", stats["chunks_with_embedding_vector"])
        c4.metric("Chunks missing embeddings", stats["chunks_missing_embedding_vector"])

        st.caption("Details")
        st.json(stats)
        
# -----------------------------
# Compare tab
# -----------------------------
with tab_compare:
    st.subheader("Run Comparison (side-by-side)")

    history = st.session_state.get("run_history", [])
    if not history:
        st.info("No runs yet. Execute a few runs, then compare them here.")
    else:
        st.caption("Select up to 3 runs to compare. Newest runs appear first.")

        options = [f"{h['run_id']} | {h['query']} | {h['backend']}/{h['mode']}" for h in history]
        selected = st.multiselect("Choose runs", options, default=options[:2])

        # Map back to summaries
        selected_runs: list[dict[str, Any]] = []
        for s in selected:
            run_id = s.split(" | ")[0].strip()
            match = next((h for h in history if h["run_id"] == run_id), None)
            if match:
                selected_runs.append(match)

        if not selected_runs:
            st.info("Select runs to compare.")
        else:
            # Comparison table
            st.markdown("### Comparison Table")
            st.dataframe(
                [
                    {
                        "run_id": r["run_id"],
                        "query": r["query"],
                        "backend": r["backend"],
                        "mode": r["mode"],
                        "verified": r["verified"],
                        "duration_ms": r["run_duration_ms"],
                        "steps": r["num_steps"],
                        "results": r["num_results"],
                        "warnings": r["warnings_count"],
                        "top_score": f"{r['top_score']:.3f}",
                    }
                    for r in selected_runs
                ],
                use_container_width=True,
                hide_index=True,
            )

            # Side-by-side panels
            st.markdown("### Side-by-side Detail")
            cols = st.columns(min(3, len(selected_runs)))
            for col, r in zip(cols, selected_runs, strict=False):
                with col:
                    st.markdown(f"**Run:** `{r['run_id']}`")
                    st.write(f"**Query:** {r['query']}")
                    st.write(f"**Backend/Mode:** {r['backend']} / {r['mode']}")
                    st.write(f"**Verified:** {r['verified']}")
                    st.write(f"**Duration (ms):** {r['run_duration_ms']}")
                    st.write(f"**Results:** {r['num_results']}")
                    st.write(f"**Warnings:** {r['warnings_count']}")

                    score = evaluation_score(r)
                    st.metric("Score (0-100)", score["score_100"])

                    # Step latency view
                    if r["step_latencies"]:
                        st.caption("Step latencies (ms)")
                        st.dataframe(
                            [
                                {"step_id": s["step_id"], "tool": s["tool_name"], "status": s["status"], "latency_ms": s["latency_ms"]}
                                for s in r["step_latencies"]
                            ],
                            use_container_width=True,
                            hide_index=True,
                        )

                    # Trace file download
                    trace_path = Path(r["events_path"])
                    content = read_text(trace_path)
                    if content:
                        st.download_button(
                            label="Download trace JSONL",
                            data=content,
                            file_name=trace_path.name,
                            mime="application/jsonl",
                            key=f"dl_{r['run_id']}",
                        )