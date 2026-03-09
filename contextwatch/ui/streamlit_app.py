"""ContextWatch — Streamlit UI for live inference monitoring.

Launch with::

    streamlit run contextwatch/ui/streamlit_app.py

Provides four panels:
    1. Run Panel — configure and launch inference
    2. Live Metrics — token counts, context %, latency, memory
    3. Graphs — latency / context / memory over generation steps
    4. Forecast — resource-exhaustion warnings
"""

from __future__ import annotations

try:
    import streamlit as st
except ImportError:
    raise ImportError(
        "Streamlit is required for the ContextWatch UI. "
        "Install it with:  pip install contextwatch[ui]"
    ) from None

try:
    import pandas as pd
except ImportError:
    raise ImportError(
        "Pandas is required for the ContextWatch UI. "
        "Install it with:  pip install contextwatch[ui]"
    ) from None

st.set_page_config(
    page_title="ContextWatch",
    page_icon="🔍",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS for a polished look
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp { font-family: 'Inter', sans-serif; }
    .metric-card {
        background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%);
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.3rem 0;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .metric-label { color: #a0a0b0; font-size: 0.85rem; margin-bottom: 0.2rem; }
    .metric-value { color: #ffffff; font-size: 1.6rem; font-weight: 700; }
    .metric-sub { color: #70e0a0; font-size: 0.8rem; }
    .warn-box {
        background: rgba(255, 170, 0, 0.12);
        border-left: 4px solid #ffaa00;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        color: #ffcc44;
    }
    .ok-box {
        background: rgba(0, 200, 100, 0.10);
        border-left: 4px solid #00c864;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        color: #70e0a0;
    }
    .mode-badge {
        display: inline-block;
        padding: 0.2rem 0.75rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        margin-left: 0.5rem;
        vertical-align: middle;
    }
    .mode-hf {
        background: rgba(255, 165, 0, 0.18);
        color: #ffaa44;
        border: 1px solid rgba(255, 165, 0, 0.35);
    }
    .mode-vllm {
        background: rgba(80, 160, 255, 0.18);
        color: #60b0ff;
        border: 1px solid rgba(80, 160, 255, 0.35);
    }
    .info-card {
        background: rgba(80, 160, 255, 0.08);
        border: 1px solid rgba(80, 160, 255, 0.2);
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin: 0.3rem 0;
        color: #90c0ff;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header with mode badge
# ---------------------------------------------------------------------------
active_mode = st.session_state.get("mode", None)
if active_mode:
    badge_cls = "mode-hf" if active_mode == "hf" else "mode-vllm"
    badge_label = "HuggingFace" if active_mode == "hf" else "vLLM"
    st.markdown(
        f'# 🔍 ContextWatch <span class="mode-badge {badge_cls}">{badge_label}</span>',
        unsafe_allow_html=True,
    )
else:
    st.title("🔍 ContextWatch")
st.caption("Real-time LLM inference monitoring — context, latency, and memory")

# ---------------------------------------------------------------------------
# 1. Run Panel
# ---------------------------------------------------------------------------
st.sidebar.header("⚙️ Run Configuration")

mode = st.sidebar.selectbox("Backend", ["hf", "vllm"], index=0)

if mode == "vllm":
    endpoint = st.sidebar.text_input("vLLM Endpoint", value="http://localhost:8000")
else:
    endpoint = ""

model_name = st.sidebar.text_input(
    "Model",
    value="distilgpt2" if mode == "hf" else "mistralai/Mistral-7B-v0.1",
)
prompt = st.sidebar.text_area("Prompt", value="Explain transformers", height=100)
max_tokens = st.sidebar.slider("Max Tokens", min_value=10, max_value=500, value=50)
warn_threshold = st.sidebar.slider(
    "Warning Threshold", min_value=0.5, max_value=1.0, value=0.75, step=0.05,
)

# Forecast limits (in sidebar)
st.sidebar.markdown("---")
st.sidebar.subheader("Forecast Limits")
memory_limit_gb = st.sidebar.number_input(
    "Memory Limit (GB)", min_value=0.0, max_value=128.0, value=8.0, step=0.5,
)
latency_limit_ms = st.sidebar.number_input(
    "Latency Limit (ms)", min_value=0.0, max_value=10000.0, value=100.0, step=10.0,
)

run_button = st.sidebar.button("🚀 Run Inference", use_container_width=True, type="primary")


# ---------------------------------------------------------------------------
# Run inference on button click
# ---------------------------------------------------------------------------
if run_button:
    with st.spinner("Running inference..."):
        try:
            if mode == "hf":
                from contextwatch.core.hf_adapter import run_hf

                result = run_hf(
                    model_name=model_name,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    warn_threshold=warn_threshold,
                )
            else:
                from contextwatch.core.vllm_adapter import run_vllm

                result = run_vllm(
                    endpoint=endpoint,
                    model=model_name,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    warn_threshold=warn_threshold,
                )

            st.session_state["result"] = result
            st.session_state["mode"] = mode
            st.session_state["memory_limit_mb"] = memory_limit_gb * 1024.0
            st.session_state["latency_limit_ms"] = latency_limit_ms
            st.session_state["error"] = None
        except Exception as e:
            st.session_state["error"] = str(e)
            st.session_state["result"] = None


# ---------------------------------------------------------------------------
# Error display
# ---------------------------------------------------------------------------
if st.session_state.get("error"):
    st.error(f"Inference failed: {st.session_state['error']}")

# ---------------------------------------------------------------------------
# Display results if available
# ---------------------------------------------------------------------------
result = st.session_state.get("result")

if result is None:
    st.info("Configure your run in the sidebar and click **Run Inference** to start.")
    st.stop()

# ---------------------------------------------------------------------------
# Generated text (main area, prominent)
# ---------------------------------------------------------------------------
st.subheader("Generated Text")
generated_text = result.generated_text or ""
non_ws_len = len(generated_text.strip())

meta_col1, meta_col2 = st.columns(2)
with meta_col1:
    st.caption(f"Characters: {len(generated_text)}")
with meta_col2:
    st.caption(f"Generated tokens: {result.generated_token_count}")

if non_ws_len == 0:
    st.warning("Model output contains only whitespace/special characters.")
    st.code(repr(generated_text), language=None)
else:
    st.code(generated_text, language=None)

# ---------------------------------------------------------------------------
# 2. Live Metrics
# ---------------------------------------------------------------------------
st.header("📊 Live Metrics")

ctx = result.context_summary
lat = result.latency_summary
mem = result.memory_summary

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Prompt Tokens", result.prompt_token_count)
with col2:
    st.metric("Generated Tokens", result.generated_token_count)
with col3:
    if ctx:
        pct = round(ctx.context_used_pct * 100, 1)
        st.metric("Context Used", f"{pct}%", delta=f"{ctx.remaining_tokens} remaining")
    else:
        st.metric("Context Used", "N/A")
with col4:
    if lat and lat.current_token_latency_ms is not None:
        st.metric("Current Latency", f"{lat.current_token_latency_ms:.1f} ms")
    else:
        st.metric("Current Latency", "N/A")

# Second row
col5, col6, col7, col8 = st.columns(4)

with col5:
    st.metric("Total Tokens", result.total_token_count)
with col6:
    if lat and lat.ttft_ms is not None:
        st.metric("TTFT", f"{lat.ttft_ms:.1f} ms")
    else:
        st.metric("TTFT", "N/A")
with col7:
    if lat and lat.rolling_avg_ms is not None:
        trend_delta = None
        if lat.trend_ms_per_100_tokens is not None:
            sign = "+" if lat.trend_ms_per_100_tokens >= 0 else ""
            trend_delta = f"{sign}{lat.trend_ms_per_100_tokens:.1f} ms/100tok"
        st.metric("Rolling Avg Latency", f"{lat.rolling_avg_ms:.1f} ms", delta=trend_delta)
    else:
        st.metric("Rolling Avg Latency", "N/A")
with col8:
    if ctx:
        risk = ctx.context_used_pct
        if risk >= 0.9:
            st.metric("Risk Score", "🔴 Critical", delta=f"{pct}% context")
        elif risk >= 0.75:
            st.metric("Risk Score", "🟡 Elevated", delta=f"{pct}% context")
        else:
            st.metric("Risk Score", "🟢 Normal", delta=f"{pct}% context")
    else:
        st.metric("Risk Score", "N/A")

# Third row — Memory
mem_col1, mem_col2 = st.columns(2)
with mem_col1:
    if mem:
        if mem.current_memory_mb >= 1024:
            st.metric("Memory", f"{mem.current_memory_mb / 1024:.2f} GB")
        else:
            st.metric("Memory", f"{mem.current_memory_mb:.0f} MB")
    else:
        st.markdown(
            '<div class="info-card">'
            "💡 <strong>Memory tracking unavailable</strong><br>"
            "vLLM manages GPU memory server-side. "
            "Use <code>nvidia-smi</code> or the vLLM dashboard to monitor GPU memory."
            "</div>",
            unsafe_allow_html=True,
        )
with mem_col2:
    if mem:
        st.metric("Peak Memory", f"{mem.peak_memory_mb / 1024:.2f} GB" if mem.peak_memory_mb >= 1024 else f"{mem.peak_memory_mb:.0f} MB")

# Full generated text in expander for copy convenience
with st.expander("📋 Copy-friendly generated text", expanded=False):
    st.code(result.generated_text, language=None)


# ---------------------------------------------------------------------------
# 3. Graphs
# ---------------------------------------------------------------------------
st.header("📈 Graphs")

graph_col1, graph_col2 = st.columns(2)

# Latency vs Token
with graph_col1:
    st.subheader("Latency per Token")
    if lat and lat.per_step_snapshots:
        lat_df = pd.DataFrame([
            {"Step": s.step, "Latency (ms)": s.latency_ms}
            for s in lat.per_step_snapshots
            if s.latency_ms is not None
        ])
        if not lat_df.empty:
            st.line_chart(lat_df.set_index("Step"), color="#4f8ff7")
        else:
            st.info("No latency data.")
    else:
        st.info("No latency snapshots available.")

# Context usage vs Token
with graph_col2:
    st.subheader("Context Usage")
    if ctx and ctx.per_step_snapshots:
        ctx_df = pd.DataFrame([
            {"Step": s.step, "Context Used (%)": s.context_used_pct * 100}
            for s in ctx.per_step_snapshots
        ])
        if not ctx_df.empty:
            st.line_chart(ctx_df.set_index("Step"), color="#4fc76f")
        else:
            st.info("No context data.")
    else:
        st.info("No per-step context snapshots (vLLM mode provides summary only).")

# Memory growth (full width below)
st.subheader("Memory Growth")
if mem and mem.per_step_snapshots:
    mem_df = pd.DataFrame([
        {"Step": s.step, "RSS (MB)": s.rss_mb}
        for s in mem.per_step_snapshots
    ])
    if not mem_df.empty:
        st.line_chart(mem_df.set_index("Step"), color="#f7734f")
    else:
        st.info("No memory data.")
elif mem is None:
    st.markdown(
        '<div class="info-card">'
        "💡 Memory graph unavailable in vLLM mode — "
        "vLLM manages GPU memory server-side."
        "</div>",
        unsafe_allow_html=True,
    )
else:
    st.info("No per-step memory snapshots recorded.")


# ---------------------------------------------------------------------------
# 4. Forecast Panel
# ---------------------------------------------------------------------------
st.header("🔮 Forecast")

if ctx:
    from contextwatch.monitor.advisor import build_diagnosis
    from contextwatch.monitor.forecaster import compute_forecast

    latency_slope: float | None = None
    if lat and lat.trend_ms_per_100_tokens is not None:
        latency_slope = lat.trend_ms_per_100_tokens / 100.0

    memory_limit_mb = st.session_state.get("memory_limit_mb", memory_limit_gb * 1024.0)
    latency_limit = st.session_state.get("latency_limit_ms", latency_limit_ms)

    forecast = compute_forecast(
        total_tokens=result.total_token_count,
        max_context=ctx.max_context,
        current_memory_mb=mem.current_memory_mb if mem else None,
        avg_growth_per_token_mb=mem.avg_growth_per_token_mb if mem else None,
        memory_limit_mb=memory_limit_mb,
        current_latency_ms=lat.current_token_latency_ms if lat else None,
        latency_slope_per_token_ms=latency_slope,
        latency_threshold_ms=latency_limit,
    )

    fc1, fc2, fc3 = st.columns(3)

    with fc1:
        st.subheader("Context")
        if forecast.context_already_saturated:
            st.markdown(
                '<div class="warn-box">⚠ Context window already saturated</div>',
                unsafe_allow_html=True,
            )
        elif forecast.tokens_until_context_limit is not None:
            if forecast.tokens_until_context_limit < 200:
                st.markdown(
                    f'<div class="warn-box">⚠ Context saturation in ~{forecast.tokens_until_context_limit} tokens</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="ok-box">✓ Context saturation in ~{forecast.tokens_until_context_limit} tokens</div>',
                    unsafe_allow_html=True,
                )

    with fc2:
        st.subheader("Memory")
        if mem is None:
            st.markdown(
                '<div class="info-card">'
                "💡 <strong>Memory forecast unavailable</strong><br>"
                "vLLM manages GPU memory server-side. Memory forecasting "
                "requires local process memory tracking (HF mode)."
                "</div>",
                unsafe_allow_html=True,
            )
        elif forecast.memory_already_exceeded:
            st.markdown(
                '<div class="warn-box">⚠ Memory limit already exceeded</div>',
                unsafe_allow_html=True,
            )
        elif forecast.tokens_until_memory_limit is not None:
            if forecast.tokens_until_memory_limit < 200:
                st.markdown(
                    f'<div class="warn-box">⚠ Memory limit in ~{forecast.tokens_until_memory_limit} tokens</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="ok-box">✓ Memory limit in ~{forecast.tokens_until_memory_limit} tokens</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div class="ok-box">✓ Memory not growing — limit unlikely</div>',
                unsafe_allow_html=True,
            )

    with fc3:
        st.subheader("Latency")
        if forecast.latency_already_exceeded:
            st.markdown(
                f'<div class="warn-box">⚠ Latency already exceeds {latency_limit:.0f}ms</div>',
                unsafe_allow_html=True,
            )
        elif forecast.tokens_until_latency_threshold is not None:
            if forecast.tokens_until_latency_threshold < 200:
                st.markdown(
                    f'<div class="warn-box">⚠ Latency >{latency_limit:.0f}ms in ~{forecast.tokens_until_latency_threshold} tokens</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="ok-box">✓ Latency >{latency_limit:.0f}ms in ~{forecast.tokens_until_latency_threshold} tokens</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div class="ok-box">✓ No latency degradation trend</div>',
                unsafe_allow_html=True,
            )

    diagnosis = build_diagnosis(
        context_used_pct=ctx.context_used_pct,
        latency_trend_ms_per_100_tokens=(
            lat.trend_ms_per_100_tokens if lat else None
        ),
        forecast=forecast,
        mode=st.session_state.get("mode", mode),
    )

    st.header("🧭 Diagnosis")
    score_col, status_col = st.columns(2)
    with score_col:
        st.metric("Risk Score", f"{diagnosis.risk_score}/100")
    with status_col:
        st.metric("Status", diagnosis.status.upper())

    st.subheader("Findings")
    if diagnosis.findings:
        for finding in diagnosis.findings:
            if finding.severity in {"critical", "high"}:
                st.error(f"[{finding.severity.upper()}] {finding.area}: {finding.message}")
            elif finding.severity == "medium":
                st.warning(f"[MEDIUM] {finding.area}: {finding.message}")
            elif finding.severity == "info":
                st.info(f"[INFO] {finding.area}: {finding.message}")
            else:
                st.write(f"[{finding.severity.upper()}] {finding.area}: {finding.message}")
    else:
        st.success("No active risks detected.")

    st.subheader("Recommendations")
    for rec in diagnosis.recommendations:
        st.write(f"- {rec}")
else:
    st.warning("No context data available for forecasting.")
