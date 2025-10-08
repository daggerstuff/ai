import streamlit as st
import json
import argparse
import pandas as pd
import altair as alt
import plotly.express as px
from wordcloud import WordCloud
from io import BytesIO
import base64

# --- Argument Parsing ---
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, required=True)
parser.add_argument("--log", type=str, required=False)
args = parser.parse_args()

# --- Page Config ---
st.set_page_config(
    page_title="Synthetic Dialogue Dashboard",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Load Data ---
with open(args.data, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]
df = pd.DataFrame(data)

# --- Load Logs ---
logs = None
if args.log:
    with open(args.log, "r", encoding="utf-8") as f:
        logs = f.read()

# --- Dark Mode / Kali Theme CSS ---
st.markdown(
    """
    <style>
    body, .stApp {
        background-color: #181825 !important;
        color: #e0e6f0 !important;
    }
    .glass-card {
        background: rgba(24,24,37,0.85);
        box-shadow: 0 8px 32px 0 rgba(0,255,255,0.10);
        backdrop-filter: blur(8px);
        border-radius: 18px;
        border: 1.5px solid #1e293b;
        padding: 2rem 1.5rem 1.5rem 1.5rem;
        margin-bottom: 2rem;
    }
    .metric-anim {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg,#84cc16,#06b6d4,#a21caf,#00fff7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shine 2s linear infinite;
    }
    @keyframes shine {
        0% { filter: brightness(1.1); }
        50% { filter: brightness(1.5); }
        100% { filter: brightness(1.1); }
    }
    .stTabs [data-baseweb="tab-list"] {
        background: #232336;
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #a6e3fa;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: #1e1e2e;
        color: #84cc16;
    }
    .stDataFrame, .stTable {
        background: #232336 !important;
        color: #e0e6f0 !important;
    }
    .stMetric {
        background: #232336 !important;
        border-radius: 10px;
        color: #84cc16 !important;
    }
    .st-cg, .st-cj, .st-bb, .st-bc {
        background: #232336 !important;
        color: #e0e6f0 !important;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# --- Sidebar: Summary Stats ---
st.sidebar.title("💡 Dashboard Summary")
st.sidebar.metric("Total Q&A Pairs", len(df))
if "prompt" in df.columns:
    avg_prompt_len = int(df["prompt"].str.len().mean())
    st.sidebar.metric("Avg. Prompt Length", avg_prompt_len)
if "response" in df.columns:
    avg_response_len = int(df["response"].str.len().mean())
    st.sidebar.metric("Avg. Response Length", avg_response_len)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style='font-size:0.95em; color:#7df9ff;'>
    <b>Kali Dark Mode Dashboard</b><br>
    Powered by Streamlit, Altair, Plotly, and WordCloud.
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Tabs ---
tabs = st.tabs(["Overview", "Data Explorer", "Visualizations", "Logs"])

# --- Overview Tab ---
with tabs[0]:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='metric-anim'>Total Q&A Pairs: {len(df):,}</div>", unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("Sample Q&A Pairs")
    sample_df = df.sample(n=min(3, len(df)), random_state=42) if len(df) > 3 else df
    for i, row in sample_df.iterrows():
        st.markdown(
            f"""
        <div class='glass-card'>
        <b>Prompt:</b> <span style='color:#06b6d4'>{row['prompt']}</span><br>
        <b>Response:</b> <span style='color:#84cc16'>{row['response']}</span>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### Key Stats")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Avg. Prompt Length", avg_prompt_len)
    with col2:
        st.metric("Avg. Response Length", avg_response_len)
    with col3:
        st.metric("Unique Prompts", df["prompt"].nunique())

# --- Data Explorer Tab ---
with tabs[1]:
    st.subheader("Data Explorer")
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.markdown(
        "<small style='color:#7df9ff;'>Tip: Use the search box to filter Q&A pairs.</small>",
        unsafe_allow_html=True,
    )

# --- Visualizations Tab ---
with tabs[2]:
    st.subheader("Visualizations")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Q&A Length Distribution")
        if "prompt" in df.columns and "response" in df.columns:
            length_df = pd.DataFrame(
                {
                    "Prompt Length": df["prompt"].str.len(),
                    "Response Length": df["response"].str.len(),
                }
            )
            chart = (
                alt.Chart(length_df.reset_index())
                .transform_fold(["Prompt Length", "Response Length"], as_=["Type", "Length"])
                .mark_area(opacity=0.5, interpolate="step")
                .encode(
                    x=alt.X(
                        "Length:Q",
                        bin=alt.Bin(maxbins=40),
                        axis=alt.Axis(labelColor="#7df9ff", titleColor="#7df9ff"),
                    ),
                    y=alt.Y(
                        "count()",
                        stack=None,
                        axis=alt.Axis(labelColor="#7df9ff", titleColor="#7df9ff"),
                    ),
                    color=alt.Color("Type:N", scale=alt.Scale(range=["#06b6d4", "#84cc16"])),
                )
                .properties(background="#181825", width=400, height=250)
            )
            st.altair_chart(chart, use_container_width=True)
    with col2:
        st.markdown("#### Word Cloud (Prompts)")
        if "prompt" in df.columns:
            text = " ".join(df["prompt"].dropna().astype(str))
            wc = WordCloud(
                width=400, height=200, background_color="#181825", mode="RGBA", colormap="winter"
            ).generate(text)
            buf = BytesIO()
            wc.to_image().save(buf, format="PNG")
            img_b64 = base64.b64encode(buf.getvalue()).decode()
            st.markdown(
                f"<img src='data:image/png;base64,{img_b64}' style='width:100%; border-radius:12px; box-shadow:0 2px 12px #00fff733;' />",
                unsafe_allow_html=True,
            )
    st.markdown("---")
    st.markdown("#### Prompt vs. Response Length Scatter Plot")
    if "prompt" in df.columns and "response" in df.columns:
        scatter = px.scatter(
            df,
            x=df["prompt"].str.len(),
            y=df["response"].str.len(),
            labels={"x": "Prompt Length", "y": "Response Length"},
            title="Prompt vs. Response Length",
            color_discrete_sequence=["#00fff7"],
        )
        scatter.update_layout(
            paper_bgcolor="#181825",
            plot_bgcolor="#232336",
            font_color="#7df9ff",
            title_font_color="#84cc16",
            xaxis=dict(color="#7df9ff", gridcolor="#232336"),
            yaxis=dict(color="#7df9ff", gridcolor="#232336"),
        )
        st.plotly_chart(scatter, use_container_width=True)

# --- Logs Tab ---
with tabs[3]:
    st.subheader("Logs")
    if logs:
        st.code(logs, language="log")
    else:
        st.info("No log file provided.")

# --- Footer ---
st.markdown(
    """
    <hr style='margin-top:2rem; margin-bottom:1rem; border:0; border-top:1px solid #232336;'>
    <div style='text-align:center; color:#7df9ff; font-size:0.95em;'>
        <b>Synthetic Dialogue Dashboard</b> &copy; 2025 &mdash; Kali Dark Mode, Streamlit, Altair, Plotly, WordCloud
    </div>
    """,
    unsafe_allow_html=True,
)
