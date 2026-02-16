# app.py — Invention Morphology Navigator + Patent Search + Patent Compare (Streamlit Cloud ready)

import os
import re
import ast
from collections import Counter
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from openai import OpenAI
from pathlib import Path

# -----------------------------
# PAGE CONFIG (ONLY ONCE)
# -----------------------------
st.set_page_config(page_title="Invention Morphology Navigator", layout="wide")

# -----------------------------
# PATHS (repo-bundled)
# -----------------------------
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
ASSETS_DIR = ROOT / "assets"

CSV_DEFAULT = DATA_DIR / "merged_functions_with_subsystems_semantic_4.csv"
XLSX_PATH = DATA_DIR / "First set.xlsx"
IMAGE_PATH = ASSETS_DIR / "Picture1.jpg"

# -----------------------------
# OPENAI KEY (Secrets first, then env var)
# -----------------------------
api_key = None
try:
    api_key = st.secrets.get("OPENAI_API_KEY", None)
except Exception:
    api_key = None
api_key = api_key or os.getenv("OPENAI_API_KEY")

# Only create client when needed (compare tab)
def get_openai_client() -> Optional[OpenAI]:
    if not api_key:
        return None
    return OpenAI(api_key=api_key)

# Models (adjust if desired)
MODEL_TEXT = "gpt-5.2"
MODEL_EMBED = "text-embedding-3-large"

# -----------------------------
# HELPERS (Morphology)
# -----------------------------
PIPE_SPLIT_RE = re.compile(r"\s*\|\s*")

def split_pipe(s: str) -> List[str]:
    if s is None:
        return []
    s = str(s).strip()
    if not s:
        return []
    parts = [p.strip() for p in PIPE_SPLIT_RE.split(s) if p.strip()]
    seen = set()
    out = []
    for p in parts:
        if not p:
            continue
        if str(p).lower() == "nan":
            continue
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out

def safe_int(x, default=0) -> int:
    try:
        if pd.isna(x):
            return default
        return int(x)
    except Exception:
        return default

def unique_count_from_pipe_series(series: pd.Series) -> int:
    uniq = set()
    for v in series.fillna("").astype(str):
        for t in split_pipe(v):
            uniq.add(t)
    return len(uniq)

def flatten_tokens_from_lists(series_of_lists: pd.Series) -> List[str]:
    out = []
    for lst in series_of_lists:
        if isinstance(lst, list):
            out.extend([t for t in lst if t and str(t).lower() != "nan"])
    return out

def apply_list_filter(series_of_lists: pd.Series, selected: List[str], mode: str) -> pd.Series:
    if not selected:
        return pd.Series([True] * len(series_of_lists), index=series_of_lists.index)
    selected = [s for s in selected if s and str(s).lower() != "nan"]
    if not selected:
        return pd.Series([True] * len(series_of_lists), index=series_of_lists.index)

    sets = series_of_lists.apply(lambda xs: set(xs) if isinstance(xs, list) else set())
    if mode.upper() == "AND":
        return sets.apply(lambda s: all(t in s for t in selected))
    return sets.apply(lambda s: any(t in s for t in selected))

def treemap_or_bar_subsystems(df: pd.DataFrame, weight_mode: str, chart_type: str):
    rows = []
    for _, r in df.iterrows():
        subs = r["__subsystems_list"] if isinstance(r["__subsystems_list"], list) else []
        w = 1 if weight_mode == "Unique Functions" else r["num_paragraphs"]
        for s in subs:
            if not s or str(s).lower() == "nan":
                continue
            rows.append((s, w))
    if not rows:
        return None

    agg = pd.DataFrame(rows, columns=["Subsystem", "Weight"]).groupby("Subsystem", as_index=False)["Weight"].sum()
    agg = agg.sort_values("Weight", ascending=False)

    if chart_type == "Treemap":
        fig = px.treemap(agg, path=["Subsystem"], values="Weight")
        fig.update_layout(height=450)
        return fig

    fig = go.Figure(data=[go.Bar(x=agg["Subsystem"], y=agg["Weight"])])
    fig.update_layout(
        height=450,
        xaxis_title="Subsystem",
        yaxis_title=("Count" if weight_mode == "Unique Functions" else "Sum(num_paragraphs)")
    )
    return fig

def top_terms_bar(counter: Counter, title: str, top_n: int = 20):
    items = [(k, v) for k, v in counter.most_common(top_n) if k and str(k).lower() != "nan"]
    if not items:
        return None
    x = [k for k, _ in items]
    y = [v for _, v in items]
    fig = go.Figure(data=[go.Bar(x=x, y=y)])
    fig.update_layout(height=450, title=title, xaxis_title="", yaxis_title="Count")
    return fig

# ---- Restored Sankey (simple labels, no prefixes/shortening) ----
def make_sankey_from_rows(
    df_rows: pd.DataFrame,
    weight_mode: str,
    top_mech: int,
    top_struct: int,
) -> Optional[go.Figure]:
    if df_rows.empty:
        return None

    mech_counts = Counter(flatten_tokens_from_lists(df_rows["__mech_list"]))
    struct_counts = Counter(flatten_tokens_from_lists(df_rows["__struct_list"]))

    mech_scope = [t for t, _ in mech_counts.most_common(top_mech)]
    struct_scope = [t for t, _ in struct_counts.most_common(top_struct)]

    rows_out = []
    for _, r in df_rows.iterrows():
        subs = r["__subsystems_list"] if isinstance(r["__subsystems_list"], list) else []
        mechs = [m for m in (r["__mech_list"] if isinstance(r["__mech_list"], list) else []) if m in mech_scope]
        structs = [s for s in (r["__struct_list"] if isinstance(r["__struct_list"], list) else []) if s in struct_scope]
        if not subs or not mechs or not structs:
            continue

        w = 1.0 if weight_mode == "Unique Functions" else float(r["num_paragraphs"])
        for ss in subs:
            for m in mechs:
                for s in structs:
                    rows_out.append((ss, m, s, w))

    if not rows_out:
        return None

    links = pd.DataFrame(rows_out, columns=["subsystem", "mechanism", "structure", "weight"])

    left_nodes = links["subsystem"].dropna().astype(str).unique().tolist()
    mid_nodes = links["mechanism"].dropna().astype(str).unique().tolist()
    right_nodes = links["structure"].dropna().astype(str).unique().tolist()

    nodes = left_nodes + mid_nodes + right_nodes
    idx = {n: i for i, n in enumerate(nodes)}

    lm = links.groupby(["subsystem", "mechanism"], dropna=False)["weight"].sum().reset_index()
    mr = links.groupby(["mechanism", "structure"], dropna=False)["weight"].sum().reset_index()

    source, target, value = [], [], []

    for _, rr in lm.iterrows():
        a, b, w = str(rr["subsystem"]), str(rr["mechanism"]), float(rr["weight"])
        if a.lower() == "nan" or b.lower() == "nan" or w <= 0:
            continue
        source.append(idx[a]); target.append(idx[b]); value.append(w)

    for _, rr in mr.iterrows():
        a, b, w = str(rr["mechanism"]), str(rr["structure"]), float(rr["weight"])
        if a.lower() == "nan" or b.lower() == "nan" or w <= 0:
            continue
        source.append(idx[a]); target.append(idx[b]); value.append(w)

    fig = go.Figure(
        data=[go.Sankey(
            arrangement="snap",
            node=dict(pad=14, thickness=16, line=dict(width=0.5), label=nodes),
            link=dict(source=source, target=target, value=value),
        )]
    )
    fig.update_layout(height=680, title=f"Sankey weighted by {weight_mode}")
    return fig

@st.cache_data(show_spinner=True)
def load_and_prepare(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    required = ["Subsystems", "CanonicalFunction", "MergedFunctions", "Mechanism", "Structure", "paragraph_ids"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["row_id"] = np.arange(len(df))

    df["num_paragraphs"] = df["num_paragraphs"].apply(safe_int) if "num_paragraphs" in df.columns else 1
    df["num_phrases"] = df["num_phrases"].apply(safe_int) if "num_phrases" in df.columns else 1

    df["__subsystems_list"] = df["Subsystems"].apply(split_pipe)
    df["__mech_list"] = df["Mechanism"].apply(split_pipe)
    df["__struct_list"] = df["Structure"].apply(split_pipe)
    df["__patents_list"] = df["paragraph_ids"].apply(split_pipe)

    # requested display fields
    df["num_patents"] = df["num_paragraphs"]
    df["num_function_phrases"] = df["num_phrases"]
    return df

# -----------------------------
# HELPERS (Excel search + compare)
# -----------------------------
@st.cache_data(show_spinner=True)
def load_excel():
    s1 = pd.read_excel(XLSX_PATH, sheet_name="Sheet1", dtype=object)
    s2 = pd.read_excel(XLSX_PATH, sheet_name="Sheet2", dtype=object)
    return s1, s2

def parse_list_cell(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return []
    try:
        return list(map(int, ast.literal_eval(str(v))))
    except Exception:
        return []

def clean_text(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return re.sub(r"\s+", " ", str(x)).strip()

def get_patent_record(df_pat: pd.DataFrame, pub: str):
    pub = clean_text(pub)
    hit = df_pat[df_pat["Publication number"].astype(str).str.strip() == pub]
    if hit.empty:
        hit = df_pat[df_pat["Publication number"].astype(str).str.contains(re.escape(pub), na=False)]
    return hit.iloc[0] if not hit.empty else None

def embed_one(client: OpenAI, text: str) -> np.ndarray:
    r = client.embeddings.create(model=MODEL_EMBED, input=[text])
    return np.array(r.data[0].embedding, dtype=np.float32)

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def gpt_compare_abstracts(client: OpenAI, a: dict, b: dict) -> str:
    prompt = f"""
Compare these two patent abstracts for technical overlap and synthesis.

PATENT A
Publication number: {a.get('Publication number','')}
Title: {a.get('Title','')}
CPC classification: {a.get('CPC classification','')}
Abstract: {a.get('Abstract','')}

PATENT B
Publication number: {b.get('Publication number','')}
Title: {b.get('Title','')}
CPC classification: {b.get('CPC classification','')}
Abstract: {b.get('Abstract','')}

Return:
1) Similarities (bullets)
2) Differences (bullets)
3) Synthesis: a combined approach that merges complementary strengths + implementation outline (high level; not legal advice)
Ground everything in the abstracts (no outside assumptions).
"""
    r = client.responses.create(model=MODEL_TEXT, input=prompt)
    return r.output_text

# =========================================================
# APP LAYOUT
# =========================================================
st.title("Invention Morphology Navigator")
st.caption("Subsystem overview → drill-down → Sankey view → patent lookup → abstract comparison (GPT).")

tabs = st.tabs(["Morphology Navigator", "Search Patent (Excel)", "Patent Compare (Abstract)"])

# =========================================================
# TAB 1: Morphology Navigator
# =========================================================
with tabs[0]:
    if IMAGE_PATH.exists():
        st.image(Image.open(IMAGE_PATH), use_container_width=True)

    with st.sidebar:
        st.header("Data")
        csv_path = st.text_input("CSV path (repo-relative recommended)", value=str(CSV_DEFAULT))

        st.divider()
        st.header("Overview controls")
        weight_mode = st.radio(
            "Count / size by",
            ["Unique Functions", "Function coverage in patents"],
            index=0,
            help="Affects subsystem chart sizing and Sankey thickness."
        )
        chart_type = st.radio("Subsystem chart type", ["Treemap", "Bar"], index=0)

        st.divider()
        st.header("Drill-down controls — instructions")
        st.markdown(
            "- **Subsystem selector** filters the entire drill-down view.\n"
            "- **Search** finds keywords across Canonical/Merged/Mechanism/Structure/Patents.\n"
            "- **Facet filters** narrow to rows containing selected Mechanisms/Structures.\n"
            "- Use **AND** to require all selected terms, **OR** to match any.\n"
            "- Sort by **num_patents (num_paragraphs)** to see most-supported functions first."
        )
        top_n_facets = st.slider("Facet Top N", 10, 150, 30, 5)
        facet_mode = st.radio("Facet mode", ["AND", "OR"], horizontal=True)

        st.divider()
        st.header("Sankey controls — instructions")
        st.markdown(
            "- Sankey uses the **current filtered dataset**.\n"
            "- Increase top-N to show more terms, but too high can be cluttered.\n"
            "- Use **Function coverage in patents** if thickness should reflect support."
        )
        sankey_top_mech = st.slider("Top mechanisms in Sankey", 5, 80, 25, 5)
        sankey_top_struct = st.slider("Top structures in Sankey", 5, 80, 25, 5)

    # Load CSV
    try:
        df = load_and_prepare(csv_path)
    except Exception as e:
        st.error(f"Failed to load CSV: {e}")
        st.stop()

    # 1) Subsystem Overview
    st.subheader("1) Subsystem Overview")

    unique_functions = len(df)
    unique_merged_functions = unique_count_from_pipe_series(df["MergedFunctions"])
    unique_mechanisms = unique_count_from_pipe_series(df["Mechanism"])
    unique_structures = unique_count_from_pipe_series(df["Structure"])
    unique_patents = unique_count_from_pipe_series(df["paragraph_ids"])

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Functions", f"{unique_functions:,}")
    k2.metric("Function Types", f"{unique_merged_functions:,}")
    k3.metric("Mechanisms", f"{unique_mechanisms:,}")
    k4.metric("Structures", f"{unique_structures:,}")
    k5.metric("Patents", f"{unique_patents:,}")

    c1, c2 = st.columns([1.3, 1])
    with c1:
        st.markdown("**Subsystem distribution** (size by selected mode)")
        fig_sub = treemap_or_bar_subsystems(df, weight_mode, chart_type)
        if fig_sub is None:
            st.info("No subsystem data available.")
        else:
            st.plotly_chart(fig_sub, use_container_width=True)

    with c2:
        st.markdown("**Global Top Mechanisms / Structures**")
        mech_counter = Counter(flatten_tokens_from_lists(df["__mech_list"]))
        struct_counter = Counter(flatten_tokens_from_lists(df["__struct_list"]))
        t = st.tabs(["Top Mechanisms", "Top Structures"])
        with t[0]:
            figm = top_terms_bar(mech_counter, "Top Mechanisms (global)", top_n=20)
            st.plotly_chart(figm, use_container_width=True) if figm else st.info("No mechanisms found.")
        with t[1]:
            figs = top_terms_bar(struct_counter, "Top Structures (global)", top_n=20)
            st.plotly_chart(figs, use_container_width=True) if figs else st.info("No structures found.")

    st.divider()

    # 2) Drill-down
    st.subheader("2) Subsystem Drill-down")

    all_subsystems = sorted({s for lst in df["__subsystems_list"] for s in (lst or [])})
    selected_subsystem = st.selectbox(
        "Select a subsystem to drill down (filters all views below)",
        ["(All)"] + all_subsystems,
        index=0
    )

    base = df.copy()
    if selected_subsystem != "(All)":
        base = base[base["__subsystems_list"].apply(lambda xs: selected_subsystem in xs if isinstance(xs, list) else False)]

    search_q = st.text_input("Search (e.g. hover, direction change etc.)", value="").strip()
    if search_q:
        ql = search_q.lower()
        def match_row(r):
            fields = [
                str(r.get("CanonicalFunction", "")),
                str(r.get("MergedFunctions", "")),
                str(r.get("Mechanism", "")),
                str(r.get("Structure", "")),
                str(r.get("Subsystems", "")),
                str(r.get("paragraph_ids", "")),
            ]
            return any(ql in f.lower() for f in fields)
        base = base[base.apply(match_row, axis=1)]

    mech_counts_local = Counter(flatten_tokens_from_lists(base["__mech_list"]))
    struct_counts_local = Counter(flatten_tokens_from_lists(base["__struct_list"]))
    top_mech_terms = [t for t, _ in mech_counts_local.most_common(top_n_facets)]
    top_struct_terms = [t for t, _ in struct_counts_local.most_common(top_n_facets)]

    fcol1, fcol2, fcol3 = st.columns([1, 1, 1])
    with fcol1:
        selected_mechs = st.multiselect("Mechanisms filter", options=top_mech_terms, default=[])
    with fcol2:
        selected_structs = st.multiselect("Structures filter", options=top_struct_terms, default=[])
    with fcol3:
        sort_by = st.selectbox("Sort by", ["num_patents", "num_function_phrases", "CanonicalFunction"], index=0)
        sort_dir = st.selectbox("Direction", ["desc", "asc"], index=0)

    mask_m = apply_list_filter(base["__mech_list"], selected_mechs, facet_mode)
    mask_s = apply_list_filter(base["__struct_list"], selected_structs, facet_mode)
    base = base[mask_m & mask_s]

    st.markdown("**Drill-down KPIs (current filter)**")
    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric("Functions", f"{len(base):,}")
    d2.metric("Mapped functions", f"{unique_count_from_pipe_series(base['MergedFunctions']):,}")
    d3.metric("Mechanisms", f"{unique_count_from_pipe_series(base['Mechanism']):,}")
    d4.metric("Structures", f"{unique_count_from_pipe_series(base['Structure']):,}")
    d5.metric("Patents", f"{unique_count_from_pipe_series(base['paragraph_ids']):,}")

    st.markdown("### A. Function list (table)")
    ascending = (sort_dir == "asc")
    table = base.sort_values(by=sort_by, ascending=ascending, kind="mergesort").copy()

    display_cols = [
        "Subsystems", "CanonicalFunction", "MergedFunctions",
        "Mechanism", "Structure",
        "num_patents", "num_function_phrases",
        "paragraph_ids",
    ]
    display_cols = [c for c in display_cols if c in table.columns]

    max_rows = st.slider("Rows to display", 100, 5000, 1000, 100)
    st.dataframe(table[display_cols].head(max_rows), use_container_width=True, hide_index=True)

    st.markdown("### B. Mechanism + Structure facets")
    fc1, fc2 = st.columns(2)
    with fc1:
        mech_df = pd.DataFrame(mech_counts_local.most_common(top_n_facets), columns=["Mechanism", "Count"])
        mech_df = mech_df[mech_df["Mechanism"].astype(str).str.lower() != "nan"]
        fig = go.Figure(data=[go.Bar(x=mech_df["Mechanism"], y=mech_df["Count"])])
        fig.update_layout(height=420, title="Top Mechanisms (current scope)", xaxis_title="", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    with fc2:
        struct_df = pd.DataFrame(struct_counts_local.most_common(top_n_facets), columns=["Structure", "Count"])
        struct_df = struct_df[struct_df["Structure"].astype(str).str.lower() != "nan"]
        fig = go.Figure(data=[go.Bar(x=struct_df["Structure"], y=struct_df["Count"])])
        fig.update_layout(height=420, title="Top Structures (current scope)", xaxis_title="", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### C. Relationship view")
    fig_sankey = make_sankey_from_rows(
        df_rows=base,
        weight_mode=weight_mode,
        top_mech=sankey_top_mech,
        top_struct=sankey_top_struct,
    )
    if fig_sankey is None:
        st.info("Not enough data to build Sankey under current filters. Try widening filters or increasing top-N.")
    else:
        st.plotly_chart(fig_sankey, use_container_width=True)

    st.divider()
    st.subheader("Export current filtered dataset")
    base_export = base.drop(columns=[c for c in base.columns if c.startswith("__")], errors="ignore")
    out_csv = base_export.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered rows CSV", data=out_csv, file_name="filtered_view.csv", mime="text/csv")


# =========================================================
# TAB 2: Search Patent (Excel)
# =========================================================
with tabs[1]:
    st.header("Search Patent (Sheet2 → Sheet1 mapping)")

    if not XLSX_PATH.exists():
        st.error(f"Excel file not found at {XLSX_PATH}. Put it in data/ and redeploy.")
        st.stop()

    sheet1, sheet2 = load_excel()
    sheet1 = sheet1.copy()
    sheet1["_original_row"] = sheet1.index + 2  # Excel-style row numbering

    row_num = st.number_input(
        "Enter _dedup_csv_row (Sheet2). In some cases it could be above number + 1.",
        min_value=2,
        step=1,
        value=2
    )

    match = sheet2[sheet2["_dedup_csv_row"] == row_num]
    if match.empty:
        st.warning("Row not found in Sheet2")
        st.stop()

    orig_list = parse_list_cell(match.iloc[0]["original_csv_rows"])
    if not orig_list:
        st.warning("No original rows listed")
        st.stop()

    first_orig = orig_list[0]
    row = sheet1[sheet1["_original_row"] == first_orig]
    if row.empty:
        st.error("Original row not found in Sheet1")
        st.stop()

    row = row.iloc[0]

    st.subheader("Publication Details")
    st.write("**Publication number:**", row.get("Publication number", ""))
    st.write("**Title:**", row.get("Title", ""))
    with st.expander("Abstract", expanded=True):
        st.write(row.get("Abstract", ""))
    st.write("**CPC classification:**", row.get("CPC classification", ""))

    with st.expander("Debug: row mapping"):
        st.write("Sheet2 original_csv_rows:", orig_list)
        st.write("Chosen first original row:", first_orig)


# =========================================================
# TAB 3: Patent Compare (Abstract-only)
# =========================================================
with tabs[2]:
    st.header("Patent Compare (Abstract-only)")

    if not XLSX_PATH.exists():
        st.error(f"Excel file not found at {XLSX_PATH}. Put it in data/ and redeploy.")
        st.stop()

    if not api_key:
        st.error("OpenAI key not found. Add OPENAI_API_KEY in Streamlit Cloud Secrets.")
        st.stop()

    client = get_openai_client()
    if client is None:
        st.error("OpenAI client could not be initialized (missing key).")
        st.stop()

    df_pat = pd.read_excel(XLSX_PATH, sheet_name="Sheet1", dtype=object)

    c1, c2 = st.columns(2)
    with c1:
        pub1 = st.text_input("Patent / Publication number A", "")
    with c2:
        pub2 = st.text_input("Patent / Publication number B", "")

    run = st.button("Compare abstracts")

    if run:
        rec1 = get_patent_record(df_pat, pub1)
        rec2 = get_patent_record(df_pat, pub2)

        if rec1 is None:
            st.error(f"Could not find Patent A ({pub1}) in Sheet1.")
            st.stop()
        if rec2 is None:
            st.error(f"Could not find Patent B ({pub2}) in Sheet1.")
            st.stop()

        meta1 = {
            "Publication number": clean_text(rec1.get("Publication number", "")),
            "Title": clean_text(rec1.get("Title", "")),
            "Abstract": clean_text(rec1.get("Abstract", "")),
            "CPC classification": clean_text(rec1.get("CPC classification", "")),
        }
        meta2 = {
            "Publication number": clean_text(rec2.get("Publication number", "")),
            "Title": clean_text(rec2.get("Title", "")),
            "Abstract": clean_text(rec2.get("Abstract", "")),
            "CPC classification": clean_text(rec2.get("CPC classification", "")),
        }

        if not meta1["Abstract"] or not meta2["Abstract"]:
            st.warning("One of the abstracts is empty; similarity + comparison may be weak.")

        with st.spinner("Computing abstract similarity (embeddings)..."):
            e1 = embed_one(client, meta1["Abstract"])
            e2 = embed_one(client, meta2["Abstract"])
            sim = cosine(e1, e2)

        st.subheader("Abstract similarity")
        st.metric("Cosine similarity", f"{sim:.4f}")

        with st.spinner("Asking GPT for similarities/differences/synthesis..."):
            analysis = gpt_compare_abstracts(client, meta1, meta2)

        st.subheader("GPT analysis")
        st.write(analysis)

        with st.expander("Patent A"):
            st.write("**Publication number:**", meta1["Publication number"])
            st.write("**Title:**", meta1["Title"])
            st.write("**CPC classification:**", meta1["CPC classification"])
            st.write("**Abstract:**")
            st.write(meta1["Abstract"])

        with st.expander("Patent B"):
            st.write("**Publication number:**", meta2["Publication number"])
            st.write("**Title:**", meta2["Title"])
            st.write("**CPC classification:**", meta2["CPC classification"])
            st.write("**Abstract:**")
            st.write(meta2["Abstract"])
