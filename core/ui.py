# core/ui.py
# Design system: CSS injection + reusable layout helpers.
# Aesthetic: "precision engineering" — dark graphite surfaces, IBM Plex Sans
# body, JetBrains Mono for numeric/label detail, single amber accent.

from __future__ import annotations

import html
from textwrap import dedent
from typing import Iterable

import streamlit as st


_CSS = dedent(
    """
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --bg: #0B0C0E;
        --surface: #141519;
        --surface-2: #1C1D22;
        --surface-3: #24252B;
        --border: #26272D;
        --border-hover: #3A3B42;
        --text: #F2F2F3;
        --text-2: #A0A0A8;
        --text-3: #6A6A72;
        --accent: #F5A524;
        --accent-dim: #C88818;
        --accent-soft: rgba(245, 165, 36, 0.10);
        --accent-line: rgba(245, 165, 36, 0.35);
        --success: #22C55E;
        --warning: #EAB308;
        --danger: #EF4444;
        --info: #60A5FA;
    }

    /* ---------- Base typography ---------- */
    html, body, [class*="css"], .stApp, .stMarkdown, button, input, textarea, select {
        font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, system-ui, sans-serif !important;
        font-feature-settings: 'cv02', 'cv03', 'cv04', 'cv11', 'ss01';
    }

    code, pre, kbd, .mono {
        font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, Consolas, monospace !important;
    }

    .stApp {
        background:
            radial-gradient(1200px 600px at 80% -10%, rgba(245, 165, 36, 0.035), transparent 60%),
            radial-gradient(900px 500px at -10% 110%, rgba(96, 165, 250, 0.025), transparent 60%),
            var(--bg);
    }

    /* Hide chrome */
    #MainMenu, footer {visibility: hidden; height: 0;}
    header[data-testid="stHeader"] {
        background: transparent;
        height: 0;
    }
    [data-testid="stToolbar"] {
        right: 1rem;
        top: 0.5rem;
    }

    /* Content container */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 5rem !important;
        padding-left: 2.5rem !important;
        padding-right: 2.5rem !important;
        max-width: 1440px;
    }

    /* ---------- Headings ---------- */
    h1, .stMarkdown h1 {
        font-weight: 600 !important;
        font-size: 1.75rem !important;
        letter-spacing: -0.02em !important;
        color: var(--text) !important;
        margin: 0 !important;
        line-height: 1.2 !important;
    }
    h2, .stMarkdown h2 {
        font-weight: 500 !important;
        font-size: 1.125rem !important;
        letter-spacing: -0.01em !important;
        color: var(--text) !important;
        margin-top: 2rem !important;
    }
    h3, .stMarkdown h3 {
        font-weight: 500 !important;
        font-size: 0.9375rem !important;
        color: var(--text) !important;
        margin-top: 1rem !important;
    }

    /* Label restyle */
    label, .stCheckbox label, .stRadio label,
    [data-baseweb="form-control-label"] {
        color: var(--text-2) !important;
        font-family: 'IBM Plex Sans' !important;
        font-size: 0.8125rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.005em !important;
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: #08090B;
        border-right: 1px solid var(--border);
        width: 260px !important;
        min-width: 260px !important;
    }
    section[data-testid="stSidebar"] > div {
        padding-top: 0;
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        padding-top: 0 !important;
    }
    section[data-testid="stSidebar"] .block-container {
        padding: 0 !important;
    }

    /* Sidebar brand */
    .brand {
        padding: 1.5rem 1.25rem 1.25rem;
        border-bottom: 1px solid var(--border);
    }
    .brand-row {
        display: flex;
        align-items: center;
        gap: 0.625rem;
        margin-bottom: 0.25rem;
    }
    .brand-mark {
        width: 22px;
        height: 22px;
        background: linear-gradient(135deg, var(--accent) 0%, #D98A0A 100%);
        border-radius: 3px;
        position: relative;
        box-shadow: 0 0 14px rgba(245, 165, 36, 0.4);
    }
    .brand-mark::after {
        content: '';
        position: absolute;
        inset: 4px;
        background: var(--bg);
        border-radius: 1px;
        box-shadow: inset 0 0 0 1px var(--accent);
    }
    .brand-name {
        font-weight: 600;
        font-size: 0.9375rem;
        letter-spacing: -0.01em;
        color: var(--text);
    }
    .brand-sub {
        font-family: 'JetBrains Mono' !important;
        font-size: 0.6875rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--text-3);
        padding-left: 2rem;
    }

    /* Nav section label */
    .nav-group {
        padding: 1.25rem 1.25rem 0.375rem;
        font-family: 'JetBrains Mono' !important;
        font-size: 0.625rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: var(--text-3);
    }

    /* Sidebar nav buttons override */
    section[data-testid="stSidebar"] .stButton {
        padding: 0 0.625rem;
    }
    section[data-testid="stSidebar"] .stButton > button {
        width: 100%;
        background: transparent !important;
        color: var(--text-2) !important;
        border: 1px solid transparent !important;
        border-radius: 6px;
        padding: 0.5rem 0.75rem;
        text-align: left;
        justify-content: flex-start;
        font-weight: 400 !important;
        font-size: 0.875rem !important;
        transition: all 0.12s ease;
        line-height: 1.3;
        box-shadow: none !important;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: var(--surface) !important;
        color: var(--text) !important;
        border-color: transparent !important;
    }
    section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: var(--accent-soft) !important;
        color: var(--accent) !important;
        border-color: rgba(245, 165, 36, 0.15) !important;
        font-weight: 500 !important;
        box-shadow: inset 2px 0 0 var(--accent) !important;
    }
    section[data-testid="stSidebar"] .stButton > button p {
        margin: 0 !important;
        font-family: 'IBM Plex Sans' !important;
    }

    /* Sidebar status */
    .sb-status {
        margin: 1.25rem 0.75rem 0;
        padding: 0.875rem 1rem;
        border: 1px solid var(--border);
        border-radius: 8px;
        background: var(--surface);
    }
    .sb-status-label {
        font-family: 'JetBrains Mono' !important;
        font-size: 0.625rem;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: var(--text-3);
        margin-bottom: 0.375rem;
    }
    .sb-status-row {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.8125rem;
        color: var(--text);
    }
    .sb-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
    }
    .sb-status-on .sb-dot {
        background: var(--success);
        box-shadow: 0 0 8px rgba(34, 197, 94, 0.7);
    }
    .sb-status-off .sb-dot {
        background: var(--text-3);
    }

    /* ---------- Page header ---------- */
    .page-head {
        padding-bottom: 1.5rem;
        margin-bottom: 2rem;
        border-bottom: 1px solid var(--border);
        display: flex;
        align-items: flex-end;
        justify-content: space-between;
        gap: 2rem;
    }
    .page-head-left {
        min-width: 0;
    }
    .page-eyebrow {
        font-family: 'JetBrains Mono' !important;
        font-size: 0.6875rem;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: var(--accent);
        margin-bottom: 0.5rem;
    }
    .page-title {
        font-weight: 600;
        font-size: 1.75rem;
        letter-spacing: -0.025em;
        color: var(--text);
        line-height: 1.15;
    }
    .page-desc {
        color: var(--text-2);
        font-size: 0.9375rem;
        margin-top: 0.5rem;
        max-width: 720px;
        line-height: 1.5;
    }
    .page-head-right {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        align-items: center;
    }

    /* ---------- Section header ---------- */
    .section {
        display: flex;
        align-items: baseline;
        gap: 1rem;
        margin: 2.5rem 0 1.25rem;
    }
    .section-num {
        font-family: 'JetBrains Mono' !important;
        font-size: 0.6875rem;
        letter-spacing: 0.12em;
        color: var(--accent);
        font-weight: 500;
    }
    .section-title {
        font-weight: 500;
        font-size: 1rem;
        color: var(--text);
        letter-spacing: -0.005em;
    }
    .section-rule {
        flex: 1;
        height: 1px;
        background: linear-gradient(90deg, var(--border), transparent);
    }
    .section-hint {
        font-size: 0.8125rem;
        color: var(--text-3);
    }

    /* ---------- KPI card ---------- */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 0.75rem;
    }
    .kpi {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 1.125rem 1.25rem 1.25rem;
        position: relative;
        overflow: hidden;
        transition: border-color 0.15s ease, transform 0.15s ease;
    }
    .kpi:hover {
        border-color: var(--border-hover);
    }
    .kpi::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent 10%, var(--accent-line) 50%, transparent 90%);
    }
    .kpi-label {
        font-family: 'JetBrains Mono' !important;
        font-size: 0.6875rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--text-3);
        margin-bottom: 0.625rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .kpi-value {
        font-weight: 600;
        font-size: 1.875rem;
        letter-spacing: -0.025em;
        color: var(--text);
        font-variant-numeric: tabular-nums;
        line-height: 1;
    }
    .kpi-unit {
        font-family: 'JetBrains Mono' !important;
        font-size: 0.875rem;
        font-weight: 400;
        color: var(--text-3);
        margin-left: 0.25rem;
    }
    .kpi-hint {
        font-size: 0.75rem;
        color: var(--text-3);
        margin-top: 0.5rem;
        line-height: 1.4;
    }
    .kpi.accent::before {
        background: linear-gradient(90deg, transparent 10%, var(--accent) 50%, transparent 90%);
        opacity: 0.7;
    }
    .kpi.accent .kpi-value {
        color: var(--accent);
    }

    /* ---------- Streamlit stMetric restyle ---------- */
    [data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 1rem 1.25rem 1.125rem;
        position: relative;
        overflow: hidden;
    }
    [data-testid="stMetric"]::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--accent-line), transparent);
    }
    [data-testid="stMetricLabel"] {
        margin-bottom: 0.5rem !important;
    }
    [data-testid="stMetricLabel"] > div,
    [data-testid="stMetricLabel"] p {
        font-family: 'JetBrains Mono' !important;
        font-size: 0.6875rem !important;
        letter-spacing: 0.14em !important;
        text-transform: uppercase !important;
        color: var(--text-3) !important;
        font-weight: 500 !important;
    }
    [data-testid="stMetricValue"] {
        font-weight: 600 !important;
        font-size: 1.75rem !important;
        letter-spacing: -0.025em !important;
        color: var(--text) !important;
        font-variant-numeric: tabular-nums !important;
        line-height: 1.1 !important;
    }
    [data-testid="stMetricDelta"] {
        font-family: 'JetBrains Mono' !important;
        font-size: 0.75rem !important;
    }

    /* ---------- Buttons (main area — sidebar selectors above win by specificity) ---------- */
    .stButton > button {
        background: var(--surface-2);
        color: var(--text);
        border: 1px solid var(--border);
        font-family: 'IBM Plex Sans' !important;
        font-weight: 500 !important;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-size: 0.875rem;
        transition: all 0.15s ease;
        box-shadow: none;
    }
    .stButton > button:hover {
        background: var(--surface-3);
        border-color: var(--border-hover);
        transform: translateY(-1px);
    }
    .stButton > button[kind="primary"] {
        background: var(--accent) !important;
        color: #0B0C0E !important;
        border: 1px solid var(--accent) !important;
        font-weight: 600 !important;
        box-shadow: 0 1px 0 rgba(0,0,0,0.2), 0 6px 20px -8px rgba(245, 165, 36, 0.5);
    }
    .stButton > button[kind="primary"]:hover {
        background: var(--accent-dim) !important;
        border-color: var(--accent-dim) !important;
        transform: translateY(-1px);
    }
    .stButton > button[kind="primary"]:active {
        transform: translateY(0);
    }

    /* Download button */
    .stDownloadButton > button {
        background: var(--surface-2) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
        font-weight: 500 !important;
        border-radius: 6px !important;
    }
    .stDownloadButton > button:hover {
        background: var(--surface-3) !important;
        border-color: var(--accent) !important;
        color: var(--accent) !important;
    }

    /* ---------- Inputs ---------- */
    .stNumberInput input,
    .stTextInput input,
    .stDateInput input {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        color: var(--text) !important;
        font-family: 'IBM Plex Sans' !important;
        border-radius: 6px !important;
        font-size: 0.875rem !important;
        padding: 0.5rem 0.75rem !important;
        transition: border-color 0.15s, box-shadow 0.15s;
    }
    .stNumberInput input:focus,
    .stTextInput input:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px rgba(245, 165, 36, 0.12) !important;
        outline: none !important;
    }
    .stNumberInput button {
        background: var(--surface-2) !important;
        border-color: var(--border) !important;
        color: var(--text-2) !important;
    }
    .stNumberInput button:hover {
        background: var(--surface-3) !important;
        color: var(--accent) !important;
        border-color: var(--border-hover) !important;
    }

    /* Selectbox */
    .stSelectbox [data-baseweb="select"] > div {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 6px !important;
        min-height: 36px;
        font-size: 0.875rem;
    }
    .stSelectbox [data-baseweb="select"]:hover > div {
        border-color: var(--border-hover) !important;
    }
    .stSelectbox [data-baseweb="select"] > div[aria-expanded="true"] {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px rgba(245, 165, 36, 0.12) !important;
    }

    /* Multiselect tokens */
    .stMultiSelect [data-baseweb="tag"] {
        background: var(--accent-soft) !important;
        border: 1px solid rgba(245, 165, 36, 0.25) !important;
        color: var(--accent) !important;
        font-family: 'JetBrains Mono' !important;
        font-size: 0.75rem !important;
    }
    .stMultiSelect [data-baseweb="tag"] svg {
        color: var(--accent) !important;
    }

    /* Slider */
    .stSlider [role="slider"] {
        background: var(--accent) !important;
        box-shadow: 0 0 0 4px rgba(245, 165, 36, 0.2) !important;
    }
    .stSlider [data-baseweb="slider"] [role="progressbar"] > div {
        background: var(--accent) !important;
    }

    /* Checkbox */
    .stCheckbox [data-baseweb="checkbox"] [data-checked="true"] {
        background: var(--accent) !important;
        border-color: var(--accent) !important;
    }

    /* ---------- Expander ---------- */
    [data-testid="stExpander"] {
        background: var(--surface);
        border: 1px solid var(--border) !important;
        border-radius: 8px;
        box-shadow: none !important;
        overflow: hidden;
    }
    [data-testid="stExpander"] summary {
        font-weight: 500 !important;
        padding: 0.75rem 1rem !important;
        font-size: 0.875rem !important;
        color: var(--text) !important;
    }
    [data-testid="stExpander"] summary:hover {
        color: var(--accent) !important;
    }
    [data-testid="stExpander"] > details[open] > summary {
        border-bottom: 1px solid var(--border);
    }

    /* ---------- Alerts ---------- */
    [data-testid="stAlert"] {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        border-left-width: 3px !important;
        box-shadow: none !important;
    }
    [data-testid="stAlert"] [data-testid="stMarkdownContainer"] p {
        font-size: 0.875rem !important;
        color: var(--text-2) !important;
        margin: 0 !important;
    }
    [data-testid="stAlert"][kind="info"] { border-left-color: var(--info) !important; }
    [data-testid="stAlert"][kind="warning"] { border-left-color: var(--warning) !important; }
    [data-testid="stAlert"][kind="success"] { border-left-color: var(--success) !important; }
    [data-testid="stAlert"][kind="error"] { border-left-color: var(--danger) !important; }

    /* ---------- File uploader ---------- */
    [data-testid="stFileUploader"] section {
        background: var(--surface) !important;
        border: 1px dashed var(--border) !important;
        border-radius: 10px !important;
        padding: 2rem 1.5rem !important;
        transition: all 0.15s ease;
    }
    [data-testid="stFileUploader"] section:hover {
        border-color: var(--accent) !important;
        background: linear-gradient(var(--surface), var(--surface)) padding-box, var(--accent-soft);
    }
    [data-testid="stFileUploader"] section > button {
        background: var(--surface-2) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
        font-size: 0.8125rem !important;
        font-weight: 500 !important;
    }
    [data-testid="stFileUploader"] section > button:hover {
        border-color: var(--accent) !important;
        color: var(--accent) !important;
    }
    [data-testid="stFileUploader"] small {
        color: var(--text-3) !important;
    }

    /* ---------- Dataframe / table ---------- */
    .stDataFrame, [data-testid="stDataFrame"] {
        border: 1px solid var(--border) !important;
        border-radius: 8px;
        overflow: hidden;
    }
    .stDataFrame [data-testid="stTable"] {
        background: var(--surface) !important;
    }
    div[data-testid="stDataFrameResizable"] * {
        font-family: 'IBM Plex Sans' !important;
        font-size: 0.8125rem;
    }

    /* ---------- Tabs (inline, e.g. Sales / Ops view) ---------- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 1px solid var(--border);
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: var(--text-2) !important;
        font-family: 'IBM Plex Sans' !important;
        font-weight: 500 !important;
        font-size: 0.8125rem !important;
        padding: 0.625rem 1rem !important;
        border: none !important;
        border-bottom: 2px solid transparent !important;
        margin-bottom: -1px !important;
        letter-spacing: 0.005em !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text) !important;
    }
    .stTabs [aria-selected="true"] {
        color: var(--accent) !important;
        border-bottom-color: var(--accent) !important;
    }
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 1.5rem;
    }

    /* ---------- Chips ---------- */
    .chip {
        display: inline-flex;
        align-items: center;
        gap: 0.375rem;
        padding: 0.1875rem 0.625rem 0.25rem;
        border-radius: 999px;
        font-family: 'JetBrains Mono' !important;
        font-size: 0.6875rem;
        font-weight: 500;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        border: 1px solid;
        line-height: 1;
        vertical-align: middle;
    }
    .chip-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        flex-shrink: 0;
    }
    .chip-neutral { background: rgba(160, 160, 168, 0.08); border-color: rgba(160, 160, 168, 0.2); color: var(--text-2); }
    .chip-neutral .chip-dot { background: var(--text-2); }
    .chip-success { background: rgba(34, 197, 94, 0.08); border-color: rgba(34, 197, 94, 0.25); color: var(--success); }
    .chip-success .chip-dot { background: var(--success); }
    .chip-warning { background: rgba(234, 179, 8, 0.08); border-color: rgba(234, 179, 8, 0.25); color: var(--warning); }
    .chip-warning .chip-dot { background: var(--warning); }
    .chip-danger { background: rgba(239, 68, 68, 0.08); border-color: rgba(239, 68, 68, 0.25); color: var(--danger); }
    .chip-danger .chip-dot { background: var(--danger); }
    .chip-accent { background: rgba(245, 165, 36, 0.10); border-color: rgba(245, 165, 36, 0.3); color: var(--accent); }
    .chip-accent .chip-dot { background: var(--accent); }
    .chip-info { background: rgba(96, 165, 250, 0.08); border-color: rgba(96, 165, 250, 0.25); color: var(--info); }
    .chip-info .chip-dot { background: var(--info); }

    /* ---------- Result summary block ---------- */
    .result-hero {
        background:
            radial-gradient(800px 200px at 90% -10%, rgba(245, 165, 36, 0.06), transparent 60%),
            linear-gradient(180deg, var(--surface) 0%, var(--bg) 100%);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 2rem 2.25rem;
        margin: 1rem 0 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .result-hero::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--accent), transparent);
    }
    .result-hero-label {
        font-family: 'JetBrains Mono' !important;
        font-size: 0.6875rem;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: var(--text-3);
        margin-bottom: 0.75rem;
    }
    .result-hero-row {
        display: flex;
        align-items: baseline;
        gap: 2rem;
        flex-wrap: wrap;
    }
    .result-hero-main {
        font-weight: 600;
        font-size: 3rem;
        letter-spacing: -0.03em;
        color: var(--text);
        font-variant-numeric: tabular-nums;
        line-height: 1;
    }
    .result-hero-main .mono-unit {
        font-family: 'JetBrains Mono' !important;
        font-size: 1.125rem;
        color: var(--text-3);
        margin-left: 0.375rem;
        font-weight: 400;
    }
    .result-hero-meta {
        display: flex;
        gap: 1.5rem;
        flex-wrap: wrap;
    }
    .result-hero-item {
        display: flex;
        flex-direction: column;
        gap: 0.25rem;
    }
    .result-hero-item-label {
        font-family: 'JetBrains Mono' !important;
        font-size: 0.625rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--text-3);
    }
    .result-hero-item-val {
        font-family: 'JetBrains Mono' !important;
        font-size: 0.9375rem;
        font-weight: 500;
        color: var(--text);
        font-variant-numeric: tabular-nums;
    }

    /* ---------- Empty state ---------- */
    .empty {
        background: var(--surface);
        border: 1px dashed var(--border);
        border-radius: 10px;
        padding: 2.5rem 2rem;
        text-align: center;
        color: var(--text-2);
    }
    .empty-icon {
        width: 40px;
        height: 40px;
        margin: 0 auto 0.75rem;
        border: 1px solid var(--border);
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: var(--surface-2);
        color: var(--text-3);
        font-family: 'JetBrains Mono' !important;
        font-size: 1.25rem;
    }
    .empty-title {
        color: var(--text);
        font-weight: 500;
        font-size: 0.9375rem;
        margin-bottom: 0.375rem;
    }
    .empty-desc {
        font-size: 0.8125rem;
        color: var(--text-3);
        max-width: 420px;
        margin: 0 auto;
        line-height: 1.5;
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    ::-webkit-scrollbar-thumb {
        background: var(--surface-3);
        border-radius: 10px;
        border: 2px solid var(--bg);
    }
    ::-webkit-scrollbar-thumb:hover {
        background: var(--border-hover);
    }

    /* Altair chart padding */
    .stAltairChart, .stVegaLiteChart {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 1rem;
    }

    .stBarChart, .stLineChart, .stScatterChart, [data-testid="stVegaLiteChart"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 1rem;
    }

    /* Divider */
    hr {
        border: none !important;
        height: 1px !important;
        background: var(--border) !important;
        margin: 2rem 0 !important;
    }
    [data-testid="stMarkdownContainer"] hr {
        background: var(--border) !important;
    }

    /* Column gap spacing */
    [data-testid="stHorizontalBlock"] {
        gap: 1rem;
    }

    /* Caption styling */
    .stCaption, [data-testid="stCaptionContainer"] {
        color: var(--text-3) !important;
        font-size: 0.8125rem !important;
        font-family: 'IBM Plex Sans' !important;
    }

    /* Spinner */
    .stSpinner > div {
        border-color: var(--accent) transparent transparent transparent !important;
    }

    /* ---------- Form-group wrapper (used inside Single Quote) ---------- */
    .form-group-spacer {
        height: 0.5rem;
    }

    /* Lean column dividers inside dense forms */
    [data-testid="stHorizontalBlock"] > [data-testid="column"] {
        padding: 0 !important;
    }
    """
).strip()


def inject_css() -> None:
    """Inject the design-system CSS once per page render."""
    st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _render(html_str: str) -> None:
    """Emit HTML via st.markdown without leading whitespace that would
    otherwise be parsed as a fenced code block."""
    st.markdown(html_str, unsafe_allow_html=True)


def sidebar_brand(name: str = "Matrix", subtitle: str = "Quote Estimation v1.0") -> None:
    """Render the brand mark at the top of the sidebar."""
    _render(
        f'<div class="brand">'
        f'<div class="brand-row">'
        f'<div class="brand-mark"></div>'
        f'<div class="brand-name">{html.escape(name)}</div>'
        f'</div>'
        f'<div class="brand-sub">{html.escape(subtitle)}</div>'
        f'</div>'
    )


def nav_group_label(label: str) -> None:
    _render(f'<div class="nav-group">{html.escape(label)}</div>')


def sidebar_status(label: str, value: str, active: bool) -> None:
    """Render a small status line at the bottom of the sidebar."""
    cls = "sb-status-on" if active else "sb-status-off"
    _render(
        f'<div class="sb-status {cls}">'
        f'<div class="sb-status-label">{html.escape(label)}</div>'
        f'<div class="sb-status-row">'
        f'<span class="sb-dot"></span>'
        f'<span>{html.escape(value)}</span>'
        f'</div>'
        f'</div>'
    )


def page_header(
    eyebrow: str,
    title: str,
    description: str | None = None,
    right_chips: Iterable[tuple[str, str]] | None = None,
) -> None:
    """Render the page header block.

    *right_chips* is an iterable of (label, variant) tuples rendered on the right.
    """
    desc_html = (
        f'<div class="page-desc">{html.escape(description)}</div>'
        if description
        else ""
    )
    chips_html = ""
    if right_chips:
        chips_html = "".join(_chip_html(label, variant) for label, variant in right_chips)
    _render(
        f'<div class="page-head">'
        f'<div class="page-head-left">'
        f'<div class="page-eyebrow">{html.escape(eyebrow)}</div>'
        f'<div class="page-title">{html.escape(title)}</div>'
        f'{desc_html}'
        f'</div>'
        f'<div class="page-head-right">{chips_html}</div>'
        f'</div>'
    )


def section(number: str, title: str, hint: str | None = None) -> None:
    """Render a section header with a numeric eyebrow and a rule line."""
    hint_html = f'<div class="section-hint">{html.escape(hint)}</div>' if hint else ""
    _render(
        f'<div class="section">'
        f'<span class="section-num">{html.escape(number)}</span>'
        f'<span class="section-title">{html.escape(title)}</span>'
        f'<span class="section-rule"></span>'
        f'{hint_html}'
        f'</div>'
    )


def kpi_grid(cards: Iterable[dict]) -> None:
    """Render a responsive KPI grid.

    Each card dict accepts: label, value, unit, hint, accent (bool).
    """
    parts = ['<div class="kpi-grid">']
    for c in cards:
        cls = "kpi accent" if c.get("accent") else "kpi"
        unit_html = (
            f'<span class="kpi-unit">{html.escape(str(c["unit"]))}</span>'
            if c.get("unit")
            else ""
        )
        hint_html = (
            f'<div class="kpi-hint">{html.escape(str(c["hint"]))}</div>'
            if c.get("hint")
            else ""
        )
        parts.append(
            f'<div class="{cls}">'
            f'<div class="kpi-label">{html.escape(str(c["label"]))}</div>'
            f'<div class="kpi-value">{html.escape(str(c["value"]))}{unit_html}</div>'
            f'{hint_html}'
            f'</div>'
        )
    parts.append('</div>')
    _render("".join(parts))


def _chip_html(label: str, variant: str = "neutral") -> str:
    variant = variant if variant in {"neutral", "success", "warning", "danger", "accent", "info"} else "neutral"
    return (
        f'<span class="chip chip-{variant}">'
        f'<span class="chip-dot"></span>{html.escape(label)}</span>'
    )


def chip(label: str, variant: str = "neutral") -> None:
    _render(_chip_html(label, variant))


def empty_state(title: str, description: str, glyph: str = "·") -> None:
    _render(
        f'<div class="empty">'
        f'<div class="empty-icon">{html.escape(glyph)}</div>'
        f'<div class="empty-title">{html.escape(title)}</div>'
        f'<div class="empty-desc">{html.escape(description)}</div>'
        f'</div>'
    )


def result_hero(
    label: str,
    value: str,
    unit: str,
    meta: Iterable[tuple[str, str]] | None = None,
) -> None:
    """Render the primary result block: big number + labeled metadata."""
    meta_html = ""
    if meta:
        items = "".join(
            (
                f'<div class="result-hero-item">'
                f'<div class="result-hero-item-label">{html.escape(m_label)}</div>'
                f'<div class="result-hero-item-val">{html.escape(m_val)}</div>'
                f'</div>'
            )
            for m_label, m_val in meta
        )
        meta_html = f'<div class="result-hero-meta">{items}</div>'
    _render(
        f'<div class="result-hero">'
        f'<div class="result-hero-label">{html.escape(label)}</div>'
        f'<div class="result-hero-row">'
        f'<div class="result-hero-main">'
        f'{html.escape(value)}<span class="mono-unit">{html.escape(unit)}</span>'
        f'</div>'
        f'{meta_html}'
        f'</div>'
        f'</div>'
    )


def confidence_variant(confidence: str) -> str:
    """Map a confidence string to a chip variant."""
    c = confidence.lower().strip()
    if c.startswith("high"):
        return "success"
    if c.startswith("med"):
        return "warning"
    if c.startswith("low"):
        return "danger"
    return "neutral"
