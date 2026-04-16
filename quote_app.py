# quote_app.py
# Matrix Quote App — Streamlit UI with sidebar navigation and a "precision
# engineering" dark design system (see core/ui.py for CSS + helpers).

import os
import math
import pandas as pd
import altair as alt
import streamlit as st

from core.config import (
    QUOTE_NUM_FEATURES,
    QUOTE_CAT_FEATURES,
    TARGETS,
    REQUIRED_TRAINING_COLS,
    SALES_BUCKETS,
)
from core.schemas import QuoteInput
from core.features import engineer_features_for_training
from core.models import train_one_op, load_model
from core.ui import (
    inject_css,
    sidebar_brand,
    nav_group_label,
    sidebar_status,
    page_header,
    section,
    kpi_grid,
    result_hero,
    empty_state,
)
from service.predict_lib import predict_quote, predict_quotes_df


# ------------------------------ Paths / config -----------------------------
MASTER_DATA_PATH = os.path.join("data", "master", "projects_master.parquet")
UPLOADS_LOG_PATH = os.path.join("data", "master", "uploads_log.csv")
METRICS_PATH = os.path.join("models", "metrics_summary.csv")


st.set_page_config(
    page_title="Matrix · Quote Estimation",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None,
)
inject_css()


# ------------------------------ Altair theme -------------------------------

def _altair_theme():
    return {
        "config": {
            "background": "transparent",
            "view": {"stroke": "transparent"},
            "axis": {
                "labelColor": "#A0A0A8",
                "titleColor": "#A0A0A8",
                "labelFont": "IBM Plex Sans, system-ui, sans-serif",
                "titleFont": "IBM Plex Sans, system-ui, sans-serif",
                "labelFontSize": 11,
                "titleFontSize": 11,
                "titleFontWeight": 500,
                "titlePadding": 10,
                "gridColor": "#26272D",
                "gridOpacity": 0.6,
                "domainColor": "#26272D",
                "tickColor": "#26272D",
            },
            "legend": {
                "labelColor": "#A0A0A8",
                "titleColor": "#A0A0A8",
                "labelFont": "IBM Plex Sans, system-ui, sans-serif",
                "titleFont": "IBM Plex Sans, system-ui, sans-serif",
                "labelFontSize": 11,
                "titleFontSize": 11,
            },
            "title": {
                "color": "#F2F2F3",
                "font": "IBM Plex Sans, system-ui, sans-serif",
                "fontWeight": 500,
                "fontSize": 13,
            },
            "range": {
                "category": [
                    "#F5A524",
                    "#60A5FA",
                    "#22C55E",
                    "#EAB308",
                    "#EF4444",
                    "#A0A0A8",
                ],
            },
            "bar": {"color": "#F5A524"},
            "point": {"color": "#F5A524", "filled": True, "size": 50},
        }
    }


alt.themes.register("matrix_dark", _altair_theme)
alt.themes.enable("matrix_dark")


# ------------------------------ Session state ------------------------------

if "models_ready" not in st.session_state:
    st.session_state["models_ready"] = False

if not st.session_state["models_ready"]:
    if os.path.exists(METRICS_PATH):
        try:
            _metrics = pd.read_csv(METRICS_PATH)
            if not _metrics.empty:
                st.session_state["models_ready"] = True
        except Exception:
            pass

if "active_page" not in st.session_state:
    st.session_state["active_page"] = "Single Quote"


# ------------------------------ Data helpers -------------------------------

def _load_master():
    if os.path.exists(MASTER_DATA_PATH):
        return pd.read_parquet(MASTER_DATA_PATH)
    return None


def _get_dropdown_options(column: str, defaults: list[str]) -> list[str]:
    df = _load_master()
    if df is not None and column in df.columns:
        values = sorted(df[column].dropna().unique().tolist())
        if values:
            return values
    return defaults


def _load_metrics():
    if os.path.exists(METRICS_PATH):
        return pd.read_csv(METRICS_PATH)
    return None


def _reset_app_state() -> None:
    for path in [MASTER_DATA_PATH, UPLOADS_LOG_PATH, METRICS_PATH]:
        if os.path.exists(path):
            os.remove(path)
    models_dir = "models"
    if os.path.exists(models_dir) and os.path.isdir(models_dir):
        for fname in os.listdir(models_dir):
            if fname.endswith(".joblib"):
                try:
                    os.remove(os.path.join(models_dir, fname))
                except OSError:
                    pass
    st.session_state["models_ready"] = False


def _log_upload(rows_raw: int, rows_train: int, rows_master_total: int) -> None:
    upload_info = {
        "rows_raw": rows_raw,
        "rows_train": rows_train,
        "rows_master_total": rows_master_total,
    }
    log_row = pd.DataFrame([upload_info])
    os.makedirs(os.path.dirname(UPLOADS_LOG_PATH), exist_ok=True)
    if os.path.exists(UPLOADS_LOG_PATH):
        df_log_old = pd.read_csv(UPLOADS_LOG_PATH)
        df_log_new = pd.concat([df_log_old, log_row], ignore_index=True)
    else:
        df_log_new = log_row
    df_log_new.to_csv(UPLOADS_LOG_PATH, index=False)


# ------------------------------ Sidebar ------------------------------------

NAV = [
    ("ESTIMATE", ["Single Quote", "Batch Quotes"]),
    ("ANALYZE", ["Overview", "Data Explorer", "Model Performance", "Drivers & Similar"]),
    ("ADMIN", ["Upload & Train"]),
]


with st.sidebar:
    sidebar_brand("Matrix", "Quote Estimation")

    for group_label, items in NAV:
        nav_group_label(group_label)
        for item in items:
            is_active = st.session_state["active_page"] == item
            if st.button(
                item,
                key=f"nav_{item}",
                type="primary" if is_active else "secondary",
                use_container_width=True,
            ):
                st.session_state["active_page"] = item
                st.rerun()

    sidebar_status(
        "Model status",
        "Ready" if st.session_state["models_ready"] else "Not trained",
        st.session_state["models_ready"],
    )


# ------------------------------ Pages --------------------------------------

def render_overview() -> None:
    df_master = _load_master()
    metrics_df = _load_metrics()

    n_projects = len(df_master) if df_master is not None else 0
    n_ops = (
        metrics_df["target"].nunique()
        if metrics_df is not None and not metrics_df.empty
        else 0
    )
    avg_mae = (
        metrics_df["mae"].mean()
        if metrics_df is not None and not metrics_df.empty
        else None
    )
    avg_r2 = (
        metrics_df["r2"].mean()
        if metrics_df is not None and "r2" in (metrics_df.columns if metrics_df is not None else []) and not metrics_df.empty
        else None
    )

    chips = []
    if st.session_state["models_ready"]:
        chips.append(("Models Ready", "success"))
    else:
        chips.append(("Models Not Trained", "warning"))
    if n_projects:
        chips.append((f"{n_projects} Projects", "accent"))

    page_header(
        eyebrow="Dashboard",
        title="System Overview",
        description=(
            "Dataset health and model status at a glance. Use the navigation to explore "
            "data, review metrics, or generate new quotes."
        ),
        right_chips=chips,
    )

    cards = [
        {
            "label": "Projects in master",
            "value": f"{n_projects}",
            "hint": "Records available for training",
        },
        {
            "label": "Operations modeled",
            "value": f"{n_ops}",
            "unit": "/ 12",
            "hint": "Per-operation GBR models on disk",
        },
        {
            "label": "Average MAE",
            "value": f"{avg_mae:.1f}" if avg_mae is not None else "—",
            "unit": "hrs" if avg_mae is not None else "",
            "hint": "Lower is better",
        },
        {
            "label": "Average R²",
            "value": f"{avg_r2:.2f}" if avg_r2 is not None else "—",
            "accent": True,
            "hint": "1.0 = perfect fit",
        },
    ]
    kpi_grid(cards)

    section("01", "Recent uploads", "Last 10 training dataset refreshes")
    if os.path.exists(UPLOADS_LOG_PATH):
        df_log = pd.read_csv(UPLOADS_LOG_PATH)
        st.dataframe(df_log.tail(10), use_container_width=True, hide_index=True)
    else:
        empty_state(
            "No uploads logged yet",
            "Visit the Upload & Train page to ingest your first project-hours dataset.",
        )

    section("02", "Per-operation model metrics", "MAE, R², and sample counts")
    if metrics_df is not None and not metrics_df.empty:
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    else:
        empty_state(
            "No models trained",
            "Upload a dataset on the Upload & Train page to train per-operation models.",
        )


def render_data_explorer() -> None:
    df_master = _load_master()

    chips = []
    if df_master is not None and not df_master.empty:
        chips.append((f"{len(df_master)} Projects", "accent"))

    page_header(
        eyebrow="Dataset",
        title="Data Explorer",
        description="Filter the master training dataset and inspect per-operation distributions.",
        right_chips=chips,
    )

    if df_master is None or df_master.empty:
        empty_state(
            "Master dataset is empty",
            "Upload a project-hours Excel file on the Upload & Train page to populate the master dataset.",
        )
        return

    industries = (
        sorted(df_master["industry_segment"].dropna().unique())
        if "industry_segment" in df_master.columns
        else []
    )
    systems = (
        sorted(df_master["system_category"].dropna().unique())
        if "system_category" in df_master.columns
        else []
    )

    section("01", "Filters")
    c1, c2 = st.columns(2)
    with c1:
        sel_industries = (
            st.multiselect("Industry segment", industries, default=industries)
            if industries
            else []
        )
    with c2:
        sel_systems = (
            st.multiselect("System category", systems, default=systems)
            if systems
            else []
        )

    df_filtered = df_master.copy()
    if industries and sel_industries:
        df_filtered = df_filtered[df_filtered["industry_segment"].isin(sel_industries)]
    if systems and sel_systems:
        df_filtered = df_filtered[df_filtered["system_category"].isin(sel_systems)]

    section("02", "Filtered projects", f"{len(df_filtered)} of {len(df_master)} total")
    st.dataframe(df_filtered.head(50), use_container_width=True, hide_index=True)

    ops_with_data = [t for t in TARGETS if t in df_filtered.columns]
    if not ops_with_data:
        return

    section("03", "Operation analysis", "Per-project hours and relationships")
    op_choice = st.selectbox("Operation", ops_with_data)
    c1, c2 = st.columns(2)
    with c1:
        st.caption(f"Hours by project · {op_choice}")
        if "project_id" in df_filtered.columns:
            proj_df = df_filtered[["project_id", op_choice]].dropna()
            proj_df = proj_df.set_index("project_id")
            st.bar_chart(proj_df[op_choice])
        else:
            empty_state("Missing project_id", "Cannot label bars without project_id column.")
    with c2:
        if "robot_count" in df_filtered.columns:
            st.caption(f"robot_count vs {op_choice}")
            scatter_df = df_filtered[["robot_count", op_choice]].dropna()
            scatter_df = scatter_df.rename(columns={op_choice: "hours"})
            chart = (
                alt.Chart(scatter_df)
                .mark_circle(size=70, opacity=0.8)
                .encode(
                    x=alt.X("robot_count:Q", title="Robot count"),
                    y=alt.Y("hours:Q", title="Hours"),
                    tooltip=["robot_count", "hours"],
                )
                .properties(height=300)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            empty_state("Missing robot_count", "robot_count not present in filtered data.")


def render_model_perf() -> None:
    metrics_df = _load_metrics()

    chips = []
    if metrics_df is not None and not metrics_df.empty:
        chips.append((f"{metrics_df['target'].nunique()} Models", "accent"))

    page_header(
        eyebrow="Model",
        title="Model Performance",
        description="Per-operation MAE, R², and sample counts from the latest training run.",
        right_chips=chips,
    )

    if metrics_df is None or metrics_df.empty:
        empty_state(
            "No models trained",
            "Train models on the Upload & Train page to see performance metrics here.",
        )
        return

    section("01", "Per-operation metrics")
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    section("02", "Visual comparison", "MAE (lower is better) and R² (higher is better)")
    c1, c2 = st.columns(2)
    with c1:
        st.caption("MAE by operation")
        mae_chart = (
            alt.Chart(metrics_df)
            .mark_bar()
            .encode(
                x=alt.X("target:N", sort="-y", title=None),
                y=alt.Y("mae:Q", title="MAE (hours)"),
                tooltip=["target", "mae"],
            )
            .properties(height=300)
        )
        st.altair_chart(mae_chart, use_container_width=True)
    with c2:
        st.caption("R² by operation")
        if "r2" in metrics_df.columns:
            r2_chart = (
                alt.Chart(metrics_df)
                .mark_bar(color="#60A5FA")
                .encode(
                    x=alt.X("target:N", sort="-y", title=None),
                    y=alt.Y("r2:Q", title="R²"),
                    tooltip=["target", "r2"],
                )
                .properties(height=300)
            )
            st.altair_chart(r2_chart, use_container_width=True)
        else:
            empty_state("No R² data", "Metrics file does not contain R² column.")


def render_drivers() -> None:
    df_master = _load_master()

    page_header(
        eyebrow="Insight",
        title="Drivers & Similar Projects",
        description=(
            "Inspect what drives each model's predictions and find historical projects "
            "that resemble a target profile."
        ),
    )

    if df_master is None or df_master.empty:
        empty_state(
            "Master dataset is empty",
            "Upload training data on the Upload & Train page first.",
        )
        return

    left, right = st.columns([1.1, 1])

    with left:
        section("01", "Global drivers by operation")
        modeled_ops = [
            t
            for t in TARGETS
            if os.path.exists(os.path.join("models", f"{t}_v1.joblib"))
        ]
        if not modeled_ops:
            empty_state("No trained models", "No .joblib files found in ./models.")
        else:
            target_choice = st.selectbox(
                "Operation", modeled_ops, key="drivers_op_select"
            )
            bundle = load_model(target_choice)
            fi_df = None
            if bundle is None:
                st.warning("Model file not found for this operation.")
            else:
                # Prefer the precomputed importances stored in bundle meta
                # (present on CQR-style bundles). Fall back to the raw model
                # attribute when available.
                meta = bundle.get("meta", {}) if isinstance(bundle, dict) else {}
                fi_list = meta.get("feature_importances") if isinstance(meta, dict) else None
                if fi_list:
                    fi_df = pd.DataFrame(fi_list)
                else:
                    model = None
                    pre = None
                    if "pipeline" in bundle:
                        pipe = bundle["pipeline"]
                        pre = pipe.named_steps.get("preprocess")
                        model = pipe.named_steps.get("model")
                    else:
                        pre = bundle.get("preprocessor")
                        model = bundle.get("model_mid") or bundle.get("model")
                    importances = getattr(model, "feature_importances_", None)
                    if importances is not None:
                        try:
                            feature_names = (
                                pre.get_feature_names_out() if pre is not None else None
                            )
                        except Exception:
                            feature_names = None
                        if feature_names is None or len(feature_names) != len(importances):
                            feature_names = [f"f_{i}" for i in range(len(importances))]
                        fi_df = pd.DataFrame(
                            {"feature": feature_names, "importance": importances}
                        )

            if fi_df is not None and not fi_df.empty:
                fi_df = (
                    fi_df.sort_values("importance", ascending=False)
                    .reset_index(drop=True)
                )
                top15 = fi_df.head(15)
                nonzero = fi_df[fi_df["importance"] > 0]
                st.caption(
                    f"Top 15 features by importance · {len(nonzero)} with nonzero weight"
                )
                chart = (
                    alt.Chart(top15)
                    .mark_bar()
                    .encode(
                        y=alt.Y("feature:N", sort="-x", title=None),
                        x=alt.X("importance:Q", title="Importance"),
                        tooltip=["feature", "importance"],
                    )
                    .properties(height=min(len(top15) * 24 + 40, 440))
                )
                st.altair_chart(chart, use_container_width=True)
                with st.expander("Raw importances"):
                    st.dataframe(top15, use_container_width=True, hide_index=True)
            elif bundle is not None:
                empty_state(
                    "Importances unavailable",
                    "The model bundle has no precomputed feature importances and "
                    "the estimator does not expose feature_importances_.",
                )

    with right:
        section("02", "Similar projects finder")
        industries = (
            sorted(df_master["industry_segment"].dropna().unique())
            if "industry_segment" in df_master.columns
            else []
        )
        systems = (
            sorted(df_master["system_category"].dropna().unique())
            if "system_category" in df_master.columns
            else []
        )

        sel_industry = st.selectbox(
            "Industry segment", ["(any)"] + industries, index=0
        )
        sel_system = st.selectbox(
            "System category", ["(any)"] + systems, index=0
        )

        rc1, rc2 = st.columns(2)
        with rc1:
            min_robots = st.number_input(
                "Min robot count", min_value=0, value=0, step=1
            )
        with rc2:
            max_robots = st.number_input(
                "Max robot count", min_value=0, value=10, step=1
            )

        if st.button(
            "Find similar projects", type="primary", use_container_width=True
        ):
            df_sim = df_master.copy()
            if sel_industry != "(any)" and "industry_segment" in df_sim.columns:
                df_sim = df_sim[df_sim["industry_segment"] == sel_industry]
            if sel_system != "(any)" and "system_category" in df_sim.columns:
                df_sim = df_sim[df_sim["system_category"] == sel_system]
            if "robot_count" in df_sim.columns:
                df_sim = df_sim[
                    (df_sim["robot_count"] >= min_robots)
                    & (df_sim["robot_count"] <= max_robots)
                ]
            st.caption(f"Matched {len(df_sim)} projects")
            cols_to_show = [
                c
                for c in [
                    "project_id",
                    "industry_segment",
                    "system_category",
                    "robot_count",
                    "stations_count",
                ]
                if c in df_sim.columns
            ]
            for t in TARGETS:
                if t in df_sim.columns:
                    cols_to_show.append(t)
                    break
            st.dataframe(
                df_sim[cols_to_show].head(50),
                use_container_width=True,
                hide_index=True,
            )


def render_single_quote() -> None:
    page_header(
        eyebrow="Estimate",
        title="Single Quote",
        description=(
            "Enter quote-time project parameters to generate an hour estimate with "
            "confidence intervals per sales bucket."
        ),
        right_chips=(
            [("Models Ready", "success")]
            if st.session_state["models_ready"]
            else [("Models Not Trained", "warning")]
        ),
    )

    if not st.session_state["models_ready"]:
        empty_state(
            "Models are not trained",
            "Train models on the Upload & Train page before generating quotes.",
        )
        return

    # ------------------ 01 · Project classification ----------------------
    section(
        "01",
        "Project classification",
        "Segment, system type, and project flags",
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        industry_segment = st.selectbox(
            "Industry segment",
            _get_dropdown_options(
                "industry_segment",
                ["Automotive", "Food & Beverage", "General Industry"],
            ),
        )
    with c2:
        system_category = st.selectbox(
            "System category",
            _get_dropdown_options(
                "system_category",
                [
                    "Machine Tending",
                    "End of Line Automation",
                    "Robotic Metal Finishing",
                    "Engineered Manufacturing Systems",
                    "Other",
                ],
            ),
        )
    with c3:
        automation_level = st.selectbox(
            "Automation level",
            _get_dropdown_options(
                "automation_level",
                ["Semi-Automatic", "Robotic", "Hard Automation"],
            ),
        )

    c4, c5, c6, c7 = st.columns(4)
    with c4:
        has_controls = st.checkbox("Includes controls work", value=True)
    with c5:
        has_robotics = st.checkbox("Includes robotics work", value=True)
    with c6:
        retrofit = st.checkbox("Retrofit project")
    with c7:
        duplicate = st.checkbox("Duplicate of prior project")

    # ------------------ 02 · Physical scale ------------------------------
    section(
        "02",
        "Physical scale",
        "Stations, robots, fixtures, and footprint",
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        stations_count = st.number_input("Stations count", min_value=0, step=1)
        part_types = st.number_input("Part types", min_value=0, step=1)
        safety_doors = st.number_input("Safety doors", min_value=0, step=1)
    with c2:
        robot_count = st.number_input("Robot count", min_value=0, step=1)
        weldment_perimeter_ft = st.number_input(
            "Weldment perimeter (ft)", min_value=0.0
        )
        safety_devices_count = st.number_input(
            "Safety devices count", min_value=0, step=1
        )
    with c3:
        fixture_sets = st.number_input("Fixture sets", min_value=0, step=1)
        fence_length_ft = st.number_input("Fence length (ft)", min_value=0.0)
        conveyor_length_ft = st.number_input(
            "Conveyor length (ft)", min_value=0.0
        )

    # ------------------ 03 · Controls & automation -----------------------
    section(
        "03",
        "Controls & automation",
        "PLC/HMI platform, vision, panels, and drives",
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        plc_family = st.text_input("PLC family", "AB Compact Logix")
        panel_count = st.number_input("Panel count", min_value=0, step=1)
        servo_axes = st.number_input("Servo axes", min_value=0, step=1)
    with c2:
        hmi_family = st.text_input("HMI family", "AB PanelView Plus")
        drive_count = st.number_input("Drive count", min_value=0, step=1)
        pneumatic_devices = st.number_input(
            "Pneumatic devices", min_value=0, step=1
        )
    with c3:
        vision_type = st.text_input("Vision type", "None")
        vision_systems_count = st.number_input(
            "Vision systems count", min_value=0, step=1
        )

    # ------------------ 04 · Product & process ---------------------------
    section(
        "04",
        "Product & process",
        "Familiarity, uncertainty, and product characteristics",
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        product_familiarity_score = st.slider(
            "Product familiarity", 1, 5, 3
        )
        bulk_rigidity_score = st.slider("Bulk rigidity", 1, 5, 3)
    with c2:
        product_rigidity = st.slider("Product rigidity", 1, 5, 3)
        process_uncertainty_score = st.slider(
            "Process uncertainty", 1, 5, 3
        )
    with c3:
        changeover_time_min = st.number_input(
            "Changeover time (min)", min_value=0.0
        )

    c4, c5, c6 = st.columns(3)
    with c4:
        is_product_deformable = st.checkbox("Product deformable")
    with c5:
        is_bulk_product = st.checkbox("Bulk product")
    with c6:
        has_tricky_packaging = st.checkbox("Tricky packaging")

    # ------------------ 05 · Complexity & indices ------------------------
    section(
        "05",
        "Complexity & indices",
        "Overall complexity, custom %, and derived indices",
    )
    c1, c2 = st.columns(2)
    with c1:
        complexity_score_1_5 = st.slider("Overall complexity", 1, 5, 3)
    with c2:
        custom_pct = st.slider("Custom %", 0, 100, 50)

    c3, c4, c5, c6 = st.columns(4)
    with c3:
        stations_robot_index = st.number_input(
            "Stations/robot idx", min_value=0.0, help="Optional derived index"
        )
    with c4:
        mech_complexity_index = st.number_input(
            "Mech complexity idx",
            min_value=0.0,
            help="Optional derived index",
        )
    with c5:
        controls_complexity_index = st.number_input(
            "Controls complexity idx",
            min_value=0.0,
            help="Optional derived index",
        )
    with c6:
        physical_scale_index = st.number_input(
            "Physical scale idx",
            min_value=0.0,
            help="Optional derived index",
        )

    # ------------------ 06 · Cost ----------------------------------------
    section("06", "Cost", "Materials estimate (log-transformed internally)")
    estimated_materials_cost = st.number_input(
        "Estimated materials cost ($)", min_value=0.0
    )

    # ------------------ Optional quoted hours comparison ----------------
    with st.expander("Compare to your quoted hours (optional)"):
        _quoted_hours_inputs = {}
        cols_row = st.columns(3)
        for idx, bucket in enumerate(SALES_BUCKETS):
            with cols_row[idx % 3]:
                _quoted_hours_inputs[bucket] = st.number_input(
                    f"{bucket} quoted hours",
                    min_value=0.0,
                    value=0.0,
                    step=1.0,
                    key=f"quoted_hours_{bucket}",
                )

    st.markdown('<div style="height: 1rem"></div>', unsafe_allow_html=True)

    if st.button("Estimate hours", type="primary"):
        _quoted = {
            b: v for b, v in _quoted_hours_inputs.items() if v and v > 0
        }
        st.session_state["quoted_hours_by_bucket"] = _quoted

        log_cost = float(math.log1p(estimated_materials_cost))

        q = QuoteInput(
            industry_segment=industry_segment,
            system_category=system_category,
            automation_level=automation_level,
            plc_family=plc_family,
            hmi_family=hmi_family,
            vision_type=vision_type,
            stations_count=stations_count,
            robot_count=robot_count,
            fixture_sets=fixture_sets,
            part_types=part_types,
            servo_axes=servo_axes,
            pneumatic_devices=pneumatic_devices,
            safety_doors=safety_doors,
            weldment_perimeter_ft=weldment_perimeter_ft,
            fence_length_ft=fence_length_ft,
            conveyor_length_ft=conveyor_length_ft,
            product_familiarity_score=product_familiarity_score,
            product_rigidity=product_rigidity,
            is_product_deformable=int(is_product_deformable),
            is_bulk_product=int(is_bulk_product),
            bulk_rigidity_score=bulk_rigidity_score,
            has_tricky_packaging=int(has_tricky_packaging),
            process_uncertainty_score=process_uncertainty_score,
            changeover_time_min=changeover_time_min,
            safety_devices_count=safety_devices_count,
            custom_pct=custom_pct,
            duplicate=int(duplicate),
            has_controls=int(has_controls),
            has_robotics=int(has_robotics),
            Retrofit=int(retrofit),
            complexity_score_1_5=complexity_score_1_5,
            vision_systems_count=vision_systems_count,
            panel_count=panel_count,
            drive_count=drive_count,
            stations_robot_index=stations_robot_index,
            mech_complexity_index=mech_complexity_index,
            controls_complexity_index=controls_complexity_index,
            physical_scale_index=physical_scale_index,
            log_quoted_materials_cost=log_cost,
        )
        pred = predict_quote(q)
        _render_quote_results(pred, _quoted)


def _render_quote_results(pred, quoted_hours_by_bucket: dict) -> None:
    """Render the results block for a single-quote estimation."""
    sales_rows = []
    for bucket in SALES_BUCKETS:
        bucket_pred = pred.sales_buckets.get(bucket)
        if bucket_pred is None:
            continue
        sales_rows.append(
            {
                "Sales bucket": bucket,
                "p10_hours": bucket_pred.p10,
                "p50_hours": bucket_pred.p50,
                "p90_hours": bucket_pred.p90,
                "confidence": bucket_pred.confidence,
            }
        )

    has_quoted_hours = bool(quoted_hours_by_bucket)
    total_model_hours = pred.total_p50

    total_quoted = None
    total_delta = None
    status_label = None
    if has_quoted_hours:
        total_quoted = sum(
            v for v in quoted_hours_by_bucket.values() if isinstance(v, (int, float))
        )
        total_delta = total_quoted - total_model_hours
        threshold_total = max(0.1 * abs(total_model_hours), 10)
        if abs(total_delta) <= threshold_total:
            status_label = "Aligned with model"
        elif total_delta > 0:
            status_label = "Quote above model"
        else:
            status_label = "Quote below model"

    # Hero summary
    meta = []
    if has_quoted_hours:
        meta.append(("Your quote", f"{total_quoted:.1f} h"))
        meta.append(("Delta", f"{total_delta:+.1f} h"))
        if status_label:
            meta.append(("Status", status_label))

    result_hero(
        label="Recommended total · P50",
        value=f"{total_model_hours:,.1f}",
        unit="hours",
        meta=meta,
    )

    # Build summary rows
    sales_summary_rows = []
    for row in sales_rows:
        role = row["Sales bucket"]
        p10, p50, p90 = row["p10_hours"], row["p50_hours"], row["p90_hours"]
        confidence = row["confidence"]
        summary_row = {
            "Role": role,
            "Recommended (P50)": round(p50, 1),
            "Range (P10–P90)": f"{p10:.1f}–{p90:.1f}",
            "Confidence": confidence.title(),
        }
        if has_quoted_hours:
            quoted_val = quoted_hours_by_bucket.get(role)
            if quoted_val is not None:
                delta = quoted_val - p50
                threshold = max(0.1 * abs(p50), 5)
                if abs(delta) <= threshold:
                    delta_status = "Close"
                elif delta > 0:
                    delta_status = "Over"
                else:
                    delta_status = "Under"
                summary_row.update(
                    {
                        "Quoted": round(quoted_val, 1),
                        "Delta": round(delta, 1),
                        "Status": delta_status,
                    }
                )
        sales_summary_rows.append(summary_row)

    sales_tab, ops_tab = st.tabs(["Sales view", "Operations view"])

    with sales_tab:
        section("01", "Sales-bucket rollup")
        if sales_summary_rows:
            df_sales = pd.DataFrame(sales_summary_rows).sort_values(
                "Recommended (P50)", ascending=False
            )
            if has_quoted_hours:
                total_row = {
                    "Role": "TOTAL",
                    "Recommended (P50)": round(
                        df_sales["Recommended (P50)"].sum(), 1
                    ),
                    "Range (P10–P90)": "—",
                    "Confidence": "—",
                    "Quoted": round(df_sales["Quoted"].sum(), 1),
                    "Delta": round(df_sales["Delta"].sum(), 1),
                    "Status": "—",
                }
                df_sales = pd.concat(
                    [df_sales, pd.DataFrame([total_row])], ignore_index=True
                )

            display_cols = [
                "Role",
                "Recommended (P50)",
                "Range (P10–P90)",
                "Confidence",
            ]
            if has_quoted_hours:
                display_cols += ["Quoted", "Delta", "Status"]
            st.dataframe(
                df_sales[display_cols],
                use_container_width=True,
                hide_index=True,
            )

            if has_quoted_hours:
                section("02", "Model vs. quote")
                df_chart = df_sales[df_sales["Role"] != "TOTAL"][
                    ["Role", "Recommended (P50)", "Quoted"]
                ]
                if not df_chart.empty:
                    chart_data = df_chart.melt(
                        id_vars="Role",
                        value_vars=["Recommended (P50)", "Quoted"],
                        var_name="Source",
                        value_name="Hours",
                    )
                    chart = (
                        alt.Chart(chart_data)
                        .mark_bar()
                        .encode(
                            x=alt.X("Role:N", sort="-y", title=None),
                            xOffset=alt.X("Source:N"),
                            y=alt.Y("Hours:Q", title="Hours"),
                            color=alt.Color(
                                "Source:N",
                                scale=alt.Scale(
                                    domain=["Recommended (P50)", "Quoted"],
                                    range=["#F5A524", "#60A5FA"],
                                ),
                                legend=alt.Legend(title=None, orient="top"),
                            ),
                            tooltip=["Role", "Source", "Hours"],
                        )
                        .properties(height=340)
                    )
                    st.altair_chart(chart, use_container_width=True)
        else:
            empty_state(
                "No sales rollup available",
                "The model produced no sales-bucket results for this quote.",
            )

    with ops_tab:
        section("01", "Per-operation predictions")
        rows = []
        for op, op_pred in pred.ops.items():
            rows.append(
                {
                    "Operation": op,
                    "P10": round(op_pred.p10, 1),
                    "P50": round(op_pred.p50, 1),
                    "P90": round(op_pred.p90, 1),
                    "Std": round(op_pred.std, 2),
                    "Rel width": round(op_pred.rel_width, 2),
                    "Confidence": op_pred.confidence.title(),
                }
            )
        df_out = pd.DataFrame(rows)
        st.dataframe(df_out, use_container_width=True, hide_index=True)


def render_batch_quotes() -> None:
    page_header(
        eyebrow="Estimate",
        title="Batch Quotes",
        description=(
            "Upload a CSV or Excel file with many project rows and get predictions "
            "for all of them in one pass."
        ),
        right_chips=(
            [("Models Ready", "success")]
            if st.session_state["models_ready"]
            else [("Models Not Trained", "warning")]
        ),
    )

    if not st.session_state["models_ready"]:
        empty_state(
            "Models are not trained",
            "Train models on the Upload & Train page before running batch predictions.",
        )
        return

    section("01", "Upload file")
    st.caption(
        "Required columns: "
        + ", ".join(QUOTE_NUM_FEATURES + QUOTE_CAT_FEATURES)
    )

    uploaded = st.file_uploader(
        "Drop a CSV or Excel file, or click to browse",
        type=["csv", "xlsx", "xls"],
        key="batch_uploader",
    )
    if uploaded is None:
        return

    name = uploaded.name.lower()
    if name.endswith(".csv"):
        df_in = pd.read_csv(uploaded)
    else:
        xls = pd.ExcelFile(uploaded)
        sheet_name = st.selectbox(
            "Select sheet for quote inputs", xls.sheet_names, key="batch_sheet"
        )
        df_in = pd.read_excel(xls, sheet_name=sheet_name)

    section("02", "Input preview", f"{len(df_in)} rows loaded")
    st.dataframe(df_in.head(), use_container_width=True, hide_index=True)

    required_cols = set(QUOTE_NUM_FEATURES + QUOTE_CAT_FEATURES)
    missing = [c for c in required_cols if c not in df_in.columns]

    if missing:
        st.error(f"Missing required columns: {missing}")
        return

    if st.button("Run predictions on all rows", type="primary"):
        df_out = predict_quotes_df(df_in)
        section(
            "03",
            "Predictions",
            f"{len(df_out)} rows scored",
        )
        st.dataframe(df_out.head(), use_container_width=True, hide_index=True)
        csv_bytes = df_out.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download predictions CSV",
            data=csv_bytes,
            file_name="quote_predictions.csv",
            mime="text/csv",
        )


def render_admin() -> None:
    page_header(
        eyebrow="Admin",
        title="Upload & Train",
        description=(
            "Upload the latest project-hours Excel export. The app will merge it into "
            "the master dataset (dedup by project_id) and retrain all per-operation models."
        ),
    )

    section("01", "Upload dataset")
    uploaded_file = st.file_uploader(
        "Drop the latest project-hours Excel file, or click to browse",
        type=["xlsx", "xls"],
        key="training_uploader",
    )

    if uploaded_file is not None:
        xls = pd.ExcelFile(uploaded_file)
        sheet_name = st.selectbox("Sheet", xls.sheet_names)
        df_raw = pd.read_excel(xls, sheet_name=sheet_name)

        section("02", "Dataset preview", f"{len(df_raw)} rows")
        st.dataframe(df_raw.head(), use_container_width=True, hide_index=True)

        missing = [c for c in REQUIRED_TRAINING_COLS if c not in df_raw.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            section("03", "Merge & train", "Adds rows to master, then retrains all models")
            if st.button(
                "Merge into master & train models", type="primary"
            ):
                with st.spinner("Processing upload and training models..."):
                    rows_raw = len(df_raw)
                    df_train = engineer_features_for_training(df_raw)

                    targets_present = [
                        t for t in TARGETS if t in df_train.columns
                    ]
                    if targets_present:
                        hours_mat = (
                            df_train[targets_present]
                            .apply(pd.to_numeric, errors="coerce")
                            .fillna(0)
                        )
                        has_any_hours = hours_mat.gt(0).any(axis=1)
                        df_train = df_train[has_any_hours]

                    rows_train = len(df_train)

                    if rows_train == 0:
                        rows_master_total = (
                            len(pd.read_parquet(MASTER_DATA_PATH))
                            if os.path.exists(MASTER_DATA_PATH)
                            else 0
                        )
                        _log_upload(rows_raw, rows_train, rows_master_total)
                        st.warning(
                            "Upload contained no rows with non-zero actual hours. "
                            "Master dataset and models were left unchanged."
                        )
                    else:
                        os.makedirs(
                            os.path.dirname(MASTER_DATA_PATH), exist_ok=True
                        )

                        if os.path.exists(MASTER_DATA_PATH):
                            df_master_old = pd.read_parquet(MASTER_DATA_PATH)
                            df_all = pd.concat(
                                [df_master_old, df_train], ignore_index=True
                            )
                        else:
                            df_all = df_train

                        if "project_id" in df_all.columns:
                            df_all = df_all.sort_index()
                            df_master_new = df_all.drop_duplicates(
                                subset=["project_id"], keep="last"
                            )
                        else:
                            df_master_new = df_all

                        df_master_new.to_parquet(MASTER_DATA_PATH, index=False)
                        rows_master_total = len(df_master_new)
                        _log_upload(rows_raw, rows_train, rows_master_total)

                        metrics_all = []
                        for target in TARGETS:
                            m = train_one_op(
                                df_master_new,
                                target,
                                models_dir="models",
                                version="v1",
                            )
                            if m:
                                metrics_all.append(m)

                        if metrics_all:
                            metrics_df = pd.DataFrame(metrics_all)
                            os.makedirs("models", exist_ok=True)
                            metrics_df.to_csv(METRICS_PATH, index=False)
                            st.session_state["models_ready"] = True

                            st.success(
                                "Master dataset updated and models trained. "
                                "Quoting pages now use the latest models."
                            )
                            st.dataframe(
                                metrics_df,
                                use_container_width=True,
                                hide_index=True,
                            )
                            csv_bytes = metrics_df.to_csv(index=False).encode(
                                "utf-8"
                            )
                            st.download_button(
                                label="Download metrics_summary.csv",
                                data=csv_bytes,
                                file_name="metrics_summary.csv",
                                mime="text/csv",
                            )
                            st.rerun()
                        else:
                            st.warning(
                                "No models were trained (not enough data for any operation). "
                                "Check that actual-hours columns have non-zero values."
                            )
    else:
        st.info("Upload your project dataset (Excel) to enable training.")

    st.markdown('<div style="height: 2rem"></div>', unsafe_allow_html=True)

    section(
        "04",
        "Reset app state",
        "Destructive — clears master dataset, upload log, and all model artifacts",
    )
    if st.button("Reset master dataset and models"):
        _reset_app_state()
        st.success(
            "Master dataset, upload log, and model artifacts have been cleared. "
            "The app is now in a blank state."
        )
        st.rerun()


# ------------------------------ Dispatch -----------------------------------

PAGE_DISPATCH = {
    "Single Quote": render_single_quote,
    "Batch Quotes": render_batch_quotes,
    "Overview": render_overview,
    "Data Explorer": render_data_explorer,
    "Model Performance": render_model_perf,
    "Drivers & Similar": render_drivers,
    "Upload & Train": render_admin,
}

PAGE_DISPATCH.get(st.session_state["active_page"], render_single_quote)()
