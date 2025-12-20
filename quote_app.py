# quote_app.py
# Streamlit UI with:
# - Overview
# - Data Explorer
# - Model Performance
# - Drivers & Similar Projects
# - Single Quote
# - Batch Quotes
# - Admin: Upload & Train

import math
import os

import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline

from core.config import (
    QUOTE_NUM_FEATURES,
    QUOTE_CAT_FEATURES,
    TARGETS,
    REQUIRED_TRAINING_COLS,
)
from core.schemas import QuoteInput
from core.features import engineer_features_for_training
from core.models import train_one_op, load_model
from service.predict_lib import predict_quote, predict_quotes_df

MASTER_DATA_PATH = os.path.join("data", "master", "projects_master.parquet")
UPLOADS_LOG_PATH = os.path.join("data", "master", "uploads_log.csv")
METRICS_PATH = os.path.join("models", "metrics_summary.csv")
METRICS_COLUMNS = [
    "target",
    "version",
    "trained",
    "rows",
    "mae",
    "coverage",
    "interval_width",
    "qhat",
    "alpha",
    "r2",
]

BUCKET_MAP = {
    "me": "Mechanical",
    "ee": "Electrical",
    "cp": "Controls",
    "rb": "Robotics",
    "bld": "Build",
    "shp": "Shop",
    "inst": "Install",
    "trv": "Travel",
    "doc": "Documentation",
    "pm": "Project Management",
}

LABELS = {
    "typical_miss": "Typical miss (hours)",
    "range_reliability": "Range reliability",
}
LABEL_TYPICAL_MISS = LABELS["typical_miss"]
LABEL_RANGE_RELIABILITY = LABELS["range_reliability"]

SINGLE_QUOTE_HELP = {
    "industry_segment": "Customer industry category. Choose the closest match.",
    "system_category": "Type of system being built (ex: end-of-line, tending, other).",
    "automation_level": "How automated the system is. Choose Robotic if robots do the work.",
    "plc_family": "PLC platform expected (used to align with similar historical projects).",
    "hmi_family": "HMI platform expected (used to align with similar historical projects).",
    "vision_type": "Vision approach (none / 2D / 3D).",
    "stations_count": "Approx number of stations or cells in the system.",
    "robot_count": "Number of robots included in the scope.",
    "fixture_sets": "How many unique fixture sets are needed.",
    "part_types": "How many different part types the system must handle.",
    "servo_axes": "Count of servo-controlled axes (excluding robots).",
    "pneumatic_devices": "Approx count of pneumatic actuators/devices.",
    "safety_doors": "Count of safety gates/doors.",
    "weldment_perimeter_ft": "Approx weldment perimeter in feet (rough estimate is fine).",
    "fence_length_ft": "Approx total safety fence/guarding length in feet.",
    "conveyor_length_ft": "Approx total conveyor length in feet.",
    "product_familiarity_score": "How familiar the team is with the product (1=new, 5=very familiar).",
    "product_rigidity": "How rigid the product is (1=flexible, 5=rigid).",
    "is_product_deformable": "Check if the product can deform during handling (bags, soft packaging).",
    "is_bulk_product": "Check if the process handles bulk/loose product.",
    "bulk_rigidity_score": "Rigidity of bulk material (1=loose, 5=rigid).",
    "has_tricky_packaging": "Check if packaging is difficult to handle or stabilize.",
    "process_uncertainty_score": "How uncertain the process is (1=known, 5=experimental).",
    "changeover_time_min": "Expected changeover time between products/parts (minutes).",
    "safety_devices_count": "Count of safety devices (light curtains, scanners, E-stops, etc.).",
    "custom_pct": "Percent of the system that is custom vs repeatable.",
    "duplicate": "Check if this is very similar to a previous project.",
    "has_controls": "Controls/PLC/HMI work is included in the quote.",
    "has_robotics": "Robotics integration/programming is included in the quote.",
    "retrofit": "Check if modifying an existing system (vs new build).",
    "complexity_score_1_5": "Overall complexity (1=straightforward, 5=high complexity).",
    "vision_systems_count": "Number of vision systems/cameras.",
    "panel_count": "Number of electrical panels.",
    "drive_count": "Number of drives (VFD/servo drives).",
    "quoted_materials_cost_usd": "Approx materials/equipment cost being quoted. Use 0 if unknown.",
}


def fmt_hours(x) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{x:.1f} h"


def fmt_range(lo, hi) -> str:
    if lo is None or hi is None or pd.isna(lo) or pd.isna(hi):
        return "—"
    return f"{lo:.1f}–{hi:.1f} h"


def compute_buffer(est, hi) -> float:
    if est is None or hi is None or pd.isna(est) or pd.isna(hi):
        return float("nan")
    return hi - est


def bucket_for_op(op_name: str) -> str:
    op_prefix = "".join(ch for ch in op_name.lower() if ch.isalpha())
    return BUCKET_MAP.get(op_prefix, "Other")


def build_bucket_summary(pred_ops: dict) -> pd.DataFrame:
    rows = []
    for op_name, op_pred in pred_ops.items():
        rows.append(
            {
                "Bucket": bucket_for_op(op_name),
                "Low": op_pred.p10,
                "Estimate": op_pred.p50,
                "High": op_pred.p90,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=["Bucket", "Low", "Estimate", "High", "Buffer"]
        )
    df = pd.DataFrame(rows)
    summary = df.groupby("Bucket", as_index=False).sum(numeric_only=True)
    summary["Buffer"] = summary.apply(
        lambda row: compute_buffer(row["Estimate"], row["High"]), axis=1
    )
    return summary[["Bucket", "Low", "Estimate", "High", "Buffer"]]


def _sum_columns(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    if not columns:
        return pd.Series(0.0, index=df.index)
    return df[columns].sum(axis=1)


def build_batch_summary(df_in: pd.DataFrame, df_out: pd.DataFrame) -> pd.DataFrame:
    df_summary = df_in.copy()
    df_summary["total_low"] = df_out["total_p10"]
    df_summary["total_estimate"] = df_out["total_p50"]
    df_summary["total_high"] = df_out["total_p90"]
    df_summary["total_buffer"] = df_summary["total_high"] - df_summary["total_estimate"]

    for bucket in BUCKET_MAP:
        low_cols = [
            col
            for col in df_out.columns
            if col.startswith(f"{bucket}_") and col.endswith("_p10")
        ]
        estimate_cols = [
            col
            for col in df_out.columns
            if col.startswith(f"{bucket}_") and col.endswith("_p50")
        ]
        high_cols = [
            col
            for col in df_out.columns
            if col.startswith(f"{bucket}_") and col.endswith("_p90")
        ]

        df_summary[f"{bucket}_low"] = _sum_columns(df_out, low_cols)
        df_summary[f"{bucket}_estimate"] = _sum_columns(df_out, estimate_cols)
        df_summary[f"{bucket}_high"] = _sum_columns(df_out, high_cols)
        df_summary[f"{bucket}_buffer"] = (
            df_summary[f"{bucket}_high"] - df_summary[f"{bucket}_estimate"]
        )

    return df_summary


def _load_master():
    """Load the master training dataset if it exists."""
    if os.path.exists(MASTER_DATA_PATH):
        return pd.read_parquet(MASTER_DATA_PATH)
    return None


def _load_metrics():
    """Load the per-operation metrics file if it exists."""
    if os.path.exists(METRICS_PATH):
        return pd.read_csv(METRICS_PATH)
    return None


def _models_ready_from_disk():
    if os.path.exists(METRICS_PATH):
        return True
    return any(
        os.path.exists(os.path.join("models", f"{target}_v1.joblib"))
        for target in TARGETS
    )


def _rerun_app():
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()


def main():
    st.set_page_config(page_title="Matrix Quote App", layout="wide")
    st.title("Matrix Quote App")

    if not st.session_state.get("models_ready", False):
        st.session_state["models_ready"] = _models_ready_from_disk()

    tabs = st.tabs(
        [
            "Overview",
            "Data Explorer",
            "Model Performance",
            "Drivers & Similar Projects",
            "Single Quote",
            "Batch Quotes",
            "Admin: Upload & Train",
        ]
    )

    (
        tab_overview,
        tab_data,
        tab_perf,
        tab_drivers,
        tab_single,
        tab_batch,
        tab_admin,
    ) = tabs


    # Overview tab: high-level status
    with tab_overview:
        st.header("Overview")

        df_master = _load_master()
        metrics_df = _load_metrics()

        col1, col2, col3 = st.columns(3)

        with col1:
            if df_master is not None:
                st.metric("Projects in master dataset", f"{len(df_master)}")
            else:
                st.metric("Projects in master dataset", "0")

        with col2:
            if metrics_df is not None and not metrics_df.empty:
                if "trained" in metrics_df.columns:
                    trained_ops = metrics_df.loc[
                        metrics_df["trained"].fillna(False), "target"
                    ].nunique()
                else:
                    trained_ops = metrics_df["target"].nunique()
                st.metric("Operations with models", f"{trained_ops}")
            else:
                st.metric("Operations with models", "0")

        with col3:
            if (
                metrics_df is not None
                and not metrics_df.empty
                and "mae" in metrics_df.columns
            ):
                avg_mae = metrics_df["mae"].mean()
                st.metric("Typical miss (avg hours)", f"{avg_mae:.1f}")
            else:
                st.metric("Typical miss (avg hours)", "N/A")

        st.markdown("---")

        colA, colB = st.columns(2)

        with colA:
            with st.expander(
                "Technical (admin): upload history", expanded=False
            ):
                st.subheader("Upload history")
                if os.path.exists(UPLOADS_LOG_PATH):
                    df_log = pd.read_csv(UPLOADS_LOG_PATH)
                    st.dataframe(df_log.tail(10))
                else:
                    st.info(
                        "No uploads logged yet. Use the Admin tab to upload data."
                    )

        with colB:
            with st.expander(
                "Technical (admin): per-operation metrics", expanded=False
            ):
                st.subheader("Model metrics snapshot")
                if metrics_df is not None and not metrics_df.empty:
                    st.dataframe(metrics_df)
                else:
                    st.info(
                        "No models trained yet. Use the Admin tab after uploading data."
                    )

    # Data Explorer tab: explore master dataset
    with tab_data:
        st.header("Data Explorer")

        df_master = _load_master()
        metrics_df = _load_metrics()
        if df_master is None or df_master.empty:
            st.info(
                "Master dataset is empty. Upload and train in the Admin tab first."
            )
        else:
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

            col_filters1, col_filters2 = st.columns(2)

            with col_filters1:
                sel_industries = (
                    st.multiselect(
                        "Filter by industry_segment",
                        industries,
                        default=industries,
                    )
                    if industries
                    else []
                )

            with col_filters2:
                sel_systems = (
                    st.multiselect(
                        "Filter by system_category", systems, default=systems
                    )
                    if systems
                    else []
                )

            df_filtered = df_master.copy()
            if industries and sel_industries:
                df_filtered = df_filtered[
                    df_filtered["industry_segment"].isin(sel_industries)
                ]
            if systems and sel_systems:
                df_filtered = df_filtered[
                    df_filtered["system_category"].isin(sel_systems)
                ]

            st.subheader(f"Filtered projects: {len(df_filtered)}")
            st.dataframe(df_filtered.head(50))

            st.markdown("---")

            ops_with_data = [t for t in TARGETS if t in df_filtered.columns]
            if ops_with_data:
                op_choice = st.selectbox(
                    "Select operation to explore", ops_with_data
                )
                col_charts1, col_charts2 = st.columns(2)

                with col_charts1:
                    st.write(f"Histogram of {op_choice}")
                    st.bar_chart(df_filtered[op_choice].dropna())

                with col_charts2:
                    st.subheader("Operation performance")
                    if metrics_df is None or metrics_df.empty:
                        st.info(
                            "Train models to see operation performance here."
                        )
                    else:
                        op_metrics = metrics_df.loc[
                            metrics_df["target"] == op_choice
                        ]
                        if op_metrics.empty:
                            st.info(
                                "Train models to see operation performance here."
                            )
                        else:
                            op_row = op_metrics.iloc[0]
                            trained_val = op_row.get("trained", True)
                            trained = (
                                "Yes"
                                if bool(trained_val)
                                else "No"
                            )
                            rows = op_row.get("rows")
                            mae = op_row.get("mae")
                            coverage = op_row.get("coverage")
                            coverage_pct = (
                                f"{coverage * 100:.0f}%"
                                if coverage is not None
                                and not pd.isna(coverage)
                                else "—"
                            )
                            rows_label = (
                                f"{int(rows)}"
                                if rows is not None
                                and not pd.isna(rows)
                                else "—"
                            )

                            st.write(f"**Trained:** {trained}")
                            st.write(
                                f"**Training examples (rows):** {rows_label}"
                            )
                            st.write(
                                f"**{LABEL_TYPICAL_MISS}:** {fmt_hours(mae)}"
                            )
                            st.write(
                                f"**{LABEL_RANGE_RELIABILITY}:** {coverage_pct}"
                            )
                            if not bool(trained_val):
                                st.caption(
                                    "Not enough history yet for this operation."
                                )
                            elif (
                                rows is not None
                                and not pd.isna(rows)
                                and rows < 15
                            ):
                                st.caption(
                                    "Limited history; expect higher variability."
                                )

                    if "robot_count" in df_filtered.columns:
                        with st.expander(
                            "Optional: robot count comparison",
                            expanded=False,
                        ):
                            st.write(f"robot_count vs {op_choice}")
                            scatter_df = df_filtered[
                                ["robot_count", op_choice]
                            ].dropna()
                            scatter_df = scatter_df.rename(
                                columns={
                                    "robot_count": "robot_count",
                                    op_choice: "hours",
                                }
                            )
                            st.scatter_chart(
                                scatter_df, x="robot_count", y="hours"
                            )
            else:
                st.info("No operation hours columns found in master dataset.")

    # Model Performance tab: show per-op metrics
    with tab_perf:
        st.header("Model Performance")

        metrics_df = _load_metrics()
        if metrics_df is None or metrics_df.empty:
            st.info("No models trained yet. Use the Admin tab after uploading data.")
        else:
            st.subheader("Model health summary")
            if "trained" in metrics_df.columns:
                trained_targets = metrics_df.loc[
                    metrics_df["trained"].fillna(False), "target"
                ].tolist()
                untrained_from_metrics = metrics_df.loc[
                    ~metrics_df["trained"].fillna(False), "target"
                ].tolist()
            else:
                trained_targets = metrics_df["target"].tolist()
                untrained_from_metrics = []

            not_trained_ops = [
                target
                for target in TARGETS
                if target not in trained_targets
            ]
            not_trained_ops.extend(
                [
                    target
                    for target in untrained_from_metrics
                    if target not in not_trained_ops
                ]
            )

            st.write(f"Models trained: {len(trained_targets)}")
            if not_trained_ops:
                st.write(
                    f"Not trained operations ({len(not_trained_ops)}): "
                    + ", ".join(not_trained_ops)
                )
            else:
                st.write("Not trained operations: 0")

            if "mae" in metrics_df.columns:
                typical_miss = metrics_df["mae"].mean()
                st.write(f"Typical miss (avg hours): {typical_miss:.1f}")

            with st.expander(
                "Technical (admin): full metrics and charts", expanded=False
            ):
                st.subheader("Per-operation metrics")
                st.dataframe(metrics_df)

                col_perf1, col_perf2 = st.columns(2)

                with col_perf1:
                    if "mae" in metrics_df.columns:
                        st.write("MAE by operation")
                        mae_chart = metrics_df[["target", "mae"]].set_index(
                            "target"
                        )
                        st.bar_chart(mae_chart)
                    else:
                        st.info("MAE metrics not available.")

                with col_perf2:
                    if "r2" in metrics_df.columns:
                        st.write("R² by operation")
                        r2_chart = metrics_df[["target", "r2"]].set_index(
                            "target"
                        )
                        st.bar_chart(r2_chart)
                    else:
                        st.info("R² metrics not available.")


    # Drivers & Similar Projects tab
    with tab_drivers:
        st.header("Drivers & Similar Projects")

        df_master = _load_master()
        if df_master is None or df_master.empty:
            st.info("Master dataset is empty. Upload and train in the Admin tab first.")
        else:
            col_dr1, col_dr2 = st.columns(2)

            # Drivers: feature importance by operation
            with col_dr1:
                with st.expander(
                    "Technical (admin): model drivers", expanded=False
                ):
                    st.subheader("Global drivers by operation")

                    modeled_ops = [
                        t
                        for t in TARGETS
                        if os.path.exists(
                            os.path.join("models", f"{t}_v1.joblib")
                        )
                    ]
                    if not modeled_ops:
                        st.info("No trained models found in ./models.")
                    else:
                        target_choice = st.selectbox(
                            "Select operation",
                            modeled_ops,
                            key="drivers_op_select",
                        )

                        pipe = load_model(target_choice)
                        pre = None
                        model = None

                        if pipe is None:
                            st.warning(
                                "No model artifact loaded for this operation."
                            )
                        elif isinstance(pipe, dict) and {
                            "preprocessor",
                            "model_mid",
                        }.issubset(pipe):
                            pre = pipe["preprocessor"]
                            model = pipe["model_mid"]
                        elif isinstance(pipe, Pipeline):
                            pre = pipe.named_steps.get("preprocess")
                            model = pipe.named_steps.get("model")
                        else:
                            st.info(
                                "Loaded artifact is not compatible with feature importance."
                            )

                        if model is None or not hasattr(
                            model, "feature_importances_"
                        ):
                            st.info(
                                "Selected model does not expose feature_importances_."
                            )
                        else:
                            try:
                                feature_names = pre.get_feature_names_out()
                            except Exception:
                                feature_names = [
                                    f"f_{i}"
                                    for i in range(
                                        len(model.feature_importances_)
                                    )
                                ]

                            importances = model.feature_importances_
                            fi_df = (
                                pd.DataFrame(
                                    {
                                        "feature": feature_names,
                                        "importance": importances,
                                    }
                                )
                                .sort_values("importance", ascending=False)
                                .reset_index(drop=True)
                            )

                            st.write("Top 15 features by importance")
                            st.dataframe(fi_df.head(15))
                            st.bar_chart(
                                fi_df.head(15).set_index("feature")[
                                    "importance"
                                ]
                            )

            # Similar projects: filter-based helper
            with col_dr2:
                st.subheader("Similar projects explorer")

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
                    "Industry segment (filter)",
                    options=["(any)"] + industries,
                    index=0,
                )
                sel_system = st.selectbox(
                    "System category (filter)",
                    options=["(any)"] + systems,
                    index=0,
                )

                min_robots = st.number_input(
                    "Min robot_count", min_value=0, value=0, step=1
                )
                max_robots = st.number_input(
                    "Max robot_count", min_value=0, value=10, step=1
                )

                if st.button("Find similar projects"):
                    df_sim = df_master.copy()

                    if (
                        sel_industry != "(any)"
                        and "industry_segment" in df_sim.columns
                    ):
                        df_sim = df_sim[df_sim["industry_segment"] == sel_industry]

                    if (
                        sel_system != "(any)"
                        and "system_category" in df_sim.columns
                    ):
                        df_sim = df_sim[df_sim["system_category"] == sel_system]

                    if "robot_count" in df_sim.columns:
                        df_sim = df_sim[
                            (df_sim["robot_count"] >= min_robots)
                            & (df_sim["robot_count"] <= max_robots)
                        ]

                    st.write(f"Found {len(df_sim)} similar projects")
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

                    st.dataframe(df_sim[cols_to_show].head(50))


    # Single Quote tab
    with tab_single:
        st.header("Single quote estimation")

        if not st.session_state["models_ready"]:
            st.warning("Models are not trained yet. Go to 'Admin: Upload & Train' first.")
        else:
            with st.form("single_quote_form"):
                st.subheader("Core inputs")
                industry_segment = st.selectbox(
                    "Industry segment",
                    ["Automotive", "Food & Beverage", "General Industry"],
                    help=SINGLE_QUOTE_HELP["industry_segment"],
                )
                system_category = st.selectbox(
                    "System category",
                    ["End of Line Automation", "Machine Tending", "Other"],
                    help=SINGLE_QUOTE_HELP["system_category"],
                )
                automation_level = st.selectbox(
                    "Automation level",
                    ["Semi-Automatic", "Robotic"],
                    help=SINGLE_QUOTE_HELP["automation_level"],
                )
                stations_count = st.number_input(
                    "Stations count",
                    min_value=0,
                    step=1,
                    help=SINGLE_QUOTE_HELP["stations_count"],
                )
                robot_count = st.number_input(
                    "Robot count",
                    min_value=0,
                    step=1,
                    help=SINGLE_QUOTE_HELP["robot_count"],
                )
                complexity_score_1_5 = st.slider(
                    "Overall complexity (1–5)",
                    1,
                    5,
                    3,
                    help=SINGLE_QUOTE_HELP["complexity_score_1_5"],
                )
                custom_pct = st.slider(
                    "Custom %",
                    0,
                    100,
                    50,
                    help=SINGLE_QUOTE_HELP["custom_pct"],
                )
                has_controls = st.checkbox(
                    "Includes controls work?",
                    value=True,
                    help=SINGLE_QUOTE_HELP["has_controls"],
                )
                has_robotics = st.checkbox(
                    "Includes robotics work?",
                    value=True,
                    help=SINGLE_QUOTE_HELP["has_robotics"],
                )
                retrofit = st.checkbox(
                    "Retrofit project?", help=SINGLE_QUOTE_HELP["retrofit"]
                )

                with st.expander(
                    "Optional inputs (only if known)", expanded=False
                ):
                    plc_family = st.text_input(
                        "PLC family",
                        "AB Compact Logix",
                        help=SINGLE_QUOTE_HELP["plc_family"],
                    )
                    hmi_family = st.text_input(
                        "HMI family",
                        "AB PanelView Plus",
                        help=SINGLE_QUOTE_HELP["hmi_family"],
                    )
                    vision_type = st.text_input(
                        "Vision type",
                        "None",
                        help=SINGLE_QUOTE_HELP["vision_type"],
                    )
                    fixture_sets = st.number_input(
                        "Fixture sets",
                        min_value=0,
                        step=1,
                        help=SINGLE_QUOTE_HELP["fixture_sets"],
                    )
                    part_types = st.number_input(
                        "Part types",
                        min_value=0,
                        step=1,
                        help=SINGLE_QUOTE_HELP["part_types"],
                    )
                    servo_axes = st.number_input(
                        "Servo axes",
                        min_value=0,
                        step=1,
                        help=SINGLE_QUOTE_HELP["servo_axes"],
                    )
                    pneumatic_devices = st.number_input(
                        "Pneumatic devices",
                        min_value=0,
                        step=1,
                        help=SINGLE_QUOTE_HELP["pneumatic_devices"],
                    )
                    safety_doors = st.number_input(
                        "Safety doors",
                        min_value=0,
                        step=1,
                        help=SINGLE_QUOTE_HELP["safety_doors"],
                    )
                    weldment_perimeter_ft = st.number_input(
                        "Weldment perimeter (ft)",
                        min_value=0.0,
                        help=SINGLE_QUOTE_HELP["weldment_perimeter_ft"],
                    )
                    fence_length_ft = st.number_input(
                        "Fence length (ft)",
                        min_value=0.0,
                        help=SINGLE_QUOTE_HELP["fence_length_ft"],
                    )
                    conveyor_length_ft = st.number_input(
                        "Conveyor length (ft)",
                        min_value=0.0,
                        help=SINGLE_QUOTE_HELP["conveyor_length_ft"],
                    )
                    product_familiarity_score = st.slider(
                        "Product familiarity (1–5)",
                        1,
                        5,
                        3,
                        help=SINGLE_QUOTE_HELP["product_familiarity_score"],
                    )
                    product_rigidity = st.slider(
                        "Product rigidity (1–5)",
                        1,
                        5,
                        3,
                        help=SINGLE_QUOTE_HELP["product_rigidity"],
                    )
                    is_product_deformable = st.checkbox(
                        "Product deformable?",
                        help=SINGLE_QUOTE_HELP["is_product_deformable"],
                    )
                    is_bulk_product = st.checkbox(
                        "Bulk product?",
                        help=SINGLE_QUOTE_HELP["is_bulk_product"],
                    )
                    bulk_rigidity_score = st.slider(
                        "Bulk rigidity score (1–5)",
                        1,
                        5,
                        3,
                        help=SINGLE_QUOTE_HELP["bulk_rigidity_score"],
                    )
                    has_tricky_packaging = st.checkbox(
                        "Tricky packaging?",
                        help=SINGLE_QUOTE_HELP["has_tricky_packaging"],
                    )
                    process_uncertainty_score = st.slider(
                        "Process uncertainty (1–5)",
                        1,
                        5,
                        3,
                        help=SINGLE_QUOTE_HELP["process_uncertainty_score"],
                    )
                    changeover_time_min = st.number_input(
                        "Changeover time (min)",
                        min_value=0.0,
                        help=SINGLE_QUOTE_HELP["changeover_time_min"],
                    )
                    safety_devices_count = st.number_input(
                        "Safety devices count",
                        min_value=0,
                        step=1,
                        help=SINGLE_QUOTE_HELP["safety_devices_count"],
                    )
                    duplicate = st.checkbox(
                        "Duplicate of prior project?",
                        help=SINGLE_QUOTE_HELP["duplicate"],
                    )
                    vision_systems_count = st.number_input(
                        "Vision systems count",
                        min_value=0,
                        step=1,
                        help=SINGLE_QUOTE_HELP["vision_systems_count"],
                    )
                    panel_count = st.number_input(
                        "Panel count",
                        min_value=0,
                        step=1,
                        help=SINGLE_QUOTE_HELP["panel_count"],
                    )
                    drive_count = st.number_input(
                        "Drive count",
                        min_value=0,
                        step=1,
                        help=SINGLE_QUOTE_HELP["drive_count"],
                    )
                    quoted_materials_cost_usd = st.number_input(
                        "Quoted materials cost ($)",
                        min_value=0.0,
                        help=SINGLE_QUOTE_HELP["quoted_materials_cost_usd"],
                    )

                submitted = st.form_submit_button("Estimate hours")

            log_quoted_materials_cost = math.log1p(quoted_materials_cost_usd)

            if submitted:
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
                    log_quoted_materials_cost=log_quoted_materials_cost,
                )
                pred = predict_quote(q)

                st.subheader("Quote Summary")
                summary_cols = st.columns(3)
                summary_cols[0].metric(
                    "Estimate (hours)", fmt_hours(pred.total_p50)
                )
                summary_cols[1].metric(
                    "Likely range (hours)",
                    fmt_range(pred.total_p10, pred.total_p90),
                )
                summary_cols[2].metric(
                    "Suggested buffer (hours)",
                    fmt_hours(compute_buffer(pred.total_p50, pred.total_p90)),
                )

                st.subheader("Bucket breakdown")
                bucket_summary = build_bucket_summary(pred.ops)
                st.dataframe(bucket_summary, use_container_width=True)

                with st.expander("Detailed breakdown", expanded=False):
                    rows = []
                    for op, op_pred in pred.ops.items():
                        rows.append(
                            {
                                "Operation": op,
                                "Bucket": bucket_for_op(op),
                                "Low": op_pred.p10,
                                "Estimate": op_pred.p50,
                                "High": op_pred.p90,
                                "Trained": op_pred.trained,
                            }
                        )
                    df_out = pd.DataFrame(rows)
                    st.dataframe(df_out, use_container_width=True)


    # Batch Quotes tab
    with tab_batch:
        st.header("Batch estimation via CSV")

        if not st.session_state["models_ready"]:
            st.warning("Models are not trained yet. Go to 'Admin: Upload & Train' first.")
        else:
            mode = st.radio(
                "Export format",
                ["Quote Summary", "Full Detail"],
                index=0,
                horizontal=True,
            )
            st.markdown(
                "Your CSV must include at least these columns: "
                f"`{', '.join(QUOTE_NUM_FEATURES + QUOTE_CAT_FEATURES)}`"
            )

            uploaded = st.file_uploader(
                "Upload quote CSV", type=["csv"], key="batch_uploader"
            )
            if uploaded is not None:
                df_in = pd.read_csv(uploaded)
                st.subheader("Input preview")
                st.dataframe(df_in.head())

                required_cols = set(QUOTE_NUM_FEATURES + QUOTE_CAT_FEATURES)
                missing = [c for c in required_cols if c not in df_in.columns]

                if missing:
                    st.error(f"Missing required columns: {missing}")
                else:
                    if st.button("Run predictions on all rows"):
                        df_out = predict_quotes_df(df_in)
                        if mode == "Quote Summary":
                            df_export = build_batch_summary(df_in, df_out)
                        else:
                            df_export = df_out
                        st.subheader("Output preview")
                        st.dataframe(df_export.head())

                        csv_bytes = df_export.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="Download predictions CSV",
                            data=csv_bytes,
                            file_name="quote_predictions.csv",
                            mime="text/csv",
                        )


    # Admin tab: dataset upload + master merge + retrain
    with tab_admin:
        st.header("Admin: Upload dataset and train models")

        st.markdown(
            "Upload the latest project_hours_dataset.xlsx export. "
            "The app will merge it into a master dataset (dedup by project_id) and retrain models."
        )

        uploaded_file = st.file_uploader(
            "Upload project_hours_dataset.xlsx",
            type=["xlsx", "xls"],
            key="training_uploader",
        )

        if uploaded_file is not None:
            xls = pd.ExcelFile(uploaded_file)
            sheet_name = st.selectbox("Select sheet", xls.sheet_names)
            df_raw = pd.read_excel(xls, sheet_name=sheet_name)

            st.subheader("Dataset preview")
            st.dataframe(df_raw.head())

            missing = [c for c in REQUIRED_TRAINING_COLS if c not in df_raw.columns]
            if missing:
                st.error(f"Missing required columns: {missing}")
            else:
                if st.button("Merge into master & train models"):
                    with st.spinner("Processing upload and training models..."):
                        rows_raw = len(df_raw)

                        # Apply training feature engineering to the new upload
                        df_train = engineer_features_for_training(df_raw)

                        # Keep only rows that have at least one non-zero actual-hours value
                        targets_present = [t for t in TARGETS if t in df_train.columns]
                        if targets_present:
                            hours_mat = (
                                df_train[targets_present]
                                .apply(pd.to_numeric, errors="coerce")
                                .fillna(0)
                            )
                            has_any_hours = hours_mat.gt(0).any(axis=1)
                            df_train = df_train[has_any_hours]

                        rows_train = len(df_train)

                        # If nothing is trainable, log and leave master/models unchanged
                        if rows_train == 0:
                            if os.path.exists(MASTER_DATA_PATH):
                                df_master_existing = pd.read_parquet(MASTER_DATA_PATH)
                                rows_master_total = len(df_master_existing)
                            else:
                                rows_master_total = 0

                            upload_info = {
                                "rows_raw": rows_raw,
                                "rows_train": rows_train,
                                "rows_master_total": rows_master_total,
                            }
                            log_row = pd.DataFrame([upload_info])
                            os.makedirs(os.path.dirname(UPLOADS_LOG_PATH), exist_ok=True)
                            if os.path.exists(UPLOADS_LOG_PATH):
                                df_log_old = pd.read_csv(UPLOADS_LOG_PATH)
                                df_log_new = pd.concat(
                                    [df_log_old, log_row], ignore_index=True
                                )
                            else:
                                df_log_new = log_row
                            df_log_new.to_csv(UPLOADS_LOG_PATH, index=False)

                            st.warning(
                                "Upload contained no rows with non-zero actual hours. "
                                "Master dataset and models were left unchanged."
                            )
                        else:
                            os.makedirs(os.path.dirname(MASTER_DATA_PATH), exist_ok=True)

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

                            upload_info = {
                                "rows_raw": rows_raw,
                                "rows_train": rows_train,
                                "rows_master_total": rows_master_total,
                            }
                            log_row = pd.DataFrame([upload_info])
                            os.makedirs(os.path.dirname(UPLOADS_LOG_PATH), exist_ok=True)
                            if os.path.exists(UPLOADS_LOG_PATH):
                                df_log_old = pd.read_csv(UPLOADS_LOG_PATH)
                                df_log_new = pd.concat(
                                    [df_log_old, log_row], ignore_index=True
                                )
                            else:
                                df_log_new = log_row
                            df_log_new.to_csv(UPLOADS_LOG_PATH, index=False)

                            metrics_all = []
                            for target in TARGETS:
                                m = train_one_op(
                                    df_master_new,
                                    target,
                                    models_dir="models",
                                    version="v1",
                                )
                                if m is None:
                                    m = {
                                        "target": target,
                                        "version": "v1",
                                        "trained": False,
                                        "rows": 0,
                                        "mae": pd.NA,
                                        "coverage": pd.NA,
                                        "interval_width": pd.NA,
                                        "qhat": pd.NA,
                                        "alpha": pd.NA,
                                        "r2": pd.NA,
                                    }
                                metrics_all.append(m)

                            if metrics_all:
                                metrics_df = pd.DataFrame(metrics_all)
                                metrics_df = metrics_df.reindex(
                                    columns=METRICS_COLUMNS
                                )
                                os.makedirs("models", exist_ok=True)
                                metrics_df.to_csv(METRICS_PATH, index=False)
                                st.session_state["models_ready"] = True

                                st.success(
                                    "Master dataset updated and models trained. "
                                    "Quoting tabs now use the latest models. "
                                    "The app will refresh to show trained status."
                                )
                                st.subheader("Model metrics")
                                st.dataframe(metrics_df)

                                csv_bytes = metrics_df.to_csv(index=False).encode(
                                    "utf-8"
                                )
                                st.download_button(
                                    label="Download metrics_summary.csv",
                                    data=csv_bytes,
                                    file_name="metrics_summary.csv",
                                    mime="text/csv",
                                )
                                _rerun_app()
                            else:
                                st.warning(
                                    "No models were trained (not enough data for any operation). "
                                    "Check that actual-hours columns have non-zero values."
                                )
        else:
            st.info("Upload your project_hours_dataset.xlsx to enable training.")

        st.markdown("---")
        st.subheader("Reset app state")
        confirm_reset = st.checkbox(
            "I understand this will delete the master dataset and trained models."
        )

        if not confirm_reset:
            st.info("Check the confirmation box to enable reset.")

        if st.button(
            "Reset master dataset and models",
            disabled=not confirm_reset,
        ):
            paths_to_remove = [
                MASTER_DATA_PATH,
                UPLOADS_LOG_PATH,
                METRICS_PATH,
            ]
            for path in paths_to_remove:
                if os.path.exists(path):
                    os.remove(path)

            for target in TARGETS:
                model_path = os.path.join("models", f"{target}_v1.joblib")
                if os.path.exists(model_path):
                    os.remove(model_path)

            for directory in [
                os.path.dirname(MASTER_DATA_PATH),
                os.path.dirname(UPLOADS_LOG_PATH),
                "models",
            ]:
                if os.path.isdir(directory) and not os.listdir(directory):
                    os.rmdir(directory)

            st.session_state["models_ready"] = False
            st.success("App state reset. Upload a dataset to train models again.")
            _rerun_app()


if __name__ == "__main__":
    main()
