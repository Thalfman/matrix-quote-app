# quote_app.py
# Streamlit UI with:
# - Quote-first experience
# - Batch quoting
# - Advanced tools (optional)
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

DEFAULT_UI = {
    "cq_industry_segment": "General Industry",
    "cq_system_category": "End of Line Automation",
    "cq_automation_level": "Robotic",
    "cq_stations_count": 2,
    "cq_robot_count": 1,
    "cq_complexity_score_1_5": 3,
    "cq_custom_pct": 50,
    "cq_has_controls": True,
    "cq_has_robotics": True,
    "cq_retrofit": False,
    "cq_plc_family": "AB Compact Logix",
    "cq_hmi_family": "AB PanelView Plus",
    "cq_vision_type": "None",
    "cq_fixture_sets": 0,
    "cq_part_types": 0,
    "cq_servo_axes": 0,
    "cq_pneumatic_devices": 0,
    "cq_safety_doors": 0,
    "cq_weldment_perimeter_ft": 0.0,
    "cq_fence_length_ft": 0.0,
    "cq_conveyor_length_ft": 0.0,
    "cq_product_familiarity_score": 3,
    "cq_product_rigidity": 3,
    "cq_is_product_deformable": False,
    "cq_is_bulk_product": False,
    "cq_bulk_rigidity_score": 3,
    "cq_has_tricky_packaging": False,
    "cq_process_uncertainty_score": 3,
    "cq_changeover_time_min": 0.0,
    "cq_safety_devices_count": 0,
    "cq_duplicate": False,
    "cq_vision_systems_count": 0,
    "cq_panel_count": 0,
    "cq_drive_count": 0,
    "cq_quoted_materials_cost_usd": 0.0,
}


def init_ui_state() -> None:
    for key, value in DEFAULT_UI.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if "last_quote_pred" not in st.session_state:
        st.session_state["last_quote_pred"] = None
    if "last_quote_input" not in st.session_state:
        st.session_state["last_quote_input"] = None
    if "show_advanced_tools" not in st.session_state:
        st.session_state["show_advanced_tools"] = False
    if "hourly_rate" not in st.session_state:
        st.session_state["hourly_rate"] = 150


def reset_quote_ui() -> None:
    for key, value in DEFAULT_UI.items():
        st.session_state[key] = value
    st.session_state["last_quote_pred"] = None
    st.session_state["last_quote_input"] = None
    _rerun_app()


def fmt_hours(x) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{x:.1f} h"


def fmt_currency(x) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"${x:,.0f}"


def fmt_range(lo, hi, suffix=" h") -> str:
    if lo is None or hi is None or pd.isna(lo) or pd.isna(hi):
        return "—"
    return f"{lo:.1f}–{hi:.1f}{suffix}"


def fmt_cost_range(lo, hi) -> str:
    if lo is None or hi is None or pd.isna(lo) or pd.isna(hi):
        return "—"
    return f"${lo:,.0f}–${hi:,.0f}"


def compute_buffer(est, hi) -> float:
    if est is None or hi is None or pd.isna(est) or pd.isna(hi):
        return float("nan")
    return hi - est


def bucket_for_op(op_name: str) -> str:
    op_prefix = "".join(ch for ch in op_name.lower() if ch.isalpha())
    return BUCKET_MAP.get(op_prefix, "Other")


def _pred_value(op_pred, field: str):
    if isinstance(op_pred, dict):
        return op_pred.get(field)
    return getattr(op_pred, field)


def build_bucket_summary(pred_ops: dict) -> pd.DataFrame:
    rows = []
    for op_name, op_pred in pred_ops.items():
        rows.append(
            {
                "Bucket": bucket_for_op(op_name),
                "Low": _pred_value(op_pred, "p10"),
                "Estimate": _pred_value(op_pred, "p50"),
                "High": _pred_value(op_pred, "p90"),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["Bucket", "Low", "Estimate", "High", "Buffer"])
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


def ensure_quote_columns(df_in: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = df_in.copy()
    missing = []

    for col in QUOTE_NUM_FEATURES:
        if col not in df.columns:
            df[col] = 0
            missing.append(col)

    for col in QUOTE_CAT_FEATURES:
        if col not in df.columns:
            df[col] = "Unknown"
            missing.append(col)

    return df, missing


def build_quote_summary_export(
    quote_input: dict,
    pred: dict,
    hourly_rate: float,
) -> pd.DataFrame:
    summary = dict(quote_input)
    summary.update(
        {
            "hourly_rate": hourly_rate,
            "hours_estimate": pred.get("total_p50"),
            "hours_low": pred.get("total_p10"),
            "hours_high": pred.get("total_p90"),
        }
    )
    summary["hours_contingency"] = summary["hours_high"] - summary["hours_estimate"]
    summary["cost_estimate"] = summary["hours_estimate"] * hourly_rate
    summary["cost_low"] = summary["hours_low"] * hourly_rate
    summary["cost_high"] = summary["hours_high"] * hourly_rate
    summary["cost_contingency"] = summary["hours_contingency"] * hourly_rate
    return pd.DataFrame([summary])


def build_quote_details_export(pred: dict, hourly_rate: float) -> pd.DataFrame:
    rows = []
    for op, op_pred in pred["ops"].items():
        rows.append(
            {
                "work_type": op,
                "bucket": bucket_for_op(op),
                "hours_low": _pred_value(op_pred, "p10"),
                "hours_estimate": _pred_value(op_pred, "p50"),
                "hours_high": _pred_value(op_pred, "p90"),
                "trained": _pred_value(op_pred, "trained"),
            }
        )
    df = pd.DataFrame(rows)
    df["hours_contingency"] = df["hours_high"] - df["hours_estimate"]
    df["cost_low"] = df["hours_low"] * hourly_rate
    df["cost_estimate"] = df["hours_estimate"] * hourly_rate
    df["cost_high"] = df["hours_high"] * hourly_rate
    df["cost_contingency"] = df["hours_contingency"] * hourly_rate

    totals = {
        "work_type": "TOTAL",
        "bucket": "All",
        "hours_low": pred.get("total_p10"),
        "hours_estimate": pred.get("total_p50"),
        "hours_high": pred.get("total_p90"),
        "trained": "—",
    }
    totals["hours_contingency"] = totals["hours_high"] - totals["hours_estimate"]
    totals["cost_low"] = totals["hours_low"] * hourly_rate
    totals["cost_estimate"] = totals["hours_estimate"] * hourly_rate
    totals["cost_high"] = totals["hours_high"] * hourly_rate
    totals["cost_contingency"] = totals["hours_contingency"] * hourly_rate

    return pd.concat([df, pd.DataFrame([totals])], ignore_index=True)


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


def render_overview():
    st.header("Status")

    df_master = _load_master()
    metrics_df = _load_metrics()

    col1, col2, col3 = st.columns(3)

    with col1:
        if df_master is not None:
            st.metric("Projects in history", f"{len(df_master)}")
        else:
            st.metric("Projects in history", "0")

    with col2:
        if metrics_df is not None and not metrics_df.empty:
            if "trained" in metrics_df.columns:
                trained_ops = metrics_df.loc[
                    metrics_df["trained"].fillna(False), "target"
                ].nunique()
            else:
                trained_ops = metrics_df["target"].nunique()
            st.metric("Work types covered", f"{trained_ops}")
        else:
            st.metric("Work types covered", "0")

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
        with st.expander("Admin details: upload history", expanded=False):
            st.subheader("Upload history")
            if os.path.exists(UPLOADS_LOG_PATH):
                df_log = pd.read_csv(UPLOADS_LOG_PATH)
                st.dataframe(df_log.tail(10))
            else:
                st.info("No uploads logged yet. Use the Admin tab to upload data.")

    with colB:
        with st.expander("Admin details: per-operation metrics", expanded=False):
            st.subheader("Model metrics snapshot")
            if metrics_df is not None and not metrics_df.empty:
                st.dataframe(metrics_df)
            else:
                st.info("No models trained yet. Use the Admin tab after uploading data.")


def render_data_explorer():
    st.header("Data Explorer")

    df_master = _load_master()
    metrics_df = _load_metrics()
    if df_master is None or df_master.empty:
        st.info("Project history is empty. Upload and train in the Admin tab first.")
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
                    "Customer industry",
                    industries,
                    default=industries,
                )
                if industries
                else []
            )

        with col_filters2:
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

        st.subheader(f"Filtered projects: {len(df_filtered)}")
        st.dataframe(df_filtered.head(50))

        st.markdown("---")

        ops_with_data = [t for t in TARGETS if t in df_filtered.columns]
        if ops_with_data:
            op_choice = st.selectbox("Work type", ops_with_data)
            col_charts1, col_charts2 = st.columns(2)

            with col_charts1:
                st.write(f"Histogram of {op_choice}")
                st.bar_chart(df_filtered[op_choice].dropna())

            with col_charts2:
                st.subheader("Operation performance")
                if metrics_df is None or metrics_df.empty:
                    st.info("Train models to see operation performance here.")
                else:
                    op_metrics = metrics_df.loc[metrics_df["target"] == op_choice]
                    if op_metrics.empty:
                        st.info("Train models to see operation performance here.")
                    else:
                        op_row = op_metrics.iloc[0]
                        trained_val = op_row.get("trained", True)
                        trained = "Yes" if bool(trained_val) else "No"
                        rows = op_row.get("rows")
                        mae = op_row.get("mae")
                        coverage = op_row.get("coverage")
                        coverage_pct = (
                            f"{coverage * 100:.0f}%"
                            if coverage is not None and not pd.isna(coverage)
                            else "—"
                        )
                        rows_label = (
                            f"{int(rows)}" if rows is not None and not pd.isna(rows) else "—"
                        )

                        st.write(f"**Trained:** {trained}")
                        st.write(f"**Training examples (rows):** {rows_label}")
                        st.write(f"**{LABEL_TYPICAL_MISS}:** {fmt_hours(mae)}")
                        st.write(f"**{LABEL_RANGE_RELIABILITY}:** {coverage_pct}")
                        if not bool(trained_val):
                            st.caption("Not enough history yet for this operation.")
                        elif rows is not None and not pd.isna(rows) and rows < 15:
                            st.caption("Limited history; expect higher variability.")

                with st.expander("Technical: key drivers", expanded=False):
                    st.subheader("Key drivers")
                    pipe = load_model(op_choice)
                    pre = None
                    model = None
                    stored_importances = None

                    if pipe is None:
                        st.info(
                            "No model artifact loaded for this operation. "
                            "Check the Drivers tab for details."
                        )
                    elif isinstance(pipe, dict) and {"preprocessor", "model_mid"}.issubset(pipe):
                        pre = pipe["preprocessor"]
                        model = pipe["model_mid"]
                        stored_importances = pipe.get("meta", {}).get("feature_importances")
                    elif isinstance(pipe, Pipeline):
                        pre = pipe.named_steps.get("preprocess")
                        model = pipe.named_steps.get("model")
                    else:
                        st.info(
                            "Loaded artifact is not compatible with feature importance."
                        )

                    if model is None or (
                        not hasattr(model, "feature_importances_") and not stored_importances
                    ):
                        st.info("Feature importances are unavailable for this operation.")
                    else:
                        if stored_importances:
                            fi_df = pd.DataFrame(stored_importances)
                        else:
                            try:
                                feature_names = pre.get_feature_names_out()
                            except Exception:
                                feature_names = [
                                    f"f_{i}" for i in range(len(model.feature_importances_))
                                ]

                            importances = model.feature_importances_
                            fi_df = pd.DataFrame(
                                {
                                    "feature": feature_names,
                                    "importance": importances,
                                }
                            )

                        fi_df = (
                            fi_df.sort_values("importance", ascending=False).reset_index(drop=True)
                        )

                        st.write("Top drivers by importance")
                        st.dataframe(fi_df.head(10))
                        st.bar_chart(fi_df.head(10).set_index("feature")["importance"])
        else:
            st.info("No operation hours columns found in project history.")


def render_model_performance():
    st.header("Model Health")

    metrics_df = _load_metrics()
    if metrics_df is None or metrics_df.empty:
        st.info("No models trained yet. Use the Admin tab after uploading data.")
    else:
        st.subheader("Coverage summary")
        if "trained" in metrics_df.columns:
            trained_targets = metrics_df.loc[metrics_df["trained"].fillna(False), "target"].tolist()
            untrained_from_metrics = metrics_df.loc[
                ~metrics_df["trained"].fillna(False), "target"
            ].tolist()
        else:
            trained_targets = metrics_df["target"].tolist()
            untrained_from_metrics = []

        not_trained_ops = [target for target in TARGETS if target not in trained_targets]
        not_trained_ops.extend(
            [target for target in untrained_from_metrics if target not in not_trained_ops]
        )

        st.write(f"Models trained: {len(trained_targets)}")
        if not_trained_ops:
            st.write(
                f"Not trained operations ({len(not_trained_ops)}): " + ", ".join(not_trained_ops)
            )
        else:
            st.write("Not trained operations: 0")

        if "mae" in metrics_df.columns:
            typical_miss = metrics_df["mae"].mean()
            st.write(f"Typical miss (avg hours): {typical_miss:.1f}")

        with st.expander("Technical: metrics and charts", expanded=False):
            st.subheader("Per-operation metrics")
            st.dataframe(metrics_df)

            col_perf1, col_perf2 = st.columns(2)

            with col_perf1:
                if "mae" in metrics_df.columns:
                    st.write("MAE by operation")
                    mae_chart = metrics_df[["target", "mae"]].set_index("target")
                    st.bar_chart(mae_chart)
                else:
                    st.info("MAE metrics not available.")

            with col_perf2:
                if "r2" in metrics_df.columns:
                    st.write("R² by operation")
                    r2_chart = metrics_df[["target", "r2"]].set_index("target")
                    st.bar_chart(r2_chart)
                else:
                    st.info("R² metrics not available.")


def render_drivers():
    st.header("Drivers")

    df_master = _load_master()
    if df_master is None or df_master.empty:
        st.info("Project history is empty. Upload and train in the Admin tab first.")
    else:
        col_dr1, col_dr2 = st.columns(2)

        # Drivers: feature importance by operation
        with col_dr1:
            with st.expander("Technical: model drivers", expanded=False):
                st.subheader("Global drivers by operation")

                modeled_ops = [
                    t
                    for t in TARGETS
                    if os.path.exists(os.path.join("models", f"{t}_v1.joblib"))
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
                    stored_importances = None

                    if pipe is None:
                        st.warning("No model artifact loaded for this operation.")
                    elif isinstance(pipe, dict) and {"preprocessor", "model_mid"}.issubset(pipe):
                        pre = pipe["preprocessor"]
                        model = pipe["model_mid"]
                        stored_importances = pipe.get("meta", {}).get("feature_importances")
                    elif isinstance(pipe, Pipeline):
                        pre = pipe.named_steps.get("preprocess")
                        model = pipe.named_steps.get("model")
                    else:
                        st.info(
                            "Loaded artifact is not compatible with feature importance."
                        )

                    if model is None or not hasattr(model, "feature_importances_") and not stored_importances:
                        st.info("Selected model does not expose feature_importances_.")
                    else:
                        if stored_importances:
                            fi_df = pd.DataFrame(stored_importances)
                        else:
                            try:
                                feature_names = pre.get_feature_names_out()
                            except Exception:
                                feature_names = [
                                    f"f_{i}" for i in range(len(model.feature_importances_))
                                ]

                            importances = model.feature_importances_
                            fi_df = pd.DataFrame(
                                {
                                    "feature": feature_names,
                                    "importance": importances,
                                }
                            )

                        fi_df = (
                            fi_df.sort_values("importance", ascending=False).reset_index(drop=True)
                        )

                        st.write("Top 15 features by importance")
                        st.dataframe(fi_df.head(15))
                        st.bar_chart(fi_df.head(15).set_index("feature")["importance"])

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
                "Customer industry",
                options=["(any)"] + industries,
                index=0,
            )
            sel_system = st.selectbox(
                "System category",
                options=["(any)"] + systems,
                index=0,
            )

            min_robots = st.number_input("Minimum robots", min_value=0, value=0, step=1)
            max_robots = st.number_input("Maximum robots", min_value=0, value=10, step=1)

            if st.button("Find similar projects"):
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


def render_create_quote(hourly_rate: float) -> None:
    st.header("Create Quote")

    if not st.session_state["models_ready"]:
        st.warning("Estimates are not available yet. Use the Admin tab to upload data.")
        return

    col_inputs, col_results = st.columns([1.15, 1])

    with col_inputs:
        with st.form("quote_form"):
            st.subheader("Project basics")
            st.selectbox(
                "Customer industry",
                ["Automotive", "Food & Beverage", "General Industry"],
                help=SINGLE_QUOTE_HELP["industry_segment"],
                key="cq_industry_segment",
            )
            st.selectbox(
                "System category",
                ["End of Line Automation", "Machine Tending", "Other"],
                help=SINGLE_QUOTE_HELP["system_category"],
                key="cq_system_category",
            )
            st.selectbox(
                "Automation level",
                ["Semi-Automatic", "Robotic"],
                help=SINGLE_QUOTE_HELP["automation_level"],
                key="cq_automation_level",
            )
            st.checkbox(
                "Retrofit (upgrade existing system)",
                help=SINGLE_QUOTE_HELP["retrofit"],
                key="cq_retrofit",
            )

            st.subheader("Scope")
            st.number_input(
                "Stations count",
                min_value=0,
                step=1,
                help=SINGLE_QUOTE_HELP["stations_count"],
                key="cq_stations_count",
            )
            st.number_input(
                "Robot count",
                min_value=0,
                step=1,
                help=SINGLE_QUOTE_HELP["robot_count"],
                key="cq_robot_count",
            )
            st.slider(
                "Overall complexity (1–5)",
                1,
                5,
                help=SINGLE_QUOTE_HELP["complexity_score_1_5"],
                key="cq_complexity_score_1_5",
            )
            st.slider(
                "Custom %",
                0,
                100,
                help=SINGLE_QUOTE_HELP["custom_pct"],
                key="cq_custom_pct",
            )

            st.subheader("Included work")
            st.checkbox(
                "Controls included",
                help=SINGLE_QUOTE_HELP["has_controls"],
                key="cq_has_controls",
            )
            st.checkbox(
                "Robotics included",
                help=SINGLE_QUOTE_HELP["has_robotics"],
                key="cq_has_robotics",
            )

            with st.expander("More details (optional, improves accuracy)", expanded=False):
                st.text_input(
                    "PLC family",
                    help=SINGLE_QUOTE_HELP["plc_family"],
                    key="cq_plc_family",
                )
                st.text_input(
                    "HMI family",
                    help=SINGLE_QUOTE_HELP["hmi_family"],
                    key="cq_hmi_family",
                )
                st.text_input(
                    "Vision type",
                    help=SINGLE_QUOTE_HELP["vision_type"],
                    key="cq_vision_type",
                )
                st.number_input(
                    "Fixture sets",
                    min_value=0,
                    step=1,
                    help=SINGLE_QUOTE_HELP["fixture_sets"],
                    key="cq_fixture_sets",
                )
                st.number_input(
                    "Part types",
                    min_value=0,
                    step=1,
                    help=SINGLE_QUOTE_HELP["part_types"],
                    key="cq_part_types",
                )
                st.number_input(
                    "Servo axes",
                    min_value=0,
                    step=1,
                    help=SINGLE_QUOTE_HELP["servo_axes"],
                    key="cq_servo_axes",
                )
                st.number_input(
                    "Pneumatic devices",
                    min_value=0,
                    step=1,
                    help=SINGLE_QUOTE_HELP["pneumatic_devices"],
                    key="cq_pneumatic_devices",
                )
                st.number_input(
                    "Safety doors",
                    min_value=0,
                    step=1,
                    help=SINGLE_QUOTE_HELP["safety_doors"],
                    key="cq_safety_doors",
                )
                st.number_input(
                    "Weldment perimeter (ft)",
                    min_value=0.0,
                    help=SINGLE_QUOTE_HELP["weldment_perimeter_ft"],
                    key="cq_weldment_perimeter_ft",
                )
                st.number_input(
                    "Fence length (ft)",
                    min_value=0.0,
                    help=SINGLE_QUOTE_HELP["fence_length_ft"],
                    key="cq_fence_length_ft",
                )
                st.number_input(
                    "Conveyor length (ft)",
                    min_value=0.0,
                    help=SINGLE_QUOTE_HELP["conveyor_length_ft"],
                    key="cq_conveyor_length_ft",
                )
                st.slider(
                    "Product familiarity (1–5)",
                    1,
                    5,
                    help=SINGLE_QUOTE_HELP["product_familiarity_score"],
                    key="cq_product_familiarity_score",
                )
                st.slider(
                    "Product rigidity (1–5)",
                    1,
                    5,
                    help=SINGLE_QUOTE_HELP["product_rigidity"],
                    key="cq_product_rigidity",
                )
                st.checkbox(
                    "Product deformable",
                    help=SINGLE_QUOTE_HELP["is_product_deformable"],
                    key="cq_is_product_deformable",
                )
                st.checkbox(
                    "Bulk product",
                    help=SINGLE_QUOTE_HELP["is_bulk_product"],
                    key="cq_is_bulk_product",
                )
                st.slider(
                    "Bulk rigidity score (1–5)",
                    1,
                    5,
                    help=SINGLE_QUOTE_HELP["bulk_rigidity_score"],
                    key="cq_bulk_rigidity_score",
                )
                st.checkbox(
                    "Tricky packaging",
                    help=SINGLE_QUOTE_HELP["has_tricky_packaging"],
                    key="cq_has_tricky_packaging",
                )
                st.slider(
                    "Process uncertainty (1–5)",
                    1,
                    5,
                    help=SINGLE_QUOTE_HELP["process_uncertainty_score"],
                    key="cq_process_uncertainty_score",
                )
                st.number_input(
                    "Changeover time (min)",
                    min_value=0.0,
                    help=SINGLE_QUOTE_HELP["changeover_time_min"],
                    key="cq_changeover_time_min",
                )
                st.number_input(
                    "Safety devices count",
                    min_value=0,
                    step=1,
                    help=SINGLE_QUOTE_HELP["safety_devices_count"],
                    key="cq_safety_devices_count",
                )
                st.checkbox(
                    "Duplicate of prior project",
                    help=SINGLE_QUOTE_HELP["duplicate"],
                    key="cq_duplicate",
                )
                st.number_input(
                    "Vision systems count",
                    min_value=0,
                    step=1,
                    help=SINGLE_QUOTE_HELP["vision_systems_count"],
                    key="cq_vision_systems_count",
                )
                st.number_input(
                    "Panel count",
                    min_value=0,
                    step=1,
                    help=SINGLE_QUOTE_HELP["panel_count"],
                    key="cq_panel_count",
                )
                st.number_input(
                    "Drive count",
                    min_value=0,
                    step=1,
                    help=SINGLE_QUOTE_HELP["drive_count"],
                    key="cq_drive_count",
                )
                st.number_input(
                    "Quoted materials cost ($)",
                    min_value=0.0,
                    help=SINGLE_QUOTE_HELP["quoted_materials_cost_usd"],
                    key="cq_quoted_materials_cost_usd",
                )

            col_submit, col_reset = st.columns(2)
            with col_submit:
                submitted = st.form_submit_button("Generate quote", type="primary")
            with col_reset:
                st.form_submit_button("Reset inputs", on_click=reset_quote_ui)

    if submitted:
        log_quoted_materials_cost = math.log1p(
            st.session_state["cq_quoted_materials_cost_usd"]
        )
        q = QuoteInput(
            industry_segment=st.session_state["cq_industry_segment"],
            system_category=st.session_state["cq_system_category"],
            automation_level=st.session_state["cq_automation_level"],
            plc_family=st.session_state["cq_plc_family"],
            hmi_family=st.session_state["cq_hmi_family"],
            vision_type=st.session_state["cq_vision_type"],
            stations_count=st.session_state["cq_stations_count"],
            robot_count=st.session_state["cq_robot_count"],
            fixture_sets=st.session_state["cq_fixture_sets"],
            part_types=st.session_state["cq_part_types"],
            servo_axes=st.session_state["cq_servo_axes"],
            pneumatic_devices=st.session_state["cq_pneumatic_devices"],
            safety_doors=st.session_state["cq_safety_doors"],
            weldment_perimeter_ft=st.session_state["cq_weldment_perimeter_ft"],
            fence_length_ft=st.session_state["cq_fence_length_ft"],
            conveyor_length_ft=st.session_state["cq_conveyor_length_ft"],
            product_familiarity_score=st.session_state["cq_product_familiarity_score"],
            product_rigidity=st.session_state["cq_product_rigidity"],
            is_product_deformable=int(st.session_state["cq_is_product_deformable"]),
            is_bulk_product=int(st.session_state["cq_is_bulk_product"]),
            bulk_rigidity_score=st.session_state["cq_bulk_rigidity_score"],
            has_tricky_packaging=int(st.session_state["cq_has_tricky_packaging"]),
            process_uncertainty_score=st.session_state["cq_process_uncertainty_score"],
            changeover_time_min=st.session_state["cq_changeover_time_min"],
            safety_devices_count=st.session_state["cq_safety_devices_count"],
            custom_pct=st.session_state["cq_custom_pct"],
            duplicate=int(st.session_state["cq_duplicate"]),
            has_controls=int(st.session_state["cq_has_controls"]),
            has_robotics=int(st.session_state["cq_has_robotics"]),
            Retrofit=int(st.session_state["cq_retrofit"]),
            complexity_score_1_5=st.session_state["cq_complexity_score_1_5"],
            vision_systems_count=st.session_state["cq_vision_systems_count"],
            panel_count=st.session_state["cq_panel_count"],
            drive_count=st.session_state["cq_drive_count"],
            log_quoted_materials_cost=log_quoted_materials_cost,
        )
        pred = predict_quote(q)
        st.session_state["last_quote_input"] = q.dict()
        st.session_state["last_quote_pred"] = {
            "total_p50": pred.total_p50,
            "total_p10": pred.total_p10,
            "total_p90": pred.total_p90,
            "ops": {
                op: {
                    "p50": op_pred.p50,
                    "p10": op_pred.p10,
                    "p90": op_pred.p90,
                    "trained": op_pred.trained,
                }
                for op, op_pred in pred.ops.items()
            },
        }

    with col_results:
        st.subheader("Quote summary")
        pred = st.session_state.get("last_quote_pred")
        quote_input = st.session_state.get("last_quote_input")

        if not pred:
            st.info("Generate a quote to see results.")
            return

        hours_est = pred.get("total_p50")
        hours_low = pred.get("total_p10")
        hours_high = pred.get("total_p90")
        hours_buffer = compute_buffer(hours_est, hours_high)

        cost_est = hours_est * hourly_rate
        cost_low = hours_low * hourly_rate
        cost_high = hours_high * hourly_rate
        cost_buffer = hours_buffer * hourly_rate

        summary_cols = st.columns(2)
        summary_cols[0].metric("Estimated hours", fmt_hours(hours_est))
        summary_cols[0].metric("Expected range", fmt_range(hours_low, hours_high))
        summary_cols[0].metric("Recommended contingency", fmt_hours(hours_buffer))
        summary_cols[1].metric("Estimated cost", fmt_currency(cost_est))
        summary_cols[1].metric("Expected cost range", fmt_cost_range(cost_low, cost_high))
        summary_cols[1].metric("Recommended contingency", fmt_currency(cost_buffer))

        pred_ops = pred.get("ops", {})
        trained_count = sum(1 for op in pred_ops.values() if _pred_value(op, "trained"))
        total_ops = len(pred_ops)
        coverage_ratio = trained_count / total_ops if total_ops else 0
        if total_ops == 0:
            quality = "Low"
        elif coverage_ratio == 1:
            quality = "High"
        elif coverage_ratio >= 0.75:
            quality = "Medium"
        else:
            quality = "Low"
        st.metric("Estimate quality", quality)
        if trained_count < total_ops:
            st.warning("Some work types aren’t covered yet; totals may be low.")

        st.subheader("Hours by work type")
        bucket_summary = build_bucket_summary(pred_ops)
        bucket_summary = bucket_summary.rename(
            columns={
                "Bucket": "Work type",
                "Low": "Low",
                "Estimate": "Estimate",
                "High": "High",
                "Buffer": "Contingency",
            }
        )
        st.dataframe(bucket_summary, use_container_width=True)

        with st.expander("Line-item details (advanced)", expanded=False):
            rows = []
            for op, op_pred in pred_ops.items():
                rows.append(
                    {
                        "Work type": op,
                        "Bucket": bucket_for_op(op),
                        "Low": _pred_value(op_pred, "p10"),
                        "Estimate": _pred_value(op_pred, "p50"),
                        "High": _pred_value(op_pred, "p90"),
                        "Trained": _pred_value(op_pred, "trained"),
                    }
                )
            df_out = pd.DataFrame(rows)
            st.dataframe(df_out, use_container_width=True)

        st.subheader("Next actions")
        if quote_input:
            summary_df = build_quote_summary_export(quote_input, pred, hourly_rate)
            details_df = build_quote_details_export(pred, hourly_rate)

            st.download_button(
                label="Download quote summary CSV",
                data=summary_df.to_csv(index=False).encode("utf-8"),
                file_name="quote_summary.csv",
                mime="text/csv",
            )
            st.download_button(
                label="Download full details CSV",
                data=details_df.to_csv(index=False).encode("utf-8"),
                file_name="quote_details.csv",
                mime="text/csv",
            )

        with st.expander("Similar past projects", expanded=False):
            df_master = _load_master()
            if df_master is None or df_master.empty:
                st.info("Project history is empty. Upload data to compare.")
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

                current_industry = st.session_state["cq_industry_segment"]
                current_system = st.session_state["cq_system_category"]
                current_robot_count = st.session_state["cq_robot_count"]

                industry_options = ["(any)"] + industries
                system_options = ["(any)"] + systems

                industry_index = (
                    industry_options.index(current_industry)
                    if current_industry in industry_options
                    else 0
                )
                system_index = (
                    system_options.index(current_system)
                    if current_system in system_options
                    else 0
                )

                sel_industry = st.selectbox(
                    "Customer industry",
                    options=industry_options,
                    index=industry_index,
                )
                sel_system = st.selectbox(
                    "System category",
                    options=system_options,
                    index=system_index,
                )

                min_default = max(0, current_robot_count - 1)
                max_default = max(current_robot_count + 1, min_default)
                min_robots = st.number_input(
                    "Minimum robots",
                    min_value=0,
                    value=min_default,
                    step=1,
                )
                max_robots = st.number_input(
                    "Maximum robots",
                    min_value=0,
                    value=max_default,
                    step=1,
                )

                if st.button("Find similar projects"):
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


def render_batch_quotes(hourly_rate: float) -> None:
    st.header("Batch Quotes (CSV)")

    if not st.session_state["models_ready"]:
        st.warning("Estimates are not available yet. Use the Admin tab to upload data.")
        return

    st.subheader("Step 1: Download template CSV")
    template_columns = (
        QUOTE_CAT_FEATURES
        + [
            "stations_count",
            "robot_count",
            "complexity_score_1_5",
            "custom_pct",
            "has_controls",
            "has_robotics",
            "Retrofit",
            "fixture_sets",
            "part_types",
            "servo_axes",
            "pneumatic_devices",
            "safety_doors",
            "weldment_perimeter_ft",
            "fence_length_ft",
            "conveyor_length_ft",
            "product_familiarity_score",
            "product_rigidity",
            "bulk_rigidity_score",
            "process_uncertainty_score",
            "changeover_time_min",
            "safety_devices_count",
            "vision_systems_count",
            "panel_count",
            "drive_count",
        ]
    )
    template_example = {col: "" for col in template_columns}
    template_example.update(
        {
            "industry_segment": "General Industry",
            "system_category": "End of Line Automation",
            "automation_level": "Robotic",
            "plc_family": "AB Compact Logix",
            "hmi_family": "AB PanelView Plus",
            "vision_type": "None",
            "stations_count": 2,
            "robot_count": 1,
            "complexity_score_1_5": 3,
            "custom_pct": 50,
            "has_controls": 1,
            "has_robotics": 1,
            "Retrofit": 0,
        }
    )
    template_df = pd.DataFrame([template_example])
    st.download_button(
        label="Download template CSV",
        data=template_df.to_csv(index=False).encode("utf-8"),
        file_name="quote_template.csv",
        mime="text/csv",
    )

    st.subheader("Step 2: Upload CSV")
    uploaded = st.file_uploader("Upload quote CSV", type=["csv"], key="batch_uploader")

    if uploaded is None:
        return

    df_in = pd.read_csv(uploaded)
    st.subheader("Input preview")
    st.dataframe(df_in.head())

    df_filled, missing = ensure_quote_columns(df_in)
    if missing:
        st.warning(
            "Missing columns were auto-filled with defaults: " + ", ".join(missing)
        )

    st.subheader("Step 3: Generate batch quotes")
    mode = st.radio(
        "Export format",
        ["Summary", "Full detail"],
        index=0,
        horizontal=True,
    )

    if st.button("Generate batch quotes"):
        df_out = predict_quotes_df(df_filled)
        if mode == "Summary":
            df_export = build_batch_summary(df_filled, df_out)
        else:
            df_export = df_out

        if {"total_p10", "total_p50", "total_p90"}.issubset(df_export.columns):
            df_export["total_cost_low"] = df_export["total_p10"] * hourly_rate
            df_export["total_cost_estimate"] = df_export["total_p50"] * hourly_rate
            df_export["total_cost_high"] = df_export["total_p90"] * hourly_rate

        st.subheader("Output preview")
        st.dataframe(df_export.head())

        csv_bytes = df_export.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download results CSV",
            data=csv_bytes,
            file_name="quote_predictions.csv",
            mime="text/csv",
        )


def render_admin():
    st.header("Admin: Upload project history & refresh estimates")

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
                            df_log_new = pd.concat([df_log_old, log_row], ignore_index=True)
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
                            df_all = pd.concat([df_master_old, df_train], ignore_index=True)
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
                            df_log_new = pd.concat([df_log_old, log_row], ignore_index=True)
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
                            metrics_df = metrics_df.reindex(columns=METRICS_COLUMNS)
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

                            csv_bytes = metrics_df.to_csv(index=False).encode("utf-8")
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
    st.subheader("Danger zone: delete history & estimates")
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


def main():
    st.set_page_config(page_title="Matrix Quote App", layout="wide")
    st.title("Matrix Quote App")

    init_ui_state()

    if not st.session_state.get("models_ready", False):
        st.session_state["models_ready"] = _models_ready_from_disk()

    st.sidebar.toggle(
        "Show advanced tools",
        value=st.session_state["show_advanced_tools"],
        help="Admins only.",
        key="show_advanced_tools",
    )
    st.sidebar.number_input(
        "Hourly rate ($/hr)",
        min_value=0,
        step=5,
        value=st.session_state["hourly_rate"],
        key="hourly_rate",
    )

    tabs = ["Create Quote", "Batch Quotes (CSV)"]
    if st.session_state["show_advanced_tools"]:
        tabs += ["Advanced", "Admin"]

    tabs_rendered = st.tabs(tabs)

    tab_create = tabs_rendered[0]
    tab_batch = tabs_rendered[1]

    with tab_create:
        render_create_quote(st.session_state["hourly_rate"])

    with tab_batch:
        render_batch_quotes(st.session_state["hourly_rate"])

    if st.session_state["show_advanced_tools"]:
        tab_advanced = tabs_rendered[2]
        tab_admin = tabs_rendered[3]

        with tab_advanced:
            adv_tabs = st.tabs(["Status", "Data Explorer", "Model Health", "Drivers"])
            with adv_tabs[0]:
                render_overview()
            with adv_tabs[1]:
                render_data_explorer()
            with adv_tabs[2]:
                render_model_performance()
            with adv_tabs[3]:
                render_drivers()

        with tab_admin:
            render_admin()


if __name__ == "__main__":
    main()
