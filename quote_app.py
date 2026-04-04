# quote_app.py
# Main Streamlit application for the Matrix Quote App.
# Provides a 7-tab UI for uploading project data, training ML models,
# and generating single or batch hour estimates using CatBoost CQR models.
#
# Tabs:
#   1. Overview        – KPI dashboard (project count, trained models, avg MAE)
#   2. Data Explorer   – Filter and visualize the master training dataset
#   3. Model Performance – Per-operation metrics (MAE, R², coverage)
#   4. Drivers & Similar – Feature importance charts + similar project finder
#   5. Single Quote    – Interactive form to quote one project at a time
#   6. Batch Quotes    – Upload CSV/Excel to quote many projects at once
#   7. Admin           – Upload training data, retrain models, reset state

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
    CONFIDENCE_LEVELS,
    DEFAULT_CONFIDENCE,
)
from core.schemas import QuoteInput
from core.features import engineer_features_for_training
from core.models import (
    CatBoostCQRBundle,
    get_feature_importance,
    load_model,
    train_one_op,
)
from service.predict_lib import predict_quote, predict_quotes_df

# ── File paths for persistent app state ──
# All of these are git-ignored and created at runtime.
MASTER_DATA_PATH = os.path.join("data", "master", "projects_master.parquet")  # combined training data
UPLOADS_LOG_PATH = os.path.join("data", "master", "uploads_log.csv")          # log of each upload
METRICS_PATH = os.path.join("models", "metrics_summary.csv")                  # per-op training metrics
VERSION_PATH = os.path.join("models", "current_version.txt")                  # current model version tag


def _current_model_version() -> str:
    """Read the current model version string, defaulting to 'v1'."""
    if os.path.exists(VERSION_PATH):
        return open(VERSION_PATH).read().strip()
    return "v1"


def _next_model_version() -> str:
    """Increment and persist the model version counter."""
    current = _current_model_version()
    num = int(current.lstrip("v")) if current.startswith("v") and current[1:].isdigit() else 0
    new_version = f"v{num + 1}"
    os.makedirs(os.path.dirname(VERSION_PATH), exist_ok=True)
    with open(VERSION_PATH, "w") as f:
        f.write(new_version)
    return new_version

# ── Page config and session bootstrap ──
st.set_page_config(page_title="Matrix Quote App", layout="wide")
st.title("Matrix Quote App")

# Track whether trained models are available. This gates the Single/Batch quote tabs.
if "models_ready" not in st.session_state:
    st.session_state["models_ready"] = False

# On a fresh browser session, check if models were previously trained and still on disk.
# This lets users refresh the page without needing to retrain.
if not st.session_state["models_ready"]:
    if os.path.exists(METRICS_PATH):
        try:
            _metrics = pd.read_csv(METRICS_PATH)
            if not _metrics.empty:
                st.session_state["models_ready"] = True
        except Exception:
            pass

# ── Create the 7-tab layout ──
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


def _reset_app_state():
    """Delete master dataset, upload log, and model artifacts; reset models_ready."""
    # Remove master dataset, upload log, metrics, and version file if present
    for path in [MASTER_DATA_PATH, UPLOADS_LOG_PATH, METRICS_PATH, VERSION_PATH]:
        if os.path.exists(path):
            os.remove(path)

    # Remove joblib model files in models/
    models_dir = "models"
    if os.path.exists(models_dir) and os.path.isdir(models_dir):
        for fname in os.listdir(models_dir):
            if fname.endswith(".joblib"):
                try:
                    os.remove(os.path.join(models_dir, fname))
                except OSError:
                    # If a file can't be removed, just skip it
                    pass

    # Mark models as not ready so Single/Batch tabs show the correct warning
    st.session_state["models_ready"] = False


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: Overview — high-level KPI dashboard
# Shows three metrics cards (project count, trained ops, average MAE) and
# two side-by-side panels for upload history and the latest metrics snapshot.
# ══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    st.header("Overview")

    df_master = _load_master()
    metrics_df = _load_metrics()

    # Display three KPI metric cards across the top of the tab
    col1, col2, col3 = st.columns(3)

    with col1:
        if df_master is not None:
            st.metric("Projects in master dataset", f"{len(df_master)}")
        else:
            st.metric("Projects in master dataset", "0")

    with col2:
        if metrics_df is not None and not metrics_df.empty:
            trained_ops = metrics_df["target"].nunique()
            st.metric("Operations with models", f"{trained_ops}")
        else:
            st.metric("Operations with models", "0")

    with col3:
        if metrics_df is not None and not metrics_df.empty:
            avg_mae = metrics_df["mae"].mean()
            st.metric("Average MAE (hours)", f"{avg_mae:.1f}")
        else:
            st.metric("Average MAE (hours)", "N/A")

    st.markdown("---")

    # Two-column layout: upload log on the left, metrics snapshot on the right
    colA, colB = st.columns(2)

    with colA:
        st.subheader("Upload history")
        if os.path.exists(UPLOADS_LOG_PATH):
            df_log = pd.read_csv(UPLOADS_LOG_PATH)
            st.dataframe(df_log.tail(10))
        else:
            st.info("No uploads logged yet. Use the Admin tab to upload data.")

    with colB:
        st.subheader("Model metrics snapshot")
        if metrics_df is not None and not metrics_df.empty:
            st.dataframe(metrics_df)
        else:
            st.info("No models trained yet. Use the Admin tab after uploading data.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: Data Explorer — browse and filter the master training dataset.
# Users can filter by industry/system, view a preview table, and see
# per-operation bar charts and scatter plots.
# ══════════════════════════════════════════════════════════════════════════════
with tab_data:
    st.header("Data Explorer")

    df_master = _load_master()
    if df_master is None or df_master.empty:
        st.info("Master dataset is empty. Upload and train in the Admin tab first.")
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
                    "Filter by industry_segment", industries, default=industries
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

        # Apply the selected filters to narrow the dataset
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

        # Per-operation visualizations: let the user pick an operation and see
        # a bar chart of hours by project and a scatter plot vs robot_count
        ops_with_data = [t for t in TARGETS if t in df_filtered.columns]
        if ops_with_data:
            op_choice = st.selectbox("Select operation to explore", ops_with_data)
            col_charts1, col_charts2 = st.columns(2)

            with col_charts1:
                st.write(f"Hours by project for {op_choice}")
                if "project_id" in df_filtered.columns:
                    proj_df = df_filtered[["project_id", op_choice]].dropna()
                    proj_df = proj_df.set_index("project_id")
                    st.bar_chart(proj_df[op_choice])
                else:
                    st.info("No project_id column found to label projects.")

            with col_charts2:
                if "robot_count" in df_filtered.columns:
                    st.write(f"robot_count vs {op_choice}")
                    scatter_df = df_filtered[["robot_count", op_choice]].dropna()
                    scatter_df = scatter_df.rename(
                        columns={"robot_count": "robot_count", op_choice: "hours"}
                    )
                    st.scatter_chart(scatter_df, x="robot_count", y="hours")
                else:
                    st.info("No robot_count column found for scatter plot.")
        else:
            st.info("No operation hours columns found in master dataset.")



# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: Model Performance — per-operation accuracy and calibration metrics.
# Shows a raw metrics table, MAE and R² bar charts, and a coverage vs target
# comparison table that helps assess whether the CQR intervals are well-calibrated.
# ══════════════════════════════════════════════════════════════════════════════
with tab_perf:
    st.header("Model Performance")

    metrics_df = _load_metrics()
    if metrics_df is None or metrics_df.empty:
        st.info("No models trained yet. Use the Admin tab after uploading data.")
    else:
        st.subheader("Per-operation metrics")
        st.dataframe(metrics_df)

        col_perf1, col_perf2 = st.columns(2)

        with col_perf1:
            st.write("MAE by operation")
            mae_chart = metrics_df[["target", "mae"]].set_index("target")
            st.bar_chart(mae_chart)

        with col_perf2:
            st.write("R² by operation")
            r2_chart = metrics_df[["target", "r2"]].set_index("target")
            st.bar_chart(r2_chart)

        st.markdown("---")
        # Build a pivot table comparing target confidence (80/90/95%) against the
        # achieved coverage and average interval width for each operation
        st.subheader("Coverage target vs achieved")

        coverage_rows = []
        for _, row in metrics_df.iterrows():
            for conf in CONFIDENCE_LEVELS:
                key = f"coverage_{int(conf * 100)}"
                width_key = f"avg_width_{int(conf * 100)}"
                coverage_rows.append(
                    {
                        "operation": row["target"],
                        "target_confidence": f"{int(conf * 100)}%",
                        "achieved_coverage": row.get(key, float("nan")),
                        "avg_interval_width": row.get(width_key, float("nan")),
                    }
                )

        df_cov = pd.DataFrame(coverage_rows)
        st.dataframe(df_cov)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: Drivers & Similar Projects
# Left column: shows CatBoost feature importance for a selected operation
#   (which inputs most influence the model's hour prediction).
# Right column: filter-based search to find historical projects similar to
#   the one being quoted (by industry, system type, robot count range).
# ══════════════════════════════════════════════════════════════════════════════
with tab_drivers:
    st.header("Drivers & Similar Projects")

    df_master = _load_master()
    if df_master is None or df_master.empty:
        st.info("Master dataset is empty. Upload and train in the Admin tab first.")
    else:
        col_dr1, col_dr2 = st.columns(2)

        # Drivers: feature importance by operation
        with col_dr1:
            st.subheader("Global drivers by operation")

            # Find which operations have trained model files on disk for the current version
            _ver = _current_model_version()
            modeled_ops = [
                t
                for t in TARGETS
                if os.path.exists(os.path.join("models", f"{t}_{_ver}.joblib"))
            ]
            if not modeled_ops:
                st.info("No trained models found in ./models.")
            else:
                target_choice = st.selectbox(
                    "Select operation", modeled_ops, key="drivers_op_select"
                )

                # Load the selected operation's model and extract feature importance scores
                model_obj = load_model(target_choice)
                fi_df = get_feature_importance(model_obj, df_master)
                if fi_df is None:
                    fi_df = pd.DataFrame()
                    st.info("Feature importance is unavailable for the selected model.")

                if not fi_df.empty:
                    st.write("Top 15 features by importance")
                    st.dataframe(fi_df.head(15))
                    st.bar_chart(
                        fi_df.head(15).set_index("feature")["importance"]
                    )
                else:
                    st.info("Feature importance is unavailable for the selected model.")

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

            # Apply filters to find projects matching the user's criteria
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

                # Show a compact table with key identifying columns + the first hours column
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


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: Single Quote — interactive form for quoting one project.
# The form is organized into collapsible expanders by category.
# Derived composite indices (stations_robot_index, etc.) are NOT shown here
# because they are auto-computed by prepare_quote_features() during prediction.
# After the user clicks "Estimate hours", results are shown in two sub-tabs:
#   - Sales view: aggregated by Sales bucket (ME, EE, PM, etc.) with optional
#     comparison against the user's manually quoted hours.
#   - Operations view: raw per-operation predictions.
# ══════════════════════════════════════════════════════════════════════════════
with tab_single:
    st.header("Single quote estimation")

    if not st.session_state["models_ready"]:
        st.warning("Models are not trained yet. Go to 'Admin: Upload & Train' first.")
    else:
        confidence_options = [int(c * 100) for c in CONFIDENCE_LEVELS]
        confidence_pct = st.select_slider(
            "Prediction confidence",
            options=confidence_options,
            value=int(DEFAULT_CONFIDENCE * 100),
            help="Select the calibrated confidence level for interval bounds.",
        )
        confidence_level = confidence_pct / 100.0

        # -- Project basics --
        with st.expander("Project basics", expanded=True):
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                industry_segment = st.selectbox(
                    "Industry segment",
                    ["Automotive", "Food & Beverage", "General Industry"],
                    help="The customer's primary industry.",
                )
                system_category = st.selectbox(
                    "System category",
                    ["Machine Tending", "End of Line Automation", "Robotic Metal Finishing", "Engineered Manufacturing Systems", "Other"],
                    help="Type of automation system being quoted.",
                )
                automation_level = st.selectbox(
                    "Automation level",
                    ["Semi-Automatic", "Robotic", "Hard Automation"],
                    help="Degree of automation in the system.",
                )
            with col_b2:
                plc_family = st.text_input("PLC family", "AB Compact Logix")
                hmi_family = st.text_input("HMI family", "AB PanelView Plus")
                vision_type = st.text_input("Vision type", "None")

        # -- Mechanical specifications --
        with st.expander("Mechanical specifications"):
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                stations_count = st.number_input("Stations count", min_value=0, step=1, help="Number of stations in the system.")
                robot_count = st.number_input("Robot count", min_value=0, step=1, help="Number of robots in the system.")
                fixture_sets = st.number_input("Fixture sets", min_value=0, step=1, help="Number of fixture tooling sets.")
                part_types = st.number_input("Part types", min_value=0, step=1, help="Number of distinct part types handled.")
                servo_axes = st.number_input("Servo axes", min_value=0, step=1, help="Total servo-driven axes.")
            with col_m2:
                pneumatic_devices = st.number_input("Pneumatic devices", min_value=0, step=1, help="Clamps, grippers, cylinders, etc.")
                safety_doors = st.number_input("Safety doors", min_value=0, step=1)
                weldment_perimeter_ft = st.number_input("Weldment perimeter (ft)", min_value=0.0)
                fence_length_ft = st.number_input("Fence length (ft)", min_value=0.0)
                conveyor_length_ft = st.number_input("Conveyor length (ft)", min_value=0.0)

        # -- Controls & electrical --
        with st.expander("Controls & electrical"):
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                vision_systems_count = st.number_input("Vision systems count", min_value=0, step=1)
                panel_count = st.number_input("Panel count", min_value=0, step=1)
                drive_count = st.number_input("Drive count", min_value=0, step=1)
            with col_c2:
                has_controls = st.checkbox("Includes controls work?", value=True)
                has_robotics = st.checkbox("Includes robotics work?", value=True)
                safety_devices_count = st.number_input("Safety devices count", min_value=0, step=1)

        # -- Product & process characteristics --
        with st.expander("Product & process characteristics"):
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                product_familiarity_score = st.slider("Product familiarity (1–5)", 1, 5, 3, help="1 = completely new, 5 = very familiar product.")
                product_rigidity = st.slider("Product rigidity (1–5)", 1, 5, 3, help="1 = very flexible, 5 = rigid.")
                is_product_deformable = st.checkbox("Product deformable?")
                is_bulk_product = st.checkbox("Bulk product?")
                bulk_rigidity_score = st.slider("Bulk rigidity score (1–5)", 1, 5, 3)
            with col_p2:
                has_tricky_packaging = st.checkbox("Tricky packaging?")
                process_uncertainty_score = st.slider("Process uncertainty (1–5)", 1, 5, 3, help="1 = well-understood process, 5 = highly uncertain.")
                changeover_time_min = st.number_input("Changeover time (min)", min_value=0.0)

        # -- Project scope --
        with st.expander("Project scope"):
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                custom_pct = st.slider("Custom %", 0, 100, 50, help="Percentage of custom (non-standard) content.")
                complexity_score_1_5 = st.slider("Overall complexity (1–5)", 1, 5, 3, help="1 = simple, 5 = very complex.")
            with col_s2:
                duplicate = st.checkbox("Duplicate of prior project?")
                retrofit = st.checkbox("Retrofit project?")
                estimated_materials_cost = st.number_input("Estimated materials cost ($)", min_value=0.0)

        # -- Quoted hours comparison (optional) --
        with st.expander("Your quoted hours (optional — for comparison)"):
            st.caption("Enter your manually estimated hours per role to compare against the model's prediction.")
            qh_cols = st.columns(3)
            quoted_hours_by_bucket = {}
            for i, bucket in enumerate(SALES_BUCKETS):
                with qh_cols[i % 3]:
                    val = st.number_input(f"{bucket} hours", min_value=0.0, value=0.0, step=1.0, key=f"qh_{bucket}")
                    if val > 0:
                        quoted_hours_by_bucket[bucket] = val

        # ── Run prediction ──
        # Assemble all form inputs into a QuoteInput object, run all 12 operation
        # models, and display the results in Sales and Operations sub-tabs.
        if st.button("Estimate hours"):

            # Log-transform the materials cost to match the training feature
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
                log_quoted_materials_cost=log_cost,
            )
            # predict_quote runs all 12 CatBoost models and returns per-op + aggregated results
            pred = predict_quote(q, confidence_level=confidence_level)

            # Collect Sales bucket predictions into a list for the summary table
            sales_rows = []
            for bucket in SALES_BUCKETS:
                bucket_pred = pred.sales_buckets.get(bucket)
                if bucket_pred is None:
                    continue
                sales_rows.append(
                    {
                        "Sales bucket": bucket,
                        "estimate_hours": bucket_pred.estimate,
                        "lo_hours": bucket_pred.lo,
                        "hi_hours": bucket_pred.hi,
                        "plus_minus": bucket_pred.plus_minus,
                    }
                )

            has_quoted_hours = bool(quoted_hours_by_bucket)

            # Build the sales summary table. If the user entered their own quoted hours,
            # add comparison columns showing the delta and a status label (Close/Over/Under).
            sales_summary_rows = []

            for row in sales_rows:
                role = row["Sales bucket"]
                estimate = row["estimate_hours"]
                lo = row["lo_hours"]
                hi = row["hi_hours"]
                plus_minus = row["plus_minus"]

                summary_row = {
                    "Role": role,
                    "Recommended hours (estimate)": estimate,
                    "Interval": f"{confidence_pct}% confident between [{lo:.1f}, {hi:.1f}]",
                    "±Hours": plus_minus,
                }

                if has_quoted_hours:
                    quoted_val = quoted_hours_by_bucket.get(role)
                    if quoted_val is not None:
                        delta = quoted_val - estimate
                        threshold = max(0.1 * abs(estimate), 5)
                        if abs(delta) <= threshold:
                            delta_status = "Close"
                        elif delta > 0:
                            delta_status = "Over model"
                        else:
                            delta_status = "Under model"

                        summary_row.update(
                            {
                                "Quoted hours": quoted_val,
                                "Delta (quoted - model)": delta,
                                "Status": delta_status,
                            }
                        )
                sales_summary_rows.append(summary_row)

            # Build the project-level summary metrics (displayed as metric cards at the top)
            total_model_hours = pred.total_estimate
            project_cols = ["Model total (estimate)", f"{confidence_pct}% interval", "± hours"]
            project_values = [
                f"{total_model_hours:.1f} h",
                f"[{pred.total_lo:.1f}, {pred.total_hi:.1f}]",
                f"±{pred.total_plus_minus:.1f} h",
            ]
            project_status = None

            # If the user entered their own quoted hours, add total-level comparison
            if has_quoted_hours:
                total_quoted = sum(
                    v for v in quoted_hours_by_bucket.values() if isinstance(v, (int, float))
                )
                total_delta = total_quoted - total_model_hours
                project_cols.extend(["Quoted total", "Delta (quoted - model)"])
                project_values.extend([f"{total_quoted:.1f} h", f"{total_delta:.1f} h"])

                threshold_total = max(0.1 * abs(total_model_hours), 10)
                if abs(total_delta) <= threshold_total:
                    project_status = "Overall close to model"
                elif total_delta > 0:
                    project_status = "Quoted hours above model"
                else:
                    project_status = "Quoted hours below model"

            # Sort the sales summary by estimated hours (highest first) and optionally
            # append a TOTAL row if the user provided their own quoted hours for comparison
            sales_summary_rows_exist = bool(sales_summary_rows)
            df_sales_summary_sorted = None
            if sales_summary_rows_exist:
                df_sales_summary = pd.DataFrame(sales_summary_rows)
                df_sales_summary_sorted = df_sales_summary.sort_values(
                    "Recommended hours (estimate)", ascending=False
                )

                if has_quoted_hours:
                    total_row = {
                        "Role": "TOTAL",
                        "Recommended hours (estimate)": df_sales_summary["Recommended hours (estimate)"].sum(),
                        "Interval": f"{confidence_pct}% confidence (rollup)",
                        "±Hours": df_sales_summary["±Hours"].sum(),
                        "Quoted hours": df_sales_summary["Quoted hours"].sum(),
                        "Delta (quoted - model)": df_sales_summary["Delta (quoted - model)"].sum(),
                        "Status": "-",
                    }
                    df_sales_summary_sorted = pd.concat(
                        [df_sales_summary_sorted, pd.DataFrame([total_row])], ignore_index=True
                    )

            # Build the per-operation detail table for the Operations sub-tab
            rows = []
            for op, op_pred in pred.ops.items():
                rows.append(
                    {
                        "operation": op,
                        "estimate_hours": op_pred.estimate,
                        "lo_hours": op_pred.lo,
                        "hi_hours": op_pred.hi,
                        "plus_minus_hours": op_pred.plus_minus,
                        "Interval": f"{confidence_pct}% confident between [{op_pred.lo:.1f}, {op_pred.hi:.1f}]",
                        "confidence": f"{confidence_pct}%",
                    }
                )
            df_out = pd.DataFrame(rows)

            # Display results in two sub-tabs: Sales-level rollup and raw Operations detail
            sales_tab, ops_tab = st.tabs(["Sales view", "Operations view"])

            with sales_tab:
                st.subheader("Project summary")
                cols = st.columns(len(project_cols))
                for col, label, val in zip(cols, project_cols, project_values):
                    col.metric(label, val)

                if project_status:
                    st.caption(project_status)
                st.caption(
                    f"{confidence_pct}% confident between [{pred.total_lo:.1f}, {pred.total_hi:.1f}] "
                    f"(±{pred.total_plus_minus:.1f} hours)"
                )

                st.subheader("Sales-level summary")
                if sales_summary_rows_exist and df_sales_summary_sorted is not None:
                    display_cols = [
                        "Role",
                        "Recommended hours (estimate)",
                        "Interval",
                        "±Hours",
                    ]
                    if has_quoted_hours:
                        display_cols += [
                            "Quoted hours",
                            "Delta (quoted - model)",
                            "Status",
                        ]
                    st.dataframe(df_sales_summary_sorted[display_cols])

                    # If quoted hours were provided, show a side-by-side bar chart
                    # comparing model estimates vs the user's manually quoted hours
                    if has_quoted_hours:
                        df_chart = df_sales_summary_sorted[
                            df_sales_summary_sorted["Role"] != "TOTAL"
                        ][["Role", "Recommended hours (estimate)", "Quoted hours"]]
                        if not df_chart.empty:
                            chart_data = df_chart.melt(
                                id_vars="Role",
                                value_vars=["Recommended hours (estimate)", "Quoted hours"],
                                var_name="Source",
                                value_name="Hours",
                            )
                            chart = (
                                alt.Chart(chart_data)
                                .mark_bar()
                                .encode(
                                    x=alt.X("Role:N", sort="-y"),
                                    y=alt.Y("Hours:Q"),
                                    color="Source:N",
                                    column=alt.Column("Source:N", header=alt.Header(title=None)),
                                    tooltip=["Role", "Source", "Hours"],
                                )
                                .resolve_scale(y="shared")
                            )
                            st.altair_chart(chart, use_container_width=True)
                else:
                    st.info("No Sales-level rollup available for this quote.")

            with ops_tab:
                st.subheader("Per-operation predictions")
                st.dataframe(df_out)

            # ── Export single quote results as CSV ──
            # Combines operation-level, bucket-level, and total-level predictions
            # into a single downloadable CSV file.
            st.markdown("---")
            export_rows = []
            for op, op_pred in pred.ops.items():
                export_rows.append({
                    "level": "operation",
                    "name": op,
                    "estimate_hours": op_pred.estimate,
                    "lo_hours": op_pred.lo,
                    "hi_hours": op_pred.hi,
                    "plus_minus_hours": op_pred.plus_minus,
                    "confidence": f"{confidence_pct}%",
                })
            for bucket in SALES_BUCKETS:
                bp = pred.sales_buckets.get(bucket)
                if bp:
                    export_rows.append({
                        "level": "sales_bucket",
                        "name": bucket,
                        "estimate_hours": bp.estimate,
                        "lo_hours": bp.lo,
                        "hi_hours": bp.hi,
                        "plus_minus_hours": bp.plus_minus,
                        "confidence": f"{confidence_pct}%",
                    })
            export_rows.append({
                "level": "total",
                "name": "PROJECT TOTAL",
                "estimate_hours": pred.total_estimate,
                "lo_hours": pred.total_lo,
                "hi_hours": pred.total_hi,
                "plus_minus_hours": pred.total_plus_minus,
                "confidence": f"{confidence_pct}%",
            })
            df_export = pd.DataFrame(export_rows)
            csv_bytes = df_export.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download quote as CSV",
                data=csv_bytes,
                file_name=f"single_quote_{confidence_pct}.csv",
                mime="text/csv",
            )

            st.caption(
                "Note: Sales bucket and total intervals are computed by summing "
                "per-operation bounds. Actual coverage may differ from the stated "
                "confidence level for aggregated totals."
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6: Batch Quotes — upload a CSV/Excel file with multiple projects and
# run predictions on all rows at once. The output DataFrame includes per-op
# predictions, Sales bucket rollups, and project totals as additional columns.
# ══════════════════════════════════════════════════════════════════════════════
with tab_batch:
    st.header("Batch estimation via CSV or Excel")

    if not st.session_state["models_ready"]:
        st.warning("Models are not trained yet. Go to 'Admin: Upload & Train' first.")
    else:
        confidence_options = [int(c * 100) for c in CONFIDENCE_LEVELS]
        confidence_pct_batch = st.select_slider(
            "Prediction confidence for this batch",
            options=confidence_options,
            value=int(DEFAULT_CONFIDENCE * 100),
        )
        confidence_level_batch = confidence_pct_batch / 100.0

        st.markdown(
            "Your file must include at least these columns: "
            f"`{', '.join(QUOTE_NUM_FEATURES + QUOTE_CAT_FEATURES)}`"
        )

        # Parse the uploaded file (CSV or multi-sheet Excel)
        uploaded = st.file_uploader(
            "Upload quote file (CSV or Excel)",
            type=["csv", "xlsx", "xls"],
            key="batch_uploader",
        )
        if uploaded is not None:
            name = uploaded.name.lower()
            if name.endswith(".csv"):
                df_in = pd.read_csv(uploaded)
            else:
                xls = pd.ExcelFile(uploaded)
                sheet_name = st.selectbox(
                    "Select sheet for quote inputs", xls.sheet_names, key="batch_sheet"
                )
                df_in = pd.read_excel(xls, sheet_name=sheet_name)

            st.subheader("Input preview")
            st.dataframe(df_in.head())

            # Fill any missing feature columns with sensible defaults so the user
            # doesn't need every single column in their upload file
            _cat_defaults = {
                "industry_segment": "General Industry",
                "system_category": "Other",
                "automation_level": "Semi-Automatic",
                "plc_family": "AB Compact Logix",
                "hmi_family": "AB PanelView Plus",
                "vision_type": "None",
            }
            missing_num = [c for c in QUOTE_NUM_FEATURES if c not in df_in.columns]
            missing_cat = [c for c in QUOTE_CAT_FEATURES if c not in df_in.columns]
            for c in missing_num:
                df_in[c] = 0
            for c in missing_cat:
                df_in[c] = _cat_defaults.get(c, "unknown")

            if missing_num or missing_cat:
                st.info(f"Filled {len(missing_num + missing_cat)} missing columns with defaults: {missing_num + missing_cat}")

            if st.button("Run predictions on all rows"):
                    df_out = predict_quotes_df(df_in, confidence_level=confidence_level_batch)
                    st.subheader("Output preview")
                    st.dataframe(df_out.head())

                    csv_bytes = df_out.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download predictions CSV",
                        data=csv_bytes,
                        file_name=f"quote_predictions_{confidence_pct_batch}.csv",
                        mime="text/csv",
                    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7: Admin — upload training data, merge into master dataset, and retrain.
# The pipeline:
#   1. User uploads an Excel file with project hours data.
#   2. App applies feature engineering (bool normalization, index computation, etc.).
#   3. Filters to rows with at least one non-zero actual-hours column.
#   4. Merges new rows into the master parquet file (dedup by project_id).
#   5. Trains all 12 operation models with the updated master dataset.
#   6. Saves metrics to CSV and increments the model version.
# Also provides a "Reset" button to wipe all data and models.
# ══════════════════════════════════════════════════════════════════════════════
with tab_admin:
    st.header("Admin: Upload dataset and train models")

    st.markdown(
        "Upload the latest project_hours_dataset Excel export. "
        "The app will merge it into a master dataset (dedup by project_id) and retrain models."
    )

    uploaded_file = st.file_uploader(
        "Upload project dataset (Excel)",
        type=["xlsx", "xls"],
        key="training_uploader",
    )

    if uploaded_file is not None:
        xls = pd.ExcelFile(uploaded_file)
        sheet_name = st.selectbox("Select sheet", xls.sheet_names)
        df_raw = pd.read_excel(xls, sheet_name=sheet_name)

        st.subheader("Dataset preview")
        st.dataframe(df_raw.head())

        # Validate that the upload has the minimum required columns before proceeding
        missing = [c for c in REQUIRED_TRAINING_COLS if c not in df_raw.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            if st.button("Merge into master & train models"):
                with st.spinner("Processing upload and training models..."):
                    rows_raw = len(df_raw)

                    # Step 1: Apply feature engineering (filter to actuals, normalize booleans,
                    # compute indices, derive log cost)
                    df_train = engineer_features_for_training(df_raw)

                    # Step 2: Keep only rows that have at least one non-zero actual-hours value
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

                    # Step 3: If no trainable rows remain, log the upload but don't touch models
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
                        # Step 4: Merge new training rows into the master parquet file.
                        # If the master already exists, concatenate and deduplicate by project_id
                        # (keeping the latest version of each project).
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

                        # Step 5: Log the upload, then retrain all 12 operation models.
                        # _next_model_version() increments the version counter (v1 -> v2 -> ...)
                        # so old model files are preserved on disk for potential rollback.

                        # Log this upload's row counts for the upload history table
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

                        # Train a CatBoost CQR model for each of the 12 operations
                        new_version = _next_model_version()
                        metrics_all = []
                        for target in TARGETS:
                            m = train_one_op(
                                df_master_new,
                                target,
                                models_dir="models",
                                version=new_version,
                            )
                            if m:
                                metrics_all.append(m)

                        # Step 6: Save training metrics and notify the user
                        if metrics_all:
                            metrics_df = pd.DataFrame(metrics_all)
                            os.makedirs("models", exist_ok=True)
                            metrics_df.to_csv(METRICS_PATH, index=False)
                            st.session_state["models_ready"] = True

                            st.success(
                                "Master dataset updated and models trained. "
                                "Quoting tabs now use the latest models."
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

                            # Force a rerun so other tabs see the new master/models
                            st.rerun()
                        else:
                            st.warning(
                                "No models were trained (not enough data for any operation). "
                                "Check that actual-hours columns have non-zero values."
                            )
    else:
        st.info("Upload your project dataset (Excel) to enable training.")

    # ── Reset section ──
    # Requires the user to check a confirmation checkbox before the reset button is enabled.
    # Deletes all uploaded data, trained models, and version files.
    st.markdown("---")
    st.subheader("Reset app state")

    confirm_reset = st.checkbox(
        "I understand this will permanently delete all uploaded data and trained models.",
        key="confirm_reset",
    )
    if st.button("Reset master dataset and models", disabled=not confirm_reset):
        _reset_app_state()
        st.success(
            "Master dataset, upload log, and model artifacts have been cleared. "
            "The app is now in a blank state."
        )
        st.rerun()
