# quote_app.py
# Streamlit UI with:
# - Overview
# - Data Explorer
# - Model Performance
# - Drivers & Similar Projects
# - Single Quote
# - Batch Quotes
# - Admin: Upload & Train

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
from service.predict_lib import predict_quote, predict_quotes_df

MASTER_DATA_PATH = os.path.join("data", "master", "projects_master.parquet")
UPLOADS_LOG_PATH = os.path.join("data", "master", "uploads_log.csv")
METRICS_PATH = os.path.join("models", "metrics_summary.csv")

st.set_page_config(page_title="Matrix Quote App", layout="wide")
st.title("Matrix Quote App")

if "models_ready" not in st.session_state:
    st.session_state["models_ready"] = False

# If this is a new session but models/metrics already exist on disk,
# mark models as ready so Single/Batch are usable without retraining.
if not st.session_state["models_ready"]:
    if os.path.exists(METRICS_PATH):
        try:
            _metrics = pd.read_csv(METRICS_PATH)
            if not _metrics.empty:
                st.session_state["models_ready"] = True
        except Exception:
            pass

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
    # Remove master dataset, upload log, and metrics file if present
    for path in [MASTER_DATA_PATH, UPLOADS_LOG_PATH, METRICS_PATH]:
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


# Data Explorer tab: explore master dataset
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



# Model Performance tab: show per-op metrics
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
                    "Select operation", modeled_ops, key="drivers_op_select"
                )

                pipe = load_model(target_choice)
                pre = pipe.named_steps["preprocess"]
                model = pipe.named_steps["model"]

                try:
                    feature_names = pre.get_feature_names_out()
                except Exception:
                    feature_names = [
                        f"f_{i}" for i in range(len(model.feature_importances_))
                    ]

                importances = model.feature_importances_
                fi_df = (
                    pd.DataFrame(
                        {"feature": feature_names, "importance": importances}
                    )
                    .sort_values("importance", ascending=False)
                    .reset_index(drop=True)
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
        industry_segment = st.selectbox(
            "Industry segment",
            ["Automotive", "Food & Beverage", "General Industry"],
        )
        system_category = st.selectbox(
            "System category",
            ["Machine Tending", "End of Line Automation", "Robotic Metal Finishing", "Engineered Manufacturing Systems", "Other"],
        )
        automation_level = st.selectbox(
            "Automation level",
            ["Semi-Automatic", "Robotic", "Hard Automation"],
        )
        plc_family = st.text_input("PLC family", "AB Compact Logix")
        hmi_family = st.text_input("HMI family", "AB PanelView Plus")
        vision_type = st.text_input("Vision type", "None")

        stations_count = st.number_input("Stations count", min_value=0, step=1)
        robot_count = st.number_input("Robot count", min_value=0, step=1)
        fixture_sets = st.number_input("Fixture sets", min_value=0, step=1)
        part_types = st.number_input("Part types", min_value=0, step=1)
        servo_axes = st.number_input("Servo axes", min_value=0, step=1)
        pneumatic_devices = st.number_input("Pneumatic devices", min_value=0, step=1)
        safety_doors = st.number_input("Safety doors", min_value=0, step=1)
        weldment_perimeter_ft = st.number_input(
            "Weldment perimeter (ft)", min_value=0.0
        )
        fence_length_ft = st.number_input("Fence length (ft)", min_value=0.0)
        conveyor_length_ft = st.number_input("Conveyor length (ft)", min_value=0.0)
        product_familiarity_score = st.slider(
            "Product familiarity (1–5)", 1, 5, 3
        )
        product_rigidity = st.slider("Product rigidity (1–5)", 1, 5, 3)
        is_product_deformable = st.checkbox("Product deformable?")
        is_bulk_product = st.checkbox("Bulk product?")
        bulk_rigidity_score = st.slider("Bulk rigidity score (1–5)", 1, 5, 3)
        has_tricky_packaging = st.checkbox("Tricky packaging?")
        process_uncertainty_score = st.slider(
            "Process uncertainty (1–5)", 1, 5, 3
        )
        changeover_time_min = st.number_input(
            "Changeover time (min)", min_value=0.0
        )
        safety_devices_count = st.number_input(
            "Safety devices count", min_value=0, step=1
        )
        custom_pct = st.slider("Custom %", 0, 100, 50)
        duplicate = st.checkbox("Duplicate of prior project?")
        has_controls = st.checkbox("Includes controls work?", value=True)
        has_robotics = st.checkbox("Includes robotics work?", value=True)
        retrofit = st.checkbox("Retrofit project?")
        complexity_score_1_5 = st.slider(
            "Overall complexity (1–5)", 1, 5, 3
        )
        vision_systems_count = st.number_input(
            "Vision systems count", min_value=0, step=1
        )
        panel_count = st.number_input("Panel count", min_value=0, step=1)
        drive_count = st.number_input("Drive count", min_value=0, step=1)
        stations_robot_index = st.number_input(
            "Stations/Robot index (optional)", min_value=0.0
        )
        mech_complexity_index = st.number_input(
            "Mechanical complexity index (optional)", min_value=0.0
        )
        controls_complexity_index = st.number_input(
            "Controls complexity index (optional)", min_value=0.0
        )
        physical_scale_index = st.number_input(
            "Physical scale index (optional)", min_value=0.0
        )
        estimated_materials_cost = st.number_input(
            "Estimated materials cost", min_value=0.0
        )
        confidence_level = st.slider(
            "Confidence level (%)",
            min_value=50,
            max_value=95,
            value=90,
            step=5,
        )

        if st.button("Estimate hours"):

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

            sales_rows = []
            for bucket in SALES_BUCKETS:
                bucket_pred = pred.sales_buckets.get(bucket)
                if bucket_pred is None:
                    continue
                sales_rows.append(
                    {
                        "Sales bucket": bucket,
                        "p50_hours": bucket_pred.p50,
                        "confidence": bucket_pred.confidence,
                    }
                )

            has_quoted_hours = False
            quoted_hours_by_bucket = st.session_state.get("quoted_hours_by_bucket")
            if isinstance(quoted_hours_by_bucket, dict) and quoted_hours_by_bucket:
                has_quoted_hours = True

            sales_summary_rows = []

            for row in sales_rows:
                role = row["Sales bucket"]
                p50 = row["p50_hours"]
                confidence = row["confidence"]

                summary_row = {
                    "Role": role,
                    "Recommended hours (P50)": p50,
                    "Confidence": confidence.title(),
                    "Confidence level": f"{confidence_level}%",
                }

                if has_quoted_hours:
                    quoted_val = quoted_hours_by_bucket.get(role)
                    if quoted_val is not None:
                        delta = quoted_val - p50
                        threshold = max(0.1 * abs(p50), 5)
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

            total_model_hours = pred.total_p50
            project_cols = ["Model total (P50)"]
            project_values = [f"{total_model_hours:.1f} h"]
            project_status = None

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

            sales_summary_rows_exist = bool(sales_summary_rows)
            df_sales_summary_sorted = None
            if sales_summary_rows_exist:
                df_sales_summary = pd.DataFrame(sales_summary_rows)
                df_sales_summary_sorted = df_sales_summary.sort_values(
                    "Recommended hours (P50)", ascending=False
                )

                if has_quoted_hours:
                    total_row = {
                        "Role": "TOTAL",
                        "Recommended hours (P50)": df_sales_summary["Recommended hours (P50)"].sum(),
                        "Confidence": "-",
                        "Confidence level": f"{confidence_level}%",
                        "Quoted hours": df_sales_summary["Quoted hours"].sum(),
                        "Delta (quoted - model)": df_sales_summary["Delta (quoted - model)"].sum(),
                        "Status": "-",
                    }
                    df_sales_summary_sorted = pd.concat(
                        [df_sales_summary_sorted, pd.DataFrame([total_row])], ignore_index=True
                    )

            rows = []
            for op, op_pred in pred.ops.items():
                rows.append(
                    {
                        "operation": op,
                        "p50_hours": op_pred.p50,
                        "confidence": op_pred.confidence,
                        "confidence_level": f"{confidence_level}%",
                    }
                )
            df_out = pd.DataFrame(rows)

            sales_tab, ops_tab = st.tabs(["Sales view", "Operations view"])

            with sales_tab:
                st.subheader("Project summary")
                cols = st.columns(len(project_cols))
                for col, label, val in zip(cols, project_cols, project_values):
                    col.metric(label, val)

                if project_status:
                    st.caption(project_status)

                st.subheader("Sales-level summary")
                st.caption(
                    f"Estimates shown at {confidence_level}% confidence."
                )
                if sales_summary_rows_exist and df_sales_summary_sorted is not None:
                    display_cols = [
                        "Role",
                        "Recommended hours (P50)",
                        "Confidence",
                        "Confidence level",
                    ]
                    if has_quoted_hours:
                        display_cols += [
                            "Quoted hours",
                            "Delta (quoted - model)",
                            "Status",
                        ]
                    st.dataframe(df_sales_summary_sorted[display_cols])

                    if has_quoted_hours:
                        df_chart = df_sales_summary_sorted[
                            df_sales_summary_sorted["Role"] != "TOTAL"
                        ][["Role", "Recommended hours (P50)", "Quoted hours"]]
                        if not df_chart.empty:
                            chart_data = df_chart.melt(
                                id_vars="Role",
                                value_vars=["Recommended hours (P50)", "Quoted hours"],
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
                st.caption(
                    f"Estimates shown at {confidence_level}% confidence."
                )
                st.dataframe(df_out)


# Batch Quotes tab
with tab_batch:
    st.header("Batch estimation via CSV or Excel")

    if not st.session_state["models_ready"]:
        st.warning("Models are not trained yet. Go to 'Admin: Upload & Train' first.")
    else:
        st.markdown(
            "Your file must include at least these columns: "
            f"`{', '.join(QUOTE_NUM_FEATURES + QUOTE_CAT_FEATURES)}`"
        )

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

            required_cols = set(QUOTE_NUM_FEATURES + QUOTE_CAT_FEATURES)
            missing = [c for c in required_cols if c not in df_in.columns]

            if missing:
                st.error(f"Missing required columns: {missing}")
            else:
                if st.button("Run predictions on all rows"):
                    df_out = predict_quotes_df(df_in)
                    st.subheader("Output preview")
                    st.dataframe(df_out.head())

                    csv_bytes = df_out.to_csv(index=False).encode("utf-8")
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
                            if m:
                                metrics_all.append(m)

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

    st.markdown("---")
    st.subheader("Reset app state")

    if st.button("Reset master dataset and models"):
        _reset_app_state()
        st.success(
            "Master dataset, upload log, and model artifacts have been cleared. "
            "The app is now in a blank state."
        )
        st.rerun()
