# Matrix Quote App

An internal Streamlit tool for estimating engineering hours on industrial automation projects (robotics, controls, mechanical, etc.). It trains per-operation RandomForest models from historical project data and produces hour estimates with confidence intervals for new quotes.

---

## What it does

1. **Ingests** historical project data from Excel uploads and stores it as Parquet.
2. **Trains** one RandomForest model per operation (12 total) using quote-time features only.
3. **Predicts** hours per operation and rolls them up into 9 sales buckets with low/mid/high confidence intervals.
4. **Explains** which features drive each estimate and surfaces similar past projects.

### Operations and sales buckets

| Operation | Sales Bucket |
|-----------|-------------|
| me10, me15, me230 | ME |
| ee20 | EE |
| rb30 | Robot |
| cp50 | Controls |
| bld100, shp150 | Build |
| inst160 | Install |
| trv180 | Travel |
| doc190 | Docs |
| pm200 | PM |

---

## Setup

### Local

```bash
pip install -r requirements.txt
streamlit run quote_app.py
```

The app opens at `http://localhost:8501`.

### GitHub Codespaces

Open the repo in a Codespace — the devcontainer installs all dependencies and starts the app automatically. Streamlit is forwarded on port 8501 and opens as an in-editor preview.

---

## How to use

1. **Admin: Upload & Train** — upload a project-hours Excel file, review the parsed data, then click **Train Models**. Models are saved to `models/` and metrics to `models/metrics_summary.csv`.
2. **Data Explorer** — browse the training dataset, filter by segment or category.
3. **Model Performance** — review per-operation R², MAE, and feature importances.
4. **Drivers & Similar Projects** — inspect what drives a specific operation and find historically similar projects.
5. **Single Quote** — enter quote parameters and get an hour estimate with a low/mid/high range per sales bucket.
6. **Batch Quotes** — upload a CSV of multiple projects and download estimates in bulk.
7. **Overview** — summary statistics and dataset health.

> Models must be trained (step 1) before Single Quote and Batch Quotes are usable.

---

## Project structure

```
matrix-quote-app/
├── .devcontainer/devcontainer.json   # GitHub Codespaces config (Python 3.11)
├── quote_app.py                      # Main Streamlit app (7-tab UI)
├── core/
│   ├── config.py                     # Feature lists, targets, sales bucket mapping
│   ├── schemas.py                    # Pydantic models (QuoteInput, QuotePrediction, etc.)
│   ├── features.py                   # Feature engineering for training & inference
│   └── models.py                     # RF training, prediction with tree-level intervals
├── service/
│   └── predict_lib.py                # Single & batch prediction orchestration
├── requirements.txt
└── .gitignore                        # Ignores data/master/, models/, __pycache__/
```

Runtime directories (git-ignored, created on first use):

```
data/master/    # Parquet dataset and upload log
models/         # Trained .joblib model files and metrics_summary.csv
```

---

## Data requirements

Training data must be uploaded as an Excel file (.xlsx). The following columns are required at minimum:

| Column | Description |
|--------|-------------|
| `project_id` | Unique identifier for each project |
| `include_in_training` | Boolean — whether to use this row for model training |
| `dataset_role` | Role of the row (e.g., `train`, `test`) |
| `industry_segment` | Industry vertical (categorical) |
| `system_category` | Type of automation system (categorical) |
| `stations_count` | Number of stations in the system |
| `robot_count` | Number of robots |
| `me10_actual_hours` | At least one actual-hours column is required |

All 12 `*_actual_hours` columns (`me10` through `pm200`) should be present for full model training. Additional quote-time feature columns (31 numeric + 6 categorical) improve prediction accuracy — see `core/config.py` for the complete lists.

---

## Tech stack

- **UI:** Streamlit, Altair
- **ML:** scikit-learn (RandomForestRegressor), joblib
- **Data:** pandas, numpy, pyarrow (Parquet), openpyxl (Excel)
- **Validation:** Pydantic
- **Runtime:** Python 3.11, port 8501
