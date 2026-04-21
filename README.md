# driftboard

![Python](https://img.shields.io/badge/python-3.11-blue)
![Evidently](https://img.shields.io/badge/evidently-0.4.33-orange)
![Streamlit](https://img.shields.io/badge/streamlit-deployed-brightgreen)
![License](https://img.shields.io/badge/license-MIT-lightgrey)
![Status](https://img.shields.io/badge/status-in%20progress-yellow)

**Production ML monitoring dashboard that detects when your model starts failing before your users notice.**

---

## Demo

> 🔗 [Live demo on Streamlit Cloud](https://driftboard.streamlit.app) *(coming soon)*

![demo](assets/demo.gif)

> Click "Simulate next day" in the sidebar to watch drift scores rise in real time across three scenarios.

---

## Problem Statement

A model that performs at 92% accuracy on test data may quietly degrade to 71% in production six weeks later — without a single error in the logs.

This happens because production data distributions shift over time (customers change, markets move, upstream pipelines break). Most teams only notice when a business KPI drops, by which point weeks of bad predictions have already caused damage.

**driftboard addresses the detection gap**: it continuously monitors feature distributions and prediction behavior, flags deviations before they become business incidents, and gives teams enough lead time to retrain rather than react.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Streamlit UI (app.py)                │
│   Overview │ Data Drift │ Performance │ Alerts           │
└────────────────────────┬────────────────────────────────┘
                         │ reads
                         ▼
┌─────────────────────────────────────────────────────────┐
│                    SQLite  (monitoring.db)               │
│  predictions_log │ drift_reports │ performance_metrics  │
│  alerts                                                  │
└──────┬────────────────┬──────────────────┬──────────────┘
       │ writes         │ writes           │ writes
       │                │                  │
┌──────▼──────┐  ┌──────▼──────┐  ┌───────▼──────┐
│  run_        │  │  Drift       │  │  Performance │
│  monitoring  │  │  Detector    │  │  Tracker     │
│  .py         │  │  (Evidently) │  │  (sklearn)   │
└──────┬──────┘  └─────────────┘  └──────────────┘
       │
       │ uses
       ▼
┌─────────────────────────────────────────────────────────┐
│   Baseline Model (GradientBoosting + joblib artifact)   │
│   Reference Data (CSV — Evidently reference window)     │
│   Data Simulator  (stable / gradual_drift / sudden)     │
└─────────────────────────────────────────────────────────┘
```

---

## Key Technical Decisions

### 1. Evidently AI over custom statistical tests

Writing KS tests and PSI from scratch is 200 lines of code that needs to be validated, maintained, and extended for every new feature type. Evidently ships battle-tested implementations for numerical (KS test) and categorical (chi-squared) features, handles edge cases, and outputs structured JSON that can be stored and queried. The tradeoff: a heavier dependency with its own release cycle. Acceptable for a monitoring layer that runs daily, not in a hot path.

### 2. SQLite over PostgreSQL

A portfolio project running on Streamlit Community Cloud has no database server. SQLite is a single file, requires zero infrastructure, and handles the write volume of a daily monitoring job without contention. The ceiling (~100 writes/day, ~10k rows/month) is never approached. PostgreSQL would be the right choice the moment multiple services write concurrently or horizontal scaling is needed — not before.

### 3. Simulated drift over a real dataset

Real-world datasets with labeled production drift are proprietary or prohibitively large for a portfolio project. Simulation gives full control over drift intensity, timing, and type, which makes the demo reproducible and explainable in an interview. The simulation parameters (mean shift, abrupt cutoff date) map directly to real scenarios: gradual demographic shifts, sudden regulatory changes, upstream data pipeline bugs.

### 4. Separation between detection and storage

`drift_detector.py` has no knowledge of SQLite. `run_monitoring.py` orchestrates both. This means the detector can be unit-tested in isolation with mocked data, and the storage layer can be swapped (e.g. to PostgreSQL or a time-series DB) without touching detection logic. Tight coupling here is a common mistake in monitoring codebases.

### 5. 30-sample minimum before computing performance metrics

In production, ground-truth labels arrive with a delay — days for fraud, weeks for churn. Computing accuracy on 5 labeled rows produces statistically meaningless results and false alerts. The 30-sample floor is configurable in `core/config.py`. The system gracefully handles the "no labels yet" state by monitoring data drift alone, which mirrors how real ML teams operate in the first weeks after a model release.

---

## Features

- **Three drift scenarios** — stable, gradual, and sudden shift, each visually and statistically distinct
- **Automated severity grading** — ok / warning / critical based on configurable thresholds
- **Performance tracking** — accuracy, F1, ROC-AUC over sliding windows, with reference baseline
- **Human-readable alerts** — messages written for operations teams, not data scientists
- **One-click simulation** — "Simulate next day" button lets anyone trigger drift during a demo

---

## Tech Stack

| Component | Technology | Why |
|---|---|---|
| Drift detection | Evidently AI 0.4.33 | Industry-standard, handles numerical + categorical features, structured JSON output |
| Baseline model | scikit-learn GradientBoosting | Universally readable, easily swappable — the point is the monitoring layer, not the model |
| Metrics storage | SQLite | Zero infrastructure, sufficient write volume, portable single file for Streamlit Cloud |
| Scheduling | APScheduler | Lightweight — no Airflow/Prefect overhead for a daily single-node job |
| Dashboard | Streamlit | Fast iteration, native Plotly integration, free cloud deployment |
| Visualizations | Plotly | Interactive time-series, color-coded severity, hover details |
| Config | pydantic-settings | Type-validated settings with `.env` support, no raw `os.getenv` calls scattered in code |
| Logging | loguru | Structured, colored, one-line setup — no `logging.basicConfig` boilerplate |

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/your-username/driftboard.git && cd driftboard
pip install -r requirements.txt

# 2. Train the baseline model and generate artifacts
jupyter nbconvert --to notebook --execute notebooks/01_baseline_training.ipynb

# 3. Pre-fill 60 days of monitoring history
python run_monitoring.py --prefill

# 4. Launch the dashboard
streamlit run app.py
```

> **Docker alternative (Streamlit only — no external services needed)**
> ```bash
> docker compose up
> ```

---

## Project Structure

```
driftboard/
├── app.py                        # Streamlit dashboard — UI only, no business logic
├── run_monitoring.py             # Daily pipeline script + --prefill mode
│
├── monitoring/
│   ├── drift_detector.py         # Evidently wrapper — data drift + concept drift
│   ├── performance_tracker.py    # sklearn metrics over sliding windows
│   └── alerting.py               # Alert evaluation and SQLite persistence
│
├── data/
│   ├── simulator.py              # Synthetic data generator (3 scenarios)
│   └── storage.py                # SQLite read/write interface
│
├── models/
│   ├── baseline_model.py         # Lazy-loading joblib wrapper
│   └── model_registry.py         # Reads model_metadata.json
│
├── core/
│   └── config.py                 # Thresholds, paths, settings singleton
│
├── artifacts/
│   ├── baseline_model.joblib     # Trained model (generated by notebook)
│   ├── reference_data.csv        # Evidently reference window
│   └── model_metadata.json       # Version + reference metrics
│
├── notebooks/
│   └── 01_baseline_training.ipynb
│
├── tests/
│   ├── test_drift_detector.py
│   ├── test_performance_tracker.py
│   └── test_simulator.py
│
├── monitoring.db                 # Pre-filled SQLite (60 days, committed)
└── requirements.txt
```

---

## Environment Variables

No API keys required. All configuration uses defaults that work out of the box.

| Variable | Default | Description |
|---|---|---|
| `DRIFT_WARNING_THRESHOLD` | `0.15` | Share of drifted features that triggers a warning |
| `DRIFT_CRITICAL_THRESHOLD` | `0.30` | Share of drifted features that triggers a critical alert |
| `PERFORMANCE_DEGRADATION_THRESHOLD` | `0.10` | F1 relative drop (10%) that triggers a performance alert |
| `MISSING_LABELS_ALERT_HOURS` | `48` | Hours without ground-truth labels before alerting |
| `MIN_SAMPLES_FOR_METRICS` | `30` | Minimum labeled rows required to compute performance metrics |
| `SHORT_WINDOW_DAYS` | `7` | Sliding window used for drift detection |
| `LONG_WINDOW_DAYS` | `30` | Retention window for dashboard charts |

Copy `.env.example` to `.env` to override any value.

---

## Roadmap

**1. Webhook notifications**
Push alerts to Slack or PagerDuty when severity reaches critical. The `AlertManager.evaluate_alerts()` return value already contains all required fields — adding a notifier is a thin wrapper around the existing interface.

**2. Per-feature drift history charts**
The current dashboard shows dataset-level drift over time. Drilling down to see which specific feature drifted on which day (and by how much) would accelerate root-cause analysis significantly.

**3. Automated retraining trigger**
When both data drift and performance degradation alerts fire on the same day for 3 consecutive days, log a retraining recommendation to SQLite. A follow-up job could pick it up and kick off a scikit-learn pipeline. The detection logic is already in place — the missing piece is the action layer.

---

## License

MIT
