# Predicting F1 Driver Style from Telemetry Data (`f1-telemetry-ml`)

A physics-informed, end-to-end machine learning and data engineering pipeline designed to predict continuous driver behavioral traits from high-frequency time-series vehicle telemetry. By training on optimal, single-lap Qualifying and Free Practice conditions and evaluating on out-of-distribution (OOD) Grand Prix race stints, this system isolates and analyzes how real-world environmental factors degrade model stability and alter driving dynamics.

---

## Project Architecture Overview

This project treats driver style modeling as a **supervised multi-output regression task**. Instead of classifying drivers into rigid binary buckets, our pipeline maps behavior across a continuous spectrum ($0.0$ to $1.0$) across three distinct physics-informed dimensions:

1. **Aggression Score:** Evaluates input jerk variance via the temporal derivatives of throttle and brake modulation ($\frac{d(\text{input})}{dt}$).
2. **Line Shape Score:** Quantifies the geometric transition through a corner, distinguishing U-shaped (high mid-corner rolling speed) from V-shaped (late braking, sharp apex rotation) racing lines.
3. **Vehicle Balance Preference:** Isolates a driver's handling tolerance between stable understeer and a hyper-responsive, loose oversteer platform by mapping calculated lateral forces against internal accelerometer sensors.



### The Environmental Stress Test Strategy

- **Training & Validation:** 100% pure Qualifying and Free Practice sessions across all 20+ grid drivers spanning the 2024–2025 seasons. This provides a baseline dataset representing uncompromised driving styles in clean air.
- **Testing & Error Analysis:** A stratified random selection of Grand Prix race laps. This serves as our OOD testing benchmark, allowing us to mathematically track how predictive stability degrades under the influence of fuel weight decay, active traffic, and progressive tire degradation.

---



## Repository Layout

```text
f1-telemetry-ml/
│
├── .gitignore                 <- Firewalls large binary files, caches, and datasets from Git
├── README.md                  <- Project blueprint and environment setup documentation
├── requirements.txt           <- Production library dependencies
├── fastf1_cache/              <- [GITIGNORED] Local API cache directory to prevent throttling
│
├── fastf1_data/               <- [GITIGNORED] Data directory
│   ├── raw/                   <- Raw telemetry arrays pulled from the FastF1 wrapper
│   └── processed/             <- Finalized resampled corner matrices and target arrays
│
├── src/                       <- Core data processing and feature engineering modules
│   ├── __init__.py
│   ├── data_ingestion.py      <- Multi-session scraping loop with caching controls
│   ├── corner_slicer.py       <- Spatial geometric indexing engine to isolate corner intervals
│   └── heuristic_engine.py    <- Physics formulas calculating the 0.0-1.0 style targets
│
├── notebooks/                 <- Sequential execution notebooks for the modeling team
│   ├── 01_data_pipeline.ipynb       <- Runs src/ modules to populate data matrices
│   ├── 02_model_baseline.ipynb      <- Establishes historical mean error benchmarks
│   ├── 03_model_ridge.ipynb         <- Linear Multi-output Ridge track
│   ├── 04_model_random_forest.ipynb <- Tree-based ensemble track
│   ├── 05_model_mlp.ipynb           <- Feedforward Deep Learning baseline
│   ├── 06_model_lstm.ipynb          <- Recurrent time-series sequence model
│   └── 07_environmental_analysis.ipynb <- Evaluates optimal models against Grand Prix data
│
└── models/                    <- [GITIGNORED] Serialized weights and model checkpoints

```

---



## Repository Setup & Installation

Follow these steps to establish a uniform local development environment across your machine and your team members' environments.

### 1. Environment Initialization

Clone the repository and spin up a localized virtual environment within your terminal to keep dependencies isolated:

```bash
# Clone the project repository
git clone https://github.com/yourusername/f1-telemetry-ml.git
cd f1-telemetry-ml

# Create a virtual environment named 'venv'
python -m venv venv

# Activate the environment
# On macOS/Linux:
source venv/bin/activate
# On Windows (Command Prompt):
venv\Scripts\activate
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1

```



### 2. Dependency Management

Install the foundational packages required for telemetry parsing, physics heuristics, and deep learning execution:

```bash
pip install -r requirements.txt

```



### 3. Verification & Pipeline Initialization

Verify your paths and ensure the workspace structure is locked down. Before running data downloading routines, verify that your `.gitignore` is successfully active by executing:

```bash
git status

```

`fastf1_cache/`*,* `fastf1_data/`*, and* `models/` *should not appear as untracked folders.*

### 4. Shared Google Colab Execution

For deep learning training inside the `notebooks/` directory using Colab Premium GPU runtimes:

1. Upload the repository structure or mount your personal GitHub fork within Colab.
2. Link your shared team Google Drive to mirror the data architecture without executing duplicate API calls:

```python
from google.colab import drive
drive.mount('/content/drive')

```

***

### Commit and Push Your Updates

Once you save the file, track the changes in Git and update your remote repository using the VS Code terminal:

```bash
git add README.md
git commit -m "Docs: Update README with comprehensive project scope and environment setup steps"
git push

```

