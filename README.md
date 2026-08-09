# Predicting F1 Driver Style from Telemetry Data (`f1-telemetry-ml`)

A physics-informed, end-to-end machine learning and data engineering pipeline designed to predict continuous driver behavioral traits from high-frequency time-series vehicle telemetry. This system processes official Formula 1 Race data from the 2024 season across 10 distinct Grand Prix tracks to quantitatively measure driver control philosophies through corners.

---

## Project Overview

This project treats driver style modeling as a **supervised multi-output regression task**. Rather than placing drivers into arbitrary discrete categories, our pipeline maps telemetry through a corner into a continuous $[0.0, 1.0]$ spectrum across three physics-informed target dimensions:

1. **Aggression Score:** Measures throttle modulation jerk variance $$\left(\frac{d(\text{Throttle})}{dt}\right)$$. Higher values indicate abrupt, aggressive pedal inputs.
2. **Line Shape Score:** Quantifies cornering geometry as the ratio of minimum apex speed to average entry/exit speed ($\frac{v_{\text{min}}}{\bar{v}_{\text{entry, exit}}}$), distinguishing $U$-shaped momentum lines (closer to $1.0$) from $V$-shaped late-braking lines (closer to $0.0$).
3. **Oversteer Preference Score:** Serves as a simplified mid-corner handling instability proxy based on peak longitudinal deceleration rates ($$-\min\left(\frac{dv}{dt}\right)$$).

### Key Data & Evaluation Strategy
- **2024 Race Sessions Only:** Focuses exclusively on Grand Prix Race (`'R'`) sessions across 10 diverse circuit geometries (*Austin, Austria, Bahrain, Belgium, Jeddah, Monaco, Monza, Silverstone, Singapore, and Suzuka*).
- **Leakage-Conscious Labeling:** Target normalization scale parameters ($\text{log-scale}$ and $\text{percentile-clip}$ bounds) are fit **strictly on training-track data** (`TRAIN_TRACKS`) and applied downstream to validation and test sets without refitting.
- **Feature Isolation:** Models are trained strictly on raw, unsmoothed telemetry channels (`Speed`, `Throttle`, `Brake`, `nGear`, `RPM`). Smoothed channels and spatial derivatives used during label generation are explicitly excluded from model inputs to prevent target leakage.
- **Dual Generalization Evaluation:** 
  - **In-Distribution Test:** Held-out $20\%$ lap slice on familiar tracks (*Monza, Silverstone*).
  - **Zero-Shot Test:** Entirely unseen track holdout (*Belgium, Jeddah*).

---

## Repository Layout

```text
f1-telemetry-ml/
│
├── .gitignore               <- Prevents tracking of large binary parquet files, CSVs, and API caches
├── README.md                <- Environment setup and workflow documentation
├── requirements.txt         <- Python library dependencies
├── fastf1_cache/            <- [GITIGNORED] Local FastF1 API cache directory
│
├── fastf1_data/             <- [GITIGNORED] Local telemetry storage
│   ├── raw/                 <- Raw whole-lap telemetry CSVs from FastF1
│   ├── processed/           <- Sliced and smoothed corner parquet files
│   └── labeled/             <- Labeled corner-instance parquet files and split datasets
│
├── src/                     <- Modular Python pipeline scripts
│   ├── __init__.py
│   ├── data_ingestion.py    <- Downloads raw race-session telemetry via FastF1 API
│   ├── corner_slicer.py     <- Cuts whole laps into spatial corner windows (-200m to +100m)
│   └── heuristic_engine.py  <- Computes raw style scores and handles train-only normalization
│
├── notebooks/               <- Sequential Jupyter execution and modeling notebooks
│   ├── 01_data_pipeline.ipynb       <- Data exploration, track partitioning, and split generation
│   ├── 02_model_baseline.ipynb      <- Evaluates training-mean baseline performance
│   ├── 03_model_ridge.ipynb         <- Multi-output Ridge Regression baseline
│   ├── 04_model_random_forest.ipynb <- Random Forest Regressor benchmark
│   ├── 05_model_mlp.ipynb           <- Multi-Layer Perceptron neural network
│   └── 06_model_lstm.ipynb          <- Sequence-based LSTM network
│
└── models/                  <- [GITIGNORED] Serialized model weights and checkpoints

```

---

## Environment Setup & Installation

Follow these steps to establish a uniform local development environment across your local machine and cloud platforms (Google Colab).

### 1. Repository Initialization & Virtual Environment

Clone the repository and set up an isolated Python virtual environment:

```bash
# Clone the repository
git clone [https://github.com/yourusername/f1-telemetry-ml.git](https://github.com/yourusername/f1-telemetry-ml.git)
cd f1-telemetry-ml

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows (Command Prompt):
venv\Scripts\activate
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1

```

### 2. Dependency Installation

Install all required dependencies (Pandas, FastF1, PyTorch, Scikit-Learn, etc.):

```bash
pip install -r requirements.txt

```

### 3. Verification & Git Safeguards

Verify that the git safeguards are actively ignoring local data caches and output directories before running data ingestion routines:

```bash
git status

```

*Ensure that `fastf1_cache/`, `fastf1_data/`, and `models/` are listed under ignored patterns and do not appear as untracked folders.*

---

## Data Pipeline & Modeling Workflow

The pipeline can be executed either via standalone terminal scripts or sequentially through Jupyter Notebooks.

### Option A: Terminal Pipeline Execution

1. **Ingest Raw Telemetry:**
Fetch whole-lap race data for the 10 target tracks of the 2024 season:
```bash
python src/data_ingestion.py

```


2. **Slice Corners & Apply Signal Smoothing:**
Segment whole laps into spatial corner windows ($-200\text{m}$ to $+100\text{m}$) and generate smoothed channels for target calculations:
```bash
python src/corner_slicer.py

```


3. **Generate & Normalize Target Labels:**
Compute raw style scores and apply train-only fitted normalization:
```bash
python src/heuristic_engine.py

```



### Option B: Interactive Notebook Execution (`notebooks/`)

1. **`01_data_pipeline.ipynb`:** Load combined labels, explore target distributions, analyze driver profiles, and produce the final `train_data.parquet`, `val_data.parquet`, `test_in_dist_data.parquet`, and `test_zero_shot.parquet` split files in `fastf1_data/labeled/`.
2. **`02_model_baseline.ipynb` through `06_model_lstm.ipynb`:** Train and evaluate each baseline and model family against the generated parquet splits.

---

## Dual Environment Execution (Local vs. Google Colab)

`notebooks/01_data_pipeline.ipynb` and all modeling notebooks support seamless switching between local development and Google Colab environments via toggle cells at the top of each notebook:

* **For Google Colab:** Run **Cell 1A** to mount Google Drive and set `data_path` to your Drive directory.
* **For Local Execution:** Run **Cell 1B** to set relative paths pointing directly to `../fastf1_data`.

### Google Colab Mount Example:

```python
from google.colab import drive
drive.mount('/content/drive')

data_path = '/content/drive/MyDrive/f1-telemetry-ml'
labeled_path = os.path.join(data_path, 'labeled')

```
