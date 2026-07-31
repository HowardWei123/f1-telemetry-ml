"""
src/corner_slicer.py

Spatial geometric indexing engine operating strictly on Race data.
Slices raw whole-lap CSVs into uniform spatial corner segments (100 points)
and routes output Parquet files directly into track-based train/val/test splits.
"""

import os
import glob
import fastf1
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

RAW_DATA_DIR = "fastf1_data/raw"
PROCESSED_DATA_DIR = "fastf1_data/processed"
CACHE_DIR = "fastf1_cache"

# Track-level splitting for Race-only data (Zero-shot evaluation strategy)
TRAIN_TRACKS = ["Monza", "Monaco", "Silverstone", "Suzuka", "Austin", "Bahrain"]
VAL_TRACKS = ["Singapore", "Austria"]
TEST_IN_DIST_TRACKS = ["Monza", "Silverstone"]
HOLDOUT_TRACKS = ["Belgium", "Jeddah"]

WINDOW_METERS_BEFORE = 200
WINDOW_METERS_AFTER = 100
TARGET_POINTS_PER_CORNER = 100
SMOOTH_WINDOW = 15
SMOOTH_POLYORDER = 2

CORNER_CACHE = {}


def setup_environment():
    os.makedirs(CACHE_DIR, exist_ok=True)
    fastf1.Cache.enable_cache(CACHE_DIR)
    for split in ["train", "val", "test_in_dist", "test_zero_shot"]:
        os.makedirs(os.path.join(PROCESSED_DATA_DIR, split), exist_ok=True)


def get_corner_markers(year: int, race_name: str) -> pd.DataFrame:
    cache_key = (year, race_name)
    if cache_key in CORNER_CACHE:
        return CORNER_CACHE[cache_key]

    try:
        session = fastf1.get_session(year, race_name, "R")
        session.load(telemetry=True, laps=True, weather=False, messages=False)
        circuit_info = session.get_circuit_info()
        corners = circuit_info.corners[["Number", "Distance"]].copy()
        CORNER_CACHE[cache_key] = corners
        return corners
    except Exception as e:
        print(f"  [CornerSlicer] Error fetching circuit info for {year} {race_name}: {e}")
        return pd.DataFrame()


def smooth_channel(series: np.ndarray, min_val=0, max_val=100) -> np.ndarray:
    if len(series) < SMOOTH_WINDOW:
        return series
    smoothed = savgol_filter(series, window_length=SMOOTH_WINDOW, polyorder=SMOOTH_POLYORDER)
    return np.clip(smoothed, min_val, max_val)


def resample_corner_segment(df_segment: pd.DataFrame, num_points: int = TARGET_POINTS_PER_CORNER) -> pd.DataFrame:
    if len(df_segment) < 5:
        return pd.DataFrame()

    distances = df_segment["Distance"].values
    _, unique_indices = np.unique(distances, return_index=True)
    if len(unique_indices) < 4:
        return pd.DataFrame()

    df_clean = df_segment.iloc[unique_indices].sort_values("Distance")
    dist_clean = df_clean["Distance"].values

    uniform_dist = np.linspace(dist_clean[0], dist_clean[-1], num_points)
    resampled_data = {"Distance": uniform_dist}

    numeric_cols = ["Speed", "Throttle", "Brake", "RPM", "nGear", "SessionTime"]
    for col in numeric_cols:
        if col in df_clean.columns:
            interp_func = interp1d(dist_clean, df_clean[col].values, kind="linear", fill_value="extrapolate")
            resampled_data[col] = interp_func(uniform_dist)

    resampled_df = pd.DataFrame(resampled_data)

    for col in ["Throttle", "Brake", "Speed"]:
        if col in resampled_df.columns:
            resampled_df[f"{col}_smooth"] = smooth_channel(resampled_df[col].to_numpy())

    # Retain identity metadata strictly for output routing and grouping
    for meta_col in ["driver", "lap_number", "session_type", "year", "race", "corner_number"]:
        if meta_col in df_segment.columns:
            resampled_df[meta_col] = df_segment[meta_col].iloc[0]

    return resampled_df


def slice_session(raw_path: str, year: int, race_name: str, session_type: str = "R") -> pd.DataFrame:
    if not os.path.exists(raw_path):
        return pd.DataFrame()

    raw = pd.read_csv(raw_path)
    if raw.empty:
        return pd.DataFrame()

    corners = get_corner_markers(year, race_name)
    if corners.empty:
        return pd.DataFrame()

    all_segments = []

    for (driver, lap_num), lap_df in raw.groupby(["driver", "lap_number"]):
        lap_df = lap_df.sort_values("Distance")

        for _, corner in corners.iterrows():
            center = corner["Distance"]
            corner_num = corner["Number"]

            mask = (lap_df["Distance"] >= center - WINDOW_METERS_BEFORE) & \
                   (lap_df["Distance"] <= center + WINDOW_METERS_AFTER)
            
            segment = lap_df[mask].copy()
            if len(segment) < 5:
                continue

            segment["corner_number"] = corner_num
            segment["session_type"] = session_type
            segment["year"] = year
            segment["race"] = race_name

            uniform_segment = resample_corner_segment(segment)
            if not uniform_segment.empty:
                all_segments.append(uniform_segment)

    if not all_segments:
        return pd.DataFrame()

    return pd.concat(all_segments, ignore_index=True)


def get_target_split_dir(race_name: str) -> str:
    """Routes file destination based on track splits for Race data."""
    if race_name in HOLDOUT_TRACKS:
        return os.path.join(PROCESSED_DATA_DIR, "test_zero_shot")
    elif race_name in VAL_TRACKS:
        return os.path.join(PROCESSED_DATA_DIR, "val")
    elif race_name in TRAIN_TRACKS:
        return os.path.join(PROCESSED_DATA_DIR, "train")
    else:
        return os.path.join(PROCESSED_DATA_DIR, "test_in_dist")


def run_slicing():
    setup_environment()
    raw_files = glob.glob(f"{RAW_DATA_DIR}/*_R.csv")

    if not raw_files:
        print(f"No raw race files found in {RAW_DATA_DIR}/ — run data_ingestion.py first.")
        return

    print(f"Found {len(raw_files)} raw Race CSV files to process...\n")

    for raw_path in raw_files:
        filename = os.path.basename(raw_path).replace(".csv", "")
        year_str, race_name, session_type = filename.split("_", 2)
        year = int(year_str)

        target_dir = get_target_split_dir(race_name)
        out_path = os.path.join(target_dir, f"corners_{filename}.parquet")

        if os.path.exists(out_path):
            print(f"Already sliced, skipping: {out_path}")
            continue

        print(f"Slicing {filename} -> Routing to {os.path.basename(target_dir)}/")
        sliced = slice_session(raw_path, year, race_name, session_type)
        
        if sliced.empty:
            print(f"  No corner segments produced for {filename}")
            continue

        sliced.to_parquet(out_path, index=False)
        print(f"  Saved {len(sliced)} corner rows to {out_path}")

    print("\n=== Race Slicing and Routing Complete ===")


if __name__ == "__main__":
    run_slicing()