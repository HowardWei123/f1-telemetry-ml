"""
src/corner_slicer.py

Spatial geometric indexing engine to isolate corner intervals.
Slices raw whole-lap CSVs into uniform spatial corner segments (100 points)
and routes output Parquet files directly into train/val/test split directories.
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

# Track subsets for directory routing
CORE_TRACKS = ["Monza", "Monaco", "Singapore", "Silverstone", "Suzuka", "Austin", "Bahrain", "Austria"]
HOLDOUT_TRACKS = ["Spa", "Jeddah"]

# Slicing Parameters
WINDOW_METERS_BEFORE = 200
WINDOW_METERS_AFTER = 100
TARGET_POINTS_PER_CORNER = 100  # Fixed spatial length for PyTorch tensors
SMOOTH_WINDOW = 15
SMOOTH_POLYORDER = 2

# In-memory lookup cache to prevent repeating FastF1 API calls
CORNER_CACHE = {}


def setup_environment():
    os.makedirs(CACHE_DIR, exist_ok=True)
    fastf1.Cache.enable_cache(CACHE_DIR)
    
    # Ensure all dataset split folders exist
    for split in ["train", "val", "test_in_dist", "test_zero_shot"]:
        os.makedirs(os.path.join(PROCESSED_DATA_DIR, split), exist_ok=True)


def get_corner_markers(year: int, race_name: str) -> pd.DataFrame:
    """
    Retrieves track corner markers, using an in-memory cache to prevent
    redundant network requests to FastF1 for the same track.
    """
    cache_key = (year, race_name)
    if cache_key in CORNER_CACHE:
        return CORNER_CACHE[cache_key]

    try:
        session = fastf1.get_session(year, race_name, "Q")
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
    """
    Interpolates varying telemetry point counts onto a uniform grid 
    (100 points per corner slice) for direct consumption by PyTorch / ML models.
    """
    if len(df_segment) < 5:
        return pd.DataFrame()

    distances = df_segment["Distance"].values
    
    # Remove duplicate spatial distance values
    _, unique_indices = np.unique(distances, return_index=True)
    if len(unique_indices) < 4:
        return pd.DataFrame()

    df_clean = df_segment.iloc[unique_indices].sort_values("Distance")
    dist_clean = df_clean["Distance"].values

    uniform_dist = np.linspace(dist_clean[0], dist_clean[-1], num_points)
    resampled_data = {"Distance": uniform_dist}

    # Interpolate numeric channels onto uniform distance grid
    numeric_cols = ["Speed", "Throttle", "Brake", "RPM", "nGear"]
    for col in numeric_cols:
        if col in df_clean.columns:
            interp_func = interp1d(dist_clean, df_clean[col].values, kind="linear", fill_value="extrapolate")
            resampled_data[col] = interp_func(uniform_dist)

    resampled_df = pd.DataFrame(resampled_data)

    # Apply Savitzky-Golay smoothing on uniform signal channels
    for col in ["Throttle", "Brake", "Speed"]:
        if col in resampled_df.columns:
            resampled_df[f"{col}_smooth"] = smooth_channel(resampled_df[col].to_numpy())

    # Retain categorical metadata across interpolated rows
    for meta_col in ["driver", "lap_number", "session_type", "year", "race", "corner_number"]:
        if meta_col in df_segment.columns:
            resampled_df[meta_col] = df_segment[meta_col].iloc[0]

    return resampled_df


def slice_session(raw_path: str, year: int, race_name: str, session_type: str) -> pd.DataFrame:
    """
    Slices a raw CSV into uniform corner segments.
    """
    if not os.path.exists(raw_path):
        return pd.DataFrame()

    raw = pd.read_csv(raw_path)
    if raw.empty:
        return pd.DataFrame()

    corners = get_corner_markers(year, race_name)
    if corners.empty:
        return pd.DataFrame()

    all_segments = []

    # Process grouped laps
    for (driver, lap_num), lap_df in raw.groupby(["driver", "lap_number"]):
        lap_df = lap_df.sort_values("Distance")

        for _, corner in corners.iterrows():
            center = corner["Distance"]
            corner_num = corner["Number"]

            # Spatial window bounding around corner center
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


def get_target_split_dir(year: int, race_name: str, session_type: str) -> str:
    """
    Determines dataset split destination directory for 2024 data:
    - Holdout Tracks (Spa, Jeddah): test_zero_shot
    - Core Tracks FP1: train
    - Core Tracks Q: val
    - Core Tracks R: test_in_dist
    """
    if race_name in HOLDOUT_TRACKS:
        return os.path.join(PROCESSED_DATA_DIR, "test_zero_shot")
    
    if race_name in CORE_TRACKS:
        if session_type == "FP1":
            return os.path.join(PROCESSED_DATA_DIR, "train")
        elif session_type == "Q":
            return os.path.join(PROCESSED_DATA_DIR, "val")
        elif session_type == "R":
            return os.path.join(PROCESSED_DATA_DIR, "test_in_dist")
                
    return PROCESSED_DATA_DIR


def run_slicing():
    setup_environment()
    raw_files = glob.glob(f"{RAW_DATA_DIR}/*.csv")

    if not raw_files:
        print(f"No raw files found in {RAW_DATA_DIR}/ — run data_ingestion.py first.")
        return

    print(f"Found {len(raw_files)} raw CSV sessions to process...\n")

    for raw_path in raw_files:
        filename = os.path.basename(raw_path).replace(".csv", "")
        
        # Strictly split on 2 underscores: {year}_{location}_{session}
        year_str, race_name, session_type = filename.split("_", 2)
        year = int(year_str)

        target_dir = get_target_split_dir(year, race_name, session_type)
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

    print("\n=== Slicing and Data Routing Complete ===")


if __name__ == "__main__":
    run_slicing()