"""
src/corner_slicer.py

Spatial geometric indexing engine to isolate corner intervals.

Takes the raw, whole-lap telemetry produced by data_ingestion.py and cuts
each lap into small windows around each corner on the track. Also applies
light smoothing to the channels that heuristic_engine.py will later
differentiate (Throttle, Brake, Speed), since raw sensor noise gets
amplified badly by derivatives if left unsmoothed.

Usage (standalone):
    python src/corner_slicer.py

Usage (from a notebook, e.g. 01_data_pipeline.ipynb):
    from src.corner_slicer import run_slicing
    run_slicing()
"""
import os
import glob
import fastf1
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter


RAW_DATA_DIR = "fastf1_data/raw"
PROCESSED_DATA_DIR = "fastf1_data/processed"

WINDOW_METERS_BEFORE = 200   # braking zones typically start well before the corner marker
WINDOW_METERS_AFTER = 100    # exit/acceleration phase needs less room
SMOOTH_WINDOW = 15   # Savitzky-Golay smoothing window (must be odd)
SMOOTH_POLYORDER = 2


def get_corner_markers(year: int, race_name: str) -> pd.DataFrame:
    """
    Get each corner's location on the track, as a distance value (meters
    from the start/finish line). FastF1 exposes this via circuit info,
    which we pull from the qualifying session for that race weekend.
    """
    session = fastf1.get_session(year, race_name, "R")
    session.load(telemetry=True, laps=True)  # both needed: get_circuit_info() computes marker distances from telemetry
    circuit_info = session.get_circuit_info()
    return circuit_info.corners  # has 'Distance' and 'Number' columns


def smooth_channel(series: np.ndarray, min_val=0, max_val=None) -> np.ndarray:
    if len(series) < SMOOTH_WINDOW:
        return series
    smoothed = savgol_filter(series, window_length=SMOOTH_WINDOW, polyorder=SMOOTH_POLYORDER)
    if max_val is not None:
        smoothed = np.clip(smoothed, min_val, max_val)
    else:
        smoothed = np.clip(smoothed, min_val, None)  # only enforce a lower bound
    return smoothed


def segment_lap_into_corners(lap_telemetry: pd.DataFrame, corner_markers: pd.DataFrame) -> list:
    """
    Cut one lap's telemetry into one small DataFrame per corner.
    Returns a list of DataFrames, each tagged with its corner_number.
    """
    segments = []
    for _, corner in corner_markers.iterrows():
        center = corner["Distance"]
        mask = (lap_telemetry["Distance"] >= center - WINDOW_METERS_BEFORE) & \
               (lap_telemetry["Distance"] <= center + WINDOW_METERS_AFTER)
        segment = lap_telemetry[mask].copy()

        if len(segment) < 5:
            continue  # not enough points captured for this corner on this lap, skip it

        CHANNEL_BOUNDS = {
            "Throttle": (0, 100),
            "Brake": (0, 100),
            "Speed": (0, None),  # km/h has no fixed practical upper bound
        }

        for col in ["Throttle", "Brake", "Speed"]:
            min_val, max_val = CHANNEL_BOUNDS[col]
            segment[f"{col}_smooth"] = smooth_channel(segment[col].to_numpy(), min_val=min_val, max_val=max_val)

        segment["corner_number"] = corner["Number"]
        segments.append(segment)
    return segments


def slice_session(year: int, race_name: str, session_type: str) -> pd.DataFrame:
    """
    Load one raw session CSV (already downloaded by data_ingestion.py),
    slice every lap into corner-sized chunks, and return the combined result.
    """
    raw_path = f"{RAW_DATA_DIR}/{year}_{race_name}_{session_type}.csv"
    if not os.path.exists(raw_path):
        print(f"  Raw file not found, skipping: {raw_path}")
        return pd.DataFrame()

    raw = pd.read_csv(raw_path)
    corners = get_corner_markers(year, race_name)

    all_segments = []
    for (driver, lap_num), lap_df in raw.groupby(["driver", "lap_number"]):
        lap_df = lap_df.sort_values("Distance")
        segments = segment_lap_into_corners(lap_df, corners)
        for seg in segments:
            seg["driver"] = driver
            seg["lap_number"] = lap_num
            seg["session_type"] = session_type
            seg["year"] = year
            seg["race"] = race_name
            all_segments.append(seg)

    if not all_segments:
        return pd.DataFrame()
    return pd.concat(all_segments, ignore_index=True)


def run_slicing():
    """
    Slice every raw CSV currently sitting in fastf1_data/raw/ and save
    one combined parquet file per session to fastf1_data/processed/.
    """
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    raw_files = glob.glob(f"{RAW_DATA_DIR}/*.csv")

    if not raw_files:
        print(f"No raw files found in {RAW_DATA_DIR}/ — run data_ingestion.py first.")
        return

    for raw_path in raw_files:
        filename = os.path.basename(raw_path).replace(".csv", "")
        year_str, race_name, session_type = filename.split("_", 2)
        year = int(year_str)

        out_path = f"{PROCESSED_DATA_DIR}/corners_{filename}.parquet"
        if os.path.exists(out_path):
            print(f"Already sliced, skipping: {out_path}")
            continue

        print(f"Slicing {filename}...")
        sliced = slice_session(year, race_name, session_type)
        if sliced.empty:
            print(f"  No corner segments produced for {filename}")
            continue

        sliced.to_parquet(out_path)
        print(f"  Saved {len(sliced)} corner-sample rows to {out_path}")


if __name__ == "__main__":
    run_slicing()