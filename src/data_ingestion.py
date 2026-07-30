"""
src/data_ingestion.py

Automated raw telemetry ingestion pipeline using FastF1.
Targeted at 10 high-variance tracks for the 2024 season.
Sessions targeted: FP1, Q, R (Standard 3-session subset).
Output format: fastf1_data/raw/{year}_{location}_{session}.csv
"""

import os
import fastf1
import pandas as pd

RAW_DATA_DIR = "fastf1_data/raw"
CACHE_DIR = "fastf1_cache"

# Core 8 Tracks + 2 Zero-Shot Holdout Tracks
TARGET_TRACKS = [
    "Monza", "Monaco", "Singapore", "Silverstone", 
    "Suzuka", "Austin", "Bahrain", "Austria", 
    "Spa", "Jeddah"
]

SEASONS = [2024]
SESSIONS = ["FP1", "Q", "R"]


def setup_environment():
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    fastf1.Cache.enable_cache(CACHE_DIR)


def fetch_session_telemetry(year: int, location: str, session_type: str) -> bool:
    """
    Downloads and exports whole-lap raw telemetry to fastf1_data/raw/{year}_{location}_{session}.csv
    """
    file_name = f"{year}_{location}_{session_type}.csv"
    output_path = os.path.join(RAW_DATA_DIR, file_name)

    if os.path.exists(output_path):
        print(f"Skipping (Already exists): {file_name}")
        return True

    print(f"Fetching: {year} {location} - Session {session_type}...")

    try:
        session = fastf1.get_session(year, location, session_type)
        session.load(telemetry=True, laps=True, weather=False, messages=False)

        laps = session.laps.pick_quicklaps()
        if laps.empty:
            print(f"  [Warning] No quicklaps found for {file_name}")
            return False

        all_laps_telemetry = []

        for _, lap in laps.iterrows():
            try:
                telemetry = lap.get_telemetry()
                if telemetry.empty:
                    continue

                # Extract essential numerical and string channels
                df_telemetry = pd.DataFrame({
                    "Distance": telemetry["Distance"],
                    "Speed": telemetry["Speed"],
                    "Throttle": telemetry["Throttle"],
                    "Brake": telemetry["Brake"],
                    "RPM": telemetry["RPM"],
                    "nGear": telemetry["nGear"],
                    "driver": lap["Driver"],
                    "lap_number": lap["LapNumber"],
                    "session_type": session_type,
                    "year": year,
                    "race": location
                })
                all_laps_telemetry.append(df_telemetry)
            except Exception:
                continue

        if not all_laps_telemetry:
            print(f"  [Warning] Could not parse lap telemetry for {file_name}")
            return False

        df_session = pd.concat(all_laps_telemetry, ignore_index=True)
        df_session.to_csv(output_path, index=False)
        print(f"  [Success] Saved -> {output_path} ({len(df_session)} rows)")
        return True

    except Exception as e:
        print(f"  [Error] Failed to fetch {year} {location} {session_type}: {e}")
        return False


def run_ingestion():
    setup_environment()
    print("=== Starting FastF1 Ingestion Pipeline (2024 Season: FP1, Q, R) ===\n")

    for year in SEASONS:
        for track in TARGET_TRACKS:
            for session in SESSIONS:
                fetch_session_telemetry(year, track, session)

    print("\n=== Data Ingestion Pipeline Complete ===")


if __name__ == "__main__":
    run_ingestion()