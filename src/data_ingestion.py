"""
src/data_ingestion.py

Automated raw telemetry ingestion pipeline using FastF1.
Targeted strictly at Race ('R') sessions for the 2024 season across 10 tracks.
Output format: fastf1_data/raw/{year}_{location}_R.csv
"""

import os
import fastf1
import pandas as pd

RAW_DATA_DIR = "fastf1_data/raw"
CACHE_DIR = "fastf1_cache"

TARGET_TRACKS = [
    "Monza", "Monaco", "Singapore", "Silverstone", 
    "Suzuka", "Austin", "Bahrain", "Austria", 
    "Belgium", "Jeddah"
]

SEASONS = [2024]
SESSIONS = ["R"]  # Exclusively Race Data


def setup_environment():
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    fastf1.Cache.enable_cache(CACHE_DIR)


def fetch_session_telemetry(year: int, location: str, session_type: str = "R") -> bool:
    """
    Downloads and exports whole-lap raw telemetry to fastf1_data/raw/{year}_{location}_R.csv
    """
    file_name = f"{year}_{location}_{session_type}.csv"
    output_path = os.path.join(RAW_DATA_DIR, file_name)

    if os.path.exists(output_path):
        print(f"Skipping (Already exists): {file_name}")
        return True

    print(f"Fetching: {year} {location} - Race (R)...")

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

                # Preserve SessionTime as string timedelta format so pandas parses it cleanly on load
                df_telemetry = pd.DataFrame({
                    "SessionTime": telemetry["SessionTime"].astype(str),
                    "Distance": telemetry["Distance"],
                    "Speed": telemetry["Speed"],
                    "Throttle": telemetry["Throttle"],
                    "Brake": telemetry["Brake"],
                    "RPM": telemetry["RPM"],
                    "nGear": telemetry["nGear"],
                    "driver": lap["Driver"],  # Kept strictly as metadata
                    "lap_number": lap["LapNumber"],
                    "session_type": session_type,
                    "year": year,
                    "race": location
                })

                # Include SteeringAngle if FastF1 dataset contains it
                if "SteeringAngle" in telemetry.columns:
                    df_telemetry["SteeringAngle"] = telemetry["SteeringAngle"]

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
        print(f"  [Error] Failed to fetch {year} {location} Race: {e}")
        return False


def run_ingestion():
    setup_environment()
    print("=== Starting FastF1 Race Ingestion Pipeline (2024 Season: Race Only) ===\n")

    for year in SEASONS:
        for track in TARGET_TRACKS:
            fetch_session_telemetry(year, track, "R")

    print("\n=== Race Data Ingestion Complete ===")


if __name__ == "__main__":
    run_ingestion()