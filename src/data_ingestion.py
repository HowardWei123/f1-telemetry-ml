"""
src/data_ingestion.py

Multi-session scraping loop with caching controls.
Pulls raw telemetry from FastF1 for a list of (year, race, session_type)
combinations and saves each session as a CSV under fastf1_data/raw/.

Usage (standalone):
    python src/data_ingestion.py

Usage (from a notebook, e.g. 01_data_pipeline.ipynb):
    from src.data_ingestion import run_ingestion
    run_ingestion()
"""
import os
import fastf1
import pandas as pd


CACHE_DIR = "fastf1_cache"
RAW_DATA_DIR = "fastf1_data/raw"

# Start small (3-5 races) while building/debugging the pipeline.
# Expand this list once everything downstream works end-to-end.
RACES = [
    (2024, "Monza"),
    (2024, "Silverstone"),
    (2024, 'Belgian Grand Prix'),
]
SESSION_TYPES = ["Q", "R"]  # Qualifying (clean baseline) + Race (drift analysis)


def setup_cache():
    """Create the cache and raw-data folders if they don't exist yet."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    fastf1.Cache.enable_cache(CACHE_DIR)


def fetch_session_telemetry(year: int, race_name: str, session_type: str) -> pd.DataFrame:
    session = fastf1.get_session(year, race_name, session_type)
    session.load(telemetry=True, laps=True, weather=False)

    all_driver_data = []
    skipped_flat = 0
    for drv in session.drivers:
        driver_laps = session.laps.pick_drivers(drv)
        for _, lap in driver_laps.iterlaps():
            try:
                tel = lap.get_car_data().add_distance()

                # Sanity check: real telemetry should have meaningful speed
                # variation. A near-constant Speed channel (e.g. FastF1
                # silently returning placeholder/degraded data instead of
                # raising when telemetry truly failed to load) would
                # otherwise slip through the try/except below undetected.
                if tel["Speed"].std() < 1.0:
                    skipped_flat += 1
                    continue

                tel["driver"] = drv
                tel["lap_number"] = lap["LapNumber"]
                tel["session_type"] = session_type
                tel["year"] = year
                tel["race"] = race_name
                all_driver_data.append(tel)
            except Exception as e:
                print(f"  Skipped lap {lap['LapNumber']} for {drv}: {e}")
                continue

    if skipped_flat:
        print(f"  Skipped {skipped_flat} laps with suspiciously flat/placeholder telemetry")

    if not all_driver_data:
        return pd.DataFrame()
    return pd.concat(all_driver_data, ignore_index=True)


def run_ingestion(races=None, session_types=None):
    """
    Run the full ingestion loop over all configured races/sessions
    and save each one as a CSV under fastf1_data/raw/.
    """
    setup_cache()
    races = races or RACES
    session_types = session_types or SESSION_TYPES

    for year, race in races:
        for session_type in session_types:
            out_path = f"{RAW_DATA_DIR}/{year}_{race}_{session_type}.csv"

            if os.path.exists(out_path):
                print(f"Already exists, skipping: {out_path}")
                continue

            print(f"Fetching {year} {race} {session_type}...")
            df = fetch_session_telemetry(year, race, session_type)
            df.to_csv(out_path, index=False)
            print(f"  Saved {len(df)} rows to {out_path}")


if __name__ == "__main__":
    run_ingestion()