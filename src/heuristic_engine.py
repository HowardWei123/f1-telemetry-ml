"""
src/heuristic_engine.py

Physics formulas calculating the 0.0-1.0 style targets.

Takes the sliced, smoothed corner-sequence data produced by corner_slicer.py
and computes three continuous style scores per corner sequence:
    - aggression_score        (throttle-rate variance / "jerk")
    - line_shape_score         (U-shaped vs V-shaped corner approach)
    - oversteer_preference_score  (simplified instability proxy — see note below)

IMPORTANT NOTE ON oversteer_preference_score:
FastF1's public telemetry does not include steering angle or true IMU-based
lateral acceleration, so a textbook-accurate understeer/oversteer measurement
isn't possible from this data alone. This formula is a SIMPLIFIED PROXY —
it flags corners where a driver decelerates unusually sharply relative to
their own typical pattern for that corner, which hints at instability but
should not be reported as a literal, physically-validated oversteer measurement.
Be upfront about this limitation in the write-up.

Usage (standalone):
    python src/heuristic_engine.py [--force]

Usage (from a notebook):
    from src.heuristic_engine import run_labeling
    run_labeling()
"""
import os
import glob
import pandas as pd
import numpy as np


PROCESSED_DATA_DIR = "fastf1_data/processed"
LABELED_DATA_DIR = "fastf1_data/labeled"

# Laps under this time (seconds) relative to the session's fastest lap are
# considered valid "pushing" laps. Out-laps / in-laps / aborted laps are
# typically much slower and would corrupt the style formulas if included.
MAX_LAPTIME_RATIO = 1.07  # e.g. 1.07 = within 7% of the fastest lap in the session


def compute_aggression_score(corner_df: pd.DataFrame) -> float:
    """
    Jerk-based aggression proxy: variance of the throttle rate of change
    (d Throttle_smooth / dt) across the corner sequence. Higher variance
    means more abrupt, less smooth input modulation.
    """
    throttle = corner_df["Throttle_smooth"].to_numpy()
    if len(throttle) < 3:
        return np.nan

    if "SessionTime" in corner_df:
        time = pd.to_timedelta(corner_df["SessionTime"]).dt.total_seconds().to_numpy()
    else:
        time = np.arange(len(throttle)).astype(float)
    dt = np.gradient(time)
    dt[dt == 0] = 1e-3  # avoid divide-by-zero on duplicate timestamps

    throttle_rate = np.gradient(throttle) / dt
    return float(np.var(throttle_rate))


def compute_line_shape_score(corner_df: pd.DataFrame) -> float:
    """
    U-shape vs V-shape proxy: ratio of minimum corner speed to the average
    of entry/exit speed. Closer to 1.0 = U-shaped (speed stays high through
    the corner). Closer to 0.0 = V-shaped (late braking, low apex speed,
    sharp acceleration out).
    """
    speed = corner_df.sort_values("Distance")["Speed"].to_numpy()
    if len(speed) < 3:
        return np.nan

    entry_speed = speed[0]
    exit_speed = speed[-1]
    min_speed = speed.min()
    avg_entry_exit = (entry_speed + exit_speed) / 2

    if avg_entry_exit == 0:
        return np.nan
    return float(min_speed / avg_entry_exit)


def compute_oversteer_proxy(corner_df: pd.DataFrame) -> float:
    """
    SIMPLIFIED instability proxy (see module docstring) — the sharpest
    deceleration rate observed in the corner. Not a true oversteer measurement.
    """
    speed = corner_df.sort_values("Distance")["Speed_smooth"].to_numpy()
    if len(speed) < 3:
        return np.nan

    decel_rate = np.min(np.gradient(speed))
    return float(decel_rate)


def normalize_log_scale(series: pd.Series) -> pd.Series:
    """Log-transform before min-max scaling — appropriate for variance-based
    metrics (like aggression_raw) which are always >= 0 and heavily
    right-skewed with a long tail of extreme values.
    NOTE: this scale is anchored to whatever data is present when run —
    recalculate for the FULL dataset whenever more races are added,
    rather than only labeling the new races."""
    log_series = np.log1p(series)  # log1p handles zero safely; series must be >= 0
    lo, hi = log_series.min(), log_series.max()
    if hi == lo:
        return series * 0
    return (log_series - lo) / (hi - lo)


def normalize_percentile_clip(series: pd.Series, lower_pct=1, upper_pct=99) -> pd.Series:
    """Min-max normalize using percentile bounds instead of true min/max —
    for metrics that CAN be negative (like oversteer_raw, a deceleration
    rate), where log-transform isn't mathematically valid. Clips extreme
    outliers on both ends so they don't dominate the scale."""
    lo = series.quantile(lower_pct / 100)
    hi = series.quantile(upper_pct / 100)
    if hi == lo:
        return series * 0
    clipped = series.clip(lo, hi)
    return (clipped - lo) / (hi - lo)


def filter_valid_laps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop out-laps, in-laps, and other non-representative laps by keeping
    only laps within MAX_LAPTIME_RATIO of that driver's fastest lap in
    this session. This is a simple heuristic, not perfect — safety car
    laps / red flags may still slip through, worth spot-checking results.

    NOTE: grouping includes year/race/session_type so laps aren't
    accidentally compared/merged across different races or sessions.
    """
    df = df.copy()
    df["SessionTime"] = pd.to_timedelta(df["SessionTime"])

    group_keys = ["year", "race", "session_type", "driver", "lap_number"]
    lap_times = df.groupby(group_keys)["SessionTime"].agg(lambda x: x.max() - x.min())
    lap_times = lap_times.reset_index(name="lap_duration")

    valid_laps = []
    for keys, group in lap_times.groupby(["year", "race", "session_type", "driver"]):
        fastest = group["lap_duration"].min()
        threshold = fastest * MAX_LAPTIME_RATIO
        keep = group[group["lap_duration"] <= threshold]
        valid_laps.append(keep)

    valid_laps_df = pd.concat(valid_laps, ignore_index=True)
    valid_keys = set(zip(
        valid_laps_df["year"], valid_laps_df["race"], valid_laps_df["session_type"],
        valid_laps_df["driver"], valid_laps_df["lap_number"]
    ))

    mask = df.apply(
        lambda row: (row["year"], row["race"], row["session_type"], row["driver"], row["lap_number"]) in valid_keys,
        axis=1
    )
    return df[mask]


def generate_labels(segmented_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute raw + normalized style scores for every
    (year, race, session_type, driver, lap, corner) group in the given
    segmented DataFrame.
    """
    filtered = filter_valid_laps(segmented_df)

    records = []
    for (year, race, session_type, driver, lap_num, corner_num), group in filtered.groupby(
        ["year", "race", "session_type", "driver", "lap_number", "corner_number"]
    ):
        records.append({
            "year": year,
            "race": race,
            "session_type": session_type,
            "driver": driver,
            "lap_number": lap_num,
            "corner_number": corner_num,
            "aggression_raw": compute_aggression_score(group),
            "line_shape_raw": compute_line_shape_score(group),
            "oversteer_raw": compute_oversteer_proxy(group),
        })

    labels_df = pd.DataFrame(records).dropna()

    labels_df["aggression_score"] = normalize_log_scale(labels_df["aggression_raw"])
    labels_df["line_shape_score"] = normalize_percentile_clip(labels_df["line_shape_raw"])
    labels_df["oversteer_preference_score"] = normalize_percentile_clip(labels_df["oversteer_raw"])

    return labels_df
    """
    Compute raw + normalized style scores for every (driver, lap, corner)
    group in the given segmented DataFrame.
    """
    filtered = filter_valid_laps(segmented_df)

    records = []
    for (year, race, session_type, driver, lap_num, corner_num), group in filtered.groupby(
    ["year", "race", "session_type", "driver", "lap_number", "corner_number"]
    ):
        records.append({
            "driver": driver,
            "lap_number": lap_num,
            "corner_number": corner_num,
            "session_type": group["session_type"].iloc[0],
            "year": group["year"].iloc[0],
            "race": group["race"].iloc[0],
            "aggression_raw": compute_aggression_score(group),
            "line_shape_raw": compute_line_shape_score(group),
            "oversteer_raw": compute_oversteer_proxy(group),
        })

    labels_df = pd.DataFrame(records).dropna()

    labels_df["aggression_score"] = normalize_log_scale(labels_df["aggression_raw"])
    labels_df["line_shape_score"] = normalize_percentile_clip(labels_df["line_shape_raw"])
    labels_df["oversteer_preference_score"] = normalize_percentile_clip(labels_df["oversteer_raw"])

    return labels_df


def run_labeling(force: bool = False):
    """
    Label every sliced parquet file currently sitting in
    fastf1_data/processed/, saving results to fastf1_data/labeled/.

    NOTE: normalization is done PER FILE here for simplicity/speed. Once
    you're happy with the formulas, run combine_and_renormalize() (below)
    to recalculate scores across the FULL combined dataset so 0.0-1.0
    means the same thing across every race/session.
    """
    os.makedirs(LABELED_DATA_DIR, exist_ok=True)
    processed_files = glob.glob(f"{PROCESSED_DATA_DIR}/corners_*.parquet")

    if not processed_files:
        print(f"No sliced files found in {PROCESSED_DATA_DIR}/ — run corner_slicer.py first.")
        return

    for path in processed_files:
        filename = os.path.basename(path).replace("corners_", "").replace(".parquet", "")
        out_path = f"{LABELED_DATA_DIR}/labels_{filename}.parquet"

        if os.path.exists(out_path) and not force:
            print(f"Already labeled, skipping: {out_path}")
            continue

        print(f"Labeling {filename}...")
        segmented = pd.read_parquet(path)
        labels = generate_labels(segmented)

        if labels.empty:
            print(f"  No labels produced for {filename} (check filter_valid_laps threshold)")
            continue

        labels.to_parquet(out_path)
        print(f"  Saved {len(labels)} labeled corner-instances to {out_path}")


def combine_and_renormalize():
    """
    Combine every per-file label set and recalculate the 0.0-1.0 scores
    across the FULL dataset, so the scale is consistent everywhere.
    Run this after run_labeling() has processed all your raw files,
    and again any time you add more races.
    """
    label_files = [f for f in glob.glob(f"{LABELED_DATA_DIR}/labels_*.parquet")
                   if "labels_combined.parquet" not in f]
    if not label_files:
        print("No per-file labels found — run run_labeling() first.")
        return

    all_labels = pd.concat([pd.read_parquet(f) for f in label_files], ignore_index=True)

    all_labels["aggression_score"] = normalize_log_scale(all_labels["aggression_raw"])
    all_labels["line_shape_score"] = normalize_percentile_clip(all_labels["line_shape_raw"])
    all_labels["oversteer_preference_score"] = normalize_percentile_clip(all_labels["oversteer_raw"])

    out_path = f"{LABELED_DATA_DIR}/labels_combined.parquet"
    all_labels.to_parquet(out_path)
    print(f"Saved {len(all_labels)} combined, consistently-scaled labels to {out_path}")
    return all_labels


if __name__ == "__main__":
    import sys
    force_flag = "--force" in sys.argv
    run_labeling(force=force_flag)
    combine_and_renormalize()