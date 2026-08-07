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

# FIX: tracks used to FIT the normalization scale. Must match whatever
# TRAIN_TRACKS is defined as in the notebook that does the actual
# train/val/test_in_dist/test_zero_shot split, so the scale is computed
# only from data the model will actually train on.
TRAIN_TRACKS = ["Monza", "Monaco", "Silverstone", "Suzuka", "Austin", "Bahrain"]


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
    speed = corner_df.sort_values("Distance")["Speed_smooth"].to_numpy()
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
    return float(-decel_rate)


# =====================================================================
# FIX: normalization split into fit (train-only) and transform (any split)
# =====================================================================

def fit_log_scale(series: pd.Series) -> tuple:
    """Compute the log-scale (lo, hi) bounds from a series — call this on
    TRAINING DATA ONLY, then use apply_log_scale() to transform any split
    using these same fitted bounds. Mirrors sklearn's fit()/transform()
    pattern, applied to this custom normalization instead of StandardScaler."""
    log_series = np.log1p(series)
    return log_series.min(), log_series.max()


def apply_log_scale(series: pd.Series, lo: float, hi: float) -> pd.Series:
    """Apply previously-fitted log-scale bounds to any series (train, val,
    test, or zero-shot). Does NOT recompute lo/hi from this series — this
    is what prevents val/test/zero-shot data from influencing the scale."""
    log_series = np.log1p(series)
    if hi == lo:
        return series * 0
    # Values outside the fitted [lo, hi] range (possible if val/test contains
    # more extreme values than train saw) are clipped rather than extrapolated,
    # keeping the output safely within [0, 1].
    return ((log_series - lo) / (hi - lo)).clip(0, 1)


def fit_percentile_clip(series: pd.Series, lower_pct=1, upper_pct=99) -> tuple:
    """Compute the percentile-clip (lo, hi) bounds from a series — call this
    on TRAINING DATA ONLY, then use apply_percentile_clip() for any split."""
    lo = series.quantile(lower_pct / 100)
    hi = series.quantile(upper_pct / 100)
    return lo, hi


def apply_percentile_clip(series: pd.Series, lo: float, hi: float) -> pd.Series:
    """Apply previously-fitted percentile-clip bounds to any series."""
    if hi == lo:
        return series * 0
    clipped = series.clip(lo, hi)
    return (clipped - lo) / (hi - lo)


def filter_valid_laps(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    time_col = None
    for col in ["SessionTime", "Time", "Date"]:
        if col in df.columns:
            time_col = col
            break

    if time_col is None:
        print("Warning: No timing column found. Skipping valid lap filtering.")
        return df

    df["_TimeDelta"] = pd.to_timedelta(df[time_col])

    group_keys = ["year", "race", "session_type", "driver", "lap_number"]
    lap_times = df.groupby(group_keys)["_TimeDelta"].agg(lambda x: x.max() - x.min())
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

    return df[mask].drop(columns=["_TimeDelta"])


def generate_labels(segmented_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute RAW style scores (no normalization yet) for every
    (year, race, session_type, driver, lap, corner) group in the given
    segmented DataFrame.

    FIX: this used to also normalize per-file, which was thrown away anyway
    once combine_and_renormalize() ran. Now it only computes raw scores —
    normalization happens exactly once, in combine_and_renormalize(),
    fit strictly on training-track data.
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

    return pd.DataFrame(records).dropna()


def run_labeling(force: bool = False):
    """
    Label every sliced parquet file currently sitting in
    fastf1_data/processed/, saving RAW (un-normalized) scores to
    fastf1_data/labeled/. Normalization happens once, in
    combine_and_renormalize(), fit only on training-track data.
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


def combine_and_renormalize(train_tracks: list = None):
    """
    Combine every per-file RAW label set, then normalize to 0.0-1.0 using
    a scale FIT ONLY on training-track rows and APPLIED to every row
    (train, val, test_in_dist, test_zero_shot alike).

    FIX (this is the actual leakage fix): previously this fit the log-scale
    and percentile-clip bounds using ALL races combined, including val/
    test_in_dist/test_zero_shot tracks — meaning held-out data influenced
    the scale used to compute training labels too. Now the fit step only
    ever looks at train_tracks rows; every other row is transformed using
    those same fitted bounds, never used to recompute them.

    train_tracks: list of race names to fit the normalization scale on.
    Defaults to TRAIN_TRACKS at the top of this file — make sure that list
    matches whatever your split notebook uses as its training tracks.
    """
    if train_tracks is None:
        train_tracks = TRAIN_TRACKS

    label_files = [f for f in glob.glob(f"{LABELED_DATA_DIR}/labels_*.parquet")
                   if "labels_combined.parquet" not in f]
    if not label_files:
        print("No per-file labels found — run run_labeling() first.")
        return

    all_labels = pd.concat([pd.read_parquet(f) for f in label_files], ignore_index=True)

    # FIX: fit bounds using ONLY train_tracks rows
    train_mask = all_labels["race"].isin(train_tracks)
    train_rows = all_labels[train_mask]

    if train_rows.empty:
        raise ValueError(
            f"No rows matched train_tracks={train_tracks} — check that these "
            f"race names exactly match the 'race' column values in your labeled data."
        )

    agg_lo, agg_hi = fit_log_scale(train_rows["aggression_raw"])
    line_lo, line_hi = fit_percentile_clip(train_rows["line_shape_raw"])
    over_lo, over_hi = fit_percentile_clip(train_rows["oversteer_raw"])

    print(f"Fitted normalization scale on {len(train_rows)} rows from tracks: {train_tracks}")
    print(f"  aggression log-scale bounds: ({agg_lo:.4f}, {agg_hi:.4f})")
    print(f"  line_shape percentile bounds: ({line_lo:.4f}, {line_hi:.4f})")
    print(f"  oversteer percentile bounds: ({over_lo:.4f}, {over_hi:.4f})")

    # FIX: apply (not re-fit) those bounds to EVERY row, train and held-out alike
    all_labels["aggression_score"] = apply_log_scale(all_labels["aggression_raw"], agg_lo, agg_hi)
    all_labels["line_shape_score"] = apply_percentile_clip(all_labels["line_shape_raw"], line_lo, line_hi)
    all_labels["oversteer_preference_score"] = apply_percentile_clip(all_labels["oversteer_raw"], over_lo, over_hi)

    out_path = f"{LABELED_DATA_DIR}/labels_combined.parquet"
    all_labels.to_parquet(out_path)
    print(f"Saved {len(all_labels)} combined, consistently-scaled labels to {out_path}")
    return all_labels


if __name__ == "__main__":
    import sys
    force_flag = "--force" in sys.argv
    run_labeling(force=force_flag)
    combine_and_renormalize()