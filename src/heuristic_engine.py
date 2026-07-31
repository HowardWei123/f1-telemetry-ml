"""
src/heuristic_engine.py

Calculates natural, bell-curve distributed (0.0 - 1.0) driver style metrics from Race data.
Uses Robust Z-Score + Sigmoid scaling to preserve realistic driving style variation
without artificial uniform ranking or outlier collapse.
"""

import os
import glob
import pandas as pd
import numpy as np

PROCESSED_DATA_DIR = "fastf1_data/processed"
LABELED_DATA_DIR = "fastf1_data/labeled"

MAX_LAPTIME_RATIO = 1.07 

MAIN_DRIVERS_2024 = {
    "VER", "PER", "LEC", "SAI", "NOR", "PIA", "RUS", "HAM",
    "ALO", "STR", "GAS", "OCO", "TSU", "RIC", "LAW", "BOT",
    "ZHO", "MAG", "HUL", "ALB", "SAR", "COL"
}


def apply_track_corner_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Normalizes raw channels relative to all grid drivers at the specific track & corner."""
    df = df.copy()
    target_cols = [c for c in ["Speed", "Throttle", "Brake", "Speed_smooth", "Throttle_smooth"] if c in df.columns]

    group_keys = ["race", "corner_number"]
    for col in target_cols:
        group_means = df.groupby(group_keys)[col].transform("mean")
        group_stds = df.groupby(group_keys)[col].transform("std").replace(0, 1e-6)
        df[f"{col}_zscore"] = (df[col] - group_means) / group_stds

    return df


def compute_aggression_score(corner_df: pd.DataFrame) -> float:
    """Throttle application variance (jerk) through braking/acceleration phases."""
    throttle = corner_df["Throttle_smooth"].to_numpy() if "Throttle_smooth" in corner_df.columns else corner_df["Throttle"].to_numpy()
    
    # Filter flat-out bends (no style choice involved)
    if np.min(throttle) > 85.0:
        return np.nan

    if "SessionTime" in corner_df.columns and not corner_df["SessionTime"].isna().all():
        time = corner_df["SessionTime"].to_numpy()
        dt = np.gradient(time)
        dt = np.where(dt <= 0, 0.01, dt)
        throttle_rate = np.gradient(throttle) / dt
    else:
        throttle_rate = np.gradient(throttle)

    return float(np.var(throttle_rate))


def compute_line_shape_score(corner_df: pd.DataFrame) -> float:
    """Ratio of minimum apex speed to entry/exit speed profile (V-shape vs U-shape)."""
    speed_col = "Speed_smooth" if "Speed_smooth" in corner_df.columns else "Speed"
    throttle_col = "Throttle_smooth" if "Throttle_smooth" in corner_df.columns else "Throttle"
    
    if corner_df[throttle_col].min() > 85.0:
        return np.nan

    speed = corner_df.sort_values("Distance")[speed_col].to_numpy()
    if len(speed) < 3:
        return np.nan

    entry_speed = speed[0]
    exit_speed = speed[-1]
    min_speed = speed.min()
    avg_entry_exit = (entry_speed + exit_speed) / 2.0

    if avg_entry_exit <= 5.0 or min_speed >= avg_entry_exit:
        return np.nan

    return float(min_speed / avg_entry_exit)


def compute_oversteer_proxy(corner_df: pd.DataFrame) -> float:
    """Deceleration rate variability on corner entry (instability proxy)."""
    speed_col = "Speed_smooth" if "Speed_smooth" in corner_df.columns else "Speed"
    throttle_col = "Throttle_smooth" if "Throttle_smooth" in corner_df.columns else "Throttle"

    if corner_df[throttle_col].min() > 85.0:
        return np.nan

    speed = corner_df.sort_values("Distance")[speed_col].to_numpy()
    if len(speed) < 3:
        return np.nan

    decel_rate = np.gradient(speed)
    # Take standard deviation of deceleration to measure braking stability
    return float(np.std(decel_rate))


def sigmoid_robust_scale(series: pd.Series, k: float = 1.0) -> pd.Series:
    """
    Maps raw metrics to smooth 0.0 - 1.0 bell curves centered at 0.5.
    Uses median and Interquartile Range (IQR) for robust outlier handling.
    """
    valid_mask = series.notna() & np.isfinite(series)
    out = pd.Series(np.nan, index=series.index)
    
    if valid_mask.sum() == 0:
        return out

    vals = series[valid_mask]
    median = vals.median()
    q75, q25 = vals.quantile(0.75), vals.quantile(0.25)
    iqr = q75 - q25

    if iqr == 0:
        iqr = 1e-6

    # Robust z-score
    z_robust = (vals - median) / iqr
    
    # Sigmoidal compression into (0, 1)
    scaled = 1.0 / (1.0 + np.exp(-k * z_robust))
    out[valid_mask] = scaled
    return out


def filter_valid_laps(df: pd.DataFrame) -> pd.DataFrame:
    """Filters out out-laps, in-laps, and non-main grid drivers."""
    df = df.copy()

    if "driver" in df.columns:
        df = df[df["driver"].isin(MAIN_DRIVERS_2024)]

    if df.empty:
        return df

    if "SessionTime" in df.columns and not df["SessionTime"].isna().all():
        group_keys = ["year", "race", "session_type", "driver", "lap_number"]
        lap_times = df.groupby(group_keys)["SessionTime"].agg(lambda x: x.max() - x.min()).reset_index(name="lap_duration")

        valid_laps = []
        for keys, group in lap_times.groupby(["year", "race", "session_type", "driver"]):
            fastest = group["lap_duration"].min()
            threshold = fastest * MAX_LAPTIME_RATIO
            keep = group[group["lap_duration"] <= threshold]
            valid_laps.append(keep)

        if valid_laps:
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

    return df


def generate_labels(segmented_df: pd.DataFrame) -> pd.DataFrame:
    filtered = filter_valid_laps(segmented_df)
    if filtered.empty:
        return pd.DataFrame()

    normalized_df = apply_track_corner_zscore(filtered)

    records = []
    group_cols = ["year", "race", "session_type", "driver", "lap_number", "corner_number"]
    
    for (year, race, session_type, driver, lap_num, corner_num), group in normalized_df.groupby(group_cols):
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

    labels_df = pd.DataFrame(records)
    return labels_df.dropna(subset=["aggression_raw", "line_shape_raw", "oversteer_raw"], how="all")


def run_labeling(force: bool = False):
    processed_files = glob.glob(f"{PROCESSED_DATA_DIR}/**/*.parquet", recursive=True)

    if not processed_files:
        print(f"No sliced files found in {PROCESSED_DATA_DIR}/ — run corner_slicer.py first.")
        return

    print(f"Found {len(processed_files)} sliced Race files to label...\n")

    for path in processed_files:
        rel_dir = os.path.dirname(os.path.relpath(path, PROCESSED_DATA_DIR))
        target_label_dir = os.path.join(LABELED_DATA_DIR, rel_dir)
        os.makedirs(target_label_dir, exist_ok=True)

        filename = os.path.basename(path).replace("corners_", "labels_")
        out_path = os.path.join(target_label_dir, filename)

        if os.path.exists(out_path) and not force:
            print(f"Already labeled, skipping: {out_path}")
            continue

        print(f"Labeling {os.path.basename(path)} -> Routing to {rel_dir}/")
        segmented = pd.read_parquet(path)
        labels = generate_labels(segmented)

        if labels.empty:
            print(f"  [Warning] No labels produced for {filename}")
            continue

        labels.to_parquet(out_path, index=False)
        print(f"  Saved {len(labels)} raw labeled corner rows to {out_path}")


def combine_and_renormalize():
    label_files = glob.glob(f"{LABELED_DATA_DIR}/**/*.parquet", recursive=True)
    label_files = [f for f in label_files if "labels_combined.parquet" not in f]

    if not label_files:
        print("No per-file labels found — run run_labeling() first.")
        return

    all_labels = pd.concat([pd.read_parquet(f) for f in label_files], ignore_index=True)

    # Global Sigmoidal Scaling to create true Gaussian distribution centered at 0.5
    all_labels["aggression_score"] = sigmoid_robust_scale(all_labels["aggression_raw"], k=0.8)
    all_labels["line_shape_score"] = sigmoid_robust_scale(all_labels["line_shape_raw"], k=0.8)
    all_labels["oversteer_preference_score"] = sigmoid_robust_scale(all_labels["oversteer_raw"], k=0.8)

    all_labels = all_labels.dropna(subset=["aggression_score", "line_shape_score", "oversteer_preference_score"])

    out_path = os.path.join(LABELED_DATA_DIR, "labels_combined.parquet")
    all_labels.to_parquet(out_path, index=False)
    print(f"\n=== Saved {len(all_labels)} globally normalized Race labels to {out_path} ===")
    return all_labels


if __name__ == "__main__":
    import sys
    force_flag = "--force" in sys.argv
    run_labeling(force=True)
    combine_and_renormalize()