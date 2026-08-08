"""
Build Final Training Dataset -- Ward-Year Table for LR / RF Baselines
--------------------------------------------------------------------------
Assembles the actual model-ready dataset from everything built so far:
  1. ward_year_master_with_flood_labels.csv  -- static terrain/LULC/road
     + annual climate + region flood-magnitude columns (from
     tier_a_annual_flood_labels.py)
  2. ward_susceptibility_index.csv           -- Tier B composite physical
     risk score (from tier_b_ward_susceptibility.py)

This covers Linear Regression and Random Forest, your two baseline
models. It does NOT cover LSTM or LSTM+GNN -- those need a ward x DAY
sequential table and a ward-adjacency graph respectively, neither of
which exist yet (see the printed NEXT STEPS note at the end of this
script's output).

WHY THE TARGET CHANGED FROM BINARY TO REGRESSION
We found empirically that "did this region flood this year" is positive
in ~84% of rows at district-year grain -- there's essentially no
negative class left, making it a poor classification target. Instead
this script builds TWO usable targets from the same underlying data:
  - flood_magnitude_target (regression): region_flood_event_count,
    which does vary meaningfully year to year (1-10 events)
  - high_flood_year_target (classification): 1 if flood_magnitude_target
    is in the top 25% for that YEAR across all regions (a relative,
    not absolute, threshold -- keeps the label meaningful even though
    raw counts trend with reporting-era differences), else 0

DROPPED COLUMNS AND WHY
  - dominant_hsg          : constant ('C') for every row -- zero information
  - region_max_severity   : 100% empty in this IFI version -- dead column
  - region_any_flood_year : the near-saturated binary discussed above --
    kept out of the FEATURE set (would leak the target trivially) but
    left in the raw merged file for reference
  - second_dominant_class : redundant text version of
    second_dominant_class_pct, which is kept

REQUIREMENTS
    pip install pandas numpy

HOW TO USE
1. Point the two input paths below at your real files
2. Run: python build_final_training_dataset.py
3. Output:
   - final_ward_year_dataset.csv        : full assembled table
   - final_ward_year_train.csv          : train split (years < TEST_START_YEAR)
   - final_ward_year_test.csv           : test split (years >= TEST_START_YEAR)
   - prints a feature/target data dictionary to the console
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------

WARD_MASTER_WITH_LABELS_CSV = "ward_year_master_with_flood_labels.csv"
SUSCEPTIBILITY_CSV = "ward_susceptibility_index.csv"

OUTPUT_FULL = "final_ward_year_dataset.csv"
OUTPUT_TRAIN = "final_ward_year_train.csv"
OUTPUT_TEST = "final_ward_year_test.csv"

TEST_START_YEAR = 2019   # last ~5 years held out, matches the daily script's split era

DEAD_COLUMNS = ["dominant_hsg", "region_max_severity"]
LEAKY_OR_REDUNDANT_COLUMNS = ["region_any_flood_year", "second_dominant_class"]

# High-flood-year classification threshold: top X% of flood_magnitude_target
# WITHIN EACH YEAR (relative, not absolute -- see docstring).
HIGH_FLOOD_QUANTILE = 0.75


# ---------------------------------------------------------------------
# 2. LOAD + MERGE
# ---------------------------------------------------------------------

def load_and_merge() -> pd.DataFrame:
    print(f"Loading {WARD_MASTER_WITH_LABELS_CSV} ...")
    master = pd.read_csv(WARD_MASTER_WITH_LABELS_CSV)
    print(f"  {master.shape[0]} rows, {master.shape[1]} columns.")

    print(f"Loading {SUSCEPTIBILITY_CSV} ...")
    susc = pd.read_csv(SUSCEPTIBILITY_CSV)
    print(f"  {susc.shape[0]} rows, {susc.shape[1]} columns.")

    # Susceptibility index is one row per ward (static snapshot), master
    # is ward x year -- broadcast the static index across every year for
    # that ward via a many-to-one merge on ward id.
    susc_cols = ["ward", "susceptibility_index", "tier"] + [
        c for c in susc.columns if c.startswith("risk_") or c.startswith("raw_")
    ]
    merged = master.merge(
        susc[susc_cols].rename(columns={"ward": "ward_uid"}),
        on="ward_uid", how="left",
    )
    n_unmatched = merged["susceptibility_index"].isna().sum()
    if n_unmatched:
        print(f"  WARNING: {n_unmatched} rows had no matching ward in "
              f"{SUSCEPTIBILITY_CSV} -- check ward_uid values line up "
              f"between the two files (e.g. naming or encoding mismatch).")
    else:
        print("  All rows matched a susceptibility score.")
    return merged


# ---------------------------------------------------------------------
# 3. CLEAN + BUILD TARGETS
# ---------------------------------------------------------------------

def clean_and_target(df: pd.DataFrame) -> pd.DataFrame:
    dropped = [c for c in DEAD_COLUMNS + LEAKY_OR_REDUNDANT_COLUMNS if c in df.columns]
    df = df.drop(columns=dropped)
    print(f"\nDropped dead/redundant columns: {dropped}")

    df["flood_magnitude_target"] = df["region_flood_event_count"]

    # Relative top-quantile threshold computed PER YEAR (across regions
    # present that year) rather than a single global cutoff -- avoids
    # the label just tracking "which years IFI happened to log more
    # events overall" (e.g. reporting improved in later decades).
    year_threshold = df.groupby("year")["flood_magnitude_target"].transform(
        lambda s: s.quantile(HIGH_FLOOD_QUANTILE)
    )
    df["high_flood_year_target"] = (
        df["flood_magnitude_target"] > year_threshold
    ).astype(int)

    pos_rate = df["high_flood_year_target"].mean()
    print(f"high_flood_year_target positive rate: {pos_rate:.1%} "
          f"(target ~{1 - HIGH_FLOOD_QUANTILE:.0%} by construction).")
    print(f"flood_magnitude_target summary:\n{df['flood_magnitude_target'].describe().to_string()}")

    return df


def one_hot_dominant_lulc(df: pd.DataFrame) -> pd.DataFrame:
    if "dominant_lulc_class" not in df.columns:
        return df
    dummies = pd.get_dummies(
        df["dominant_lulc_class"], prefix="dominant_lulc", dtype=int
    )
    print(f"\nOne-hot encoded dominant_lulc_class -> {list(dummies.columns)}")
    return pd.concat([df, dummies], axis=1)


# ---------------------------------------------------------------------
# 4. SPLIT
# ---------------------------------------------------------------------

def year_based_split(df: pd.DataFrame):
    train = df[df["year"] < TEST_START_YEAR].copy()
    test = df[df["year"] >= TEST_START_YEAR].copy()
    print(f"\nTrain: {len(train)} rows, years {train['year'].min()}-{train['year'].max()}")
    print(f"Test:  {len(test)} rows, years {test['year'].min()}-{test['year'].max()}")
    print(f"Train high_flood_year_target positive rate: "
          f"{train['high_flood_year_target'].mean():.1%}")
    print(f"Test  high_flood_year_target positive rate: "
          f"{test['high_flood_year_target'].mean():.1%}")
    return train, test


# ---------------------------------------------------------------------
# 5. DATA DICTIONARY
# ---------------------------------------------------------------------

def print_data_dictionary(df: pd.DataFrame):
    id_cols = ["region", "ward_uid", "ward", "year"]
    target_cols = ["flood_magnitude_target", "high_flood_year_target"]
    id_target = set(id_cols + target_cols)
    feature_cols = [c for c in df.columns if c not in id_target]
    numeric_features = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    non_numeric = [c for c in feature_cols if c not in numeric_features]

    print("\n" + "=" * 60)
    print("FINAL DATASET -- DATA DICTIONARY")
    print("=" * 60)
    print(f"ID columns        : {id_cols}")
    print(f"Regression target : flood_magnitude_target "
          f"(= region_flood_event_count, that region's count of IFI-"
          f"matched flood events that year)")
    print(f"Classification target : high_flood_year_target "
          f"(1 if in top {1-HIGH_FLOOD_QUANTILE:.0%} of "
          f"flood_magnitude_target within that year, else 0)")
    print(f"\nNumeric feature columns ({len(numeric_features)}):")
    print(f"  {numeric_features}")
    if non_numeric:
        print(f"\nNon-numeric columns still present, NOT auto-usable by "
              f"LR/RF as-is ({len(non_numeric)}): {non_numeric}")
        print("  -- either one-hot encode these yourself or drop them "
              "before fitting sklearn models.")
    print("=" * 60)


# ---------------------------------------------------------------------
# 6. MAIN
# ---------------------------------------------------------------------

def main():
    df = load_and_merge()
    df = clean_and_target(df)
    df = one_hot_dominant_lulc(df)

    df.to_csv(OUTPUT_FULL, index=False)
    print(f"\nSaved full dataset -> {OUTPUT_FULL}")

    train, test = year_based_split(df)
    train.to_csv(OUTPUT_TRAIN, index=False)
    test.to_csv(OUTPUT_TEST, index=False)
    print(f"Saved -> {OUTPUT_TRAIN}, {OUTPUT_TEST}")

    print_data_dictionary(df)

    print("\nNEXT STEPS for the remaining 2 models:")
    print("  - LSTM needs a ward x DAY table (rainfall/soil-moisture per")
    print("    day, broadcast from region to ward, with a day-level flood")
    print("    label) -- not built yet. tier_a_region_hazard_model.py has")
    print("    the region-day version; broadcasting it down to wards and")
    print("    joining static ward features is the next script to write.")
    print("  - LSTM+GNN additionally needs a ward-adjacency graph (which")
    print("    wards border which) -- buildable from your ward KML/GeoJSON")
    print("    polygons via shapely 'touches'/'intersects', not built yet.")


if __name__ == "__main__":
    main()