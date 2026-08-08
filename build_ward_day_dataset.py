"""
Build Ward x Day Sequential Dataset -- for the LSTM model
--------------------------------------------------------------------------
This is the script flagged as "not built yet" in build_final_training_
dataset.py's NEXT STEPS. LR/RF consume final_ward_year_dataset.csv (one
row per ward per YEAR). An LSTM needs sequence, so this script builds
one row per ward per DAY instead, by:

  1. Taking the region-day dynamic climate table already produced by
     tier_a_region_hazard_model.py (tier_a_region_day_features.csv --
     rainfall/soil-moisture/LST + the IFI-derived 'flooded' label, one
     row per region per day, 1990-2024) and BROADCASTING it to every
     ward inside that region on that date.
  2. Joining each ward's YEAR-varying static features (LULC / curve
     number, from wards_with_resolved_cn.csv, 2001-2024) by matching
     (region, ward_uid, year) -- this is why the output is capped to
     2001-2024, see CAVEAT 1 below.
  3. Joining each ward's TRULY static features (terrain from
     all_regions_ward_dem_zonal_stats.csv, road density from
     ward_year_master.csv, and the Tier B susceptibility_index) once
     per ward, since these don't change day to day.

WHY 2001-2024, NOT 1990-2024
The daily climate table goes back to 1990, but the ward-level LULC/CN
panel (wards_with_resolved_cn.csv) only starts in 2001 -- that's the
real ceiling on ward-level features, same as ward_year_master.csv. This
script drops 1990-2000 rather than forward-filling 2001's LULC values
backwards, which would fabricate 11 years of static-feature "history"
that never existed. If you want the full 1990-2024 rainfall record for
sequence length, either (a) accept 2001-2024 (still 24 monsoon
seasons/~8,700 days per ward), or (b) explicitly backfill LULC/CN from
2001 for 1990-2000 yourself and accept that caveat.

CAVEAT -- THE DAY-LEVEL FLOOD LABEL IS STILL REGION-LEVEL, BROADCAST
Exactly like tier_a_region_hazard_model.py: 'flooded' here means "IFI
recorded a flood-level event somewhere in this region's district on
this date", not "this specific ward flooded". Every ward in a region
gets the IDENTICAL label on a given day. This is a real limitation, not
a bug -- no public source gives ward-day ground truth for these cities.
Two ways to make the LSTM's job less naive about it:
  - Use ward_susceptibility_index (joined in below) as an auxiliary
    input/weight so the model has SOME ward-differentiating risk signal
    beyond the shared regional label.
  - Read this model's predictions as "ward-level RISK conditional on
    regional hazard", not a validated ward-level flood forecast.

CAVEAT -- COLUMNS DELIBERATELY EXCLUDED (LEAKAGE)
ward_year_master.csv's ANNUAL columns (annual_rainfall_mm,
annual_mean_soil_moisture_m3m3, annual_mean_lst_c, annual_mean_max_tide_m,
annual_spring_tide_days, region_flood_event_count, region_flood_days_count,
region_any_flood_year, region_total_fatalities) are aggregates computed
OVER the full year -- including days after the one a given row
represents. Joining them onto a day-level row would leak future-in-year
information into every day of that year, and region_flood_event_count /
region_any_flood_year are literally built from the same 'flooded' label
already present here at daily grain. None of those columns are pulled
into this script; only genuinely static-or-slower-changing columns are.

REQUIREMENTS
    pip install pandas numpy

HOW TO USE
1. Run tier_a_region_hazard_model.py first if tier_a_region_day_features.csv
   doesn't exist yet (this script reads that file rather than
   recomputing rainfall/soil-moisture features from scratch).
2. Point the paths in CONFIGURATION at your real files.
3. Run: python build_ward_day_dataset.py
4. Output: ward_day_dataset.csv -- one row per (region, ward_uid, date),
   2001-2024, ~2.6M rows. Also writes a train/test split by year, same
   TEST_START_YEAR convention as build_final_training_dataset.py, so
   an LSTM's held-out years line up with the LR/RF baselines' held-out
   years for a fair comparison.

A NOTE ON SIZE
~2.6M rows x ~35 columns is a few hundred MB as CSV. If that's unwieldy,
rerun with OUTPUT_FORMAT = "parquet" below (needs `pip install pyarrow`)
for a much smaller, faster-to-reload file -- same schema either way.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------

REGION_DAY_FEATURES_CSV = "tier_a_region_day_features.csv"   # from tier_a_region_hazard_model.py
WARD_YEAR_PANEL_CSV = "wards_with_resolved_cn.csv"            # ward x year LULC/CN, 2001-2024
WARD_DEM_CSV = "all_regions_ward_dem_zonal_stats.csv"         # static ward terrain
WARD_MASTER_CSV = "ward_year_master.csv"                      # for road density / area / is_coastal
SUSCEPTIBILITY_CSV = "ward_susceptibility_index.csv"          # Tier B static composite score

OUTPUT_FORMAT = "csv"          # "csv" or "parquet"
OUTPUT_FULL = "ward_day_dataset.csv"
OUTPUT_TRAIN = "ward_day_train.csv"
OUTPUT_TEST = "ward_day_test.csv"

# Matches build_final_training_dataset.py's TEST_START_YEAR so LR/RF and
# LSTM are evaluated on the same held-out years.
TEST_START_YEAR = 2019

# Static columns pulled from ward_year_master.csv -- deliberately NOT
# the annual climate/flood columns (see leakage caveat in the docstring).
SAFE_STATIC_MASTER_COLS = [
    "region", "ward_uid", "area_km2", "road_length_km",
    "road_density_km_per_km2", "is_coastal_region",
]

# Ward-year panel columns to carry onto the day table (LULC/CN -- these
# vary year to year but are NOT derived from rainfall/flood, so they're
# safe to broadcast across every day of their year).
PANEL_FEATURE_COLS = [
    "pct_impervious", "pct_pervious", "dominant_lulc_class",
    "dominant_class_pct", "lulc_diversity_normalized",
    "mean_cn_estimate", "mean_cn_estimate_soilC",
]


# ---------------------------------------------------------------------
# 2. LOAD PIECES
# ---------------------------------------------------------------------

def load_region_day(path: str) -> pd.DataFrame:
    print(f"Loading region-day climate/label table from {path} ...")
    df = pd.read_csv(path, parse_dates=["date"])
    df["year"] = df["date"].dt.year
    df = df[(df["year"] >= 2001) & (df["year"] <= 2024)].copy()
    df = df.rename(columns={"flooded": "region_day_flooded_broadcast"})
    print(f"  {len(df)} region-day rows in 2001-2024 "
          f"({df['region'].nunique()} regions).")
    return df


def load_ward_year_panel(path: str) -> pd.DataFrame:
    print(f"Loading ward x year LULC/CN panel from {path} ...")
    df = pd.read_csv(path)
    cols = ["region", "ward_uid", "ward", "year"] + [
        c for c in PANEL_FEATURE_COLS if c in df.columns
    ]
    missing = [c for c in PANEL_FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"  NOTE: panel is missing expected columns {missing} -- "
              f"skipping those, using what's present.")
    df = df[cols].copy()
    print(f"  {len(df)} ward-year rows, {df['ward_uid'].nunique()} wards, "
          f"years {df['year'].min()}-{df['year'].max()}.")
    return df


def load_static_terrain(path: str) -> pd.DataFrame:
    print(f"Loading static ward terrain from {path} ...")
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in ("region", "ward") if c in df.columns
                           and c != "ward_uid"], errors="ignore")
    # keep region/ward for readability but avoid duplicate merge keys
    print(f"  {len(df)} wards.")
    return df


def load_safe_static_master(path: str) -> pd.DataFrame:
    print(f"Loading static road/area/coastal columns from {path} ...")
    df = pd.read_csv(path)
    cols = [c for c in SAFE_STATIC_MASTER_COLS if c in df.columns]
    df = df[cols].drop_duplicates(subset=["region", "ward_uid"])
    print(f"  {len(df)} unique wards.")
    return df


def load_susceptibility(path: str) -> pd.DataFrame:
    print(f"Loading Tier B susceptibility index from {path} ...")
    df = pd.read_csv(path)
    df = df.rename(columns={"ward": "ward_uid"})
    keep = ["ward_uid", "susceptibility_index", "tier"] + [
        c for c in df.columns if c.startswith("risk_")
    ]
    df = df[[c for c in keep if c in df.columns]]
    print(f"  {len(df)} wards.")
    return df


# ---------------------------------------------------------------------
# 3. ASSEMBLE THE WARD x DAY SKELETON
# ---------------------------------------------------------------------

def build_ward_day_table() -> pd.DataFrame:
    region_day = load_region_day(REGION_DAY_FEATURES_CSV)
    panel = load_ward_year_panel(WARD_YEAR_PANEL_CSV)
    terrain = load_static_terrain(WARD_DEM_CSV)
    safe_static = load_safe_static_master(WARD_MASTER_CSV)
    susc = load_susceptibility(SUSCEPTIBILITY_CSV)

    # ward-year keys (region, ward_uid, ward, year) x every day in that
    # region+year from region_day -> this single merge IS the ward x day
    # skeleton, and brings the daily climate/label columns along with it
    # in one vectorized step (avoids a manual day-by-day/ward-by-ward loop).
    print("\nExpanding ward-year keys against region-day calendar "
          "(this is the actual ward x day cross product -- may take a "
          "little while for ~2.6M rows)...")
    ward_year_keys = panel[["region", "ward_uid", "ward", "year"]].drop_duplicates()
    skeleton = ward_year_keys.merge(region_day, on=["region", "year"], how="inner")
    print(f"  Skeleton built: {len(skeleton)} ward-day rows, "
          f"{skeleton['ward_uid'].nunique()} wards, "
          f"{skeleton['date'].min().date()} to {skeleton['date'].max().date()}.")

    print("\nJoining ward-year LULC/CN features...")
    skeleton = skeleton.merge(
        panel.drop(columns=["ward"]), on=["region", "ward_uid", "year"], how="left"
    )

    print("Joining static ward terrain...")
    skeleton = skeleton.merge(terrain, on="ward_uid", how="left", suffixes=("", "_terrain"))

    print("Joining static road density / area / coastal flag...")
    skeleton = skeleton.merge(safe_static, on=["region", "ward_uid"], how="left")

    print("Joining Tier B susceptibility index...")
    skeleton = skeleton.merge(susc, on="ward_uid", how="left")
    n_missing_susc = skeleton["susceptibility_index"].isna().sum()
    if n_missing_susc:
        print(f"  NOTE: {n_missing_susc} rows ({skeleton.loc[skeleton['susceptibility_index'].isna(), 'ward_uid'].nunique()} wards) "
              f"had no susceptibility score -- likely one of the wards "
              f"dropped from the panel-vs-DEM join upstream. Left as NaN.")

    skeleton = skeleton.sort_values(["region", "ward_uid", "date"]).reset_index(drop=True)
    return skeleton


# ---------------------------------------------------------------------
# 4. SEQUENCE-FRIENDLY ENGINEERED COLUMNS
# ---------------------------------------------------------------------

def add_sequence_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    A couple of cheap additions that specifically help an LSTM (which
    already sees the raw daily series) rather than duplicating what
    tier_a_region_hazard_model.py already computed (rainfall_3day/7day,
    already present via the region_day merge).
    """
    df["day_of_year"] = df["date"].dt.dayofyear
    df["is_monsoon_season"] = df["date"].dt.month.isin([6, 7, 8, 9]).astype(int)
    # Days since this ward's region last saw a flooded day, capped at 365
    # so the model has a "how long has it been dry" signal for the
    # build-up dynamics an LSTM is meant to learn. Computed per region
    # (same for every ward in that region on a given date -- consistent
    # with the label itself being region-broadcast).
    df = df.sort_values(["region", "date"])
    flooded_dates = (
        df[["region", "date", "region_day_flooded_broadcast"]]
        .drop_duplicates(subset=["region", "date"])
        .sort_values(["region", "date"])
    )
    flooded_dates["last_flood_date"] = flooded_dates["date"].where(
        flooded_dates["region_day_flooded_broadcast"] == 1
    )
    flooded_dates["last_flood_date"] = flooded_dates.groupby("region")["last_flood_date"].ffill()
    flooded_dates["days_since_last_regional_flood"] = (
        (flooded_dates["date"] - flooded_dates["last_flood_date"]).dt.days
    )
    flooded_dates["days_since_last_regional_flood"] = flooded_dates[
        "days_since_last_regional_flood"
    ].fillna(9999).clip(upper=365).astype(int)

    df = df.merge(
        flooded_dates[["region", "date", "days_since_last_regional_flood"]],
        on=["region", "date"], how="left",
    )
    return df.sort_values(["region", "ward_uid", "date"]).reset_index(drop=True)


# ---------------------------------------------------------------------
# 5. SPLIT + SAVE
# ---------------------------------------------------------------------

def year_based_split_and_save(df: pd.DataFrame):
    """
    Writes train/test directly from boolean masks without holding extra
    full-size copies in memory at once (the ward x day table is ~2.6M
    rows -- on memory-constrained machines, df + a .copy() of ~90% of it
    + a .copy() of the rest can exceed available RAM). Frees each split
    immediately after writing.
    """
    import gc
    train_mask = df["year"] < TEST_START_YEAR
    print(f"\nTrain: {train_mask.sum()} rows, years "
          f"{df.loc[train_mask, 'year'].min()}-{df.loc[train_mask, 'year'].max()}, "
          f"positive rate {df.loc[train_mask, 'region_day_flooded_broadcast'].mean():.2%}")
    save(df[train_mask], OUTPUT_TRAIN)
    gc.collect()

    test_mask = ~train_mask
    print(f"Test:  {test_mask.sum()} rows, years "
          f"{df.loc[test_mask, 'year'].min()}-{df.loc[test_mask, 'year'].max()}, "
          f"positive rate {df.loc[test_mask, 'region_day_flooded_broadcast'].mean():.2%}")
    save(df[test_mask], OUTPUT_TEST)
    gc.collect()


def save(df: pd.DataFrame, path: str):
    if OUTPUT_FORMAT == "parquet":
        path = path.rsplit(".", 1)[0] + ".parquet"
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
    print(f"Saved -> {path} ({len(df)} rows, {len(df.columns)} columns)")


# ---------------------------------------------------------------------
# 6. MAIN
# ---------------------------------------------------------------------

def main():
    df = build_ward_day_table()
    df = add_sequence_features(df)

    print("\n" + "=" * 60)
    print("WARD x DAY DATASET -- SUMMARY")
    print("=" * 60)
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Wards: {df['ward_uid'].nunique()}  Regions: {df['region'].nunique()}  "
          f"Date range: {df['date'].min().date()} - {df['date'].max().date()}")
    print(f"region_day_flooded_broadcast positive rate: "
          f"{df['region_day_flooded_broadcast'].mean():.2%}")
    print("=" * 60)

    save(df, OUTPUT_FULL)
    year_based_split_and_save(df)

    print("\nNEXT STEP for LSTM+GNN specifically: build the ward-adjacency "
          "graph (see build_ward_adjacency_graph.py) and, when training, "
          "group these rows by ward_uid, sort by date (already done here), "
          "and window into fixed-length sequences (e.g. 14/30-day lookback "
          "-> next-day region_day_flooded_broadcast) per ward before "
          "feeding an LSTM.")


if __name__ == "__main__":
    main()