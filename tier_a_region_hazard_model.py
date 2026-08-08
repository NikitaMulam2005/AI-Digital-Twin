"""
Tier A -- Region-Level Flood Hazard Model
--------------------------------------------------------------------------
Predicts P(this region floods on this date) using rainfall, soil
moisture, and (where available) tide features, with labels derived from
the India Flood Inventory (IFI). This is the "WHEN" half of the two-tier
design -- see tier_b_ward_susceptibility.py for the "WHERE" half.

WHY REGION-LEVEL, NOT WARD-LEVEL
IFI's `India_Flood_Inventory_v3.csv` has NO usable coordinates (the
Latitude/Longitude columns are empty for all 6,876 events) and its only
geometry (a shapefile) covers just 2014-2020 at DISTRICT resolution --
not finer than the text Districts field already gives you, and missing
your defining 2005 event entirely. District is therefore the real
ceiling of what this source can label, and Maharashtra districts don't
line up with your corporations 1:1 (Thane district covers Thane, KDMC,
AND NMMC together; Raigad district is far larger than just Panvel). This
script is honest about that: it labels REGIONS, not wards, and
Thane/KDMC/NMMC all inherit the same "Thane district flooded" label on a
given date because the source genuinely cannot distinguish them. Panvel
is dropped entirely (Raigad is too broad a proxy to be meaningful, and
Panvel has no rainfall/soil-moisture region defined in the companion
scripts anyway).

LABEL CAVEATS -- READ BEFORE TRUSTING THE OUTPUT
  1. Class imbalance: flood-labeled days are a small fraction of the
     full 1990-2024 daily record. Don't just look at accuracy --
     precision/recall/PR-AUC matter far more here.
  2. Broadcast noise: a "Thane" district event might really have been
     entirely within old Thane city, or entirely within what's now
     NMMC -- KDMC and NMMC both get the positive label regardless. This
     inflates false positives for whichever sub-region wasn't actually
     hit.
  3. Missed events: IFI is built from IMD's Disastrous Weather Events
     archive + EM-DAT + Dartmouth Flood Observatory -- it captures
     large/damaging/reported floods, not every waterlogging incident.
     Minor but real waterlogging days will show up as "flooded=0" in
     this label, which trains the model to be less sensitive than
     reality.
  4. This model's ceiling is therefore "does IMD-reported flood-level
     rainfall happen in this district", which is a genuinely useful
     region-scoped hazard signal -- just don't present its output as
     ward-level or as a complete flood catalog.

REQUIREMENTS
    pip install pandas numpy scikit-learn joblib

HOW TO USE
1. Point the file paths in CONFIGURATION at your actual files (defaults
   match the filenames produced by the companion IMD rainfall / ERA5-
   Land scripts, and the raw IFI CSV from
   github.com/hydrosenselab/India-Flood-Inventory, v3.0 folder)
2. Run: python tier_a_region_hazard_model.py
3. Output:
   - tier_a_region_day_features.csv : full engineered feature table
   - tier_a_region_day_predictions.csv : held-out test predictions
   - tier_a_region_hazard_model.joblib : trained model, reusable via
     joblib.load() for scoring new dates
"""

import re
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

# ---------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------

RAINFALL_CSV = "mmr_pune_rainfall_1990_2024_clean.csv"
SOIL_MOISTURE_CSV = "mmr_pune_soil_moisture_lst_era5land_1990_2024_clean.csv"
TIDE_CSV = "tide_features_daily_1990_2024.csv"     # optional -- skipped gracefully if missing/unrecognized
IFI_RAW_CSV = "India_Flood_Inventory_v3.csv"        # from IFI repo's v3.0 folder

OUTPUT_FEATURES_CSV = "tier_a_region_day_features.csv"
OUTPUT_PREDICTIONS_CSV = "tier_a_region_day_predictions.csv"
OUTPUT_MODEL = "tier_a_region_hazard_model.joblib"

START_YEAR = 1990
END_YEAR = 2024

# Regions this model covers -- must match the region names used in the
# rainfall/soil-moisture CSVs from the companion scripts.
REGIONS = ["BMC", "Thane", "KDMC", "NMMC", "Pune"]

# IFI District text -> which of the REGIONS above it labels as positive.
# Thane district intentionally maps to three regions (see docstring
# CAVEAT #2). Districts not listed here (e.g. Raigad) are dropped.
DISTRICT_TO_REGIONS = {
    "mumbai": ["BMC"],
    "mumbai suburban": ["BMC"],
    "thane": ["Thane", "KDMC", "NMMC"],
    "pune": ["Pune"],
}

# Train/test split by year (avoids random leakage between consecutive
# days of the same flood event landing in both train and test).
TEST_YEARS = set(range(2018, END_YEAR + 1))


# ---------------------------------------------------------------------
# 2. LABELS -- PARSE IFI, BROADCAST EVENT DATE RANGES TO REGION-DAYS
# ---------------------------------------------------------------------

def load_ifi_labels(ifi_csv: str, regions: list[str]) -> pd.DataFrame:
    """
    Read the raw IFI CSV, match each event's free-text Districts field
    against DISTRICT_TO_REGIONS, and expand every matched event into one
    row per (region, date) for every day in [Start Date, End Date].
    Returns columns: region, date, flooded (always 1 here -- the full
    0/1 calendar is built later by joining this against every date).
    """
    print(f"Loading IFI events from {ifi_csv} ...")
    df = pd.read_csv(ifi_csv)
    df["Start Date"] = pd.to_datetime(df["Start Date"], format="%d-%m-%Y %H:%M", errors="coerce")
    df["End Date"] = pd.to_datetime(df["End Date"], format="%d-%m-%Y %H:%M", errors="coerce")
    df = df.dropna(subset=["Start Date", "Districts"])
    print(f"  {len(df)} events have a parseable Start Date and non-empty Districts.")

    rows = []
    unmatched_districts = set()
    for _, ev in df.iterrows():
        end = ev["End Date"] if pd.notna(ev["End Date"]) else ev["Start Date"]
        if end < ev["Start Date"]:
            end = ev["Start Date"]
        districts = [d.strip().lower() for d in re.split(r"[;,]", str(ev["Districts"])) if d.strip()]

        matched_regions = set()
        for d in districts:
            if d in DISTRICT_TO_REGIONS:
                matched_regions.update(DISTRICT_TO_REGIONS[d])
            else:
                unmatched_districts.add(d)

        if not matched_regions:
            continue

        date_range = pd.date_range(ev["Start Date"], end, freq="D")
        for region in matched_regions:
            if region not in regions:
                continue
            for d in date_range:
                rows.append({"region": region, "date": d, "flooded": 1,
                             "event_uei": ev.get("UEI"), "severity": ev.get("Severity")})

    label_df = pd.DataFrame(rows)
    if label_df.empty:
        print("  WARNING: no events matched any target region -- check "
              "DISTRICT_TO_REGIONS spelling against your IFI file's "
              "Districts values.")
    else:
        label_df = label_df.drop_duplicates(subset=["region", "date"])
        print(f"  Matched {label_df['event_uei'].nunique()} unique events "
              f"-> {len(label_df)} region-day positive labels across "
              f"{label_df['region'].value_counts().to_dict()}")
    return label_df


def build_full_calendar(regions: list[str], start_year: int, end_year: int) -> pd.DataFrame:
    dates = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq="D")
    return pd.DataFrame(
        [(r, d) for r in regions for d in dates],
        columns=["region", "date"],
    )


# ---------------------------------------------------------------------
# 3. FEATURES -- RAINFALL, SOIL MOISTURE, TIDE
# ---------------------------------------------------------------------

def load_rainfall_features(path: str) -> pd.DataFrame:
    print(f"Loading rainfall features from {path} ...")
    df = pd.read_csv(path, parse_dates=["date"])
    agg = df.groupby(["region", "date"])["rainfall_mm"].agg(
        rainfall_mean="mean", rainfall_max="max"
    ).reset_index()
    agg = agg.sort_values(["region", "date"])
    agg["rainfall_3day"] = agg.groupby("region")["rainfall_mean"].transform(
        lambda s: s.rolling(3, min_periods=1).sum()
    )
    agg["rainfall_7day"] = agg.groupby("region")["rainfall_mean"].transform(
        lambda s: s.rolling(7, min_periods=1).sum()
    )
    print(f"  {len(agg)} region-day rainfall rows.")
    return agg


def load_soil_moisture_features(path: str) -> pd.DataFrame:
    print(f"Loading soil moisture / LST features from {path} ...")
    df = pd.read_csv(path, parse_dates=["date"])
    agg = df.groupby(["region", "date"]).agg(
        soil_moisture_mean=("soil_moisture_m3m3", "mean"),
        lst_mean=("land_surface_temp_c", "mean"),
    ).reset_index()
    print(f"  {len(agg)} region-day soil moisture rows.")
    return agg


def load_tide_features(path: str) -> pd.DataFrame:
    """
    Best-effort tide loader. Tries to find a date column and a region
    column automatically; if it can't confidently identify both, skips
    tide features entirely rather than guessing wrong and corrupting the
    merge. Returns an empty DataFrame (with region/date columns only) on
    failure so the rest of the pipeline can proceed unaffected.
    """
    if not Path(path).exists():
        print(f"NOTE: tide file '{path}' not found -- proceeding without "
              f"tide features (BMC/Thane/NMMC will lack this signal).")
        return pd.DataFrame(columns=["region", "date"])

    print(f"Loading tide features from {path} ...")
    df = pd.read_csv(path)
    date_col = next((c for c in df.columns if "date" in c.lower()), None)
    region_col = next((c for c in df.columns if "region" in c.lower()), None)

    if date_col is None or region_col is None:
        print(f"  WARNING: couldn't confidently find both a date and a "
              f"region column in {path} (columns present: "
              f"{list(df.columns)}). Skipping tide features -- merge "
              f"manually if you need them.")
        return pd.DataFrame(columns=["region", "date"])

    df = df.rename(columns={date_col: "date", region_col: "region"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    numeric_cols = [c for c in df.columns if c not in ("date", "region")
                    and pd.api.types.is_numeric_dtype(df[c])]
    print(f"  Using tide numeric columns: {numeric_cols}")
    return df[["region", "date"] + numeric_cols]


# ---------------------------------------------------------------------
# 4. MERGE + MODEL
# ---------------------------------------------------------------------

def build_master_table(regions: list[str]) -> pd.DataFrame:
    calendar = build_full_calendar(regions, START_YEAR, END_YEAR)

    labels = load_ifi_labels(IFI_RAW_CSV, regions)
    if not labels.empty:
        calendar = calendar.merge(
            labels[["region", "date", "flooded"]], on=["region", "date"], how="left"
        )
    else:
        calendar["flooded"] = np.nan
    calendar["flooded"] = calendar["flooded"].fillna(0).astype(int)

    rainfall = load_rainfall_features(RAINFALL_CSV)
    soil = load_soil_moisture_features(SOIL_MOISTURE_CSV)
    tide = load_tide_features(TIDE_CSV)

    master = calendar.merge(rainfall, on=["region", "date"], how="left")
    master = master.merge(soil, on=["region", "date"], how="left")
    if not tide.empty:
        master = master.merge(tide, on=["region", "date"], how="left")

    n_before = len(master)
    feature_cols_check = [c for c in master.columns
                           if c not in ("region", "date", "flooded")]
    master = master.dropna(subset=["rainfall_mean"]).reset_index(drop=True)
    dropped = n_before - len(master)
    if dropped:
        print(f"Dropped {dropped} region-days with no rainfall data "
              f"(likely outside your downloaded date range).")

    return master


def train_and_evaluate(master: pd.DataFrame):
    feature_cols = [c for c in master.columns
                    if c not in ("region", "date", "flooded", "event_uei", "severity")
                    and pd.api.types.is_numeric_dtype(master[c])]
    print(f"\nFeatures used: {feature_cols}")

    master["year"] = master["date"].dt.year
    train = master[~master["year"].isin(TEST_YEARS)].copy()
    test = master[master["year"].isin(TEST_YEARS)].copy()
    print(f"Train: {len(train)} rows ({train['flooded'].sum()} positive), "
          f"years < {min(TEST_YEARS)}")
    print(f"Test:  {len(test)} rows ({test['flooded'].sum()} positive), "
          f"years >= {min(TEST_YEARS)}")

    if train["flooded"].sum() == 0 or test["flooded"].sum() == 0:
        print("\nWARNING: one of train/test has ZERO positive labels. "
              "The model/metrics below will be meaningless -- check that "
              "TEST_YEARS actually splits your event dates sensibly, or "
              "that DISTRICT_TO_REGIONS is matching real IFI rows at all.")

    X_train = train[feature_cols].fillna(train[feature_cols].median())
    y_train = train["flooded"]
    X_test = test[feature_cols].fillna(train[feature_cols].median())
    y_test = test["flooded"]

    model = RandomForestClassifier(
        n_estimators=300, max_depth=8, class_weight="balanced",
        random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    print("\n" + "=" * 60)
    print("Tier A Region Hazard Model -- Evaluation (held-out years)")
    print("=" * 60)
    print(classification_report(y_test, preds, target_names=["No flood", "Flood"],
                                 zero_division=0))
    if y_test.nunique() > 1:
        print(f"ROC-AUC          : {roc_auc_score(y_test, proba):.3f}")
        print(f"Avg Precision(PR): {average_precision_score(y_test, proba):.3f}")
    print("\nFeature importances:")
    for name, imp in sorted(zip(feature_cols, model.feature_importances_),
                             key=lambda x: -x[1]):
        print(f"  {name:20s} {imp:.3f}")
    print("=" * 60)
    print("REMINDER: read the label caveats in this script's docstring "
          "before trusting these numbers -- 'flooded' here means "
          "'IFI-reported district flood', broadcast to region, not "
          "ward-level ground truth.")

    test_out = test[["region", "date", "flooded"]].copy()
    test_out["predicted_probability"] = proba
    test_out["predicted_label"] = preds
    return model, test_out


# ---------------------------------------------------------------------
# 5. MAIN
# ---------------------------------------------------------------------

def main():
    master = build_master_table(REGIONS)
    master.to_csv(OUTPUT_FEATURES_CSV, index=False)
    print(f"\nSaved full feature table -> {OUTPUT_FEATURES_CSV}")

    model, predictions = train_and_evaluate(master)

    predictions.to_csv(OUTPUT_PREDICTIONS_CSV, index=False)
    print(f"Saved test-set predictions -> {OUTPUT_PREDICTIONS_CSV}")

    joblib.dump(model, OUTPUT_MODEL)
    print(f"Saved trained model -> {OUTPUT_MODEL}")


if __name__ == "__main__":
    main()