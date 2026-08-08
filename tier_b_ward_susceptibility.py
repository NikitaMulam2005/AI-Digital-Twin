"""
Tier B -- Ward-Level Static Flood Susceptibility Index
--------------------------------------------------------------------------
Builds a composite, per-ward susceptibility score from PHYSICAL
characteristics alone -- no flood event labels required. This is the
"where is this ward inherently vulnerable" half of the two-tier design:
    Tier A predicts WHEN a region floods (needs IFI event labels)
    Tier B scores WHERE within a region risk concentrates (this script)
Multiply/combine the two downstream for a ward x date risk estimate.

METHOD
Standard Weighted Linear Combination (WLC) susceptibility index, the
same family of method used in most published India flood-susceptibility
papers (often built via AHP for the weights). For each ward:
  1. Take five physical indicators (edit COLUMN_MAP below if your
     column names differ):
       - elevation       (lower  = more risk -- water pools low)
       - slope           (flatter = more risk -- water doesn't drain)
       - curve_number    (higher = more risk -- more impervious/runoff)
       - road_density    (lower  = more risk -- proxy for drainage
                           infrastructure; sparse road network often
                           means sparse formal stormwater drainage too)
       - tide_exposure   (coastal = more risk -- tidal backup during
                           high tide can block outfall drainage)
  2. Min-max normalize each 0-1 across all wards
  3. Invert the ones where LOW raw value means HIGH risk (elevation,
     slope, road_density) so that after inversion, 1.0 always means
     "more flood-prone" for every indicator
  4. Weighted sum -> susceptibility_index (0-100), plus a 5-tier label
     (Very Low..Very High) via quantile bins

CAVEAT -- WEIGHTS ARE A JUDGMENT CALL, NOT FITTED
Because no flood labels are used here, the weights below are not learned
from data -- they're a reasonable starting point from the flood-
susceptibility literature, not a validated result. Treat this index as a
prioritization/screening tool, not a calibrated probability. If/when
Tier A produces enough real event-day labels, revisit these weights by
checking correlation between this index and actual observed flooding
where you do have ward-level ground truth (e.g. from the Sentinel-1
approach discussed separately), and adjust.

REQUIREMENTS
    pip install pandas numpy

HOW TO USE
1. Point WARD_MASTER_CSV at your merged ward_year_master.csv (or any
   CSV with one row per ward -- or per ward-year, in which case
   YEAR_TO_USE picks which year's row to treat as "current")
2. Check COLUMN_MAP below -- if the script's auto-detection picks the
   wrong column (it prints what it picked), override the mapping
   explicitly with the exact column name from your file
3. Run: python tier_b_ward_susceptibility.py
4. Output: ward_susceptibility_index.csv, one row per ward, with the
   five normalized components, the composite index, and its tier label
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------

WARD_MASTER_CSV = "ward_year_master.csv"
OUTPUT_CSV = "ward_susceptibility_index.csv"

# If your data is a ward x year panel, which year's row to use as the
# "current" static snapshot. "latest" = most recent year present per ward.
YEAR_TO_USE = "latest"          # or an int, e.g. 2024
WARD_ID_COL = None              # None = auto-detect (looks for 'ward_uid'/'ward_id')
REGION_COL = None               # None = auto-detect (looks for 'region')
YEAR_COL = None                 # None = auto-detect (looks for 'year')

# Set any of these to an exact column name from your CSV to override
# auto-detection. Leave as None to let the script guess from keywords.
COLUMN_MAP = {
    "elevation": None,      # e.g. "mean_elevation_m"
    "slope": None,          # e.g. "mean_slope_deg"
    "curve_number": "mean_cn_estimate",   # your file has no "curve" substring -- this is the resolved CN column
    "road_density": None,   # e.g. "road_density_km_per_km2"
}
# Explicit coastal flag, if your file has one (yours does: is_coastal_region).
# Used in preference to the tide-column-null heuristic below when present.
COASTAL_FLAG_COL = "is_coastal_region"
# tide_exposure has no fixed column -- it's detected as "does this ward
# have ANY non-null tide-related column", matching the master script's
# own note that KDMC/Pune wards get NaN tide columns (non-coastal).
TIDE_KEYWORDS = ["tide"]

# Direction: True = higher raw value means higher risk (no inversion
# needed). False = lower raw value means higher risk (will be inverted
# after normalization).
HIGHER_IS_RISKIER = {
    "elevation": False,
    "slope": False,
    "curve_number": True,
    "road_density": False,
    "tide_exposure": True,
}

# Starting weights (must sum to 1.0) -- see CAVEAT in the docstring above.
WEIGHTS = {
    "elevation": 0.30,
    "slope": 0.20,
    "curve_number": 0.30,
    "road_density": 0.10,
    "tide_exposure": 0.10,
}

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-6, "WEIGHTS must sum to 1.0"


# ---------------------------------------------------------------------
# 2. COLUMN AUTO-DETECTION
# ---------------------------------------------------------------------

def find_column(df: pd.DataFrame, keywords: list[str], exclude: list[str] = None) -> str:
    """
    Find the first column whose name contains ALL given keywords
    (case-insensitive). `exclude` keywords rule a column out even if it
    matches. Raises a clear error listing candidates if nothing/multiple
    plausible matches are found, so the user can hardcode COLUMN_MAP
    instead of silently guessing wrong.
    """
    exclude = exclude or []
    cols_lower = {c: c.lower() for c in df.columns}
    matches = [
        c for c, cl in cols_lower.items()
        if all(k in cl for k in keywords) and not any(e in cl for e in exclude)
    ]
    if not matches:
        raise KeyError(
            f"Couldn't auto-detect a column for keywords {keywords} "
            f"(excluding {exclude}). Available columns: {list(df.columns)}. "
            f"Set the matching entry in COLUMN_MAP explicitly instead."
        )
    if len(matches) > 1:
        # Prefer a column containing "mean" if there are several
        # candidates (e.g. mean/min/max elevation) -- otherwise just
        # take the first and tell the user what was picked.
        preferred = [c for c in matches if "mean" in c.lower()]
        chosen = preferred[0] if preferred else matches[0]
        print(f"  NOTE: multiple columns matched {keywords}: {matches} "
              f"-- using '{chosen}'. Override COLUMN_MAP if this is wrong.")
        return chosen
    print(f"  Auto-detected {keywords} -> '{matches[0]}'")
    return matches[0]


def resolve_columns(df: pd.DataFrame) -> dict:
    resolved = {}
    keyword_guesses = {
        "elevation": (["elev"], ["min", "max", "std"]),
        "slope": (["slope"], ["min", "max", "std"]),
        "curve_number": (["curve"], []),
        "road_density": (["road", "density"], []),
    }
    print("Resolving indicator columns ...")
    for key, override in COLUMN_MAP.items():
        if override:
            resolved[key] = override
            print(f"  Using configured column for '{key}': '{override}'")
            continue
        kws, excl = keyword_guesses[key]
        resolved[key] = find_column(df, kws, excl)
    return resolved


def resolve_id_columns(df: pd.DataFrame):
    ward_col = WARD_ID_COL or find_column(df, ["ward"])
    try:
        region_col = REGION_COL or find_column(df, ["region"])
    except KeyError:
        region_col = None
        print("  NOTE: no 'region' column found -- proceeding without it "
              "(susceptibility index will still work, just without a "
              "region label in the output).")
    try:
        year_col = YEAR_COL or find_column(df, ["year"])
    except KeyError:
        year_col = None
        print("  NOTE: no 'year' column found -- treating input as one "
              "row per ward already (not a ward x year panel).")
    return ward_col, region_col, year_col


# ---------------------------------------------------------------------
# 3. SNAPSHOT SELECTION (collapse ward x year panel -> one row per ward)
# ---------------------------------------------------------------------

def select_snapshot(df: pd.DataFrame, ward_col: str, year_col: str) -> pd.DataFrame:
    if year_col is None:
        if df[ward_col].duplicated().any():
            raise ValueError(
                f"No year column detected, but '{ward_col}' has duplicate "
                f"values -- this looks like a panel without a usable year "
                f"column. Set YEAR_COL explicitly."
            )
        return df

    if YEAR_TO_USE == "latest":
        idx = df.groupby(ward_col)[year_col].idxmax()
        snap = df.loc[idx].reset_index(drop=True)
        print(f"Using each ward's latest available year "
              f"(range found: {df[year_col].min()}-{df[year_col].max()}).")
    else:
        snap = df[df[year_col] == YEAR_TO_USE].reset_index(drop=True)
        if snap.empty:
            raise ValueError(
                f"YEAR_TO_USE={YEAR_TO_USE} not found in column "
                f"'{year_col}'. Available years: "
                f"{sorted(df[year_col].dropna().unique())}"
            )
        print(f"Using fixed year {YEAR_TO_USE} for all wards.")
    return snap


# ---------------------------------------------------------------------
# 4. NORMALIZE + COMPOSITE INDEX
# ---------------------------------------------------------------------

def min_max_normalize(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    if hi == lo:
        # No variation across wards -- everyone gets a neutral 0.5
        # rather than a divide-by-zero NaN.
        return pd.Series(0.5, index=series.index)
    return (series - lo) / (hi - lo)


def build_index(df: pd.DataFrame, cols: dict, ward_col: str, region_col: str) -> pd.DataFrame:
    out = pd.DataFrame()
    out["ward"] = df[ward_col]
    if region_col:
        out["region"] = df[region_col]

    # Detect tide/coastal exposure. Prefer an explicit coastal flag
    # column (e.g. is_coastal_region) if present -- more reliable than
    # inferring it from which tide columns happen to be null. Falls back
    # to the null-heuristic only if no such flag column exists.
    if COASTAL_FLAG_COL and COASTAL_FLAG_COL in df.columns:
        flag = df[COASTAL_FLAG_COL]
        if flag.dtype == object:
            out["raw_tide_exposure"] = flag.astype(str).str.lower().isin(
                ["true", "1", "yes", "y"]
            ).astype(float)
        else:
            out["raw_tide_exposure"] = pd.to_numeric(flag, errors="coerce").fillna(0).astype(float)
        print(f"  Tide/coastal exposure taken from explicit flag column "
              f"'{COASTAL_FLAG_COL}'.")
    else:
        tide_cols = [c for c in df.columns if any(k in c.lower() for k in TIDE_KEYWORDS)]
        if tide_cols:
            out["raw_tide_exposure"] = df[tide_cols].notna().any(axis=1).astype(float)
            print(f"  Tide exposure detected from columns: {tide_cols}")
        else:
            out["raw_tide_exposure"] = 0.0
            print("  WARNING: no tide-related columns found -- all wards "
                  "treated as non-coastal (raw_tide_exposure=0). If some of "
                  "your wards ARE coastal, this component will be useless "
                  "until a real tide/coastal-flag column is added.")

    for key, col in cols.items():
        out[f"raw_{key}"] = pd.to_numeric(df[col], errors="coerce")

    # Normalize + orient every component so that 1.0 = riskier
    risk_components = []
    for key in list(cols.keys()) + ["tide_exposure"]:
        raw = out[f"raw_{key}"]
        norm = min_max_normalize(raw)
        if not HIGHER_IS_RISKIER[key]:
            norm = 1 - norm
        out[f"risk_{key}"] = norm
        risk_components.append(key)

    # Weighted composite. Rows with any missing indicator get an
    # NaN-safe weighted average over whatever indicators ARE present
    # (re-normalizing weights over available columns) rather than
    # silently dropping the whole ward.
    weight_arr = np.array([WEIGHTS[k] for k in risk_components])
    risk_matrix = out[[f"risk_{k}" for k in risk_components]].to_numpy()
    valid = ~np.isnan(risk_matrix)
    weights_per_row = np.where(valid, weight_arr, 0.0)
    weight_sums = weights_per_row.sum(axis=1)
    weighted_vals = np.nan_to_num(risk_matrix) * weights_per_row
    composite = np.divide(
        weighted_vals.sum(axis=1), weight_sums,
        out=np.full(len(out), np.nan), where=weight_sums > 0
    )
    out["susceptibility_index"] = composite * 100
    n_incomplete = (weight_sums < weight_arr.sum() - 1e-9).sum()
    if n_incomplete:
        print(f"  NOTE: {n_incomplete} ward(s) had at least one missing "
              f"indicator -- their index used a re-normalized subset of "
              f"weights rather than being dropped.")

    out["tier"] = pd.qcut(
        out["susceptibility_index"], q=5,
        labels=["Very Low", "Low", "Moderate", "High", "Very High"],
        duplicates="drop",
    )
    return out.sort_values("susceptibility_index", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------
# 5. MAIN
# ---------------------------------------------------------------------

def main():
    print(f"Loading {WARD_MASTER_CSV} ...")
    df = pd.read_csv(WARD_MASTER_CSV)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns.")

    ward_col, region_col, year_col = resolve_id_columns(df)
    cols = resolve_columns(df)
    snap = select_snapshot(df, ward_col, year_col)

    print(f"\nBuilding susceptibility index for {len(snap)} wards ...")
    result = build_index(snap, cols, ward_col, region_col)

    print("\n" + "=" * 60)
    print("Ward Susceptibility Index -- Summary")
    print("=" * 60)
    print(result["tier"].value_counts().reindex(
        ["Very High", "High", "Moderate", "Low", "Very Low"]
    ).to_string())
    print("\nTop 10 most susceptible wards:")
    show_cols = ["ward"] + (["region"] if region_col else []) + ["susceptibility_index", "tier"]
    print(result[show_cols].head(10).to_string(index=False))

    result.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()