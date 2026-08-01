"""
Ward-level Zonal Statistics -- LULC, All Regions (detailed version)
--------------------------------------------------------------------------
Spatially joins the pixel-level MODIS LULC CSV to each corporation's ward
boundary file and aggregates to one row per ward (or one row per ward PER
YEAR, if you point it at the multi-year output from modis_lulc_timeseries.py
-- auto-detected via a `year` column).

FIXES APPLIED IN THIS VERSION (ported from ward_zonal_dem_fixed.py, which
hit and resolved the identical issues -- see that script for full detail):

  1. Pune ward_field was "name" -> actually "Name" in wards/pune.kml.
  2. wards_mumbai.kml and wards_thane.kml store polygon coordinates as
     (lat, lon) instead of the KML standard (lon, lat). Confirmed against
     the real files: BMC ward bounds came back as x=18.89-19.27 (that's
     latitude in the longitude slot). load_ward_boundaries() now
     auto-detects this per-file and corrects it before anything else
     touches the geometry -- verified against the actual uploaded KMLs
     to produce 100% self-match on synthetic in-ward points.
  3. groupby().apply() silently breaks when a region matches 0 pixels --
     pandas can't infer output shape from zero groups, so it returns the
     input frame's columns unchanged instead of calling the summarizer,
     which caused `summary.insert(0, "region", ...)` to collide with a
     `region` column already present. Replaced with an explicit per-group
     loop so a 0-match join just yields 0 correctly-shaped rows.
  4. sorted(missing_wards) crashed on KML files with blank ward-name
     fields (NaN mixed with str). Now uses a type-safe sort key.
  5. Added the same bounds diagnostic used in the DEM script, so any
     future coordinate-order problem shows up immediately instead of as
     a mysterious low match rate.

WHY THESE PARTICULAR FEATURES
A single "pct_impervious + dominant class" per ward throws away a lot of
what actually matters for the SCS Curve Number formula and flood risk
more broadly: Curve Number depends on BOTH land cover AND hydrologic
soil group (A/B/C/D), not soil group C alone -- so this computes a CN
estimate for all four groups, not just one. A ward that's "60% urban,
40% forest" behaves very differently from one that's "95% urban, 5%
forest" even though both might round to the same "dominant_lulc_class".
dominant_class_pct and a second-dominant class capture that; a Shannon
diversity index captures general heterogeneity in one number.

Land-use CHANGE over time is arguably more predictive of rising flood
risk than a static snapshot -- a ward that went from 40% to 75%
impervious over 20 years is a very different risk story than one that's
been at 75% the whole time. If you feed this script the multi-year CSV,
it computes that trend automatically.

The raw 17-class IGBP breakdown is noisy at ward scale. Broader
CN-standard groups (Urban / Cropland / Forest / Shrub-Grassland /
Wetland / Water / Barren) are what most flood models and NRCS tables
actually key off of, so those are included alongside (not instead of)
the raw breakdown.

FEATURES PRODUCED PER WARD (per year, if input has one):
    total_cells
    pct_impervious, pct_pervious
    dominant_lulc_class, dominant_class_pct
    second_dominant_class, second_dominant_class_pct
    lulc_diversity_shannon, lulc_diversity_normalized (0-1 evenness)
    mean_cn_estimate_soilA/B/C/D, cn_estimate_soilC_std
    pct_Urban, pct_Cropland, pct_Forest, pct_Shrub_Grassland, pct_Wetland,
        pct_Water, pct_Barren_or_Snow   (broad CN groups)
    pct_vegetated  (Forest + Shrub_Grassland + Cropland + Wetland)
    pct_<raw IGBP class name>  (full original breakdown, kept for
        completeness)

IF THE INPUT CSV HAS A `year` COLUMN, you ALSO get a full ward x year
panel, AND a separate trend summary CSV per region + merged, with:
    earliest_year, latest_year, pct_impervious_earliest,
    pct_impervious_latest, pct_impervious_change, pct_vegetated_earliest,
    pct_vegetated_latest, pct_vegetated_change,
    avg_annual_pct_impervious_change

CN CAVEAT: the A/B/C/D estimates are illustrative NRCS TR-55-style
approximations by land-cover class only. Real CN selection needs an
actual soil survey plus land treatment/condition info this script has no
way to know. Treat as a rough starting range, not a final input.

Same per-corporation ward-field mapping as the DEM script:
    BMC   (wards_mumbai.kml)       -> "Name"
    Thane (wards_thane.kml)        -> "lgd_name"
    KDMC  (kdmc.kml)               -> "ward_lgd_name"
    NMMC  (wards_navi_mumbai.kml)  -> "ward_lgd_name"
    Pune  (pune.kml)               -> "Name"   # FIX: was "name" (lowercase)

REQUIREMENTS
    pip install geopandas fiona shapely pandas numpy

HOW TO USE
    python ward_zonal_stats_lulc_fixed.py
    -- point PIXEL_CSV at either mmr_pune_lulc_clean.csv (single year) or
       mmr_pune_lulc_timeseries_clean.csv (multi-year) -- both work,
       multi-year just additionally produces the trend CSVs
    -- expects the ward KMLs in ./wards/ (same layout as the DEM script)
"""

import fiona
import geopandas as gpd
import numpy as np
import pandas as pd
from pathlib import Path
from shapely.ops import transform as shapely_transform

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

# FIX: default now points at the multi-year output, matching what you
# actually just generated. Point back at mmr_pune_lulc_clean.csv if you
# only want a single-year snapshot.
PIXEL_CSV = "mmr_pune_lulc_timeseries_clean.csv"

REGION_CONFIGS = {
    "BMC":   {"boundary_file": "wards/wards_mumbai.kml",       "ward_field": "Name"},
    "Thane": {"boundary_file": "wards/wards_thane.kml",        "ward_field": "lgd_name"},
    "KDMC":  {"boundary_file": "wards/kdmc.kml",               "ward_field": "ward_lgd_name"},
    "NMMC":  {"boundary_file": "wards/wards_navi_mumbai.kml",  "ward_field": "ward_lgd_name"},
    # FIX: pune.kml has "Name" (capital N), not "name".
    "Pune":  {"boundary_file": "wards/pune.kml",               "ward_field": "Name"},
}

MERGED_OUTPUT_CSV = "all_regions_ward_lulc_zonal_stats.csv"
MERGED_TREND_CSV = "all_regions_ward_lulc_trend.csv"

# Rough India lat/lon envelope, used only to auto-detect swapped KML
# coordinates -- not a precision check. Same constants as the DEM script.
INDIA_LAT_RANGE = (6.0, 38.0)
INDIA_LON_RANGE = (66.0, 100.0)

IGBP_CLASSES = {
    1: "Evergreen Needleleaf Forest", 2: "Evergreen Broadleaf Forest",
    3: "Deciduous Needleleaf Forest", 4: "Deciduous Broadleaf Forest",
    5: "Mixed Forest", 6: "Closed Shrublands", 7: "Open Shrublands",
    8: "Woody Savannas", 9: "Savannas", 10: "Grasslands",
    11: "Permanent Wetlands", 12: "Croplands", 13: "Urban and Built-up",
    14: "Cropland/Natural Vegetation Mosaic", 15: "Snow and Ice",
    16: "Barren or Sparsely Vegetated", 17: "Water Bodies",
}

# Broader CN-standard groupings -- what most flood models/NRCS tables
# actually key off of, vs. the noisy raw 17-class breakdown at ward scale.
IGBP_TO_GROUP = {
    1: "Forest", 2: "Forest", 3: "Forest", 4: "Forest", 5: "Forest",
    6: "Shrub_Grassland", 7: "Shrub_Grassland", 8: "Shrub_Grassland",
    9: "Shrub_Grassland", 10: "Shrub_Grassland",
    11: "Wetland",
    12: "Cropland", 14: "Cropland",
    13: "Urban",
    15: "Barren_or_Snow", 16: "Barren_or_Snow",
    17: "Water",
}
VEGETATED_GROUPS = {"Forest", "Shrub_Grassland", "Cropland", "Wetland"}

# Illustrative NRCS TR-55-style Curve Numbers by (Hydrologic Soil Group
# A, B, C, D), AMC II average condition -- approximate, see docstring.
CN_TABLE_ABCD = {
    1: (30, 55, 70, 77), 2: (30, 55, 70, 77), 3: (30, 55, 70, 77),
    4: (30, 55, 70, 77), 5: (30, 55, 70, 77),
    6: (35, 56, 70, 77), 7: (35, 56, 70, 77), 8: (35, 56, 70, 77),
    9: (35, 56, 70, 77),
    10: (49, 69, 79, 84),
    11: (77, 86, 91, 94),
    12: (67, 78, 85, 89),
    13: (81, 88, 91, 93),
    14: (55, 70, 80, 85),
    15: (np.nan, np.nan, np.nan, np.nan),
    16: (77, 86, 91, 94),
    17: (np.nan, np.nan, np.nan, np.nan),
}

IMPERVIOUS_CODES = {13}


# ---------------------------------------------------------------------
# LOAD BOUNDARIES (with swap auto-fix)
# ---------------------------------------------------------------------

def _looks_like_swapped_latlon(gdf: gpd.GeoDataFrame) -> bool:
    """See ward_zonal_dem_fixed.py -- same detector, same confirmed cause
    (wards_mumbai.kml / wards_thane.kml store x=lat, y=lon)."""
    minx, miny, maxx, maxy = gdf.total_bounds
    x_is_lat = INDIA_LAT_RANGE[0] <= minx and maxx <= INDIA_LAT_RANGE[1]
    y_is_lon = INDIA_LON_RANGE[0] <= miny and maxy <= INDIA_LON_RANGE[1]
    x_is_lon = INDIA_LON_RANGE[0] <= minx and maxx <= INDIA_LON_RANGE[1]
    return x_is_lat and y_is_lon and not x_is_lon


def load_ward_boundaries(path: str, ward_field: str) -> gpd.GeoDataFrame:
    suffix = Path(path).suffix.lower()
    if suffix == ".kml":
        fiona.drvsupport.supported_drivers["KML"] = "rw"
        gdf = gpd.read_file(path, driver="KML")
    else:
        gdf = gpd.read_file(path)

    if ward_field not in gdf.columns:
        raise KeyError(f"'{ward_field}' not found in {path}. Columns available: "
                        f"{list(gdf.columns)}")

    gdf[ward_field] = gdf[ward_field].astype(str).str.strip()
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")

    # FIX: auto-detect and correct swapped lat/lon coordinates.
    if _looks_like_swapped_latlon(gdf):
        print(f"  FIXED: {path} has swapped lat/lon coordinates "
              f"(x=lat, y=lon instead of x=lon, y=lat) -- correcting now.")
        gdf["geometry"] = gdf["geometry"].apply(
            lambda geom: shapely_transform(lambda x, y, z=None: (y, x), geom)
        )

    # FIX: surface blank/NaN ward names immediately.
    blank_mask = gdf[ward_field].isin(["", "nan", "None"])
    if blank_mask.any():
        print(f"  WARNING: {blank_mask.sum()} ward polygon(s) in {path} have a "
              f"blank/null '{ward_field}' value -- these wards can never be "
              f"matched by name and should be fixed at the source KML.")

    return gdf[[ward_field, "geometry"]].rename(columns={ward_field: "ward"})


# ---------------------------------------------------------------------
# LOAD PIXELS + SPATIAL JOIN
# ---------------------------------------------------------------------

def load_pixel_points(csv_path: str, region_filter: str) -> tuple:
    df = pd.read_csv(csv_path)
    has_year = "year" in df.columns
    if region_filter and "region" in df.columns:
        before = len(df)
        df = df[df["region"] == region_filter].reset_index(drop=True)
        print(f"  Filtered to region={region_filter}: {len(df)}/{before} rows")

    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["longitude"], df["latitude"]), crs="EPSG:4326"
    )
    return gdf, has_year


def join_pixels_to_wards(pixels: gpd.GeoDataFrame, wards: gpd.GeoDataFrame,
                          region_name: str) -> gpd.GeoDataFrame:
    # FIX: bounds diagnostic, same as the DEM script.
    wminx, wminy, wmaxx, wmaxy = wards.total_bounds
    pminx, pminy, pmaxx, pmaxy = pixels.total_bounds
    print(f"  Ward bounds  (minx,miny,maxx,maxy): "
          f"({wminx:.4f}, {wminy:.4f}, {wmaxx:.4f}, {wmaxy:.4f})")
    print(f"  Pixel bounds (minx,miny,maxx,maxy): "
          f"({pminx:.4f}, {pminy:.4f}, {pmaxx:.4f}, {pmaxy:.4f})")

    x_overlap = (wminx <= pmaxx) and (pminx <= wmaxx)
    y_overlap = (wminy <= pmaxy) and (pminy <= wmaxy)
    if not (x_overlap and y_overlap):
        print(f"  WARNING: ward bounds and pixel bounds do NOT overlap for "
              f"{region_name} even after the lat/lon-swap auto-fix. Check CRS "
              f"and that both files actually cover {region_name}.")

    joined = gpd.sjoin(pixels, wards, how="inner", predicate="within")
    matched, total = len(joined), len(pixels)
    rate = matched / total if total else 0.0
    print(f"  Matched {matched}/{total} pixels to a ward ({rate:.1%})")
    return joined


# ---------------------------------------------------------------------
# AGGREGATE TO WARD LEVEL -- detailed feature set
# ---------------------------------------------------------------------

def shannon_diversity(class_counts: pd.Series) -> tuple:
    p = class_counts / class_counts.sum()
    p = p[p > 0]
    h = -(p * np.log(p)).sum()
    h_norm = h / np.log(len(IGBP_CLASSES))  # normalized 0-1 evenness vs all 17 possible classes
    return round(h, 4), round(h_norm, 4)


def summarize_ward_lulc(df: pd.DataFrame) -> dict:
    total = len(df)
    out = {"total_cells": total}

    # --- Impervious / pervious ---
    out["pct_impervious"] = round(df["is_impervious"].mean() * 100, 2)
    out["pct_pervious"] = round(100 - out["pct_impervious"], 2)

    # --- Dominant / second-dominant class, with confidence (%) ---
    class_counts = df["lulc_code"].value_counts()
    class_names = class_counts.index.map(IGBP_CLASSES)
    if len(class_counts) >= 1:
        out["dominant_lulc_class"] = class_names[0]
        out["dominant_class_pct"] = round(class_counts.iloc[0] / total * 100, 2)
    if len(class_counts) >= 2:
        out["second_dominant_class"] = class_names[1]
        out["second_dominant_class_pct"] = round(class_counts.iloc[1] / total * 100, 2)
    else:
        out["second_dominant_class"] = None
        out["second_dominant_class_pct"] = 0.0

    # --- Diversity / heterogeneity ---
    out["lulc_diversity_shannon"], out["lulc_diversity_normalized"] = shannon_diversity(class_counts)

    # --- CN estimates across all 4 hydrologic soil groups ---
    cn_df = df["lulc_code"].map(CN_TABLE_ABCD)
    cn_a = cn_df.map(lambda t: t[0])
    cn_b = cn_df.map(lambda t: t[1])
    cn_c = cn_df.map(lambda t: t[2])
    cn_d = cn_df.map(lambda t: t[3])
    out["mean_cn_estimate_soilA"] = round(cn_a.mean(skipna=True), 2)
    out["mean_cn_estimate_soilB"] = round(cn_b.mean(skipna=True), 2)
    out["mean_cn_estimate_soilC"] = round(cn_c.mean(skipna=True), 2)
    out["mean_cn_estimate_soilD"] = round(cn_d.mean(skipna=True), 2)
    out["cn_estimate_soilC_std"] = round(cn_c.std(skipna=True), 2)

    # --- Broad CN-standard groups ---
    group = df["lulc_code"].map(IGBP_TO_GROUP)
    group_pct = (group.value_counts(normalize=True) * 100)
    for g in ["Urban", "Cropland", "Forest", "Shrub_Grassland", "Wetland", "Water", "Barren_or_Snow"]:
        out[f"pct_{g}"] = round(group_pct.get(g, 0.0), 2)
    out["pct_vegetated"] = round(sum(out[f"pct_{g}"] for g in VEGETATED_GROUPS), 2)

    # --- Full raw IGBP class breakdown (kept alongside the broad groups) ---
    class_pct = (df["lulc_class"].value_counts(normalize=True) * 100)
    class_pct.index = [f"pct_{c.replace(' ', '_').replace('/', '_')}" for c in class_pct.index]
    out.update(class_pct.round(2).to_dict())

    return out


def build_ward_summary(joined: pd.DataFrame, has_year: bool) -> pd.DataFrame:
    """
    FIX: replaces `joined.groupby(keys).apply(summarize_ward_lulc)`, which
    silently breaks on a 0-match join (see ward_zonal_dem_fixed.py for the
    full explanation of why). Builds rows explicitly per group instead, so
    an empty `joined` just produces zero well-formed rows.
    """
    group_keys = ["ward", "year"] if has_year else ["ward"]
    rows = []
    if not joined.empty:
        for key, group in joined.groupby(group_keys):
            stats = summarize_ward_lulc(group)
            if has_year:
                stats["ward"], stats["year"] = key
            else:
                stats["ward"] = key
            rows.append(stats)

    if rows:
        summary = pd.DataFrame(rows)
        # Put id columns first; leave the rest (which vary by which LULC
        # classes are actually present) in whatever order pandas built.
        id_cols = ["ward", "year"] if has_year else ["ward"]
        other_cols = [c for c in summary.columns if c not in id_cols]
        summary = summary[id_cols + other_cols]
    else:
        id_cols = ["ward", "year"] if has_year else ["ward"]
        summary = pd.DataFrame(columns=id_cols + ["total_cells"])

    return summary


def _safe_sort_key(w):
    """FIX: same NaN-safe sort key as the DEM script -- ward names read
    from a KML with blank entries come through as NaN (float) mixed with
    normal strings, and Python can't compare across those types."""
    is_nan = w is None or (isinstance(w, float) and pd.isna(w))
    return (is_nan, "" if is_nan else str(w))


def compute_trend(panel: pd.DataFrame) -> pd.DataFrame:
    """
    panel: ward x year rows (already has pct_impervious, pct_vegetated).
    Returns one row per ward summarizing change from earliest to latest
    year available for that ward.
    """
    rows = []
    for ward, g in panel.groupby("ward"):
        g = g.sort_values("year")
        first, last = g.iloc[0], g.iloc[-1]
        n_years = last["year"] - first["year"]
        imp_change = last["pct_impervious"] - first["pct_impervious"]
        rows.append({
            "ward": ward,
            "earliest_year": int(first["year"]),
            "latest_year": int(last["year"]),
            "pct_impervious_earliest": first["pct_impervious"],
            "pct_impervious_latest": last["pct_impervious"],
            "pct_impervious_change": round(imp_change, 2),
            "avg_annual_pct_impervious_change": round(imp_change / n_years, 4) if n_years else np.nan,
            "pct_vegetated_earliest": first["pct_vegetated"],
            "pct_vegetated_latest": last["pct_vegetated"],
            "pct_vegetated_change": round(last["pct_vegetated"] - first["pct_vegetated"], 2),
        })
    return pd.DataFrame(rows)


def process_region(region_name: str, cfg: dict) -> tuple:
    print(f"\n--- {region_name} ---")
    wards = load_ward_boundaries(cfg["boundary_file"], cfg["ward_field"])
    print(f"  Loaded {len(wards)} ward polygons from {cfg['boundary_file']}")

    pixels, has_year = load_pixel_points(PIXEL_CSV, region_name)
    joined = join_pixels_to_wards(pixels, wards, region_name)

    summary = build_ward_summary(joined, has_year)

    matched_wards = set(summary["ward"])
    missing_wards = set(wards["ward"]) - matched_wards
    if missing_wards:
        # FIX: type-safe sort so NaN ward names don't crash this line.
        preview = sorted(missing_wards, key=_safe_sort_key)[:5]
        print(f"  WARNING: {len(missing_wards)} ward(s) got zero matched pixels: "
              f"{preview}{'...' if len(missing_wards) > 5 else ''}")

    # FIX: summary never carries a leftover `region` column now, so this
    # insert is always safe.
    summary.insert(0, "region", region_name)
    summary.insert(1, "ward_uid", region_name + "_" + summary["ward"].astype(str))

    out_path = f"{region_name.lower()}_ward_lulc_zonal_stats.csv"
    summary.to_csv(out_path, index=False)
    print(f"  Saved -> {out_path}")

    trend = None
    if has_year and not summary.empty:
        trend = compute_trend(summary)
        trend.insert(0, "region", region_name)
        trend.insert(1, "ward_uid", region_name + "_" + trend["ward"].astype(str))
        trend_path = f"{region_name.lower()}_ward_lulc_trend.csv"
        trend.to_csv(trend_path, index=False)
        print(f"  Saved trend -> {trend_path}")

    return summary, trend


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    all_summaries, all_trends = [], []
    for region_name, cfg in REGION_CONFIGS.items():
        try:
            summary, trend = process_region(region_name, cfg)
            all_summaries.append(summary)
            if trend is not None:
                all_trends.append(trend)
        except Exception as e:
            print(f"WARNING: skipping {region_name} -- {e}")

    if not all_summaries:
        print("\nNo regions produced usable data -- nothing to merge.")
        return

    merged = pd.concat(all_summaries, ignore_index=True)
    merged.to_csv(MERGED_OUTPUT_CSV, index=False)
    print(f"\nSaved merged all-region ward-level LULC CSV -> {MERGED_OUTPUT_CSV}")
    print(f"Total rows (wards x years if applicable): {len(merged)}")
    preview_cols = ["region", "ward", "ward_uid", "pct_impervious",
                     "dominant_lulc_class", "dominant_class_pct",
                     "lulc_diversity_normalized", "mean_cn_estimate_soilC"]
    print(merged[[c for c in preview_cols if c in merged.columns]].head(10).to_string(index=False))

    if all_trends:
        merged_trend = pd.concat(all_trends, ignore_index=True)
        merged_trend.to_csv(MERGED_TREND_CSV, index=False)
        print(f"\nSaved merged all-region LULC trend CSV -> {MERGED_TREND_CSV}")
        print(merged_trend.sort_values("pct_impervious_change", ascending=False)
              .head(10).to_string(index=False))
        print("(sorted by biggest impervious-area increase -- these wards saw the "
              "fastest urbanization over the available years)")

    print("\nNOTE: kept separate from the DEM ward CSV on purpose -- join later "
          "on region+ward (`ward_uid`) once rainfall/soil-moisture ward CSVs "
          "exist too.")


if __name__ == "__main__":
    main()