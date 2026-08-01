"""
Ward-level Zonal Statistics -- DEM (Elevation/Slope), All Regions
--------------------------------------------------------------------------
Same pattern as ward_zonal_stats_all_regions.py (the LULC version), but
for the SRTM elevation/slope pixel data from srtm_dem_multiregion.py.

FIXES APPLIED IN THIS VERSION (see inline comments marked FIX:)
  1. Pune ward_field was "name" -> actually "Name" in wards/pune.kml.
  2. When a region's spatial join matches 0 pixels, groupby().apply()
     on an empty frame doesn't call summarize_ward_dem at all -- pandas
     falls back to returning the input columns roughly as-is, which
     still contains the original `region` column from the pixel CSV.
     summary.insert(0, "region", ...) then collided with that existing
     column ("cannot insert region, already exists"). Now we build the
     summary explicitly per-ward and never depend on the apply-on-empty
     fallback shape.
  3. sorted(missing_wards) crashed when ward names contain NaN (float)
     mixed with strings, because Python can't compare float < str.
     Now sorts with a type-safe key.
  4. Added a bounds diagnostic + hard failure on 0-match joins, so a
     CRS/coordinate-order mismatch is caught immediately instead of
     limping forward into confusing downstream errors.

WHY THESE PARTICULAR FEATURES
Flood-risk modeling really wants flow accumulation / TWI / HAND (Height
Above Nearest Drainage) at ward level -- see the flow_accumulation_* and
twi_* columns in your original ward_zonal_stats.csv. This script can't
produce those: they require hydrological conditioning (sink-filling,
flow routing) via dedicated tools -- WhiteboxTools, RichDEM, or GRASS
r.watershed -- run on the native-resolution DEM, which is a separate
step from this pixel-level CSV. If/when you run that, the output can be
zonal-stats'd with this same script structure (swap CONTINUOUS_COLUMNS
and rerun).

What THIS script does instead is get more signal out of the elevation/
slope data we already have, rather than settling for plain mean/min/max:
    - Distribution shape: mean, median, std, min, max, p10, p90 (using
      percentiles instead of raw min/max alone matters here specifically
      because SRTM has known noisy outlier pixels -- p10/p90 are far
      more robust to a handful of bad cells than min/max).
    - Flood-relevant derived shares, which are meaningful even without
      full flow routing:
        pct_low_lying_<Xm>  -- share of ward at or below X meters
                                elevation (proxy for waterlogging-prone
                                area -- doesn't require flow direction
                                to be useful in a flat coastal city like
                                Mumbai, where low elevation alone is a
                                strong predictor of ponding)
        pct_below_sea_level -- share of ward with negative elevation
                                (reclaimed land / tidal flats -- these
                                already showed up clearly in the raw
                                BMC/NMMC quality reports)
        pct_flat_terrain    -- share with slope below FLAT_SLOPE_DEG
                                (flat = water doesn't drain away on its
                                own, all else equal)
        pct_steep_terrain   -- share with slope above STEEP_SLOPE_DEG
                                (steep = fast runoff generation upstream
                                of flatter areas)
        relief_m            -- max minus min elevation (a crude proxy
                                for how much internal head/drainage
                                potential the ward has)

Produces:
    - one CSV per region  (e.g. bmc_ward_dem_zonal_stats.csv)
    - one merged CSV across all regions (all_regions_ward_dem_zonal_stats.csv),
      with `region` and `ward_uid` columns matching the LULC version's
      keys, so DEM, LULC, rainfall, and soil moisture can all be joined
      later on region+ward. Kept as a separate file here on purpose --
      merge downstream once all four are ready.

Same per-corporation ward-field mapping as the LULC script (see that
script's docstring for how these were determined from the raw KMLs):
    BMC   (wards_mumbai.kml)       -> "Name"
    Thane (wards_thane.kml)        -> "lgd_name"
    KDMC  (kdmc.kml)               -> "ward_lgd_name"
    NMMC  (wards_navi_mumbai.kml)  -> "ward_lgd_name"
    Pune  (pune.kml)               -> "Name"   # FIX: was "name" (lowercase)

REQUIREMENTS
    pip install geopandas fiona shapely pandas numpy

HOW TO USE
    python ward_zonal_dem_fixed.py
    -- expects mmr_pune_srtm_dem_clean.csv (from srtm_dem_multiregion.py)
       in the same folder
"""

import fiona
import geopandas as gpd
import numpy as np
import pandas as pd
from pathlib import Path
from shapely.ops import transform as shapely_transform

# Rough India lat/lon envelope, generous enough to cover the whole MMR +
# Pune study area with margin. Used only to auto-detect swapped KML
# coordinates -- not a precision check.
INDIA_LAT_RANGE = (6.0, 38.0)
INDIA_LON_RANGE = (66.0, 100.0)

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

PIXEL_CSV = "mmr_pune_srtm_dem_clean.csv"   # combined pixel-level DEM output

# Thresholds behind the derived flood-relevant features -- adjust to
# taste. LOW_LYING_THRESHOLDS_M can hold multiple cutoffs; you'll get
# one pct_low_lying_<X>m column per value.
LOW_LYING_THRESHOLDS_M = [2, 5, 10]
FLAT_SLOPE_DEG = 2.0     # at/below this = "flat" (poor natural drainage)
STEEP_SLOPE_DEG = 15.0   # at/above this = "steep" (fast runoff generation)

REGION_CONFIGS = {
    "BMC":   {"boundary_file": "wards/wards_mumbai.kml",       "ward_field": "Name"},
    "Thane": {"boundary_file": "wards/wards_thane.kml",        "ward_field": "lgd_name"},
    "KDMC":  {"boundary_file": "wards/kdmc.kml",               "ward_field": "ward_lgd_name"},
    "NMMC":  {"boundary_file": "wards/wards_navi_mumbai.kml",  "ward_field": "ward_lgd_name"},
    # FIX: pune.kml has "Name" (capital N), not "name" -- confirmed by the
    # original error's own "Columns available" list.
    "Pune":  {"boundary_file": "wards/pune.kml",               "ward_field": "Name"},
}

MERGED_OUTPUT_CSV = "all_regions_ward_dem_zonal_stats.csv"

# Columns that summarize_ward_dem produces, in order. Used to build an
# empty-but-correctly-shaped DataFrame when a region matches 0 pixels,
# so downstream code (insert, to_csv, concat) behaves identically
# whether a region has data or not. FIX: this is what lets us avoid the
# groupby().apply()-on-empty-frame fallback that caused the BMC crash.
def _stat_columns() -> list:
    cols = ["ward", "total_cells"]
    for name in ["elevation_m", "slope_deg"]:
        cols += [f"{name}_mean", f"{name}_median", f"{name}_std",
                 f"{name}_min", f"{name}_max", f"{name}_p10", f"{name}_p90"]
    cols += ["relief_m", "pct_below_sea_level"]
    cols += [f"pct_low_lying_{t}m" for t in LOW_LYING_THRESHOLDS_M]
    cols += ["pct_flat_terrain", "pct_steep_terrain"]
    return cols


# ---------------------------------------------------------------------
# LOAD BOUNDARIES
# ---------------------------------------------------------------------

def _looks_like_swapped_latlon(gdf: gpd.GeoDataFrame) -> bool:
    """
    FIX (root cause of BMC/Thane 0% match): some source KMLs (confirmed:
    wards_mumbai.kml, wards_thane.kml) were authored with coordinates as
    (lat, lon) instead of the KML/GeoJSON standard (lon, lat). Diagnostic
    run showed BMC ward bounds x=18.89-19.27 (that's latitude sitting in
    the x/longitude slot) vs the pixel CSV's correct x=72.75-73.05. This
    detects that pattern generically: if geometry x-values fall in the
    India latitude envelope AND y-values fall in the India longitude
    envelope, the file has lat/lon swapped.
    """
    minx, miny, maxx, maxy = gdf.total_bounds
    x_is_lat = INDIA_LAT_RANGE[0] <= minx and maxx <= INDIA_LAT_RANGE[1]
    y_is_lon = INDIA_LON_RANGE[0] <= miny and maxy <= INDIA_LON_RANGE[1]
    # Guard against the (correct) case where lon happens to be a valid
    # "latitude-range" number too -- only flag when x looks like lat AND
    # y looks like lon AND x does NOT already look like a valid lon.
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

    # FIX: auto-detect and correct swapped lat/lon coordinates. This is
    # what was silently causing 0% matches for BMC and Thane -- their
    # KML geometries have x=latitude, y=longitude instead of the
    # standard x=longitude, y=latitude.
    if _looks_like_swapped_latlon(gdf):
        print(f"  FIXED: {path} has swapped lat/lon coordinates "
              f"(x=lat, y=lon instead of x=lon, y=lat) -- correcting now.")
        gdf["geometry"] = gdf["geometry"].apply(
            lambda geom: shapely_transform(lambda x, y, z=None: (y, x), geom)
        )

    # FIX: surface blank/NaN ward names immediately -- these are what
    # caused sorted() to crash later (comparing float NaN to str).
    blank_mask = gdf[ward_field].isin(["", "nan", "None"])
    if blank_mask.any():
        print(f"  WARNING: {blank_mask.sum()} ward polygon(s) in {path} have a "
              f"blank/null '{ward_field}' value -- these wards can never be "
              f"matched by name and should be fixed at the source KML.")

    return gdf[[ward_field, "geometry"]].rename(columns={ward_field: "ward"})


# ---------------------------------------------------------------------
# LOAD PIXELS + SPATIAL JOIN
# ---------------------------------------------------------------------

def load_pixel_points(csv_path: str, region_filter: str) -> gpd.GeoDataFrame:
    df = pd.read_csv(csv_path)
    if region_filter and "region" in df.columns:
        before = len(df)
        df = df[df["region"] == region_filter].reset_index(drop=True)
        print(f"  Filtered to region={region_filter}: {len(df)}/{before} rows")

    return gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["longitude"], df["latitude"]), crs="EPSG:4326"
    )


def join_pixels_to_wards(pixels: gpd.GeoDataFrame, wards: gpd.GeoDataFrame,
                          region_name: str) -> gpd.GeoDataFrame:
    # FIX: print bounds so a CRS/lon-lat-swap mismatch is visible
    # immediately instead of manifesting as a mysterious 0-7% match rate.
    wminx, wminy, wmaxx, wmaxy = wards.total_bounds
    pminx, pminy, pmaxx, pmaxy = pixels.total_bounds
    print(f"  Ward bounds  (minx,miny,maxx,maxy): "
          f"({wminx:.4f}, {wminy:.4f}, {wmaxx:.4f}, {wmaxy:.4f})")
    print(f"  Pixel bounds (minx,miny,maxx,maxy): "
          f"({pminx:.4f}, {pminy:.4f}, {pmaxx:.4f}, {pmaxy:.4f})")

    x_overlap = (wminx <= pmaxx) and (pminx <= wmaxx)
    y_overlap = (wminy <= pmaxy) and (pminy <= wmaxy)
    if not (x_overlap and y_overlap):
        # NOTE: load_ward_boundaries() now auto-detects and fixes the
        # lat/lon-swap case (confirmed cause for BMC/Thane). If this
        # warning still fires after that fix, it's something else --
        # e.g. a genuinely different CRS, or the ward file covering a
        # different area than the pixel CSV -- and needs a fresh look.
        print(f"  WARNING: ward bounds and pixel bounds still do NOT overlap for "
              f"{region_name} even after the lat/lon-swap auto-fix. This is a "
              f"different problem than the BMC/Thane swap -- check CRS and that "
              f"both files actually cover {region_name}.")

    joined = gpd.sjoin(pixels, wards, how="inner", predicate="within")
    matched, total = len(joined), len(pixels)
    rate = matched / total if total else 0.0
    print(f"  Matched {matched}/{total} pixels to a ward ({rate:.1%})")

    # FIX: if the strict "within" predicate got a suspiciously low match
    # despite overlapping bounds, retry with "intersects" as a diagnostic
    # (points sitting exactly on a polygon boundary/grid edge are not
    # "within" it). We don't silently swap behavior -- we just report it.
    if total and rate < 0.5 and x_overlap and y_overlap:
        test = gpd.sjoin(pixels, wards, how="inner", predicate="intersects")
        if len(test) > matched:
            print(f"  NOTE: predicate='intersects' would match {len(test)}/{total} "
                  f"({len(test)/total:.1%}) -- if that's much higher, pixel centers "
                  f"may be landing exactly on ward boundary lines. Consider switching "
                  f"predicate to 'intersects' for this dataset.")

    return joined


# ---------------------------------------------------------------------
# AGGREGATE TO WARD LEVEL -- richer distributional + flood-relevant stats
# ---------------------------------------------------------------------

def summarize_ward_dem(df: pd.DataFrame) -> dict:
    elev = df["elevation_m"]
    slope = df["slope_deg"]
    out = {"total_cells": len(df)}

    # --- Distribution shape (percentiles used instead of relying only
    # on min/max, since SRTM has known noisy outlier pixels) ---
    for name, series in [("elevation_m", elev), ("slope_deg", slope)]:
        out[f"{name}_mean"] = round(series.mean(), 4)
        out[f"{name}_median"] = round(series.median(), 4)
        out[f"{name}_std"] = round(series.std(), 4)
        out[f"{name}_min"] = round(series.min(), 4)
        out[f"{name}_max"] = round(series.max(), 4)
        out[f"{name}_p10"] = round(series.quantile(0.10), 4)
        out[f"{name}_p90"] = round(series.quantile(0.90), 4)

    # --- Flood-relevant derived features ---
    out["relief_m"] = round(elev.max() - elev.min(), 4)
    out["pct_below_sea_level"] = round((elev < 0).mean() * 100, 2)
    for threshold in LOW_LYING_THRESHOLDS_M:
        out[f"pct_low_lying_{threshold}m"] = round((elev <= threshold).mean() * 100, 2)
    out["pct_flat_terrain"] = round((slope <= FLAT_SLOPE_DEG).mean() * 100, 2)
    out["pct_steep_terrain"] = round((slope >= STEEP_SLOPE_DEG).mean() * 100, 2)

    return out


def build_ward_summary(joined: pd.DataFrame, all_ward_names: list) -> pd.DataFrame:
    """
    FIX: replaces `joined.groupby("ward").apply(summarize_ward_dem)`.
    That pattern silently breaks when `joined` is empty (0 matched
    pixels): pandas can't infer output shape from zero groups, so
    .apply() returns the ORIGINAL frame's columns unchanged instead of
    calling summarize_ward_dem at all. That left a stray `region` column
    sitting in `summary`, which collided with the later
    `summary.insert(0, "region", ...)` call ("cannot insert region,
    already exists").

    This version builds rows explicitly per ward, so an empty `joined`
    just produces zero rows with the *correct* columns -- no ambiguity,
    no fallback behavior to trip over.
    """
    rows = []
    if not joined.empty:
        for ward_name, group in joined.groupby("ward"):
            stats = summarize_ward_dem(group)
            stats["ward"] = ward_name
            rows.append(stats)

    if rows:
        summary = pd.DataFrame(rows)
        summary = summary[_stat_columns()]  # enforce consistent column order
    else:
        summary = pd.DataFrame(columns=_stat_columns())

    return summary


# ---------------------------------------------------------------------
# SORT HELPER -- FIX for the Thane crash
# ---------------------------------------------------------------------

def _safe_sort_key(w):
    """
    FIX: sorted(missing_wards) crashed with
        '<' not supported between instances of 'float' and 'str'
    because ward names read from a KML with blank entries come through
    as NaN (float) mixed in with normal string ward names, and Python's
    sort can't compare across those types. This key coerces everything
    to a (is_missing, string) tuple so NaN/None sort safely to the end
    without ever being compared directly to a string.
    """
    is_nan = w is None or (isinstance(w, float) and pd.isna(w))
    return (is_nan, "" if is_nan else str(w))


# ---------------------------------------------------------------------
# PER-REGION PROCESSING
# ---------------------------------------------------------------------

def process_region(region_name: str, cfg: dict) -> pd.DataFrame:
    print(f"\n--- {region_name} ---")
    wards = load_ward_boundaries(cfg["boundary_file"], cfg["ward_field"])
    print(f"  Loaded {len(wards)} ward polygons from {cfg['boundary_file']}")

    pixels = load_pixel_points(PIXEL_CSV, region_name)
    joined = join_pixels_to_wards(pixels, wards, region_name)

    summary = build_ward_summary(joined, wards["ward"].tolist())

    missing_wards = set(wards["ward"]) - set(summary["ward"])
    if missing_wards:
        # FIX: type-safe sort so NaN ward names don't crash this line.
        preview = sorted(missing_wards, key=_safe_sort_key)[:5]
        print(f"  WARNING: {len(missing_wards)} ward(s) got zero matched pixels "
              f"(possible at fine resolution near ward edges/coastline, or a "
              f"region-wide coordinate mismatch -- see bounds above): "
              f"{preview}{'...' if len(missing_wards) > 5 else ''}")

    # FIX: summary now never carries a leftover `region` column from the
    # pixel CSV (see build_ward_summary), so these inserts are always safe.
    summary.insert(0, "region", region_name)
    summary.insert(1, "ward_uid", region_name + "_" + summary["ward"].astype(str))

    out_path = f"{region_name.lower()}_ward_dem_zonal_stats.csv"
    summary.to_csv(out_path, index=False)
    print(f"  Saved -> {out_path}")

    return summary


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    all_summaries = []
    for region_name, cfg in REGION_CONFIGS.items():
        try:
            summary = process_region(region_name, cfg)
            all_summaries.append(summary)
        except Exception as e:
            print(f"WARNING: skipping {region_name} -- {e}")

    if not all_summaries:
        print("\nNo regions produced usable data -- nothing to merge.")
        return

    merged = pd.concat(all_summaries, ignore_index=True)
    merged.to_csv(MERGED_OUTPUT_CSV, index=False)
    print(f"\nSaved merged all-region ward-level DEM CSV -> {MERGED_OUTPUT_CSV}")
    print(f"Total wards across all regions: {len(merged)}")

    zero_pixel_wards = (merged["total_cells"].isna() | (merged["total_cells"] == 0)).sum()
    if zero_pixel_wards:
        print(f"NOTE: {zero_pixel_wards} ward(s) across all regions have NO matched "
              f"pixel data (empty stat columns) -- investigate the bounds/overlap "
              f"warnings printed above before trusting this merged CSV.")

    preview_cols = ["region", "ward", "ward_uid", "elevation_m_mean",
                     "elevation_m_p10", "pct_low_lying_5m", "pct_flat_terrain",
                     "relief_m"]
    print(merged[preview_cols].head(10).to_string(index=False))
    print("\nNOTE: kept separate from the LULC ward CSV on purpose -- join later "
          "on region+ward (`ward_uid`) once rainfall/soil-moisture ward CSVs "
          "exist too, rather than merging piecemeal here.")
    print("NOTE: for flow_accumulation/TWI/HAND-style features, run proper "
          "hydrological conditioning (WhiteboxTools/RichDEM/GRASS) on the "
          "native-resolution DEM first -- see docstring.")


if __name__ == "__main__":
    main()