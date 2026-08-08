"""
merge_ward_data_v2.py
----------------------
Builds one master ward-level dataset for the AI-Digital-Twin project,
now including a real spatial join for roads and ward boundaries, plus tide.

Combines (waterlogging is still skipped):
  1. all_regions_ward_dem_zonal_stats.csv   -> static ward terrain
  2. all_regions_ward_lulc_trend.csv        -> static ward LULC trend summary
  3. wards_with_resolved_cn.csv             -> ward x year LULC / runoff-curve-number panel (2001-2024)
  4. wards/*.kml                            -> ward boundary polygons (parsed + matched to ward_uid)
  5. road_network/*.geojson                 -> roads, SPATIALLY clipped to ward polygons ->
                                                 road_length_km and road_density_km_per_km2 per ward
  6. rainfall_per_mc/*.csv                  -> daily grid rainfall -> region x year -> broadcast to wards
  7. soil_moisture_per_mc/*.csv             -> daily grid soil moisture/LST -> region x year -> broadcast
  8. tide_features_daily_1990_2024.csv      -> daily tide (Mumbai/Thane/Navi Mumbai only) ->
                                                 region x year -> broadcast to wards in coastal regions only
                                                 (KDMC and Pune wards get NaN tide columns - not coastal)

Still skipped:
  - waterlogging/*  (out of scope, per current request)

IMPORTANT DATA NOTE ON KML COORDINATE ORDER:
  wards_mumbai.kml and wards_thane.kml store coordinates as (lat, lon) instead of the
  standard (lon, lat). This script auto-corrects those two files. If you add new ward
  KML files later, always check gdf.total_bounds before trusting it - Maharashtra should
  show longitude ~72-74 as the first pair and latitude ~18-20 as the second.

CAVEAT ON ROAD DATA COMPLETENESS:
  After the spatial join, BMC (Mumbai) shows a much lower road density than the other
  four cities. That's not a bug in this script - mumbai_roads.geojson itself only
  contains ~398 km of road for the whole city, which is far below what OSM actually has
  for Mumbai. Worth re-checking/re-exporting that source file before using BMC road
  density in a model.

Output:
  ward_year_master.csv   -> one row per (region, ward_uid, ward, year), 2001-2024, all
                             static + panel + climate + tide + road-density columns

Requires: pandas, geopandas, shapely, fiona (pip install geopandas shapely fiona pyogrio)

Usage:
  python merge_ward_data_v2.py --data-dir /path/to/AI-Digital-Twin-main --out ward_year_master.csv
"""

import argparse
import glob
import os

import fiona
import geopandas as gpd
import pandas as pd
from shapely.ops import transform

fiona.drvsupport.supported_drivers["KML"] = "rw"

UTM43N = "EPSG:32643"  # metric CRS covering the whole MMR + Pune area, for length/area math

# region -> (kml path, ward-name field in that KML, needs lat/lon swap)
REGION_KML = {
    "BMC":   ("wards/wards_mumbai.kml", "Name2", True),
    "KDMC":  ("wards/kdmc.kml", "ward_lgd_name", False),
    "Pune":  ("wards/pune.kml", "Name", False),
    "NMMC":  ("wards/wards_navi_mumbai.kml", "ward_lgd_name", False),
    "Thane": ("wards/wards_thane.kml", "lgd_name", True),
}

REGION_ROADS = {
    "BMC":   "road_network/mumbai_roads.geojson",
    "KDMC":  "road_network/kdmc_roads.geojson",
    "Pune":  "road_network/pune_roads.geojson",
    "NMMC":  "road_network/nmmc_roads.geojson",
    "Thane": "road_network/thane_roads.geojson",
}

TIDE_CITY_TO_REGION = {"Mumbai": "BMC", "Thane": "Thane", "Navi Mumbai": "NMMC"}


def _swap_xy(geom):
    return transform(lambda x, y, z=None: (y, x), geom)


def load_static_ward_table(data_dir: str) -> pd.DataFrame:
    """DEM zonal stats + LULC trend -> one row per ward (no year dimension)."""
    dem = pd.read_csv(os.path.join(data_dir, "all_regions_ward_dem_zonal_stats.csv"))
    trend = pd.read_csv(os.path.join(data_dir, "all_regions_ward_lulc_trend.csv"))
    static = dem.merge(trend, on=["region", "ward_uid", "ward"], how="outer", suffixes=("", "_trend"))
    return static


def load_ward_year_panel(data_dir: str) -> pd.DataFrame:
    """wards_with_resolved_cn.csv -> one row per ward per year, 2001-2024."""
    return pd.read_csv(os.path.join(data_dir, "wards_with_resolved_cn.csv"))


def load_ward_boundaries(data_dir: str) -> gpd.GeoDataFrame:
    """Parse every ward KML, fix coordinate order where needed, match to ward_uid."""
    ward_ids = pd.read_csv(os.path.join(data_dir, "all_regions_ward_dem_zonal_stats.csv"))[
        ["region", "ward_uid", "ward"]
    ]

    all_gdfs = []
    for region, (fname, namefield, needs_swap) in REGION_KML.items():
        gdf = gpd.read_file(os.path.join(data_dir, fname), driver="KML")
        gdf["ward_name"] = gdf[namefield].astype(str).str.strip()
        if needs_swap:
            gdf["geometry"] = gdf["geometry"].apply(_swap_xy)
        gdf = gdf.set_crs(4326, allow_override=True)

        ids = ward_ids[ward_ids["region"] == region][["ward_uid", "ward"]]
        merged = gdf[["ward_name", "geometry"]].merge(ids, left_on="ward_name", right_on="ward", how="inner")
        merged["region"] = region

        n_unmatched = len(gdf) - len(merged)
        if n_unmatched:
            print(f"[wards/{fname}] {n_unmatched} polygon(s) did not match any ward_uid - dropped.")

        all_gdfs.append(merged[["region", "ward_uid", "ward", "geometry"]])

    wards_gdf = pd.concat(all_gdfs, ignore_index=True)
    return gpd.GeoDataFrame(wards_gdf, geometry="geometry", crs=4326)


def compute_road_density(data_dir: str, wards_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Spatially clip each region's roads to its ward polygons; sum length per ward."""
    wards_utm = wards_gdf.to_crs(UTM43N)
    wards_utm["area_km2"] = wards_utm.geometry.area / 1e6

    results = []
    for region, fname in REGION_ROADS.items():
        roads = gpd.read_file(os.path.join(data_dir, fname))[["fclass", "name", "geometry"]].to_crs(UTM43N)
        roads = roads[roads.geometry.notna() & ~roads.geometry.is_empty]

        w = wards_utm[wards_utm["region"] == region][["ward_uid", "area_km2", "geometry"]]
        clipped = gpd.overlay(roads, w, how="intersection")
        clipped["length_m"] = clipped.geometry.length

        agg = clipped.groupby("ward_uid", as_index=False)["length_m"].sum()
        agg = w[["ward_uid", "area_km2"]].merge(agg, on="ward_uid", how="left")
        agg["length_m"] = agg["length_m"].fillna(0)
        agg["road_length_km"] = agg["length_m"] / 1000
        agg["road_density_km_per_km2"] = agg["road_length_km"] / agg["area_km2"]
        agg["region"] = region
        results.append(agg[["region", "ward_uid", "area_km2", "road_length_km", "road_density_km_per_km2"]])

    return pd.concat(results, ignore_index=True)


def load_region_year_climate(data_dir: str, subfolder: str, pattern: str, value_cols: dict) -> pd.DataFrame:
    """Aggregate a folder of per-region daily grid CSVs to one row per region per year."""
    files = sorted(glob.glob(os.path.join(data_dir, subfolder, pattern)))
    if not files:
        raise FileNotFoundError(f"No files matched {subfolder}/{pattern}")

    frames = []
    for f in files:
        df = pd.read_csv(f, parse_dates=["date"])
        df["year"] = df["date"].dt.year
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    return combined.groupby(["region", "year"], as_index=False).agg(value_cols)


def load_tide_region_year(data_dir: str) -> pd.DataFrame:
    """Daily tide (Mumbai/Thane/Navi Mumbai only) -> region x year. KDMC/Pune are not coastal."""
    tide = pd.read_csv(os.path.join(data_dir, "tide_features_daily_1990_2024.csv"), parse_dates=["date"])
    tide["year"] = tide["date"].dt.year
    tide["region"] = tide["city"].map(TIDE_CITY_TO_REGION)
    tide = tide.dropna(subset=["region"])

    agg = tide.groupby(["region", "year"], as_index=False).agg(
        annual_mean_max_tide_m=("max_tide_height_m", "mean"),
        annual_mean_tide_range_m=("tide_range_m", "mean"),
        annual_spring_tide_days=("is_spring_tide", "sum"),
    )
    return agg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=".", help="Path to the cloned AI-Digital-Twin repo")
    parser.add_argument("--out", default="ward_year_master.csv", help="Output CSV path")
    args = parser.parse_args()
    data_dir = args.data_dir

    print("Loading static ward table (DEM + LULC trend)...")
    static_ward = load_static_ward_table(data_dir)

    print("Loading ward x year LULC/CN panel...")
    panel = load_ward_year_panel(data_dir)

    print("Parsing ward boundary KMLs and matching to ward_uid...")
    wards_gdf = load_ward_boundaries(data_dir)

    print("Spatially clipping roads to ward polygons (this takes a minute for Pune)...")
    road_density = compute_road_density(data_dir, wards_gdf)

    print("Merging static ward attributes + road density...")
    static_ward = static_ward.merge(road_density, on=["region", "ward_uid"], how="left")

    print("Merging static ward table onto the ward x year panel...")
    master = panel.merge(static_ward, on=["region", "ward_uid", "ward"], how="left", suffixes=("", "_static"))

    print("Aggregating rainfall grids to region x year (annual total mm)...")
    rainfall_yr = load_region_year_climate(
        data_dir, "rainfall_per_mc", "*_rainfall_1990_2024_clean.csv",
        value_cols={"rainfall_mm": "sum"},
    ).rename(columns={"rainfall_mm": "annual_rainfall_mm"})

    print("Aggregating soil moisture / LST grids to region x year (annual mean)...")
    soil_yr = load_region_year_climate(
        data_dir, "soil_moisture_per_mc", "*_soil_moisture_lst_era5land_1990_2024_clean.csv",
        value_cols={"soil_moisture_m3m3": "mean", "land_surface_temp_c": "mean"},
    ).rename(columns={
        "soil_moisture_m3m3": "annual_mean_soil_moisture_m3m3",
        "land_surface_temp_c": "annual_mean_lst_c",
    })

    print("Aggregating tide data to region x year (coastal regions only)...")
    tide_yr = load_tide_region_year(data_dir)

    print("Broadcasting region-level climate + tide onto every ward in that region/year...")
    master = master.merge(rainfall_yr, on=["region", "year"], how="left")
    master = master.merge(soil_yr, on=["region", "year"], how="left")
    master = master.merge(tide_yr, on=["region", "year"], how="left")  # NaN for KDMC/Pune - expected

    master["is_coastal_region"] = master["region"].isin(TIDE_CITY_TO_REGION.values())

    master = master.sort_values(["region", "ward_uid", "year"]).reset_index(drop=True)

    print(f"\nFinal shape: {master.shape[0]} rows x {master.shape[1]} columns")
    print(f"Wards covered: {master['ward_uid'].nunique()} | Years: {master['year'].min()}-{master['year'].max()}")

    master.to_csv(args.out, index=False)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()