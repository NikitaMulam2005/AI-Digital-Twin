"""
Compute Topographic Wetness Index (TWI) from the existing SRTM DEM point grids
in dem_per_mc/*_srtm_dem_clean.csv, and aggregate to ward level to match the
schema of dem_per_ward/*_ward_dem_zonal_stats.csv.

TWI = ln( a / tan(slope) )
where a = specific catchment area (upslope contributing area per unit contour width)

Method:
1. Reshape the regular lat/lon/elevation point grid into a 2D raster array.
2. Write as GeoTIFF (WGS84) so pysheds can consume it.
3. Fill depressions + resolve flats (standard DEM hydro-conditioning).
4. D8 flow direction -> flow accumulation (in # of contributing cells).
5. Convert accumulation to specific catchment area using cell size in metres.
6. TWI = ln(a / tan(slope)), slope floored to avoid log(0)/div-by-0 on flat cells.
7. Zonal stats per ward polygon (from wards/*.kml), same stat set as the
   existing elevation/slope zonal stats file, so it can be merged in directly.
"""
import numpy as np
if not hasattr(np, "in1d"):        # pysheds 0.5 uses the numpy<2.0 alias
    np.in1d = np.isin
import os
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from pysheds.grid import Grid
import warnings
warnings.filterwarnings("ignore")

# Each region's ward KML uses a different attribute for the ward name, and only
# BMC/PUNE's field matches the dem_per_ward CSV naming directly.
REGIONS = {
    "BMC":   ("dem_per_mc/bmc_srtm_dem_clean.csv",   "wards/wards_mumbai.kml",       "Name",   None),
    "THANE": ("dem_per_mc/thane_srtm_dem_clean.csv", "wards/wards_thane.kml",        "sourcewardcode", "Ward No.{}"),
    "KDMC":  ("dem_per_mc/kdmc_srtm_dem_clean.csv",  "wards/kdmc.kml",               "ward_lgd_name", None),
    "NMMC":  ("dem_per_mc/nmmc_srtm_dem_clean.csv",  "wards/wards_navi_mumbai.kml",  "ward_lgd_name", None),
    "PUNE":  ("dem_per_mc/pune_srtm_dem_clean.csv",  "wards/pune.kml",               "Name",   None),
}

CELL_DEG = 0.0027777777777779  # confirmed grid spacing (10 arc-sec)

# Intermediate GeoTIFFs written next to this script (created if missing) --
# not hardcoded to any one machine's folder layout.
TIF_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tif_cache")
os.makedirs(TIF_OUTPUT_DIR, exist_ok=True)


def grid_from_points(df):
    """Turn a lat/lon/elevation point CSV into a north-up 2D array + affine transform."""
    lats = np.sort(df["latitude"].unique())[::-1]  # north -> south (row 0 = north)
    lons = np.sort(df["longitude"].unique())        # west -> east
    nrows, ncols = len(lats), len(lons)

    lat_idx = {v: i for i, v in enumerate(lats)}
    lon_idx = {v: i for i, v in enumerate(lons)}

    arr = np.full((nrows, ncols), np.nan, dtype="float32")
    ri = df["latitude"].map(lat_idx).to_numpy()
    ci = df["longitude"].map(lon_idx).to_numpy()
    arr[ri, ci] = df["elevation_m"].to_numpy()

    # SRTM ocean/void cells in this dataset are flagged -4.0; treat as nodata
    arr[arr == -4.0] = np.nan

    transform = from_origin(lons.min() - CELL_DEG / 2, lats.max() + CELL_DEG / 2,
                             CELL_DEG, CELL_DEG)
    return arr, transform, lats, lons


def write_geotiff(arr, transform, path):
    nodata = -9999.0
    out = np.where(np.isnan(arr), nodata, arr).astype("float32")
    with rasterio.open(
        path, "w", driver="GTiff", height=out.shape[0], width=out.shape[1],
        count=1, dtype="float32", crs="EPSG:4326", transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(out, 1)


def compute_twi_raster(tif_path, mean_lat_deg):
    grid = Grid.from_raster(tif_path)
    dem = grid.read_raster(tif_path)

    # Hydrologic conditioning
    pit_filled = grid.fill_pits(dem)
    flooded = grid.fill_depressions(pit_filled)
    inflated = grid.resolve_flats(flooded)

    # D8 flow direction + accumulation (# upstream cells, includes self)
    fdir = grid.flowdir(inflated)
    acc = grid.accumulation(fdir)

    # Cell size in metres (approx, fine at this scale)
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * np.cos(np.radians(mean_lat_deg))
    cell_x_m = CELL_DEG * m_per_deg_lon
    cell_y_m = CELL_DEG * m_per_deg_lat
    cell_area_m2 = cell_x_m * cell_y_m
    contour_width_m = (cell_x_m + cell_y_m) / 2.0

    specific_catchment_area = (np.asarray(acc) * cell_area_m2) / contour_width_m

    # Slope: recompute from the conditioned DEM for consistency with accumulation grid
    dz_dx, dz_dy = np.gradient(np.asarray(inflated), cell_x_m, cell_y_m)
    slope_rad = np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2))
    slope_rad_floored = np.maximum(slope_rad, np.radians(0.1))  # avoid tan(0)

    twi = np.log(np.maximum(specific_catchment_area, 1.0) / np.tan(slope_rad_floored))
    return np.asarray(twi), np.asarray(dem)


def zonal_stats(points_gdf, wards_gdf, value_col, prefix):
    joined = gpd.sjoin(points_gdf, wards_gdf[["ward_name", "geometry"]],
                        how="inner", predicate="within")
    g = joined.groupby("ward_name")[value_col]
    stats = pd.DataFrame({
        f"{prefix}_mean": g.mean(),
        f"{prefix}_median": g.median(),
        f"{prefix}_std": g.std(),
        f"{prefix}_min": g.min(),
        f"{prefix}_max": g.max(),
        f"{prefix}_p10": g.quantile(0.10),
        f"{prefix}_p90": g.quantile(0.90),
    }).reset_index().rename(columns={"ward_name": "ward"})
    return stats


def process_region(region, dem_csv, ward_kml, name_field, name_fmt):
    print(f"\n=== {region} ===")
    df = pd.read_csv(dem_csv)
    arr, transform, lats, lons = grid_from_points(df)
    tif_path = os.path.join(TIF_OUTPUT_DIR, f"{region.lower()}_dem.tif")
    write_geotiff(arr, transform, tif_path)

    twi, dem_conditioned = compute_twi_raster(tif_path, mean_lat_deg=lats.mean())
    print("TWI range:", np.nanmin(twi), "-", np.nanmax(twi))

    # Flatten grid back to points
    nrows, ncols = twi.shape
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    flat = pd.DataFrame({
        "latitude": lat_grid.ravel(),
        "longitude": lon_grid.ravel(),
        "twi": twi.ravel(),
    }).dropna()

    points_gdf = gpd.GeoDataFrame(
        flat, geometry=gpd.points_from_xy(flat["longitude"], flat["latitude"]),
        crs="EPSG:4326",
    )

    wards_gdf = gpd.read_file(ward_kml, driver="KML")
    # Some of these KMLs store coordinates as (lat, lon) instead of the standard
    # (lon, lat) -- detect by checking whether the raw x-bound looks like a
    # latitude (~18-20 for this study area) rather than a longitude (~72-74).
    minx = wards_gdf.total_bounds[0]
    if 15 < minx < 25:  # x looks like latitude -> swap x/y
        from shapely.ops import transform as shp_transform
        wards_gdf["geometry"] = wards_gdf["geometry"].apply(
            lambda g: shp_transform(lambda x, y: (y, x), g)
        )
    if name_fmt:
        wards_gdf["ward_name"] = wards_gdf[name_field].apply(lambda v: name_fmt.format(v))
    else:
        wards_gdf["ward_name"] = wards_gdf[name_field].astype(str).str.strip()

    stats = zonal_stats(points_gdf, wards_gdf, "twi", "twi")
    stats.insert(0, "region", region)
    print(stats.head())
    return stats


if __name__ == "__main__":
    all_stats = []
    for region, (dem_csv, ward_kml, name_field, name_fmt) in REGIONS.items():
        try:
            s = process_region(region, dem_csv, ward_kml, name_field, name_fmt)
            all_stats.append(s)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"FAILED {region}: {e}")

    result = pd.concat(all_stats, ignore_index=True)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "all_regions_ward_twi_zonal_stats.csv")
    result.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")
    print("\nSaved:", result.shape)
    print(result.head(10))