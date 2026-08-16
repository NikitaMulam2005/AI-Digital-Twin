"""
Build all_regions_wards.geojson from the existing per-region ward KML files
--------------------------------------------------------------------------
ONE-TIME CONVERSION -- run this once, before build_sentinel1_flood_labels.py.

This does not create new ward boundaries. It reuses the exact same KML
files and config already established in build_ward_adjacency_graph.py /
merge_ward_data_v2.py (wards/*.kml, one name field per region, an
axis-swap flag for BMC and Thane) and combines them into a single
GeoJSON with a consistent schema: 'region', 'ward', and 'geometry' --
the shape build_sentinel1_flood_labels.py's load_ward_featurecollection()
expects, with ward_uid = region + "_" + ward built the same way
compute_twi.py and everything downstream of it already builds it.

WHY THE AXIS SWAP
wards_mumbai.kml (BMC) and wards_thane.kml (Thane) store coordinates as
(lat, lon) instead of the (lon, lat) order GeoJSON/KML/most GIS tooling
expects -- this is called out explicitly in merge_ward_data_v2.py's own
docstring, and confirmed here by the same REGION flag your existing
scripts already use. Left uncorrected, those two regions' polygons
would be plotted/interpreted with swapped coordinates (e.g. a Mumbai
ward polygon would appear somewhere in the ocean or a wildly wrong
location) -- this script swaps them back before writing output, exactly
matching the AXIS_SWAP flag already used elsewhere in this project.

REQUIREMENTS
    pip install geopandas fiona shapely

KML SUPPORT NOTE
    Some fiona/GDAL builds ship the KML driver as read-only or disabled
    by default. If reading fails with a "KML" driver error, try:
        pip install fiona --force-reinstall --no-binary fiona
    or install GDAL via conda instead of pip, which more reliably
    bundles KML/LIBKML support.

USAGE
    python3 build_wards_geojson_from_kml.py
    # writes ./all_regions_wards.geojson in the current directory
"""

import geopandas as gpd
from shapely.ops import transform as shapely_transform

# Same config as build_ward_adjacency_graph.py / merge_ward_data_v2.py:
#   region -> (kml_path, name_field, needs_axis_swap)
REGION_KML_CONFIG = {
    "BMC":   ("wards/wards_mumbai.kml", "Name2", True),
    "KDMC":  ("wards/kdmc.kml", "ward_lgd_name", False),
    "Pune":  ("wards/pune.kml", "Name", False),
    "NMMC":  ("wards/wards_navi_mumbai.kml", "ward_lgd_name", False),
    "Thane": ("wards/wards_thane.kml", "lgd_name", True),
}

OUTPUT_PATH = "all_regions_wards.geojson"


def swap_xy(geom):
    """Swaps x/y (lon/lat) coordinates in-place for a shapely geometry --
    used for the two regions whose source KML stores (lat, lon) instead
    of the standard (lon, lat) order."""
    return shapely_transform(lambda x, y, z=None: (y, x), geom)


def load_region(region: str, kml_path: str, name_field: str, needs_swap: bool) -> gpd.GeoDataFrame:
    print(f"Reading {region} from {kml_path} (name field: '{name_field}', "
          f"axis swap: {needs_swap})...")
    gdf = gpd.read_file(kml_path, driver="KML")
    if name_field not in gdf.columns:
        raise KeyError(
            f"'{name_field}' not found in {kml_path} columns: {list(gdf.columns)} -- "
            f"the field name may differ from what build_ward_adjacency_graph.py expects "
            f"(KML property names can vary by export tool). Update name_field for {region} "
            f"in REGION_KML_CONFIG to match."
        )
    if needs_swap:
        gdf["geometry"] = gdf["geometry"].apply(swap_xy)

    out = gpd.GeoDataFrame({
        "region": region,
        "ward": gdf[name_field].astype(str).str.strip(),
        "geometry": gdf["geometry"],
    }, crs=gdf.crs or "EPSG:4326")
    print(f"  {region}: {len(out)} ward polygons loaded.")
    return out


def main():
    parts = []
    for region, (kml_path, name_field, needs_swap) in REGION_KML_CONFIG.items():
        parts.append(load_region(region, kml_path, name_field, needs_swap))

    combined = gpd.pd.concat(parts, ignore_index=True)
    combined = gpd.GeoDataFrame(combined, crs="EPSG:4326")

    dupes = combined[combined.duplicated(subset=["region", "ward"], keep=False)]
    if not dupes.empty:
        print(f"\n  WARNING: {len(dupes)} rows share a duplicate (region, ward) pair -- "
              f"these will collide when build_sentinel1_flood_labels.py builds ward_uid. "
              f"Inspect and dedupe before proceeding:")
        print(dupes[["region", "ward"]].to_string())

    combined.to_file(OUTPUT_PATH, driver="GeoJSON")
    print(f"\nSaved -> {OUTPUT_PATH} ({len(combined)} total ward polygons across "
          f"{combined['region'].nunique()} regions).")
    print("\nPer-region ward counts:")
    print(combined.groupby("region").size().to_string())


if __name__ == "__main__":
    main()