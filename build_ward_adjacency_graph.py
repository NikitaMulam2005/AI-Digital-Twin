"""
Build Ward Adjacency Graph -- for the LSTM+GNN model
--------------------------------------------------------------------------
This is the second piece flagged as "not built yet" in
build_final_training_dataset.py's NEXT STEPS. A GNN needs edges: which
wards border which, so risk/water can propagate across them message-
pass-style. This script derives that graph from the same ward boundary
KMLs merge_ward_data_v2.py already parses for road-density clipping,
reusing its exact KML-parsing logic (namefield-per-region, the lat/lon
axis swap for wards_mumbai.kml and wards_thane.kml, matching each
polygon to a ward_uid via all_regions_ward_dem_zonal_stats.csv).

METHOD
  1. Parse all five region KMLs into one combined GeoDataFrame of ward
     polygons in EPSG:4326, matched to ward_uid (same as
     merge_ward_data_v2.load_ward_boundaries).
  2. Reproject to a metric CRS (EPSG:32643, UTM 43N -- same one
     merge_ward_data_v2.py uses for road-length math) so buffering and
     length are in real metres, not degrees.
  3. Buffer every polygon outward by BUFFER_M metres and self-join on
     'intersects'. The buffer matters: adjacent wards are often
     digitized with small gaps or slivers between them (especially
     across different source KMLs / regions), so a strict
     'touches'/zero-buffer test silently misses real neighbors. A few
     metres of tolerance fixes that without falsely connecting wards
     that are genuinely blocks apart -- if BUFFER_M is set too large,
     unrelated nearby wards start linking; the printed edge-count sanity
     check below gives a rough gut-check on that.
  4. For each surviving pair, compute the actual shared-boundary length
     (the ORIGINAL, un-buffered boundaries, intersected against a small
     buffer of each other) as an edge weight -- two wards that share a
     500m border are a stronger GNN edge than two that barely clip
     corners.
  5. Output is a plain edge list: (ward_uid_1, region_1, ward_uid_2,
     region_2, shared_boundary_km, same_region). CROSS-region edges are
     kept deliberately (e.g. a Thane ward bordering an NMMC ward) --
     the project brief describes these cities as "interconnected", and
     a GNN that can't pass messages across a city boundary would miss
     exactly the drainage-doesn't-respect-jurisdictions effect that
     motivates modeling them together in the first place.

CAVEAT -- THIS IS GEOGRAPHIC ADJACENCY, NOT DRAINAGE-FLOW ADJACENCY
'Which wards border which' is a reasonable, defensible proxy and is
what's buildable from open ward-boundary KMLs alone. It is NOT the same
as "which wards are hydrologically connected by actual stormwater
drains / nullahs / rivers" -- two wards can share a border but sit on
opposite sides of a ridge (no real flow between them), or two wards can
be hydrologically linked via a drain/river while not being geographic
neighbors at all (e.g. a river corridor threading through several
non-adjacent wards). If/when real drainage-network data becomes
available for these cities, treat this file as the fallback graph, not
the final one -- swap in real flow-direction edges where you can, and
keep geographic adjacency only for wards that data doesn't cover.

CAVEAT -- ROAD-DATA-STYLE COMPLETENESS ISSUE MAY RECUR HERE TOO
merge_ward_data_v2.py flags that mumbai_roads.geojson under-covers
Mumbai's real road network. The ward KMLs don't have that specific
problem (they're boundary polygons, not a scraped network), but do
sanity-check total wards found per region below against the known ward
counts (BMC 24, Thane 46, KDMC ~122, NMMC ~109, Pune 15) before trusting
the graph -- a KML parsing/matching issue would show up as an
implausibly low or high edge count for that region.

REQUIREMENTS
    pip install pandas geopandas shapely fiona pyogrio

HOW TO USE
1. Point DATA_DIR at your cloned AI-Digital-Twin repo root (same
   convention as merge_ward_data_v2.py --data-dir).
2. Run: python build_ward_adjacency_graph.py
3. Output:
   - ward_adjacency_edges.csv  : one row per undirected edge
   - ward_adjacency_summary.csv: per-ward degree (neighbor count),
     useful as a sanity check and as an extra GNN input feature
"""

import argparse
import os

import fiona
import geopandas as gpd
import pandas as pd
from shapely.ops import transform

fiona.drvsupport.supported_drivers["KML"] = "rw"

UTM43N = "EPSG:32643"  # same metric CRS merge_ward_data_v2.py uses

# Buffer tolerance (metres) applied before the intersects test, to catch
# real neighbors separated by small digitization gaps. Tune down if you
# start seeing implausible non-neighbor edges; tune up if known-adjacent
# ward pairs are missing from the output.
BUFFER_M = 15

# Only keep edges with at least this much real shared-boundary length --
# filters out edges that only survived because of corner-clipping noise
# from the buffer step, not a genuine shared border.
MIN_SHARED_BOUNDARY_M = 25

# Same KML config as merge_ward_data_v2.py -- region -> (kml path,
# ward-name field in that KML, needs lat/lon swap).
REGION_KML = {
    "BMC":   ("wards/wards_mumbai.kml", "Name2", True),
    "KDMC":  ("wards/kdmc.kml", "ward_lgd_name", False),
    "Pune":  ("wards/pune.kml", "Name", False),
    "NMMC":  ("wards/wards_navi_mumbai.kml", "ward_lgd_name", False),
    "Thane": ("wards/wards_thane.kml", "lgd_name", True),
}

KNOWN_WARD_COUNTS = {"BMC": 24, "Thane": 46, "KDMC": 122, "NMMC": 109, "Pune": 15}


def _swap_xy(geom):
    return transform(lambda x, y, z=None: (y, x), geom)


# ---------------------------------------------------------------------
# 1. LOAD + MATCH WARD POLYGONS (same logic as merge_ward_data_v2.py)
# ---------------------------------------------------------------------

def load_ward_boundaries(data_dir: str) -> gpd.GeoDataFrame:
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
        n_found = len(merged)
        expected = KNOWN_WARD_COUNTS.get(region)
        flag = ""
        if expected and n_found != expected:
            flag = f"  <-- expected ~{expected}, check namefield/matching if this is way off"
        print(f"[{fname}] matched {n_found} ward polygons "
              f"({n_unmatched} unmatched, dropped).{flag}")

        all_gdfs.append(merged[["region", "ward_uid", "ward", "geometry"]])

    wards_gdf = pd.concat(all_gdfs, ignore_index=True)
    wards_gdf = gpd.GeoDataFrame(wards_gdf, geometry="geometry", crs=4326)
    return wards_gdf.to_crs(UTM43N)


# ---------------------------------------------------------------------
# 2. FIND NEIGHBOR PAIRS
# ---------------------------------------------------------------------

def find_adjacent_pairs(wards_utm: gpd.GeoDataFrame) -> pd.DataFrame:
    buffered = wards_utm.copy()
    buffered["geometry"] = buffered.geometry.buffer(BUFFER_M)

    print(f"\nSelf-joining {len(buffered)} buffered ward polygons "
          f"(buffer={BUFFER_M}m) on 'intersects'...")
    joined = gpd.sjoin(
        buffered[["ward_uid", "region", "geometry"]],
        buffered[["ward_uid", "region", "geometry"]],
        predicate="intersects", how="inner",
        # geopandas appends "_" + suffix itself, so passing "1"/"2" (not
        # "_1"/"_2") is what actually produces ward_uid_1 / ward_uid_2.
        lsuffix="1", rsuffix="2",
    )
    # drop self-matches and dedupe symmetric pairs (A-B and B-A -> keep one)
    joined = joined[joined["ward_uid_1"] != joined["ward_uid_2"]]
    joined["pair_key"] = joined.apply(
        lambda r: tuple(sorted([r["ward_uid_1"], r["ward_uid_2"]])), axis=1
    )
    joined = joined.drop_duplicates(subset="pair_key").drop(columns="pair_key")
    print(f"  {len(joined)} candidate neighbor pairs after buffering + dedup.")
    return joined.reset_index(drop=True)


def compute_shared_boundary_length(pairs: pd.DataFrame, wards_utm: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    For each candidate pair, measure real shared-boundary length using
    the ORIGINAL (un-buffered) ward outlines: intersect ward A's
    boundary with a small buffer of ward B's boundary. This is the edge
    weight, and also the final filter -- pairs below
    MIN_SHARED_BOUNDARY_M are dropped as buffer-only noise.
    """
    geom_lookup = wards_utm.set_index("ward_uid")["geometry"]
    lengths = []
    for _, row in pairs.iterrows():
        g1 = geom_lookup[row["ward_uid_1"]].boundary
        g2 = geom_lookup[row["ward_uid_2"]].boundary.buffer(BUFFER_M)
        shared = g1.intersection(g2)
        lengths.append(shared.length)
    pairs = pairs.copy()
    pairs["shared_boundary_m"] = lengths

    n_before = len(pairs)
    pairs = pairs[pairs["shared_boundary_m"] >= MIN_SHARED_BOUNDARY_M].reset_index(drop=True)
    dropped = n_before - len(pairs)
    if dropped:
        print(f"  Dropped {dropped} pairs below MIN_SHARED_BOUNDARY_M "
              f"({MIN_SHARED_BOUNDARY_M}m) -- corner-clip noise, not real borders.")

    pairs["shared_boundary_km"] = pairs["shared_boundary_m"] / 1000
    pairs["same_region"] = pairs["region_1"] == pairs["region_2"]
    return pairs[[
        "ward_uid_1", "region_1", "ward_uid_2", "region_2",
        "shared_boundary_km", "same_region",
    ]]


# ---------------------------------------------------------------------
# 3. MAIN
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=".", help="Path to the cloned AI-Digital-Twin repo")
    parser.add_argument("--out", default="ward_adjacency_edges.csv", help="Output edge-list CSV path")
    args = parser.parse_args()

    print("Parsing ward boundary KMLs and matching to ward_uid...")
    wards_utm = load_ward_boundaries(args.data_dir)
    print(f"\nTotal wards matched: {len(wards_utm)} across {wards_utm['region'].nunique()} regions.")

    pairs = find_adjacent_pairs(wards_utm)
    edges = compute_shared_boundary_length(pairs, wards_utm)

    n_cross_region = (~edges["same_region"]).sum()
    print(f"\nFinal edge count: {len(edges)} "
          f"({n_cross_region} cross-region edges -- e.g. Thane<->NMMC, Thane<->KDMC).")

    print("\nEdges per region pair:")
    edges["pair_regions"] = edges.apply(
        lambda r: "-".join(sorted([r["region_1"], r["region_2"]])), axis=1
    )
    print(edges["pair_regions"].value_counts().to_string())

    edges = edges.drop(columns="pair_regions")
    edges.to_csv(args.out, index=False)
    print(f"\nSaved edge list -> {args.out}")

    # Per-ward degree, useful as both a sanity check and a standalone
    # GNN input feature (isolated wards / very high-degree wards are
    # both worth a second look before training).
    degree = pd.concat([
        edges.rename(columns={"ward_uid_1": "ward_uid", "region_1": "region"})[["ward_uid", "region"]],
        edges.rename(columns={"ward_uid_2": "ward_uid", "region_2": "region"})[["ward_uid", "region"]],
    ])
    degree_counts = degree.groupby(["ward_uid", "region"]).size().reset_index(name="num_neighbors")

    all_wards = wards_utm[["ward_uid", "region"]].drop_duplicates()
    degree_counts = all_wards.merge(degree_counts, on=["ward_uid", "region"], how="left")
    degree_counts["num_neighbors"] = degree_counts["num_neighbors"].fillna(0).astype(int)

    isolated = degree_counts[degree_counts["num_neighbors"] == 0]
    if len(isolated):
        print(f"\nWARNING: {len(isolated)} ward(s) have ZERO neighbors found -- "
              f"worth checking manually (could be a genuinely isolated ward, "
              f"e.g. an island, or a KML boundary gap larger than BUFFER_M):")
        print(isolated.to_string(index=False))

    summary_path = "ward_adjacency_summary.csv"
    degree_counts.sort_values("num_neighbors", ascending=False).to_csv(summary_path, index=False)
    print(f"Saved per-ward neighbor counts -> {summary_path}")
    print(f"\nMedian neighbors per ward: {degree_counts['num_neighbors'].median():.0f}  "
          f"(sanity check -- most urban wards should have roughly 3-8 neighbors)")


if __name__ == "__main__":
    main()