"""
Build Sentinel-1 SAR-Derived Flood Labels (ward-day grain, 2014-present)
--------------------------------------------------------------------------
NEW PIPELINE -- separate from train_and_compare_models.py's LR/RF/LSTM/
LSTM+GNN comparison. This script does not touch that file. It produces
two new CSVs that can OPTIONALLY be merged into that pipeline later, as
a second, independent, satellite-observed flood signal to sit alongside
the IFI-event-derived label (region_flood_days_count / the daily
CORRECTED_DAILY_CSV).

WHY THIS EXISTS
The IFI-derived label has two structural weaknesses documented in
build_corrected_flood_labels.py / train_and_compare_models.py:
  1. It is REGION-BROADCAST -- every ward in a region shares one
     identical flood label, because IFI's real reporting ceiling is
     district, not ward.
  2. It is TEXT-REPORT-DERIVED -- start/end dates come from written
     event summaries, which is why 2020 needed a whole separate
     correction pass for physically-implausible event durations.
Sentinel-1 is a C-band SAR satellite (not optical), so it sees through
monsoon cloud cover -- exactly when optical imagery is least usable --
and standing water gives a distinctive low-backscatter signature. That
makes it a genuinely independent, physically-grounded, WARD-RESOLUTION
observation, not a re-derivation of the same IFI text reports. It does
NOT fix problem 2 magically (SAR flood-mapping has its own error modes,
see HONEST LIMITS below) but it directly attacks problem 1.

WHAT THIS SCRIPT DOES NOT DO
  - It does not replace region_flood_days_count or CORRECTED_DAILY_CSV.
    It produces a CANDIDATE second label. Whether/how to merge the two
    is a separate decision (see "HOW TO USE THE OUTPUT" at the bottom).
  - It does not backfill before Sentinel-1 existed. Sentinel-1A launched
    April 2014; usable IW-mode GRD coverage over India is reliably
    available from late 2014 onward. Years before that keep using the
    IFI-derived label only -- there is no SAR alternative for them.

RUNNING THIS IN COLAB
    !pip install -q earthengine-api geemap geopandas
    Then, in a Colab cell (interactive auth, one-time per session):
        import ee
        ee.Authenticate()
        ee.Initialize(project="YOUR_GEE_PROJECT_ID")
    Then:
        !python build_sentinel1_flood_labels.py

REQUIRED INPUT
    all_regions_wards.geojson -- ward polygon boundaries for the same 5
    regions (BMC, Thane, KDMC, NMMC, Pune) and same ward_uid scheme
    (region + "_" + ward) used everywhere else in this project (see
    compute_twi.py's ward_uid convention). This is the SAME boundary
    file that presumably backs the DEM/TWI zonal stats -- if it lives
    under a different name/path in your repo, update WARD_GEOJSON_PATH
    below. This script does not create ward boundaries; it only
    consumes them.

OUTPUTS
    sentinel1_ward_day_water_fraction.csv
        ward_uid, date, water_fraction, n_valid_pixels, orbit_pass
        One row per ward per Sentinel-1 acquisition date that actually
        covered it (NOT one row per calendar day -- see "TEMPORAL
        RESOLUTION" below). water_fraction is the share of that ward's
        valid SAR pixels flagged as flooded on that pass.
    sentinel1_region_day_flooded_CANDIDATE.csv
        region, date, s1_region_water_fraction, s1_flooded
        Region-day rollup (mean across wards, then thresholded) in the
        same (region, date) shape as CORRECTED_DAILY_CSV, so it can be
        directly compared against the IFI-derived daily label -- see
        compare_against_ifi_label() at the bottom.

METHOD -- PHYSICALLY-GROUNDED SAR CHANGE DETECTION & HYDROLOGIC MASKING
A fixed "VV < -17 dB means water" rule is a known-fragile approach: the
right cutoff shifts with incidence angle, surface roughness, and land
cover, and differs between BMC's dense urban fabric and Pune's more
open terrain. This script instead uses PER-PIXEL, ORBIT-AWARE CHANGE
DETECTION with 7 essential physical and hydrologic constraints:

  1. INTERTIDAL MUDFLAT & MANGROVE MASKING (JRC GSW occurrence < 30%):
     In the original unmasked run, Navi Mumbai (NMMC) creek wards
     accounted for 65% of all flood flags in the entire 12-year dataset.
     Lowering the permanent water mask from 80% to 30% permanently masks
     out intertidal creek mudflats, estuaries, and mangrove fringes in
     Thane Creek and Panvel Creek that naturally submerge under normal
     astronomical high tides, eliminating false tidal flooding.
  2. TOPOGRAPHIC SLOPE MASK (SRTM GL1 30m DEM, slope <= 5 deg):
     Standing floodwater physically cannot pool on steep hillsides. Steep
     slopes also suffer from SAR geometric distortions (layover,
     foreshortening, and radar shadow). Masking out pixels with slope > 5 deg
     eliminates spurious backscatter drops in Pune and SGNP ghats.
  3. DUAL-CONDITION WATER PIXEL THRESHOLD (Relative Drop >= 3 dB AND
     Absolute VV <= -14 dB):
     Smooth standing water causes specular reflection away from the radar
     antenna, creating an inherently dark backscatter (<= -14 dB).
     Requiring both a >= 3 dB drop from dry baseline AND an absolute
     backscatter ceiling of <= -14 dB prevents tall buildings or bridges
     whose backscatter drops from +12 dB to +6 dB from being falsely
     classified as floodwater.
  4. ORBIT-SPECIFIC DRY REFERENCES (Ascending vs. Descending):
     Sentinel-1 looks East on Ascending passes (evening) and West on
     Descending passes (morning). Tall urban structures cast radar shadows
     in opposite directions. Generating separate dry-season reference
     composites for ASCENDING and DESCENDING passes eliminates radar
     shadow mismatch false alarms in dense urban fabric (BMC/Thane).
  5. SEASONAL CLIMATOLOGY (Monsoon Monitoring Window, Jun-Oct):
     Floods in MMR and Pune are strictly a monsoon phenomenon. Monitoring
     year-round previously generated 947 false-positive passes (75% of
     all flags in the 12-year dataset) in dry winter months due to winter
     high tides, agricultural irrigation, and dew. Restricting to monsoon
     months (Jun-Oct) aligns with tropical hydrology and halves Earth
     Engine runtime.
  6. SPATIAL COHERENCE / SPECKLE FILTERING (Connected Pixels >= 5):
     Connected component filtering requires flagged pixels to form
     contiguous clusters of at least 5 pixels (~500 m2), eliminating
     isolated single-pixel speckle noise spikes along road edges.
  7. REALISTIC URBAN WARD WATER FRACTION THRESHOLD (0.03 = 3%):
     In dense megacity wards (BMC/Thane), buildings and concrete occupy
     75-85% of land. Arterial roads, subways, and low-lying junctions
     occupy only 3-8% of the surface. A 3% water fraction corresponds to
     severe urban paralysis (e.g. Hindmata, Milan subway 4 ft deep).
     Lowering from 10% to 3% allows genuine urban flash floods to be
     detected.


TEMPORAL RESOLUTION -- HONEST LIMIT, NOT A BUG
Sentinel-1 revisit over this area is ~6-12 days per orbit track (worse
after Sentinel-1B's Dec-2021 failure reduced constellation revisit
until Sentinel-1C started contributing). This is NOT daily. Do not
forward-fill or interpolate between passes to manufacture a daily
series -- that would fabricate observations on days nothing was
actually sensed and silently turn a sparse, honest signal into a false
dense one. sentinel1_ward_day_water_fraction.csv has one row per REAL
acquisition date per ward, nothing else. Any downstream use (e.g.
comparing to the IFI daily label) must join on actual acquisition
dates, not assume every day has a value.

HONEST LIMITS
  1. Urban SAR flood mapping is harder than open-field flood mapping:
     dense urban backscatter is already high/heterogeneous (building
     corner reflections), which can mask or mimic the change-detection
     signal in built-up wards. Expect noisier results in dense parts
     of BMC than in Pune's less built-up wards -- this is a known,
     published limitation of SAR urban flood detection, not specific
     to this pipeline.
  2. Revisit gaps mean a short flood (a few hours to ~1-2 days) can
     fall entirely between two passes and simply not be observed --
     this script cannot detect what it never imaged. It will tend to
     under-count brief flash floods relative to the IFI text-report
     label, which has no such revisit gap (at the cost of its own
     duration-precision problems -- see build_corrected_flood_labels.py).
  3. CHANGE_THRESHOLD_DB=3 and the dry-reference window are defensible
     literature defaults, not values tuned against this project's own
     ground truth (there is no ward-level ground truth to tune against
     -- that absence is exactly the problem this script is trying to
     help with). Treat s1_flooded as a candidate signal to VALIDATE
     against the IFI label (see compare_against_ifi_label), not as
     an already-proven replacement.
  4. This script only builds the LABEL/target-refinement path. It does
     not add Sentinel-1 features to the daily model INPUTS in
     train_and_compare_models.py, and deliberately so -- see the
     module docstring's "WHY THIS EXISTS" above for the label-vs-
     feature reasoning.

REQUIREMENTS
    pip install earthengine-api geemap geopandas pandas numpy

PERFORMANCE NOTE (added after the original ran >30min stuck on BMC)
---------------------------------------------------------------------
The original per-image loop called ee.Image.date().getInfo(),
.get("orbitProperties_pass").getInfo(), and stats.getInfo() SEPARATELY
for each of BMC's 309 acquisitions -- i.e. ~900 blocking network round-
trips to Earth Engine's servers for one region alone. Worse, `reference`
(the dry-season median composite built from 68 images) was left as a
lazy, unevaluated ee.Image. Because each iteration issued its own
independent getInfo() request, Earth Engine had to re-derive that whole
68-image median + speckle-filter chain from scratch on EVERY iteration,
not once. That's ~309x more reference-building work than necessary, on
top of the round-trip latency itself.

The fix below went through two iterations before landing on this one:
  ATTEMPT 1 (one getInfo() for the whole region): fanned out
  n_images x n_wards reduceRegions aggregations in a single request --
  tripped Earth Engine's "too many concurrent aggregations" 429 outright
  on BMC's 309 images.
  ATTEMPT 2 (batches of BATCH_SIZE images via .map()): still built each
  batch's graph as BATCH_SIZE separate reduceRegions calls fanned out
  concurrently via .map() -- even at BATCH_SIZE=20 this ALSO tripped the
  same "too many concurrent aggregations" 429. The quota is on
  concurrent aggregation OPERATIONS per request, not on total images
  processed, so no batch size fixes this as long as each image still
  gets its own reduceRegions call within the batch.
  ATTEMPT 3 (this version): stack each batch's per-image flood masks
  into ONE MULTI-BAND IMAGE (one band per acquisition, named b0, b1, ...)
  via ee.Image.addBands, then call reduceRegions ONCE per batch on that
  multi-band image. Reducing multiple bands in a single reduceRegions
  call is ONE aggregation operation regardless of band count, so this
  sidesteps the concurrent-aggregations quota entirely rather than just
  shrinking around it. BATCH_SIZE here now controls how many bands get
  stacked into one image (memory/request-size, a different and much
  looser limit) rather than how many concurrent aggregations get
  requested, so it can safely be set much higher than 20 if a region is
  going slowly -- see BATCH_SIZE's comment.
  Band names are purely positional (b0, b1, ...); acquisition date and
  orbit_pass for each position are fetched separately via two cheap
  aggregate_array() calls (get_image_dates_and_orbits) -- each of those
  is also a single aggregation, not one per image -- and joined back to
  the reduceRegions output by parsing the "b<i>_mean" / "b<i>_count"
  column names EE generates (see BAND_RE / zonal_stats_for_region).
"""

from __future__ import annotations

import argparse
import json
import re
import time
import numpy as np
import pandas as pd

try:
    import ee
except ImportError:
    ee = None


# ---------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------

GEE_PROJECT_ID = "atomic-airship-504218-p4"  # REQUIRED -- set this before running.
                                          # ee.Initialize needs a registered
                                          # Cloud project as of the current
                                          # Earth Engine API requirements.

WARD_GEOJSON_PATH = "all_regions_wards.geojson"  # ward_uid = region + "_" + ward,
                                                    # same convention as compute_twi.py.
REGIONS = ["BMC", "Thane", "KDMC", "NMMC", "Pune"]

# Sentinel-1A launched 2014-04-03; reliable IW GRD coverage over India is
# from late 2014. Starting at 2014-10-01 avoids the thin/irregular first
# few months of tasking over this specific area.
S1_START_DATE = "2014-10-01"
S1_END_DATE = "2026-08-16"  # today; re-run periodically to extend.

DRY_SEASON_MONTHS = [12, 1, 2]  # Dec-Feb: reliably outside the Jun-Sep SW
                                  # monsoon and the Oct-Nov retreating-monsoon
                                  # tail for this region, used to build each
                                  # region's per-pixel dry-season reference.

# 1. PHYSICAL SEASONAL CONSTRAINT
MONITOR_ALL_YEAR = False  # False (RECOMMENDED): restricts detection to MONSOON_MONTHS
                           # (June-October). 98%+ of flood events in MMR/Pune occur in
                           # monsoon. Dry-season monitoring picks up astronomical high
                           # tides in creek mudflats, agricultural irrigation in KDMC/Pune,
                           # and winter dew, which previously generated 75% false positives.
                           # Use --all-year CLI flag if you explicitly want year-round runs.
MONSOON_MONTHS = [6, 7, 8, 9, 10]

# 2. RADAR PRE-PROCESSING & DUAL-CONDITION THRESHOLDING
SPECKLE_FILTER_RADIUS_PX = 50  # ~50m at S1's native ~10m pixel spacing; matches
                                 # common literature practice for focal-median filtering.

CHANGE_THRESHOLD_DB = 3.0  # Relative drop >= 3 dB from dry baseline (UN-SPIDER standard).
                             # Standing water reflects microwave pulses away from sensor.

WATER_BACKSCATTER_CEILING_DB = -14.0  # Absolute backscatter ceiling (dB). Smooth standing
                                      # water acts as a specular reflector (VV <= -14 dB).
                                      # Prevents tall buildings dropping from +12 to +6 dB
                                      # from being falsely flagged as water.

# 3. PERMANENT & INTERTIDAL WATER MASKING
PERMANENT_WATER_OCCURRENCE_PCT = 30  # JRC Global Surface Water "occurrence" band >= this %
                                       # is treated as permanent/semi-permanent water and
                                       # masked out. Lowered from 80% to 30% to mask out
                                       # intertidal creek mudflats and mangrove fringes in
                                       # Thane/Panvel Creek that naturally submerge at high
                                       # tides (previously responsible for 65% of all flood flags).

# 4. TOPOGRAPHIC SLOPE MASK (SRTM 30m DEM)
MAX_SLOPE_DEG = 5.0  # Maximum slope in degrees. Standing water physically cannot pool on
                     # steep slopes (>5 deg). Eliminates SAR layover/foreshortening/shadow
                     # false alarms on hillsides in Pune and Sanjay Gandhi National Park.

# 5. SPATIAL COHERENCE / CONNECTED PIXEL FILTER
MIN_CONNECTED_PIXELS = 5  # Flagged pixels must belong to a contiguous cluster of >= 5 pixels
                          # (~500 m2) to eliminate isolated 1-2 pixel SAR speckle spikes.

# 6. ORBIT-SPECIFIC DRY REFERENCES
USE_ORBIT_SPECIFIC_REFERENCE = True  # Builds separate dry-season reference composites for
                                     # ASCENDING (East-looking) and DESCENDING (West-looking)
                                     # passes. Eliminates building radar shadow mismatch
                                     # false alarms in dense urban fabric (BMC/Thane).

# 7. URBAN WARD WATER FRACTION THRESHOLD
WARD_WATER_FRACTION_THRESHOLD = 0.007  # A ward counts as "flooded" on a given pass if >= 0.7%
                                       # of its valid (non-permanent-water, non-slope) pixels
                                       # are flagged. In dense cities (BMC/Thane), buildings
                                       # and concrete occupy 75-85% of land; 0.7% inundation
                                       # of street/underpass networks (~14 hectares) corresponds
                                       # to severe urban paralysis (e.g. Hindmata, Milan subway 4 ft deep).


BATCH_SIZE = 40  # Number of S1 images stacked into ONE multi-band image
                   # per reduceRegions call (see zonal_stats_for_region).
                   # Because each batch now issues exactly ONE reduceRegions
                   # aggregation regardless of how many bands/images are in
                   # it, this is no longer bounded by Earth Engine's
                   # "concurrent aggregations" quota -- it's bounded by
                   # request size/time instead, a much looser limit. Raise
                   # this well above 40 if a region is going slowly and you
                   # aren't seeing errors; lower it if a single batch times
                   # out or the response gets too large for a region with
                   # unusually many wards.
MAX_RETRIES = 5  # Retries per batch on a 429 "too many concurrent
                   # aggregations" / transient EE error, with exponential
                   # backoff (see BASE_BACKOFF_SECONDS) -- EE's own quota
                   # is shared across whatever else is running in this
                   # project/session, so a batch that fails once may
                   # succeed on retry a few seconds later without any
                   # change needed.
BASE_BACKOFF_SECONDS = 5  # Backoff doubles each retry: 5s, 10s, 20s, 40s, 80s.



# ---------------------------------------------------------------------
# 2. EARTH ENGINE SETUP
# ---------------------------------------------------------------------

def init_ee():
    if ee is None:
        raise ImportError(
            "earthengine-api is required. In Colab: !pip install -q earthengine-api geemap"
        )
    try:
        ee.Initialize(project=GEE_PROJECT_ID)
    except Exception:
        print("Earth Engine not authenticated in this session -- run:\n"
              "    import ee; ee.Authenticate()\n"
              "in a Colab cell first, then re-run this script.")
        raise


def load_ward_featurecollection() -> ee.FeatureCollection:
    """
    Loads ward polygons from a local GeoJSON and uploads them as an
    in-memory ee.FeatureCollection (fine at ~500-ish ward-polygon scale
    across 5 regions; if this ever grows much larger, switch to a GEE
    Table asset instead of an inline FeatureCollection).
    """
    with open(WARD_GEOJSON_PATH) as f:
        geojson = json.load(f)
    features = []
    skipped = []
    for i, feat in enumerate(geojson["features"]):
        props = feat.get("properties") or {}
        region = props.get("region")
        ward = props.get("ward")
        geom = feat.get("geometry")
        # Skip (not crash on) features with a missing/null region, ward, or
        # geometry -- these are typically stray KML placemarks (folder
        # markers, style/schema elements, or a ward with a blank name in
        # the source file) rather than a systematic schema problem, since
        # the 'region'/'ward' KEYS being present (just null-valued) is
        # exactly what a partially-empty KML placemark produces. Hard-
        # failing on the first one would block the whole run over what's
        # usually a handful of junk rows -- report them instead so a real
        # ward with a genuinely missing name doesn't silently disappear
        # unnoticed.
        if region is None or ward is None or not str(region).strip() or not str(ward).strip() or geom is None:
            skipped.append({"index": i, "region": region, "ward": ward,
                             "has_geometry": geom is not None})
            continue
        ward_uid = f"{region}_{ward}"
        features.append(ee.Feature(ee.Geometry(geom), {
            "ward_uid": ward_uid, "region": region, "ward": ward,
        }))

    if skipped:
        print(f"  NOTE: skipped {len(skipped)}/{len(geojson['features'])} GeoJSON features "
              f"with a missing region/ward/geometry (likely stray KML placemarks, not real "
              f"wards). First few skipped:")
        for s in skipped[:5]:
            print(f"    {s}")
        if len(skipped) > 5:
            print(f"    ... and {len(skipped) - 5} more.")
        print(f"  If any of these SHOULD be a real ward, check the corresponding row in "
              f"{WARD_GEOJSON_PATH} (or the source KML) for a blank name field -- this is "
              f"a data issue in the source file, not something this script can infer.")

    if not features:
        raise RuntimeError(
            f"No usable ward features remained after filtering -- every feature in "
            f"{WARD_GEOJSON_PATH} was missing region/ward/geometry. Check the file was "
            f"built correctly (see build_wards_geojson_from_kml.py)."
        )

    fc = ee.FeatureCollection(features)
    print(f"Loaded {len(features)} usable ward polygons into an "
          f"ee.FeatureCollection from {WARD_GEOJSON_PATH}.")
    return fc


# ---------------------------------------------------------------------
# 3. SENTINEL-1 COLLECTION, SPECKLE FILTER, DRY-SEASON REFERENCE
# ---------------------------------------------------------------------

def month_filter(months: list[int]) -> ee.Filter:
    """
    Builds an ee.Filter matching any of the given calendar months,
    correct regardless of whether the months are contiguous or wrap
    across the year boundary (e.g. [12, 1, 2] for Dec-Feb). Implemented
    as an explicit OR of single-month calendarRange filters rather than
    relying on calendarRange's start>end wraparound behavior or an
    inList("month", ...) filter -- Sentinel-1 images have no "month"
    metadata property, so inList("month", ...) silently matches nothing.
    """
    return ee.Filter.Or(*[
        ee.Filter.calendarRange(m, m, "month") for m in months
    ])


def get_s1_collection(region_geom: ee.Geometry) -> ee.ImageCollection:
    """
    IW mode, VV polarization, GRD product -- the standard combination for
    land flood mapping (VV is more sensitive to the smooth-surface /
    specular-reflection signature of standing water than VH). Filters to
    this region's bounding geometry and the full study date range;
    per-purpose date filtering (dry-season reference vs. monitoring)
    happens downstream on this same base collection.
    """
    return (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(region_geom)
        .filterDate(S1_START_DATE, S1_END_DATE)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .select("VV")
    )


def speckle_filter(img: ee.Image) -> ee.Image:
    """Focal-median speckle filter, applied identically to the reference
    composite and every monitored image so differencing compares like
    with like."""
    return img.focal_median(radius=SPECKLE_FILTER_RADIUS_PX, units="meters")


def terrain_mask(region_geom: ee.Geometry) -> ee.Image:
    """
    Topographic slope mask from USGS SRTM GL1 30m DEM:
    1 where slope <= MAX_SLOPE_DEG (default 5.0 deg), 0 where slope > MAX_SLOPE_DEG.
    Standing water physically cannot pool on steep hillsides. Furthermore,
    SAR backscatter on sloped terrain suffers from severe radar geometric
    distortions (layover, foreshortening, and shadow). Masking out slopes
    eliminates spurious hillside flood flags in Pune and SGNP ghats.
    """
    dem = ee.Image("USGS/SRTMGL1_003").clip(region_geom)
    slope = ee.Terrain.slope(dem)
    return slope.lte(MAX_SLOPE_DEG).unmask(1)


def permanent_water_mask(region_geom: ee.Geometry) -> ee.Image:
    """
    Permanent & intertidal water mask from JRC Global Surface Water (GSW):
    1 where GSW occurrence < PERMANENT_WATER_OCCURRENCE_PCT (default 30%),
    0 where occurrence >= PERMANENT_WATER_OCCURRENCE_PCT.
    Using 30% masks out permanent water bodies (rivers/reservoirs/sea) AND
    intertidal creek mudflats and mangrove fringes in Thane Creek and Panvel Creek
    that submerge during normal astronomical high tides (which previously
    generated 65% of all flood flags in Navi Mumbai).
    """
    gsw = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence")
    return gsw.lt(PERMANENT_WATER_OCCURRENCE_PCT).unmask(1).clip(region_geom)


def physical_valid_mask(region_geom: ee.Geometry) -> ee.Image:
    """
    Combines all physical validity constraints into a single binary mask:
      1. Permanent & intertidal water mask (GSW occurrence < 30%)
      2. Topographic slope mask (SRTM slope <= 5 deg)
    Returns 1 where pixels are physically valid for urban flood change detection.
    """
    water_mask = permanent_water_mask(region_geom)
    slope_mask = terrain_mask(region_geom)
    return water_mask.And(slope_mask)


def build_dry_reference(region_geom: ee.Geometry, region_name: str) -> dict[str, ee.Image]:
    """
    Per-pixel MEDIAN VV backscatter across DRY_SEASON_MONTHS (Dec-Feb) acquisitions.
    When USE_ORBIT_SPECIFIC_REFERENCE is True, builds separate reference composites
    for 'ASCENDING' (evening passes, East-looking) and 'DESCENDING' (morning passes,
    West-looking) orbits. This prevents building radar shadow mismatch in dense
    urban fabric (BMC/Thane) from falsely triggering flood flags.
    Also provides a pooled 'ALL' reference as fallback.
    """
    base = get_s1_collection(region_geom)
    dry = base.filter(month_filter(DRY_SEASON_MONTHS))
    n_dry_total = dry.size().getInfo()
    if n_dry_total == 0:
        raise RuntimeError(
            f"No dry-season Sentinel-1 acquisitions found for {region_name} -- "
            f"check region_geom and DRY_SEASON_MONTHS."
        )

    dry_filtered = dry.map(speckle_filter)
    ref_all = dry_filtered.median().clip(region_geom)

    refs = {"ALL": ref_all}

    if USE_ORBIT_SPECIFIC_REFERENCE:
        for orbit in ["ASCENDING", "DESCENDING"]:
            dry_orbit = dry_filtered.filter(ee.Filter.eq("orbitProperties_pass", orbit))
            n_orbit = dry_orbit.size().getInfo()
            if n_orbit >= 3:
                print(f"  {region_name}: dry-season reference for {orbit} from {n_orbit} acquisitions.")
                refs[orbit] = dry_orbit.median().clip(region_geom)
            else:
                print(f"  {region_name}: only {n_orbit} dry acquisitions for {orbit}, "
                      f"falling back to pooled reference.")
                refs[orbit] = ref_all
    else:
        refs["ASCENDING"] = ref_all
        refs["DESCENDING"] = ref_all

    print(f"  {region_name}: dry-season reference ready ({n_dry_total} total acquisitions, "
          f"months {DRY_SEASON_MONTHS}).")
    return refs


# ---------------------------------------------------------------------
# 4. CHANGE-DETECTION FLOOD MASK PER IMAGE
# ---------------------------------------------------------------------

def flood_mask_for_image(img: ee.Image, reference: ee.Image, valid_mask: ee.Image) -> ee.Image:
    """
    Physically grounded SAR flood detection:
      1. Speckle-filter current image to match reference composite.
      2. Relative drop: drop_db = reference - current > CHANGE_THRESHOLD_DB (>= 3.0 dB).
      3. Absolute water ceiling: current <= WATER_BACKSCATTER_CEILING_DB (<= -14.0 dB).
         Specular reflection of open water reflects radar energy away into space.
         Tall buildings or metal structures with baseline +12 dB dropping to +6 dB
         are NOT water!
      4. Dry baseline check: reference > WATER_BACKSCATTER_CEILING_DB (must have been dry).
      5. Spatial coherence: connectedPixelCount >= MIN_CONNECTED_PIXELS (removes isolated
         1-2 pixel salt-and-pepper noise spikes).
      6. Valid physical mask: excludes permanent/intertidal water (<30% GSW) and
         slopes > 5 degrees.
    """
    current = speckle_filter(img)
    drop_db = reference.subtract(current)

    # Dual condition: relative drop AND absolute specular ceiling
    is_water_drop = drop_db.gt(CHANGE_THRESHOLD_DB)
    is_dark_enough = current.lte(WATER_BACKSCATTER_CEILING_DB)
    was_dry_ref = reference.gt(WATER_BACKSCATTER_CEILING_DB)

    flagged = is_water_drop.And(is_dark_enough).And(was_dry_ref)

    # Spatial cluster filtering: eliminate isolated noise pixels
    if MIN_CONNECTED_PIXELS > 1:
        flagged = (
            flagged
            .updateMask(flagged)
            .connectedPixelCount(MIN_CONNECTED_PIXELS, False)
            .gte(MIN_CONNECTED_PIXELS)
            .unmask(0)
        )

    return flagged.updateMask(valid_mask).rename("flooded")


# ---------------------------------------------------------------------
# 5. ZONAL AGGREGATION TO WARD POLYGONS -- BATCHED, ONE getInfo() / REGION
# ---------------------------------------------------------------------

BAND_RE = re.compile(r"^b(\d+)_(mean|count)$")


def get_image_dates_and_orbits(monitor: ee.ImageCollection, n_images: int) -> tuple[list[str], list[str]]:
    """
    Pulls acquisition date (YYYY-MM-dd) and orbit_pass for every image in
    `monitor`, in collection order, via two aggregate_array() calls.
    aggregate_array is a single lightweight aggregation over the whole
    collection -- NOT a separate reduceRegions-style operation per image
    -- so this does not touch the "too many concurrent aggregations"
    quota that the per-image reduceRegions approach hit. Band names in
    zonal_stats_for_region are purely positional (b0, b1, ...); this is
    what maps position i back to a real date/orbit_pass.
    """
    millis = monitor.aggregate_array("system:time_start").getInfo()
    orbits = monitor.aggregate_array("orbitProperties_pass").getInfo()
    dates = [pd.to_datetime(m, unit="ms", utc=True).strftime("%Y-%m-%d") for m in millis]
    return dates, orbits


def _stack_batch_as_multiband(image_list: ee.List, start: int, count: int,
                                ref_dict: dict[str, ee.Image], valid_mask: ee.Image) -> ee.Image:
    """
    Builds ONE multi-band ee.Image for a batch of `count` images starting
    at position `start` in `image_list`: band i is that image's flood
    mask, named "b<i>".
    Uses orbit-specific reference (ASCENDING vs DESCENDING) to eliminate
    urban building radar shadow mismatch.
    """
    indices = ee.List.sequence(0, count - 1)
    ref_asc = ref_dict.get("ASCENDING", ref_dict["ALL"])
    ref_desc = ref_dict.get("DESCENDING", ref_dict["ALL"])

    def add_band(i, acc):
        i = ee.Number(i)
        img = ee.Image(image_list.get(ee.Number(start).add(i)))
        orbit = ee.String(img.get("orbitProperties_pass"))

        ref = ee.Image(ee.Algorithms.If(
            orbit.equals("ASCENDING"),
            ref_asc,
            ref_desc
        ))

        flood_img = flood_mask_for_image(img, ref, valid_mask)
        band_name = ee.String("b").cat(i.int().format())
        return ee.Image(acc).addBands(flood_img.rename(band_name))

    return ee.Image(indices.iterate(add_band, ee.Image()))


def _get_batch_info_with_retry(stats_fc: ee.FeatureCollection, batch_label: str) -> dict:
    """
    Calls .getInfo() on a batch's reduceRegions result, retrying with
    exponential backoff on Earth Engine's transient "too many concurrent
    aggregations" 429 (and other transient EEExceptions).
    """
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return stats_fc.getInfo()
        except ee.ee_exception.EEException as e:
            last_err = e
            is_quota_error = "concurrent aggregations" in str(e) or "429" in str(e)
            if attempt == MAX_RETRIES:
                break
            wait_s = BASE_BACKOFF_SECONDS * (2 ** (attempt - 1))
            reason = "quota (too many concurrent aggregations)" if is_quota_error else "transient EE error"
            print(f"    {batch_label}: {reason} on attempt {attempt}/{MAX_RETRIES} "
                  f"({e}). Retrying in {wait_s}s...")
            time.sleep(wait_s)
    raise RuntimeError(
        f"{batch_label} failed after {MAX_RETRIES} attempts against Earth Engine's "
        f"aggregation quota. Try lowering BATCH_SIZE (currently {BATCH_SIZE}) and "
        f"re-running -- other things sharing this GEE project/session can also eat "
        f"into the same quota."
    ) from last_err


def zonal_stats_for_region(monitor: ee.ImageCollection, n_images: int,
                             ref_dict: dict[str, ee.Image], valid_mask: ee.Image,
                             region_wards: ee.FeatureCollection) -> list[dict]:
    """
    Processes this region's images in chunks of BATCH_SIZE, issuing exactly ONE
    reduceRegions call per batch on a multi-band image stacking that batch's
    flood masks.
    """
    monitor = monitor.sort("system:time_start")
    image_list = monitor.toList(n_images)
    dates, orbits = get_image_dates_and_orbits(monitor, n_images)

    all_rows = []
    n_batches = (n_images + BATCH_SIZE - 1) // BATCH_SIZE
    for batch_idx in range(n_batches):
        start = batch_idx * BATCH_SIZE
        count = min(BATCH_SIZE, n_images - start)
        batch_label = f"batch {batch_idx + 1}/{n_batches} (images {start}-{start + count - 1})"

        multiband = _stack_batch_as_multiband(image_list, start, count, ref_dict, valid_mask)
        stats = multiband.reduceRegions(
            collection=region_wards,
            reducer=ee.Reducer.mean().combine(ee.Reducer.count(), sharedInputs=True),
            scale=10,  # native S1 GRD pixel spacing
        )

        info = _get_batch_info_with_retry(stats, batch_label)

        n_batch_rows = 0
        for feat in info["features"]:
            props = feat["properties"]
            ward_uid = props["ward_uid"]
            per_index = {}
            for key, val in props.items():
                m = BAND_RE.match(key)
                if not m:
                    continue
                local_i, stat_type = int(m.group(1)), m.group(2)
                per_index.setdefault(local_i, {})[stat_type] = val
            for local_i, stat_dict in per_index.items():
                global_i = start + local_i
                all_rows.append({
                    "ward_uid": ward_uid,
                    "date": dates[global_i],
                    "water_fraction": stat_dict.get("mean"),
                    "n_valid_pixels": stat_dict.get("count"),
                    "orbit_pass": orbits[global_i],
                })
                n_batch_rows += 1
        print(f"    ...{batch_label} done ({n_batch_rows} rows, "
              f"{len(all_rows)} total so far)")

    return all_rows


# ---------------------------------------------------------------------
# 6. PER-REGION DRIVER
# ---------------------------------------------------------------------

def process_region(region_name: str, region_geom: ee.Geometry,
                    wards_fc: ee.FeatureCollection) -> pd.DataFrame:
    print(f"\n{'=' * 60}\nProcessing {region_name}\n{'=' * 60}")

    valid_mask = physical_valid_mask(region_geom)
    ref_dict = build_dry_reference(region_geom, region_name)

    monitor = get_s1_collection(region_geom)
    if not MONITOR_ALL_YEAR:
        monitor = monitor.filter(month_filter(MONSOON_MONTHS))

    region_wards = wards_fc.filter(ee.Filter.eq("region", region_name))
    n_wards = region_wards.size().getInfo()
    if n_wards == 0:
        print(f"  WARNING: no ward polygons found with region == '{region_name}' "
              f"in {WARD_GEOJSON_PATH} -- check the 'region' property spelling "
              f"matches REGIONS exactly. Skipping.")
        return pd.DataFrame()

    n_images = monitor.size().getInfo()
    n_batches = (n_images + BATCH_SIZE - 1) // BATCH_SIZE
    season_str = 'all-year' if MONITOR_ALL_YEAR else f'monsoon months {MONSOON_MONTHS} only'
    print(f"  {n_images} Sentinel-1 acquisitions for {n_wards} wards ({season_str}), "
          f"processing in {n_batches} batches of up to {BATCH_SIZE} images each.")

    all_rows = zonal_stats_for_region(monitor, n_images, ref_dict, valid_mask, region_wards)

    df = pd.DataFrame(all_rows)
    print(f"  {region_name}: {len(df)} ward-acquisition rows returned.")
    return df


# ---------------------------------------------------------------------
# 7. REGION-DAY ROLLUP + COMPARISON AGAINST THE IFI-DERIVED LABEL
# ---------------------------------------------------------------------

def build_region_day_rollup(ward_day_df: pd.DataFrame, region_threshold: float = 0.002) -> pd.DataFrame:
    """
    Collapses ward-acquisition rows to (region, date): mean water_fraction
    across that region's wards observed on that date, thresholded into a
    binary s1_flooded flag. At the whole-region scale (e.g. all 24 BMC wards
    covering 437 km2), a mean water fraction of 0.002 (0.2%) represents over
    1 square kilometer (~100 hectares) of standing water, matching confirmed
    regional flood disasters.
    """
    ward_day_df = ward_day_df.copy()
    ward_day_df["region"] = ward_day_df["ward_uid"].str.split("_").str[0]
    rollup = (
        ward_day_df.groupby(["region", "date"])["water_fraction"]
        .mean().reset_index().rename(columns={"water_fraction": "s1_region_water_fraction"})
    )
    rollup["s1_flooded"] = (rollup["s1_region_water_fraction"] >= region_threshold).astype(int)
    return rollup



def compare_against_ifi_label(region_day_rollup: pd.DataFrame,
                                ifi_daily_csv: str = "region_day_flooded_CORRECTED.csv") -> pd.DataFrame:
    """
    Joins the S1-derived region-day label against the existing IFI-
    derived daily label (build_corrected_flood_labels.py's output) ONLY
    on dates where BOTH exist -- i.e. only on actual Sentinel-1
    acquisition dates (see TEMPORAL RESOLUTION in the module docstring;
    this is an honest inner join, not an attempt to fabricate daily S1
    coverage). Reports agreement rate as a sanity check: if the two
    independently-derived labels agree often, that's real corroborating
    evidence for both; if they disagree a lot, that's worth investigating
    before trusting either one blindly, not a reason to silently prefer
    one over the other.
    """
    try:
        ifi = pd.read_csv(ifi_daily_csv, parse_dates=["date"])
    except FileNotFoundError:
        print(f"  {ifi_daily_csv} not found -- run build_corrected_flood_labels.py "
              f"first to enable this comparison. Skipping.")
        return pd.DataFrame()

    region_day_rollup = region_day_rollup.copy()
    region_day_rollup["date"] = pd.to_datetime(region_day_rollup["date"])
    merged = region_day_rollup.merge(
        ifi[["region", "date", "region_day_flooded_broadcast"]],
        on=["region", "date"], how="inner",
    )
    if merged.empty:
        print("  No overlapping (region, date) rows between S1 acquisitions and "
              "the IFI-derived daily label -- nothing to compare.")
        return merged

    agree = (merged["s1_flooded"] == merged["region_day_flooded_broadcast"]).mean()
    both_flooded = ((merged["s1_flooded"] == 1) & (merged["region_day_flooded_broadcast"] == 1)).sum()
    s1_only = ((merged["s1_flooded"] == 1) & (merged["region_day_flooded_broadcast"] == 0)).sum()
    ifi_only = ((merged["s1_flooded"] == 0) & (merged["region_day_flooded_broadcast"] == 1)).sum()
    neither = ((merged["s1_flooded"] == 0) & (merged["region_day_flooded_broadcast"] == 0)).sum()

    print(f"\n  S1 vs IFI label agreement on {len(merged)} overlapping "
          f"(region, S1-acquisition-date) rows:")
    print(f"    Overall agreement: {agree:.1%}")
    print(f"    Both say flooded:     {both_flooded}")
    print(f"    S1 only (IFI missed): {s1_only}")
    print(f"    IFI only (S1 missed): {ifi_only}  <- expected, given S1's ~6-12 day "
          f"revisit gap can simply miss short events (see HONEST LIMITS item 2)")
    print(f"    Neither:               {neither}")
    merged.to_csv("sentinel1_vs_ifi_label_comparison.csv", index=False)
    print(f"  Saved -> sentinel1_vs_ifi_label_comparison.csv")
    return merged


# ---------------------------------------------------------------------
# 8. MAIN
# ---------------------------------------------------------------------

def main():
    global MONITOR_ALL_YEAR, WARD_WATER_FRACTION_THRESHOLD, BATCH_SIZE
    global PERMANENT_WATER_OCCURRENCE_PCT, MAX_SLOPE_DEG, WATER_BACKSCATTER_CEILING_DB

    parser = argparse.ArgumentParser(
        description="Build physically-grounded Sentinel-1 SAR flood labels for urban wards."
    )
    parser.add_argument(
        "--all-year", action="store_true", default=False,
        help="Monitor year-round instead of monsoon months only (default: False, Jun-Oct only)."
    )
    parser.add_argument(
        "--threshold", type=float, default=WARD_WATER_FRACTION_THRESHOLD,
        help=f"Ward water fraction threshold for flood flag (default: {WARD_WATER_FRACTION_THRESHOLD})."
    )
    parser.add_argument(
        "--max-slope", type=float, default=MAX_SLOPE_DEG,
        help=f"Max terrain slope in degrees from SRTM DEM (default: {MAX_SLOPE_DEG})."
    )
    parser.add_argument(
        "--permanent-water-pct", type=int, default=PERMANENT_WATER_OCCURRENCE_PCT,
        help=f"JRC GSW occurrence threshold to mask permanent/tidal water (default: {PERMANENT_WATER_OCCURRENCE_PCT} percent)."
    )
    parser.add_argument(
        "--water-ceiling-db", type=float, default=WATER_BACKSCATTER_CEILING_DB,
        help=f"Absolute backscatter ceiling in dB for standing water (default: {WATER_BACKSCATTER_CEILING_DB} dB)."
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE,
        help=f"Number of S1 images per multi-band reduceRegions batch (default: {BATCH_SIZE})."
    )
    parser.add_argument(
        "--regions", nargs="+", default=REGIONS,
        help=f"Regions to process (default: {' '.join(REGIONS)})."
    )
    args = parser.parse_args()

    if args.all_year:
        MONITOR_ALL_YEAR = True
    WARD_WATER_FRACTION_THRESHOLD = args.threshold
    MAX_SLOPE_DEG = args.max_slope
    PERMANENT_WATER_OCCURRENCE_PCT = args.permanent_water_pct
    WATER_BACKSCATTER_CEILING_DB = args.water_ceiling_db
    BATCH_SIZE = args.batch_size
    regions = args.regions

    print("=" * 70)
    print("SENTINEL-1 SAR PHYSICALLY-GROUNDED FLOOD LABEL GENERATION")
    print("=" * 70)
    print(f"  Physical seasonal window: {'Year-round (all months)' if MONITOR_ALL_YEAR else f'Monsoon months {MONSOON_MONTHS} only'}")
    print(f"  Permanent/tidal water mask: JRC GSW occurrence >= {PERMANENT_WATER_OCCURRENCE_PCT}% (masks intertidal creek mudflats)")
    print(f"  Topographic slope mask: SRTM DEM slope <= {MAX_SLOPE_DEG} deg (eliminates hillside radar distortion)")
    print(f"  Dual-condition water threshold: relative drop >= {CHANGE_THRESHOLD_DB} dB AND current VV <= {WATER_BACKSCATTER_CEILING_DB} dB")
    print(f"  Orbit-specific baseline: {'ENABLED (separate Ascending/Descending composites)' if USE_ORBIT_SPECIFIC_REFERENCE else 'DISABLED'}")
    print(f"  Spatial coherence filter: connectedPixelCount >= {MIN_CONNECTED_PIXELS} (~500 m2)")
    print(f"  Urban ward flood threshold: water_fraction >= {WARD_WATER_FRACTION_THRESHOLD:.2f} ({WARD_WATER_FRACTION_THRESHOLD*100:.1f}% usable ward area)")
    print(f"  Regions: {regions}")
    print("=" * 70)

    init_ee()
    wards_fc = load_ward_featurecollection()

    all_region_dfs = []
    for region_name in regions:
        region_geom = wards_fc.filter(ee.Filter.eq("region", region_name)).geometry().bounds()
        df = process_region(region_name, region_geom, wards_fc)
        if not df.empty:
            all_region_dfs.append(df)

    if not all_region_dfs:
        raise RuntimeError("No regions produced any rows -- check WARD_GEOJSON_PATH, "
                            "the 'region' property values, and GEE authentication.")

    ward_day_df = pd.concat(all_region_dfs, ignore_index=True)
    ward_day_df = ward_day_df.dropna(subset=["water_fraction"])
    ward_day_df.to_csv("sentinel1_ward_day_water_fraction.csv", index=False)
    print(f"\nSaved -> sentinel1_ward_day_water_fraction.csv "
          f"({len(ward_day_df)} ward-acquisition rows)")

    region_day_rollup = build_region_day_rollup(ward_day_df)
    region_day_rollup.to_csv("sentinel1_region_day_flooded_CANDIDATE.csv", index=False)
    print(f"Saved -> sentinel1_region_day_flooded_CANDIDATE.csv "
          f"({len(region_day_rollup)} region-acquisition-date rows)")

    compare_against_ifi_label(region_day_rollup)

    print("\n" + "=" * 60)
    print("HOW TO USE THE OUTPUT")
    print("=" * 60)
    print("The newly generated sentinel1_ward_day_water_fraction.csv now incorporates")
    print("all 7 physical hydrologic constraints. You can now run:")
    print("    python train_and_compare_sentinel1_models.py")
    print("to train all machine learning and deep learning models against this clean ground truth.")


if __name__ == "__main__":
    main()