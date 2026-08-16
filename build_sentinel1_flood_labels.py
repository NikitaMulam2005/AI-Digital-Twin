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

METHOD -- CHANGE DETECTION, NOT A FIXED ABSOLUTE THRESHOLD
A fixed "VV < -17 dB means water" rule is a known-fragile approach: the
right cutoff shifts with incidence angle, surface roughness, and land
cover, and differs between BMC's dense urban fabric and Pune's more
open terrain. This script instead uses PER-PIXEL, PER-REGION CHANGE
DETECTION against each region's own dry-season reference backscatter:
  1. For each region, build a DRY-SEASON REFERENCE composite: the
     per-pixel MEDIAN VV backscatter across all available acquisitions
     in DRY_SEASON_MONTHS (Dec-Feb, well outside monsoon), computed
     separately per region, pooled across all years in RESULT_YEAR_MIN
     to now (median is robust to the occasional stray wet/anomalous
     scene in a nominally dry month). This is each pixel's own "normal,
     dry" baseline -- not a fixed dB number shared across regions.
  2. For every monsoon-season (or any-season, see MONITOR_ALL_YEAR)
     acquisition, compute pixel-wise VV_DROP_DB = reference - current.
     Standing water causes specular reflection away from the sensor,
     so flooding shows as a LARGE POSITIVE drop from the dry baseline
     -- this is far more robust to absolute calibration differences
     than thresholding raw backscatter.
  3. Flag a pixel as flooded if VV_DROP_DB > CHANGE_THRESHOLD_DB
     (default 3 dB, a standard, conservative literature value for
     SAR flood change-detection -- see comment at CHANGE_THRESHOLD_DB).
  4. A speckle filter (focal median, SPECKLE_FILTER_RADIUS_PX) is
     applied to both the reference composite and every current image
     BEFORE differencing -- raw SAR is inherently noisy pixel-to-pixel
     ("speckle"), and differencing two noisy images without filtering
     first would flag isolated noise spikes as false floods.
  5. Permanent water bodies (rivers, reservoirs, the coastline) are
     masked out using JRC Global Surface Water (occurrence > 80%) --
     otherwise every reference-vs-current comparison over a river
     would be a false "wet" reading because the river is ALREADY water
     in the dry-season reference too, not because it just flooded, and
     any acquisition noise around its banks would misfire either way.
  6. Zonal mean of the flooded-pixel mask over each ward polygon gives
     that ward's water_fraction for that acquisition date.

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

import json
import re
import time
import numpy as np
import pandas as pd

try:
    import ee
except ImportError as e:
    raise ImportError(
        "earthengine-api is required. In Colab: !pip install -q earthengine-api geemap"
    ) from e


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

MONITOR_ALL_YEAR = True  # If True, flood detection runs on every S1 pass
                           # year-round (lets the pipeline also catch
                           # unseasonal/pre-monsoon flooding, and gives a
                           # true daily-analogue comparison against the IFI
                           # label which is not monsoon-restricted either).
                           # If False, restricts detection to MONSOON_MONTHS
                           # only, which would be faster but silently
                           # assumes IFI's out-of-season events don't exist.
MONSOON_MONTHS = [6, 7, 8, 9, 10]

SPECKLE_FILTER_RADIUS_PX = 50  # ~50m at S1's native ~10m pixel spacing;
                                 # matches common literature practice for
                                 # Lee/focal-median speckle filtering
                                 # without over-smoothing ward-scale detail.

CHANGE_THRESHOLD_DB = 3.0  # Standard, conservative SAR flood-change-
                             # detection literature value (a >=3 dB drop
                             # from a pixel's own dry baseline is a widely
                             # used flood-flag cutoff, e.g. UN-SPIDER's
                             # published SAR flood mapping recipe). See
                             # HONEST LIMITS item 3 -- not locally tuned.

PERMANENT_WATER_OCCURRENCE_PCT = 80  # JRC Global Surface Water "occurrence"
                                       # band >= this % is treated as
                                       # permanent water and masked out of
                                       # both the reference and every
                                       # current image before differencing.

WARD_WATER_FRACTION_THRESHOLD = 0.10  # A ward counts as "flooded" on a
                                        # given pass if >=10% of its valid
                                        # SAR pixels are flagged -- avoids
                                        # a single noisy pixel at a ward's
                                        # edge (post speckle-filter, still
                                        # not perfectly zero) flipping the
                                        # whole ward to "flooded". 10% is a
                                        # deliberately conservative floor;
                                        # adjust based on validation against
                                        # the IFI label once run once.

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


def permanent_water_mask(region_geom: ee.Geometry) -> ee.Image:
    """1 where NOT permanent water (safe to evaluate for flood change),
    0 where JRC Global Surface Water occurrence exceeds the configured
    threshold -- masks out rivers/reservoirs/coastline so they don't
    register as false "newly flooded" pixels every single pass."""
    gsw = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence")
    return gsw.lt(PERMANENT_WATER_OCCURRENCE_PCT).unmask(1).clip(region_geom)


def build_dry_reference(region_geom: ee.Geometry, region_name: str) -> ee.Image:
    """
    Per-pixel MEDIAN VV backscatter across every DRY_SEASON_MONTHS
    acquisition for this region, pooled across all years -- this
    region's own "what does dry ground/normal urban fabric look like"
    baseline. Median (not mean) so an occasional anomalous dry-month
    scene doesn't drag the baseline.

    NOTE: this remains a lazy ee.Image (not materialized to a client-
    side array or exported Asset). That's fine as long as it's only
    evaluated ONCE per region -- which is what process_region() now
    guarantees by folding every per-image reduceRegions call for this
    region into a single server-side graph / single getInfo(). Do not
    reintroduce a per-image Python-loop getInfo() around anything that
    depends on `reference`, or you'll reintroduce the original slowdown
    (Earth Engine will re-derive this 68-image median from scratch on
    every separate request).
    """
    base = get_s1_collection(region_geom)
    dry = base.filter(month_filter(DRY_SEASON_MONTHS))
    n_dry = dry.size().getInfo()
    if n_dry == 0:
        raise RuntimeError(
            f"No dry-season Sentinel-1 acquisitions found for {region_name} -- "
            f"check region_geom and DRY_SEASON_MONTHS."
        )
    print(f"  {region_name}: building dry-season reference from {n_dry} acquisitions "
          f"(months {DRY_SEASON_MONTHS}).")
    reference = dry.map(speckle_filter).median()
    return reference.clip(region_geom)


# ---------------------------------------------------------------------
# 4. CHANGE-DETECTION FLOOD MASK PER IMAGE
# ---------------------------------------------------------------------

def flood_mask_for_image(img: ee.Image, reference: ee.Image, water_mask: ee.Image) -> ee.Image:
    """
    flood_flag = 1 where (reference_dB - current_dB) > CHANGE_THRESHOLD_DB
    AND the pixel is not permanent water. Water causes a large POSITIVE
    drop from the dry baseline (specular reflection away from sensor),
    so this is `reference.subtract(current) > threshold`, not the other
    way around.
    """
    current = speckle_filter(img)
    drop_db = reference.subtract(current)
    flagged = drop_db.gt(CHANGE_THRESHOLD_DB)
    return flagged.updateMask(water_mask).rename("flooded")


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
                                reference: ee.Image, water_mask: ee.Image) -> ee.Image:
    """
    Builds ONE multi-band ee.Image for a batch of `count` images starting
    at position `start` in `image_list`: band i is that image's flood
    mask, named "b<i>" (i local to the batch, 0..count-1). Stacked via
    repeated addBands() inside ee.List.iterate() rather than
    ImageCollection.toBands(), so band names are exactly "b0", "b1", ...
    with no extra prefixing to parse back out.

    The point of stacking into bands (rather than mapping reduceRegions
    across `count` separate images) is that reduceRegions run once on a
    multi-band image is ONE aggregation operation no matter how many
    bands it has -- this is what avoids Earth Engine's "too many
    concurrent aggregations" quota, which is keyed on the number of
    aggregation OPERATIONS in a request, not on total pixels/images
    touched.
    """
    indices = ee.List.sequence(0, count - 1)

    def add_band(i, acc):
        i = ee.Number(i)
        img = ee.Image(image_list.get(ee.Number(start).add(i)))
        flood_img = flood_mask_for_image(img, reference, water_mask)
        band_name = ee.String("b").cat(i.int().format())
        return ee.Image(acc).addBands(flood_img.rename(band_name))

    # NOTE: the starting accumulator must be ee.Image() with NO arguments
    # (the documented way to get a genuine zero-band image). ee.Image([])
    # looks similar but is parsed client-side as "combine this list of
    # images" -- passing an empty list there raises "Cannot combine 0
    # images" immediately, before anything even reaches the server.
    return ee.Image(indices.iterate(add_band, ee.Image()))


def _get_batch_info_with_retry(stats_fc: ee.FeatureCollection, batch_label: str) -> dict:
    """
    Calls .getInfo() on a batch's reduceRegions result, retrying with
    exponential backoff on Earth Engine's transient "too many concurrent
    aggregations" 429 (and other transient EEExceptions) -- these are a
    shared-quota issue, not a problem with the request itself, so a retry
    a few seconds later commonly succeeds without any change. With the
    multi-band-per-batch approach each batch is only ONE aggregation, so
    this should rarely fire now, but is kept as a safety net for whatever
    other transient EE errors (server load, timeouts) can still occur.
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
                             reference: ee.Image, water_mask: ee.Image,
                             region_wards: ee.FeatureCollection) -> list[dict]:
    """
    Processes this region's images in chunks of BATCH_SIZE, but unlike
    the earlier .map()-based version, each batch issues exactly ONE
    reduceRegions call (on a multi-band image stacking that batch's
    flood masks) instead of BATCH_SIZE separate reduceRegions calls
    fanned out concurrently -- see _stack_batch_as_multiband's docstring
    for why that's what actually avoids the "too many concurrent
    aggregations" 429 (batch size alone does not, as long as each image
    still gets its own reduceRegions call within the batch).

    `monitor` is sorted by acquisition time first so that positional
    band indices (b0, b1, ...) line up deterministically with the
    date/orbit_pass lists pulled separately in get_image_dates_and_orbits.
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

        multiband = _stack_batch_as_multiband(image_list, start, count, reference, water_mask)
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
            # Each ward-feature has interleaved "b<i>_mean"/"b<i>_count"
            # properties, one pair per band/image in this batch. Group
            # them back into per-image rows keyed by local index i, then
            # translate i -> the batch's global image index to look up
            # the real date/orbit_pass.
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

    water_mask = permanent_water_mask(region_geom)
    reference = build_dry_reference(region_geom, region_name)

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
    print(f"  {n_images} Sentinel-1 acquisitions for {n_wards} wards "
          f"({'all-year' if MONITOR_ALL_YEAR else f'months {MONSOON_MONTHS} only'}), "
          f"processing in {n_batches} batches of up to {BATCH_SIZE} images each.")

    all_rows = zonal_stats_for_region(monitor, n_images, reference, water_mask, region_wards)

    df = pd.DataFrame(all_rows)
    print(f"  {region_name}: {len(df)} ward-acquisition rows returned.")
    return df


# ---------------------------------------------------------------------
# 7. REGION-DAY ROLLUP + COMPARISON AGAINST THE IFI-DERIVED LABEL
# ---------------------------------------------------------------------

def build_region_day_rollup(ward_day_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapses ward-acquisition rows to (region, date): mean water_fraction
    across that region's wards observed on that date, thresholded into a
    binary s1_flooded flag with WARD_WATER_FRACTION_THRESHOLD applied at
    the REGION-mean level too (consistent single rule, rather than a
    separate ward-count-based rule at this grain).
    """
    ward_day_df = ward_day_df.copy()
    ward_day_df["region"] = ward_day_df["ward_uid"].str.split("_").str[0]
    rollup = (
        ward_day_df.groupby(["region", "date"])["water_fraction"]
        .mean().reset_index().rename(columns={"water_fraction": "s1_region_water_fraction"})
    )
    rollup["s1_flooded"] = (rollup["s1_region_water_fraction"] >= WARD_WATER_FRACTION_THRESHOLD).astype(int)
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
    init_ee()
    wards_fc = load_ward_featurecollection()

    all_region_dfs = []
    for region_name in REGIONS:
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
    print("This does NOT automatically feed into train_and_compare_models.py.")
    print("Two reasonable next steps, depending on what the agreement-rate")
    print("comparison above shows:")
    print("  1. VALIDATION ONLY (safest first step): keep using the existing")
    print("     IFI-derived CORRECTED_DAILY_CSV / region_flood_days_count_CORRECTED.csv")
    print("     unchanged, and cite the agreement rate as evidence for/against")
    print("     trusting the IFI label in the report.")
    print("  2. WARD-RESOLUTION LABEL REFINEMENT (bigger change, 2014+ only):")
    print("     for ward-year rows with year >= 2015, replace the region-broadcast")
    print("     flood label with an actual per-ward one derived from")
    print("     sentinel1_ward_day_water_fraction.csv (e.g. ward-year flood-day")
    print("     count = number of distinct S1 acquisition dates that year where")
    print("     that ward's water_fraction >= WARD_WATER_FRACTION_THRESHOLD).")
    print("     This directly fixes the 'every ward in a region shares one label'")
    print("     limitation -- but changes the target definition, so it should be")
    print("     run as a clearly-labeled separate comparison, not a silent swap-in,")
    print("     given years < 2015 would still be on the old region-broadcast label.")


if __name__ == "__main__":
    main()