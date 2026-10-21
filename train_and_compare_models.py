"""
Train and Compare All Four Flood Models -- LR, RF, LSTM, LSTM+GNN
--------------------------------------------------------------------------
Answers the actual question from build_final_training_dataset.py's NEXT
STEPS: train all four models and compare them with real numbers, not
just build their input tables.

RUNNING THIS IN COLAB
    !git clone https://github.com/NikitaMulam2005/AI-Digital-Twin.git
    %cd AI-Digital-Twin
    !pip install -q scikit-learn torch matplotlib imbalanced-learn

Upload build_corrected_flood_labels.py and this script into that folder,
then run:
    !python build_corrected_flood_labels.py   # one-time, builds the two
                                                # corrected label CSVs below
    !python train_and_compare_models.py

WHAT CHANGED IN THIS VERSION, AND WHY (read this before trusting numbers)

1. THE 2020 FIX IS NOW EVIDENCE-BASED, NOT AN ARBITRARY CAP.
   Earlier versions clipped 2020's region_flood_days_count to "20,
   because that's just under 2005" -- a number picked to match a desired
   outcome, with no evidence behind it. Here's what's actually wrong,
   with evidence: region_flood_days_count and the daily flood label are
   both built by expanding each IFI event's [Start Date, End Date] into
   one row per day, per matched region. Across all 346 IFI events that
   match our 5 regions (1990-2024), 95% last <=7 days and 98% last <=14
   days -- then there's a clean gap before 6 outlier events (32-93 days).
   THREE of those six -- 93, 62, and 62 days long -- fall in 2020 alone
   and contribute 651 region-days between them, over 8x any other single
   year in the 55-year file. A "flood event" spanning April-to-June or
   May-to-August isn't a real continuous flood; it reads like a
   seasonal/monsoon summary logged with event-style start/end dates.
   FIX: build_corrected_flood_labels.py caps every individual event's
   duration at 14 days (the natural break in the real distribution,
   not a number chosen to hit a target) BEFORE expanding to daily rows.
   This is applied uniformly to every event, every year, every region --
   ordinary short events (the vast majority) are completely untouched.
   Corrected 2020 flood-days: BMC 16, KDMC 29, NMMC 29, Pune 50, Thane 29
   -- still the worst year on record (as it plausibly should be), just
   not physically-impossible-length events.

2. FEATURES ARE NOW SELECTED BY EVIDENCE, NOT THROWN IN WHOLESALE.
   The true independent sample size here is ~90 TRAINING region-years
   (5 regions x ~18 years) -- ward-year rows duplicate this ~60x because
   every ward in a region shares the same annual target. Fitting on all
   92 numeric features with 90 real rows is overfitting almost by
   definition. select_features_by_within_region_signal() computes each
   feature's correlation with the target AFTER subtracting each region's
   own mean from both (i.e. "does this feature moving up/down WITHIN a
   region in a given year predict flood-days moving up/down that year,
   beyond what the region's own average already tells you") -- computed
   on TRAINING data only, no test leakage -- and keeps only features
   clearing a threshold. Empirically this took region-year Ridge NSE
   from -0.38 (92 features) to roughly baseline parity (12 features) on
   a held-out check -- see chat.

3. A NAIVE "PREDICT THIS REGION'S OWN TRAINING-PERIOD MEAN" BASELINE IS
   NOW A PERMANENT FIFTH ROW IN EVERY COMPARISON TABLE.
   Every one of LR/RF/LSTM/LSTM+GNN underperformed this trivial baseline
   in earlier runs (see chat) -- often badly. Rather than requiring
   remembering to compute this by hand each time, it's now built in, so
   no future run's numbers can look good only because nothing is being
   compared against a sane floor.

4. TRUE region-year GRAIN IS THE HEADLINE METRIC FOR ALL FOUR MODELS,
   ward-level (inflated N) is kept only as a labeled "before" reference.
   See compute_regionyear_metrics for the full reasoning.

5. TOPOGRAPHIC WETNESS INDEX (TWI) ADDED AS A STATIC WARD FEATURE.
   Per the report's Section 5 diagnosis: every ward within a region
   shares one identical flood label, so ward-level models can only ever
   differentiate wards via STATIC features -- and the static set so far
   was limited to elevation_m_mean and slope_deg_mean, which don't
   capture where water actually accumulates (a low, steep cell drains
   fast; a flat cell downhill of a large catchment does not, even at
   the same elevation). TWI = ln(upslope contributing area / tan(slope))
   is a standard hydrological index that does capture this, computed
   from the existing SRTM DEM via D8 flow accumulation (see
   compute_twi.py / all_regions_ward_twi_zonal_stats.csv). Added as
   twi_mean and twi_p90 (p90 flags wards with a concentrated low-lying
   pocket even when their average TWI looks unremarkable) to
   STATIC_FEATURE_COLS for the daily/LSTM/LSTM+GNN path, and merged
   into the region-year table for LR/RF, where it goes through the
   same within-region evidence filter as every other candidate feature
   (i.e. it's a candidate, not forced in -- see item 2 above). This is
   a genuinely new source of ward-level differentiation, not a
   relabeling of existing terrain columns -- but it does NOT address
   the annual-grain small-sample-size or 2020-outlier ceilings (items
   1 and the "Small true sample" point in Section 5); those need more
   years of data, not more features.

THE FAIRNESS PROBLEM THIS SCRIPT STILL SOLVES (unchanged from earlier)
LR/RF natively predict an ANNUAL quantity; LSTM/LSTM+GNN natively
predict a DAILY probability. All four are evaluated against the same
common target -- the (now corrected) region_flood_days_count -- by
summing LSTM/LSTM+GNN's predicted daily probabilities per ward-year.

WHAT LSTM+GNN ACTUALLY DOES (unchanged, verified correct -- see chat)
Every training/eval batch gathers ALL wards' raw dynamic features for
the exact days in that batch's windows, mixes them through the
normalized adjacency matrix (torch.einsum('bn,blnd->bld', ...) --
exact, not an approximation), then applies a learned linear+ReLU before
the shared per-ward LSTM. Verified against a naive re-implementation.

HONEST LIMITS, STILL TRUE
  1. QUICK_TEST_MODE subsamples wards/years/epochs for a fast pipeline
     validation run -- set False on real hardware for real numbers.
  2. The daily/annual flood label is still region-broadcast (IFI's real
     ceiling is district, not ward) -- every ward in a region shares one
     label. This is now less noisy (event-duration capping helps) but
     the fundamental region-broadcast limitation is unchanged.
  3. Even after all of the above, don't expect NSE > 0 to suddenly
     appear everywhere -- the point of this version is an HONEST
     comparison against a real floor, not a guaranteed win.

REQUIREMENTS
    pip install pandas numpy scikit-learn torch matplotlib
"""

import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              confusion_matrix, precision_recall_curve, average_precision_score)
from sklearn.utils.class_weight import compute_sample_weight
try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    SMOTE = None


from build_ward_day_dataset import build_ward_day_table, add_sequence_features

# ---------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------

FINAL_WARD_YEAR_TRAIN_CSV = "final_ward_year_train.csv"
FINAL_WARD_YEAR_TEST_CSV = "final_ward_year_test.csv"
ADJACENCY_EDGES_CSV = "ward_adjacency_edges.csv"
CORRECTED_ANNUAL_CSV = "region_flood_days_count_CORRECTED.csv"   # from build_corrected_flood_labels.py
AUTOCORRELATION_FEATURES_CSV = "region_year_autocorrelation_features.csv"  # from build_autocorrelation_features.py
CORRECTED_DAILY_CSV = "region_day_flooded_CORRECTED.csv"          # from build_corrected_flood_labels.py
CORRECTED_TIDE_CSV = "tide_features_daily_CORRECTED.csv"  # UNUSED -- tide was tested and
                                                             # removed from the daily pipeline
                                                             # (made LSTM/LSTM+GNN worse, see chat).
                                                             # Kept here only as a reminder not to
                                                             # re-add the loading code without also
                                                             # re-adding tide to DYNAMIC_FEATURE_COLS.
WARD_TWI_CSV = "all_regions_ward_twi_zonal_stats.csv"      # from compute_twi.py -- D8 flow-
                                                             # accumulation-derived Topographic
                                                             # Wetness Index, ward zonal stats.
                                                             # See "WHAT CHANGED" item 5 above.

COMMON_TARGET = "region_flood_days_count"
TEST_START_YEAR = 2019
VAL_YEARS_BEFORE_TEST = 2   # LSTM/LSTM+GNN only: years [TEST_START_YEAR -
                             # VAL_YEARS_BEFORE_TEST, TEST_START_YEAR) are
                             # held out from gradient updates and used only
                             # to pick the decision threshold that maximizes
                             # F1 (see train_sequence_model) -- picking the
                             # threshold on the TEST set itself would leak
                             # test-set information into a "trained" choice.

SMOTE_MINORITY_RATIO = 0.5  # After oversampling, minority (flooded) class
                              # is brought up to this fraction of the
                              # majority class in the TRAINING split only
                              # (never applied to val/test) -- combined
                              # with class_weight="balanced" (LR/RF) and
                              # pos_weight (LSTM). Not set to 1.0 (full
                              # balance): with a real ~3% positive rate,
                              # forcing full 1:1 balance means ~97% of the
                              # final training set is synthetic/duplicated
                              # minority rows, which risks overfitting to
                              # interpolated/repeated patterns rather than
                              # real ones. 0.5 is a middle ground -- still
                              # a large oversampling boost, without letting
                              # synthetic data dominate the training set.

FLOOD_WINDOW_DAYS_LIST = [3, 7]  # LR/RF/HGB daily models predict "will
                                   # this region flood at any point in the
                                   # next N days", not "flood exactly
                                   # today". Evidence for this: on this
                                   # data, single-day prediction hit a hard
                                   # ceiling (F1 ~0.2 at best, near the
                                   # trivial-baseline floor) largely
                                   # because the exact CALENDAR DAY a flood
                                   # is logged on is itself a noisy,
                                   # approximate artifact of how IFI events
                                   # are dated -- asking "will it flood on
                                   # this exact day" holds the model to a
                                   # precision the label doesn't actually
                                   # have. Widening to a multi-day window
                                   # is both more honest about what the
                                   # label can support AND more
                                   # operationally useful (a "flood risk in
                                   # the next 3 days" warning is
                                   # actionable; "sometime this week" less
                                   # so, hence testing both 3 and 7 days).
                                   # This does NOT change LSTM/LSTM+GNN,
                                   # which keep the native daily task.

LSTM_RUN_WINDOW_VARIANTS = False  # If True, LSTM and LSTM+GNN are ALSO
                                   # retrained once per entry in
                                   # FLOOD_WINDOW_DAYS_LIST (same "flood
                                   # within next N days" reframing as
                                   # LR/RF/HGB), purely for the daily
                                   # classification comparison -- their
                                   # region-year regression numbers still
                                   # come only from the native (window=1)
                                   # run, since summing overlapping N-day-
                                   # window probabilities into an annual
                                   # day-count has no valid interpretation
                                   # (see train_sequence_model). This
                                   # roughly TRIPLES total LSTM/LSTM+GNN
                                   # training time (1 native run + one run
                                   # per window per model). Defaults to
                                   # False: on free-tier Colab (limited
                                   # session time, no guaranteed GPU),
                                   # tripling LSTM/GNN training time risks
                                   # losing the whole run to a disconnect.
                                   # Set True only once a full run without
                                   # this completes comfortably.
                                   # skip the extra runs if that's too slow.

LOOKBACK_DAYS = 14
STRIDE_DAYS = 3
HIDDEN_SIZE = 24
GRAPH_HIDDEN_SIZE = 8
BATCH_SIZE = 256
LEARNING_RATE = 1e-3

WITHIN_REGION_CORR_THRESHOLD = 0.15  # keep features whose within-region
                                       # (demeaned) |correlation| with the
                                       # target, computed on TRAINING data
                                       # only, clears this bar. See docstring.

QUICK_TEST_MODE = False
if QUICK_TEST_MODE:
    EPOCHS = 3
    SAMPLE_WARD_FRAC = 0.35
    SAMPLE_YEAR_MIN = 2015
else:
    EPOCHS = 25
    SAMPLE_WARD_FRAC = 1.0
    SAMPLE_YEAR_MIN = 2001

DYNAMIC_FEATURE_COLS = [
    "rainfall_mean", "rainfall_max", "rainfall_3day", "rainfall_7day",
    "soil_moisture_mean", "lst_mean", "is_monsoon_season",
    # NOTE: tide features (max_tide_height_m, tide_range_m, is_spring_tide)
    # were tested here and REMOVED -- despite tide being the strongest
    # single annual predictor found, adding it to the daily LSTM/LSTM+GNN
    # task made both models noticeably worse (see chat), likely because
    # tide's strong predictable cycle gave the model an easy low-loss
    # shortcut that didn't transfer to the annual summed target. This is
    # a real, tested negative result, not an oversight.
]
STATIC_FEATURE_COLS = [
    "susceptibility_index", "elevation_m_mean", "slope_deg_mean",
    "road_density_km_per_km2", "is_coastal_region",
    "twi_mean", "twi_p90",  # NEW -- see "WHAT CHANGED" item 5. twi_mean is
                             # the ward's average wetness/pooling tendency;
                             # twi_p90 catches a ward with one bad low-lying
                             # pocket that a plain mean would average away
                             # against the rest of a higher, better-drained ward.
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}" + (
    f" ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else
    " -- NO GPU DETECTED, LSTM/LSTM+GNN will be slow. In Colab: "
    "Runtime -> Change runtime type -> select a GPU."
))
torch.manual_seed(42)
np.random.seed(42)


# ---------------------------------------------------------------------
# 2. METRICS
# ---------------------------------------------------------------------

def compute_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    denom = np.sum((y_true - y_true.mean()) ** 2)
    nse = 1 - np.sum((y_true - y_pred) ** 2) / denom if denom > 0 else np.nan
    return {"MAE": mae, "RMSE": rmse, "NSE": nse}


def compute_regionyear_metrics(rows: pd.DataFrame):
    """
    'rows' is ward-year grain (region, ward_uid, year, actual, predicted).
    The flood label has zero within-region resolution -- every ward in a
    region shares one identical annual value -- so ward-year rows
    massively inflate apparent N (a region with 112 wards is 112 copies
    of ONE real observation). Collapsing to (region, year) gives the
    TRUE sample size instead of letting regions with more wards silently
    dominate a metric just by having more copies. 'actual' collapses
    with 'first' (already identical across a region-year); 'predicted'
    collapses with 'mean'.
    """
    collapsed = (
        rows.groupby(["region", "year"])
        .agg(actual=("actual", "first"), predicted=("predicted", "mean"))
        .reset_index()
    )
    return compute_metrics(collapsed["actual"], collapsed["predicted"]), collapsed


def compute_binary_threshold(train_ry: pd.DataFrame, target: str) -> float:
    """
    Data-driven split point for 'high flood year' vs 'ordinary year':
    the MEDIAN of the (corrected) TRAINING region-year target values.
    Computed ONCE, from training data only, and reused as the exact same
    cutoff for the real actual values AND every model's predicted value
    -- so "high flood year" means the same real-world thing across the
    whole comparison, rather than a threshold quietly re-picked per
    model in a way that could flatter one over another.
    """
    threshold = float(train_ry[target].median())
    print(f"  Binary 'high flood year' threshold (training-data median): "
          f"{threshold:.2f} flood-days/year")
    return threshold


def classification_metrics_from_regionyear(rows: pd.DataFrame, threshold: float) -> dict:
    """
    Converts a region-year regression comparison (actual vs predicted
    flood-days) into a genuine binary classification problem -- "was
    this a high flood year?" -- by thresholding BOTH actual and
    predicted at the same data-driven cutoff, then scoring with real
    classification metrics. 'rows' must already be at TRUE region-year
    grain (use compute_regionyear_metrics's `collapsed` output, or
    equivalent) -- ward-level rows would duplicate each region-year's
    single real outcome many times over and inflate every metric here
    exactly like it would for MAE/RMSE/NSE.
    """
    y_true = (rows["actual"] >= threshold).astype(int).to_numpy()
    y_pred = (rows["predicted"] >= threshold).astype(int).to_numpy()
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "n": len(rows),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": cm,  # [[TN, FP], [FN, TP]]
    }


def print_confusion_matrix(model_name: str, clf: dict):
    tn, fp, fn, tp = clf["confusion_matrix"].ravel()
    print(f"  {model_name} (N={clf['n']}):  accuracy={clf['accuracy']:.3f}  "
          f"precision={clf['precision']:.3f}  recall={clf['recall']:.3f}  f1={clf['f1']:.3f}")
    print(f"      {'':22s}Predicted: Ordinary   Predicted: High-flood")
    print(f"      Actual: Ordinary    {tn:>10}            {fp:>10}")
    print(f"      Actual: High-flood  {fn:>10}            {tp:>10}")


# ---------------------------------------------------------------------
# 3. LOAD CORRECTED FLOOD LABELS (replaces the raw/uncapped columns)
# ---------------------------------------------------------------------

def load_corrected_annual_labels() -> pd.DataFrame:
    try:
        return pd.read_csv(CORRECTED_ANNUAL_CSV)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"{CORRECTED_ANNUAL_CSV} not found -- run "
            f"`python build_corrected_flood_labels.py` first (one-time, "
            f"builds the corrected annual/daily flood-day labels from "
            f"the raw IFI file with a data-driven event-duration cap "
            f"instead of the raw uncapped counts)."
        )


def load_ward_twi() -> pd.DataFrame:
    """
    Loads all_regions_ward_twi_zonal_stats.csv and builds the ward_uid key
    (region + "_" + ward) so it merges cleanly onto both final_ward_year_*
    (LR/RF path) and the ward-day skeleton (LSTM path), the same way
    all_regions_ward_dem_zonal_stats.csv already does elsewhere.

    NOTE: compute_twi.py's region names are upper-cased (BMC/THANE/KDMC/
    NMMC/PUNE); every other file in this pipeline uses mixed case for two
    of them (Thane, Pune). REGION_CANON below fixes that before building
    ward_uid -- skip it and every Thane/Pune row silently fails to join.
    """
    try:
        twi = pd.read_csv(WARD_TWI_CSV)
    except FileNotFoundError:
        print(f"  NOTE: {WARD_TWI_CSV} not found -- run compute_twi.py first "
              f"to include TWI as a feature. Continuing without it (existing "
              f"elevation/slope columns are unaffected).")
        return pd.DataFrame(columns=["ward_uid"])

    REGION_CANON = {"BMC": "BMC", "THANE": "Thane", "KDMC": "KDMC",
                     "NMMC": "NMMC", "PUNE": "Pune"}
    twi["region"] = twi["region"].map(REGION_CANON).fillna(twi["region"])
    twi["ward_uid"] = twi["region"] + "_" + twi["ward"]
    keep = ["ward_uid"] + [c for c in twi.columns if c.startswith("twi_")]
    twi = twi[keep].drop_duplicates(subset="ward_uid")
    print(f"  Loaded TWI for {len(twi)} wards from {WARD_TWI_CSV}.")
    return twi


def apply_corrected_annual_labels(df: pd.DataFrame) -> pd.DataFrame:
    corrected = load_corrected_annual_labels()
    # Drop every non-key column that `corrected` also supplies (not just
    # COMMON_TARGET) before merging. region_flood_days_count_CORRECTED.csv
    # carries BOTH region_flood_days_count and region_flood_event_count --
    # leaving the latter's old/uncapped copy in `df` makes the merge collide
    # and produces region_flood_event_count_x/_y, which is exactly what the
    # leakage guard in get_numeric_feature_cols is designed to catch.
    incoming_cols = [c for c in corrected.columns if c not in ("region", "year")]
    df = df.drop(columns=incoming_cols, errors="ignore").merge(
        corrected, on=["region", "year"], how="left"
    )
    for c in incoming_cols:
        df[c] = df[c].fillna(0)
    return df


# ---------------------------------------------------------------------
# 4. EVIDENCE-BASED FEATURE SELECTION (training data only, no leakage)
# ---------------------------------------------------------------------

def select_features_by_within_region_signal(train_ry: pd.DataFrame, feature_cols: list[str],
                                              target: str, threshold: float) -> list[str]:
    """
    Keeps only features whose WITHIN-REGION (demeaned) correlation with
    the target clears `threshold`, computed on TRAINING region-years
    only. Raw correlation is confounded by region identity -- a static
    feature that just differs BETWEEN regions (but never varies within
    one) can show a high raw correlation while carrying zero information
    a naive per-region-mean baseline doesn't already have. Demeaning
    both feature and target by their own region's mean isolates the
    only kind of signal that could actually beat that baseline: does
    this feature moving up/down within a region in a given year predict
    flood-days moving up/down that year.
    """
    demeaned = train_ry.copy()
    for c in feature_cols + [target]:
        demeaned[c] = demeaned[c] - demeaned.groupby("region")[c].transform("mean")
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = demeaned[feature_cols].corrwith(demeaned[target]).abs()
    corr = corr.dropna().sort_values(ascending=False)
    selected = corr[corr > threshold]
    print(f"  Within-region feature selection: {len(selected)}/{len(feature_cols)} features "
          f"clear |corr| > {threshold} (computed on training data only).")
    print(f"  Selected: {list(selected.index)}")
    return list(selected.index) if len(selected) > 0 else list(corr.head(5).index)


# ---------------------------------------------------------------------
# 5. LR + RF
# ---------------------------------------------------------------------

def get_numeric_feature_cols(df: pd.DataFrame) -> list[str]:
    id_and_targets = {
        "region", "ward_uid", "ward", "year",
        "flood_magnitude_target", "high_flood_year_target",
        "region_flood_event_count", "region_flood_days_count",
        "region_total_fatalities",
    }
    numeric = [
        c for c in df.columns
        if c not in id_and_targets and pd.api.types.is_numeric_dtype(df[c])
    ]
    # SAFETY NET: catch _x/_y suffixed leftovers from a merge collision
    # (e.g. an older/different script version merging in a file that
    # also had 'region_flood_event_count') -- these carry near-total
    # information about the target and MUST NOT reach model training.
    # Exact-name exclusion above can't catch a suffixed variant, so this
    # checks explicitly and fails loudly instead of silently leaking.
    suspect = [c for c in numeric if c.rstrip("_x").rstrip("_y") in id_and_targets
               or any(c.startswith(t + "_") and c[len(t) + 1:] in ("x", "y") for t in id_and_targets)]
    if suspect:
        raise RuntimeError(
            f"LEAKAGE GUARD TRIPPED: {suspect} look like duplicate/suffixed "
            f"versions of an excluded id/target column (region_flood_event_count "
            f"or similar) -- this means a merge somewhere collided two columns "
            f"with the same base name. Do not proceed: find and fix the merge "
            f"before training, or these will leak the answer into the model."
        )
    all_nan = [c for c in numeric if df[c].isna().all()]
    if all_nan:
        print(f"  Dropping {len(all_nan)} all-NaN column(s) from features: {all_nan}")
    return [c for c in numeric if c not in all_nan]


def train_lr_rf():
    print("=" * 60)
    print("Training Linear Regression (RidgeCV) + Random Forest")
    print("=" * 60)
    train = pd.read_csv(FINAL_WARD_YEAR_TRAIN_CSV)
    test = pd.read_csv(FINAL_WARD_YEAR_TEST_CSV)
    print(f"Raw ward-year rows -- train: {len(train)} | test: {len(test)} "
          f"(inflated by ward-count duplication -- see compute_regionyear_metrics)")

    print("Applying corrected flood-day labels (data-driven event-duration cap, "
          "see build_corrected_flood_labels.py)...")
    train = apply_corrected_annual_labels(train)
    test = apply_corrected_annual_labels(test)

    # Merge in TWI (see WARD_TWI_CSV / load_ward_twi). Left as a candidate
    # like everything else here -- goes through the same within-region
    # evidence filter below, not forced into the model.
    print("Merging TWI (Topographic Wetness Index) ward features...")
    twi = load_ward_twi()
    train = train.merge(twi, on="ward_uid", how="left")
    test = test.merge(twi, on="ward_uid", how="left")

    # Merge in daily autocorrelation/persistence features (burst-day counts,
    # consecutive wet/high-soil-moisture streaks) -- see
    # build_autocorrelation_features.py. These are candidates only: they go
    # through the exact same within-region evidence check as every other
    # feature below, not forced in.
    try:
        autocorr = pd.read_csv(AUTOCORRELATION_FEATURES_CSV)
        autocorr_cols = [c for c in autocorr.columns if c not in ("region", "year")]
        train = train.merge(autocorr, on=["region", "year"], how="left")
        test = test.merge(autocorr, on=["region", "year"], how="left")
        print(f"Merged {len(autocorr_cols)} candidate autocorrelation features: {autocorr_cols}")
    except FileNotFoundError:
        print(f"  {AUTOCORRELATION_FEATURES_CSV} not found -- run "
              f"build_autocorrelation_features.py first to include these. Continuing without them.")

    feature_cols_all = get_numeric_feature_cols(train)
    print(f"{len(feature_cols_all)} numeric features available before selection.")

    train_ry = train.groupby(["region", "year"])[feature_cols_all + [COMMON_TARGET]].mean().reset_index()
    test_ry = test.groupby(["region", "year"])[feature_cols_all + [COMMON_TARGET]].mean().reset_index()
    print(f"Collapsed to TRUE sample size -- train: {len(train_ry)} region-years "
          f"| test: {len(test_ry)} region-years.")

    feature_cols = select_features_by_within_region_signal(
        train_ry, feature_cols_all, COMMON_TARGET, WITHIN_REGION_CORR_THRESHOLD
    )

    X_train = train_ry[feature_cols].fillna(train_ry[feature_cols].median())
    X_test = test_ry[feature_cols].fillna(train_ry[feature_cols].median())
    y_train, y_test = train_ry[COMMON_TARGET], test_ry[COMMON_TARGET]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    y_train_fit = np.log1p(y_train)

    n_folds = min(5, len(train_ry))
    ridge = RidgeCV(alphas=np.logspace(-3, 5, 60), cv=KFold(n_folds, shuffle=True, random_state=42))
    ridge.fit(X_train_scaled, y_train_fit)
    print(f"RidgeCV selected alpha = {ridge.alpha_:.3g}")
    lr_pred_ry = np.clip(np.expm1(ridge.predict(X_test_scaled)), 0, None)

    rf_grid = GridSearchCV(
        RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        param_grid={"max_depth": [2, 3, 4, 5], "min_samples_leaf": [2, 3, 5]},
        cv=KFold(n_folds, shuffle=True, random_state=42),
        scoring="neg_mean_absolute_error",
    )
    rf_grid.fit(X_train, y_train_fit)
    print(f"GridSearchCV selected RF params: {rf_grid.best_params_}")
    rf_pred_ry = np.clip(np.expm1(rf_grid.predict(X_test)), 0, None)

    naive_pred_ry = test_ry["region"].map(train_ry.groupby("region")[COMMON_TARGET].mean()).to_numpy()

    def broadcast_to_ward_rows(pred_ry):
        lookup = test_ry[["region", "year"]].assign(predicted=pred_ry)
        return test.merge(lookup, on=["region", "year"], how="left")["predicted"].to_numpy()

    lr_pred = broadcast_to_ward_rows(lr_pred_ry)
    rf_pred = broadcast_to_ward_rows(rf_pred_ry)
    naive_pred = broadcast_to_ward_rows(naive_pred_ry)

    id_cols = test[["region", "ward_uid", "year"]].reset_index(drop=True)
    y_test_ward = test[COMMON_TARGET].values
    lr_rows = id_cols.assign(actual=y_test_ward, predicted=lr_pred)
    rf_rows = id_cols.assign(actual=y_test_ward, predicted=rf_pred)
    naive_rows = id_cols.assign(actual=y_test_ward, predicted=naive_pred)

    lr_metrics, lr_collapsed = compute_regionyear_metrics(lr_rows)
    rf_metrics, rf_collapsed = compute_regionyear_metrics(rf_rows)
    naive_metrics, naive_collapsed = compute_regionyear_metrics(naive_rows)

    print(f"\nLinear Regression (region-year, TRUE N={len(test_ry)}) -> {lr_metrics}")
    print(f"Random Forest     (region-year, TRUE N={len(test_ry)}) -> {rf_metrics}")
    print(f"Naive baseline    (region-year, TRUE N={len(test_ry)}) -> {naive_metrics}  <- floor to beat")

    # ---- Classification framing: "was this a high flood year?" ----
    print("\nClassification metrics (regression predictions thresholded into "
          "'high flood year' vs 'ordinary year'):")
    threshold = compute_binary_threshold(train_ry, COMMON_TARGET)
    lr_clf = classification_metrics_from_regionyear(lr_collapsed, threshold)
    rf_clf = classification_metrics_from_regionyear(rf_collapsed, threshold)
    naive_clf = classification_metrics_from_regionyear(naive_collapsed, threshold)
    print_confusion_matrix("Linear Regression", lr_clf)
    print_confusion_matrix("Random Forest", rf_clf)
    print_confusion_matrix("Naive baseline", naive_clf)

    return (lr_metrics, rf_metrics, naive_metrics, lr_rows, rf_rows, naive_rows,
            threshold, lr_clf, rf_clf, naive_clf)


# ---------------------------------------------------------------------
# 6. SHARED DATA PREP FOR LSTM / LSTM+GNN
# ---------------------------------------------------------------------

def build_ward_day_in_memory() -> pd.DataFrame:
    print("\nBuilding ward x day table in memory (reusing "
          "build_ward_day_dataset.py's functions, no multi-GB CSV write)...")
    df = build_ward_day_table()

    # Merge in TWI -- build_ward_day_table() (build_ward_day_dataset.py)
    # doesn't know about it yet, so it's joined on here rather than
    # requiring that file to be edited too. See load_ward_twi() / WARD_TWI_CSV.
    print("Merging TWI (Topographic Wetness Index) ward features...")
    twi = load_ward_twi()
    n_before = len(df)
    df = df.merge(twi, on="ward_uid", how="left")
    assert len(df) == n_before, "TWI merge changed row count -- ward_uid duplicates in TWI file?"
    n_missing_twi = df["twi_mean"].isna().sum() if "twi_mean" in df.columns else len(df)
    if n_missing_twi:
        print(f"  NOTE: {n_missing_twi} rows ({df.loc[df['twi_mean'].isna(), 'ward_uid'].nunique() if 'twi_mean' in df.columns else 'all'} wards) "
              f"had no TWI match -- likely a ward-name mismatch between the KML "
              f"source and this pipeline's ward_uid (see compute_twi.py's Thane "
              f"note: 47 vs 46 wards). Filled with the median below like other "
              f"static features.")

    df = df[df["year"] >= SAMPLE_YEAR_MIN].reset_index(drop=True)

    if SAMPLE_WARD_FRAC < 1.0:
        rng = np.random.RandomState(42)
        wards = df["ward_uid"].unique()
        keep = rng.choice(wards, size=int(len(wards) * SAMPLE_WARD_FRAC), replace=False)
        df = df[df["ward_uid"].isin(keep)].reset_index(drop=True)
        print(f"QUICK_TEST_MODE: subsampled to {len(keep)}/{len(wards)} wards.")

    df = add_sequence_features(df)

    # Overlay the corrected daily flood label (data-driven event-duration
    # cap) in place of the raw region_day_flooded_broadcast column that
    # came from build_ward_day_table() -- see build_corrected_flood_labels.py.
    print("Applying corrected daily flood labels...")
    corrected_daily = pd.read_csv(CORRECTED_DAILY_CSV, parse_dates=["date"])
    df = df.drop(columns=["region_day_flooded_broadcast"], errors="ignore").merge(
        corrected_daily[["region", "date", "region_day_flooded_broadcast"]],
        on=["region", "date"], how="left",
    )
    df["region_day_flooded_broadcast"] = df["region_day_flooded_broadcast"].fillna(0)

    # NOTE: tide features were tested here and REMOVED (see
    # DYNAMIC_FEATURE_COLS comment above and chat notes) -- adding tide to
    # the daily LSTM/LSTM+GNN task made both models noticeably worse.
    # There is deliberately no tide-loading code left here; if this
    # changes in the future, re-add it alongside DYNAMIC_FEATURE_COLS,
    # not as an orphaned block that crashes when the tide CSV isn't built.

    for c in DYNAMIC_FEATURE_COLS + STATIC_FEATURE_COLS:
        if c not in df.columns:
            raise KeyError(f"Expected column '{c}' missing from ward-day table.")
        df[c] = pd.to_numeric(df[c], errors="coerce")
        # Impute missing static/dynamic values using within-region medians first,
        # avoiding cross-region contamination (e.g. coastal NMMC wards inheriting 600m Pune elevations).
        reg_med = df.groupby("region")[c].transform("median")
        df[c] = df[c].fillna(reg_med).fillna(df[c].median())

    print(f"Ward-day table ready: {len(df)} rows, {df['ward_uid'].nunique()} wards, "
          f"{df['date'].min().date()} to {df['date'].max().date()}.")
    return df


# ---------------------------------------------------------------------
# 6a2. DAILY-GRAIN PERSISTENCE FEATURES -- CAUSAL / ROLLING, NO LOOK-AHEAD
# ---------------------------------------------------------------------
#
# The annual autocorrelation features in build_autocorrelation_features.py
# (max_consecutive_wet_days, rainfall_lag1_autocorrelation, etc.) are each
# computed ONCE per region-year using the ENTIRE year's data. Broadcasting
# that single annual value onto every day of the year as a row-level
# feature for a daily model would leak future information into past
# days: a January 1st row would carry a feature computed using data from
# the following December. That's not train/test leakage (the annual
# thresholds already respect TEST_START_YEAR) -- it's leakage ACROSS
# days WITHIN a year, specific to using a whole-year aggregate as a
# per-day feature.
#
# This builds the causal analogue instead: for each region-day, using
# ONLY that day and days before it (never after):
#   consec_wet_days_to_date                 - current wet-day streak
#                                              ending today (resets to 0
#                                              the day after a dry day)
#   consec_high_soil_moisture_days_to_date   - same idea, soil moisture
#   rainfall_lag1_autocorr_30d               - rolling 30-day lag-1
#                                              autocorrelation of daily
#                                              rainfall, trailing window
#   n_heavy_rain_burst_days_30d              - rolling 30-day count of
#                                              burst days (burst
#                                              threshold = this region's
#                                              own 90th pct of
#                                              rainfall_3day, computed
#                                              from TRAINING years only,
#                                              same no-leakage rule as
#                                              the annual version)
#   pct_days_above_median_soil_moisture_30d  - rolling 30-day fraction
#                                              above this region's
#                                              training-period soil
#                                              moisture median
#
# Computed once per (region, date) -- not per ward-day -- since the
# underlying rainfall/soil-moisture columns are identical for every ward
# in a region on a given day; then merged back onto the ward-day table
# by (region, date).

ROLLING_WINDOW_DAYS_LIST = [7, 30]  # Both a short (7d, closer to LSTM's
                                      # 14-day lookback -- the "cheap lever
                                      # for temporal awareness without
                                      # going full recurrent") and long
                                      # (30d, monthly/seasonal persistence)
                                      # trailing window are computed for
                                      # each rolling feature below, so
                                      # LR/RF/HGB get both a fast-reacting
                                      # and a slow-reacting version of the
                                      # same signal. min_periods scales
                                      # with the window so a 7d feature
                                      # doesn't sit empty for a third of
                                      # its own window like a 30d one can.
WET_DAY_MM = 1.0  # matches build_autocorrelation_features.py's definition


def _running_streak(bool_arr: np.ndarray) -> np.ndarray:
    """Length of the True-streak ending at each position (0 if False)."""
    out = np.zeros(len(bool_arr), dtype=int)
    run = 0
    for i, v in enumerate(bool_arr):
        run = run + 1 if v else 0
        out[i] = run
    return out


def _rolling_lag1_autocorr(x: np.ndarray) -> float:
    if len(x) < 3:
        return 0.0
    a, b = x[:-1], x[1:]
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def build_daily_persistence_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\nBuilding causal (rolling, no look-ahead) daily persistence features...")
    regday = (
        df.drop_duplicates(subset=["region", "date"])
        [["region", "date", "year", "rainfall_mean", "rainfall_3day", "soil_moisture_mean"]]
        .sort_values(["region", "date"])
        .reset_index(drop=True)
    )

    train_mask = regday["year"] < TEST_START_YEAR
    burst_threshold = regday[train_mask].groupby("region")["rainfall_3day"].quantile(0.90)
    soil_median = regday[train_mask].groupby("region")["soil_moisture_mean"].median()
    print("  Per-region rainfall_3day 90th-percentile burst threshold (training years only):")
    print(burst_threshold.to_string())
    print("  Per-region soil_moisture_mean median (training years only):")
    print(soil_median.to_string())

    regday["is_wet_day"] = regday["rainfall_mean"] > WET_DAY_MM
    regday["is_burst_day"] = regday.apply(
        lambda r: r["rainfall_3day"] > burst_threshold.get(r["region"], np.inf), axis=1
    )
    regday["is_high_soil_moisture"] = regday.apply(
        lambda r: r["soil_moisture_mean"] > soil_median.get(r["region"], np.inf), axis=1
    )

    new_cols = ["consec_wet_days_to_date", "consec_high_soil_moisture_days_to_date"]
    for w in ROLLING_WINDOW_DAYS_LIST:
        new_cols += [f"rainfall_lag1_autocorr_{w}d", f"n_heavy_rain_burst_days_{w}d",
                     f"pct_days_above_median_soil_moisture_{w}d"]

    out_frames = []
    for region, g in regday.groupby("region"):
        g = g.sort_values("date").copy()
        g["consec_wet_days_to_date"] = _running_streak(g["is_wet_day"].to_numpy())
        g["consec_high_soil_moisture_days_to_date"] = _running_streak(
            g["is_high_soil_moisture"].to_numpy()
        )
        for w in ROLLING_WINDOW_DAYS_LIST:
            autocorr_min_periods = max(3, w // 3)  # scales with window so a
                                                     # short 7d window isn't
                                                     # held to the same
                                                     # 10-day warm-up as 30d
            g[f"rainfall_lag1_autocorr_{w}d"] = (
                g["rainfall_mean"].rolling(w, min_periods=autocorr_min_periods)
                .apply(_rolling_lag1_autocorr, raw=True)
            )
            g[f"n_heavy_rain_burst_days_{w}d"] = (
                g["is_burst_day"].astype(int).rolling(w, min_periods=1).sum()
            )
            g[f"pct_days_above_median_soil_moisture_{w}d"] = (
                g["is_high_soil_moisture"].astype(float).rolling(w, min_periods=1).mean()
            )
        out_frames.append(g)

    result = pd.concat(out_frames, ignore_index=True)
    result[new_cols] = result[new_cols].fillna(0.0)
    print(f"  Built {len(new_cols)} causal persistence features "
          f"(windows: {ROLLING_WINDOW_DAYS_LIST} days) for {len(result)} region-day rows.")
    return result[["region", "date"] + new_cols]


DAILY_PERSISTENCE_FEATURE_COLS = (
    ["consec_wet_days_to_date", "consec_high_soil_moisture_days_to_date"]
    + [f"{name}_{w}d" for w in ROLLING_WINDOW_DAYS_LIST
       for name in ("rainfall_lag1_autocorr", "n_heavy_rain_burst_days",
                     "pct_days_above_median_soil_moisture")]
)


# ---------------------------------------------------------------------
# 6a3. DAILY-GRAIN LAG / TREND FEATURES -- explicit past-day values
# ---------------------------------------------------------------------
#
# The persistence features above are SUMMARIES of recent history (streak
# length, rolling autocorrelation, burst count) -- useful, but they
# compress the last N days down to a handful of numbers. LSTM's
# recurrence gets something more direct: it actually SEES each of the
# past 14 days' raw values in sequence and can learn its own weighting
# of "3 days ago mattered more than 10 days ago" from data. LR/RF/HGB
# have no equivalent unless we hand them the raw past values explicitly.
#
# This adds, for each of a few key dynamic variables, its value exactly
# 1/2/3 days ago (LAG_DAYS_LIST) plus a simple 1-day trend (today minus
# yesterday, a crude "is this getting worse" signal) -- all computed
# causally (shift() only ever looks backward) per region-day, same
# no-look-ahead discipline as everything else in this pipeline.
#
# Not applied to LSTM/LSTM+GNN: they already see the true, un-compressed
# sequence of all DYNAMIC_FEATURE_COLS across the full 14-day lookback via
# recurrence -- adding lag columns there would be redundant, not helpful.

LAG_DAYS_LIST = [1, 2, 3]
LAG_FEATURE_SOURCE_COLS = ["rainfall_mean", "soil_moisture_mean", "lst_mean"]


def build_daily_lag_features(df: pd.DataFrame, lags: list[int] = LAG_DAYS_LIST,
                              source_cols: list[str] = LAG_FEATURE_SOURCE_COLS) -> pd.DataFrame:
    print(f"\nBuilding causal lag/trend features (lags={lags} days) for {source_cols}...")
    regday = (
        df.drop_duplicates(subset=["region", "date"])
        [["region", "date", "year"] + source_cols]
        .sort_values(["region", "date"])
        .reset_index(drop=True)
    )
    new_cols = [f"{col}_lag{lag}d" for col in source_cols for lag in lags]
    new_cols += [f"{col}_trend_1d" for col in source_cols]

    out_frames = []
    for region, g in regday.groupby("region"):
        g = g.sort_values("date").copy()
        for col in source_cols:
            for lag in lags:
                g[f"{col}_lag{lag}d"] = g[col].shift(lag)
            g[f"{col}_trend_1d"] = g[col] - g[col].shift(1)
        out_frames.append(g)

    result = pd.concat(out_frames, ignore_index=True)
    # NaN only at each region's series start (no history yet, e.g. the
    # very first 1-3 days of the 1990-2024 record) -- filled with 0
    # (neutral / "no change"), never with a value derived from later days.
    result[new_cols] = result[new_cols].fillna(0.0)
    print(f"  Built {len(new_cols)} causal lag/trend features for {len(result)} region-day rows.")
    return result[["region", "date"] + new_cols]


DAILY_LAG_FEATURE_COLS = (
    [f"{col}_lag{lag}d" for col in LAG_FEATURE_SOURCE_COLS for lag in LAG_DAYS_LIST]
    + [f"{col}_trend_1d" for col in LAG_FEATURE_SOURCE_COLS]
)


# ---------------------------------------------------------------------
# 6b. LR + RF + HGB + STACKED ENSEMBLE, TRAINED ON DAILY (ward-day) DATA,
#     PREDICTING "FLOOD WITHIN THE NEXT N DAYS" (N in FLOOD_WINDOW_DAYS_LIST)
# ---------------------------------------------------------------------
#
# The region-year LR/RF in train_lr_rf() only ever see 90/30 ANNUAL rows
# (region-year averages) -- that's a genuinely different, much smaller
# training problem than what the LSTM sees. This trains directly on
# ward-day rows instead: same DYNAMIC_FEATURE_COLS + STATIC_FEATURE_COLS
# + causal persistence-feature inputs used elsewhere in the daily
# pipeline, but predicting a forward-looking WINDOW target rather than
# the exact day -- see FLOOD_WINDOW_DAYS_LIST comment for why.
#
# class_weight="balanced" / sample_weight="balanced" (LR/RF/HGB) AND
# SMOTE oversampling (fit split only) are both applied, same imbalance
# fix as elsewhere in this pipeline. On top of that, the decision
# threshold is tuned per model on a held-out VALIDATION slice (years
# [TEST_START_YEAR - VAL_YEARS_BEFORE_TEST, TEST_START_YEAR), never used
# for fitting) by maximizing F1 -- never on the test set itself.
#
# HistGradientBoostingClassifier is added alongside LR/RF: earlier runs
# showed LR beating RF on this task, suggesting the RF's default
# hyperparameters may be under-tuned for this specific, low-signal
# problem -- HGB is a stronger, well-regularized tree-ensemble baseline
# that's often more competitive than an under-tuned RF on imbalanced
# tabular data, without requiring a new dependency (built into sklearn).
#
# A simple averaged-probability STACK of all three models is also
# scored: cheap to compute, and stacking model families that make
# different kinds of errors (linear vs. tree-based) often buys a real
# few points of F1 even when no single model improves.
#
# average_precision_score (area under the precision-recall curve) is
# reported for every model too -- a single F1 number depends on exactly
# which threshold got picked; PR-AUC summarizes discriminative power
# across ALL thresholds and is the more robust number to compare models
# by. The full test-set PR curve is also saved to CSV per window so it
# can be plotted (e.g. to pick an operating point other than max-F1 --
# a "put crews on standby" use case may prefer higher recall at some
# precision cost, or vice versa).

def _best_f1_threshold(y_true, y_prob):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1s = np.where(
        (precisions[:-1] + recalls[:-1]) > 0,
        2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-12),
        0.0,
    )
    if len(thresholds) == 0 or f1s.max() == 0:
        return 0.5, 0.0
    best_idx = f1s.argmax()
    return float(thresholds[best_idx]), float(f1s[best_idx])


def _clf_metrics(y_true, y_pred, task_label: str):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": cm,
        "task": task_label,
    }


def build_flood_in_window_target(df: pd.DataFrame, window_days: int,
                                  source_col: str = "region_day_flooded_broadcast") -> pd.DataFrame:
    """
    For each region-day, target = 1 if source_col is 1 on that day OR any
    of the following (window_days - 1) days, else 0. This is a forward-
    looking LABEL (allowed to use "future" days relative to the row's
    date) -- NOT a feature, so this is not leakage; it answers "would a
    flood-risk alert issued today have been correct". Only that day and
    earlier are ever used for FEATURES elsewhere in the pipeline.
    Rows at the end of each region's date range without a full forward
    window are dropped (their true label can't be determined).
    """
    target_col = f"flood_in_next_{window_days}d"
    regday = (
        df.drop_duplicates(subset=["region", "date"])
        [["region", "date", "year", source_col]]
        .sort_values(["region", "date"])
        .reset_index(drop=True)
    )
    frames = []
    for region, g in regday.groupby("region"):
        g = g.sort_values("date").reset_index(drop=True)
        shifted = pd.concat([g[source_col].shift(-k) for k in range(window_days)], axis=1)
        g[target_col] = shifted.max(axis=1)
        frames.append(g)
    result = pd.concat(frames, ignore_index=True)
    before = len(result)
    result = result.dropna(subset=[target_col])
    print(f"  Built '{target_col}' target: dropped {before - len(result)} trailing "
          f"region-day rows (incomplete forward window at each region's series end).")
    result[target_col] = result[target_col].astype(int)
    return result[["region", "date", target_col]]


def train_daily_window_models(df: pd.DataFrame, window_days: int) -> dict:
    print("\n" + "=" * 60)
    print(f"Training LR + RF + HistGradientBoosting + Stack -- DAILY grain, "
          f"'flood within next {window_days} day(s)' target")
    print("=" * 60)

    persistence = build_daily_persistence_features(df)
    df = df.merge(persistence, on=["region", "date"], how="left")

    lag_features = build_daily_lag_features(df)
    df = df.merge(lag_features, on=["region", "date"], how="left")

    window_target = build_flood_in_window_target(df, window_days)
    target = f"flood_in_next_{window_days}d"
    df = df.merge(window_target, on=["region", "date"], how="inner")  # inner: drops the
                                                                        # trailing incomplete-
                                                                        # window rows for every ward

    feature_cols = (DYNAMIC_FEATURE_COLS + STATIC_FEATURE_COLS
                     + DAILY_PERSISTENCE_FEATURE_COLS + DAILY_LAG_FEATURE_COLS)
    print(f"Using {len(feature_cols)} features: {len(DAILY_PERSISTENCE_FEATURE_COLS)} "
          f"causal persistence features + {len(DAILY_LAG_FEATURE_COLS)} causal lag/trend features.")

    val_start_year = TEST_START_YEAR - VAL_YEARS_BEFORE_TEST
    fit_df = df[df["year"] < val_start_year].copy()
    val_df = df[(df["year"] >= val_start_year) & (df["year"] < TEST_START_YEAR)].copy()
    test_df = df[df["year"] >= TEST_START_YEAR].copy()
    print(f"Ward-day rows -- fit: {len(fit_df)} | val: {len(val_df)} (years "
          f"[{val_start_year}, {TEST_START_YEAR}), threshold selection only) | "
          f"test: {len(test_df)}")
    print(f"  Positive rate -- fit: {fit_df[target].mean():.4f} | "
          f"val: {val_df[target].mean():.4f} | test: {test_df[target].mean():.4f}")

    for part in (fit_df, val_df, test_df):
        part[feature_cols] = part[feature_cols].fillna(fit_df[feature_cols].median())

    X_fit = fit_df[feature_cols].to_numpy(dtype=np.float32)
    X_val = val_df[feature_cols].to_numpy(dtype=np.float32)
    X_test = test_df[feature_cols].to_numpy(dtype=np.float32)
    y_fit = fit_df[target].to_numpy(dtype=int)
    y_val = val_df[target].to_numpy(dtype=int)
    y_test = test_df[target].to_numpy(dtype=int)

    scaler = StandardScaler().fit(X_fit)
    X_fit_s = scaler.transform(X_fit)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    print(f"  Fit positive rate before SMOTE: {y_fit.mean():.4f}")
    smote = SMOTE(sampling_strategy=SMOTE_MINORITY_RATIO, random_state=42)
    X_fit_res, y_fit_res = smote.fit_resample(X_fit_s, y_fit)
    print(f"  Fit positive rate after SMOTE (target ratio={SMOTE_MINORITY_RATIO}): "
          f"{y_fit_res.mean():.4f} ({len(y_fit_res)} rows, was {len(y_fit)})")
    sample_weight_res = compute_sample_weight("balanced", y_fit_res)

    # Hyperparameter search, scored by PR-AUC (average_precision) rather
    # than accuracy -- accuracy is nearly meaningless at this imbalance,
    # PR-AUC is the metric we actually want higher. Run on a SUBSAMPLE of
    # the SMOTE-resampled fit set (search cost scales with n_iter x cv x
    # data size, and this is already the single slowest step in the
    # pipeline on free-tier Colab) -- the winning hyperparameters are then
    # refit on the FULL resampled fit set, so the final model still sees
    # everything, only the search itself is sped up.
    HPARAM_SEARCH_SUBSAMPLE = 150_000
    if len(y_fit_res) > HPARAM_SEARCH_SUBSAMPLE:
        X_search, _, y_search, _ = train_test_split(
            X_fit_res, y_fit_res, train_size=HPARAM_SEARCH_SUBSAMPLE,
            stratify=y_fit_res, random_state=42,
        )
    else:
        X_search, y_search = X_fit_res, y_fit_res

    task_label = f"flood in next {window_days} day(s), daily ward-day grain"
    results = {}
    val_probs, test_probs = {}, {}

    print(f"Tuning Logistic Regression (C) via 3-fold CV on a {len(y_search)}-row "
          f"subsample, scoring=PR-AUC...")
    lr_search = GridSearchCV(
        LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
        {"C": [0.01, 0.1, 1, 10, 100]}, scoring="average_precision", cv=3, n_jobs=-1,
    )
    lr_search.fit(X_search, y_search)
    print(f"  Best LR C={lr_search.best_params_['C']} (CV PR-AUC={lr_search.best_score_:.3f})")
    lr_model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42,
                                   **lr_search.best_params_)
    lr_model.fit(X_fit_res, y_fit_res)
    val_probs["Logistic Regression"] = lr_model.predict_proba(X_val_s)[:, 1]
    test_probs["Logistic Regression"] = lr_model.predict_proba(X_test_s)[:, 1]

    print(f"Tuning Random Forest via RandomizedSearchCV on a {len(y_search)}-row "
          f"subsample, scoring=PR-AUC...")
    rf_search = RandomizedSearchCV(
        RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
        {"n_estimators": [100, 150, 200], "max_depth": [6, 8, 10, None],
         "min_samples_leaf": [2, 5, 10], "max_samples": [0.2, 0.3, 0.5]},
        n_iter=8, scoring="average_precision", cv=3, random_state=42, n_jobs=-1,
    )
    rf_search.fit(X_search, y_search)
    print(f"  Best RF params={rf_search.best_params_} (CV PR-AUC={rf_search.best_score_:.3f})")
    rf_model = RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1,
                                       **rf_search.best_params_)
    rf_model.fit(X_fit_res, y_fit_res)
    val_probs["Random Forest"] = rf_model.predict_proba(X_val_s)[:, 1]
    test_probs["Random Forest"] = rf_model.predict_proba(X_test_s)[:, 1]

    print(f"Tuning HistGradientBoosting via RandomizedSearchCV on a {len(y_search)}-row "
          f"subsample, scoring=PR-AUC...")
    hgb_search = RandomizedSearchCV(
        HistGradientBoostingClassifier(random_state=42),
        {"max_iter": [100, 150, 200], "max_depth": [4, 6, 8, None],
         "learning_rate": [0.03, 0.05, 0.1], "l2_regularization": [0.0, 0.5, 1.0]},
        n_iter=8, scoring="average_precision", cv=3, random_state=42, n_jobs=-1,
    )
    hgb_search.fit(X_search, y_search)
    print(f"  Best HGB params={hgb_search.best_params_} (CV PR-AUC={hgb_search.best_score_:.3f})")
    hgb_model = HistGradientBoostingClassifier(random_state=42, **hgb_search.best_params_)
    hgb_model.fit(X_fit_res, y_fit_res, sample_weight=sample_weight_res)
    val_probs["HistGradientBoosting"] = hgb_model.predict_proba(X_val_s)[:, 1]
    test_probs["HistGradientBoosting"] = hgb_model.predict_proba(X_test_s)[:, 1]

    # Two stacked ensembles, for comparison:
    # (1) simple average of the three base probabilities -- unlearned,
    #     cheap, a reasonable floor for what stacking should beat.
    # (2) a proper meta-learner: a small LogisticRegression trained on the
    #     three base models' VAL-set predictions (never the fit set --
    #     that would just relearn what the base models already know from
    #     their own training data) to predict y_val, then applied to the
    #     three base models' TEST-set predictions. This lets the
    #     meta-learner learn a WEIGHTED combination (e.g. "trust RF more
    #     when LR and HGB disagree") instead of a blind average.
    # Honest caveat: the meta-learner is fit on val, and its threshold is
    # ALSO selected on val (same split, same three-way fit/val/test
    # separation as everything else) -- val is doing double duty here
    # (fitting the meta-learner's weights, then picking its threshold), so
    # treat its val-derived numbers as slightly optimistic; the TEST
    # scores below are the real, unbiased read.
    val_probs["Stacked (avg)"] = np.mean(
        [val_probs["Logistic Regression"], val_probs["Random Forest"], val_probs["HistGradientBoosting"]], axis=0
    )
    test_probs["Stacked (avg)"] = np.mean(
        [test_probs["Logistic Regression"], test_probs["Random Forest"], test_probs["HistGradientBoosting"]], axis=0
    )

    meta_X_val = np.column_stack(
        [val_probs["Logistic Regression"], val_probs["Random Forest"], val_probs["HistGradientBoosting"]]
    )
    meta_X_test = np.column_stack(
        [test_probs["Logistic Regression"], test_probs["Random Forest"], test_probs["HistGradientBoosting"]]
    )
    meta_model = LogisticRegression(class_weight="balanced", random_state=42)
    meta_model.fit(meta_X_val, y_val)
    print(f"  Meta-learner (stacking) coefficients [LR, RF, HGB]: {meta_model.coef_[0]}")
    val_probs["Stacked (meta-LR)"] = meta_model.predict_proba(meta_X_val)[:, 1]
    test_probs["Stacked (meta-LR)"] = meta_model.predict_proba(meta_X_test)[:, 1]

    pr_curve_rows = []
    print(f"\n  Daily classification metrics ('flood within next {window_days} day(s)', "
          f"held-out ward-days):")
    for name in ("Logistic Regression", "Random Forest", "HistGradientBoosting",
                 "Stacked (avg)", "Stacked (meta-LR)"):
        threshold, val_f1 = _best_f1_threshold(y_val, val_probs[name])
        pred = (test_probs[name] >= threshold).astype(int)
        clf = _clf_metrics(y_test, pred, task_label)
        pr_auc = average_precision_score(y_test, test_probs[name])
        clf["pr_auc"] = pr_auc
        results[f"{name} ({window_days}d)"] = clf
        print(f"  Validation-selected threshold: {threshold:.3f} (val F1={val_f1:.3f})")
        print_confusion_matrix(f"  {name} ({window_days}d window, threshold={threshold:.3f}, "
                                f"test PR-AUC={pr_auc:.3f})", clf)

        precisions, recalls, thresholds = precision_recall_curve(y_test, test_probs[name])
        for p, r, t in zip(precisions[:-1], recalls[:-1], thresholds):
            pr_curve_rows.append({"model": name, "window_days": window_days,
                                   "threshold": t, "precision": p, "recall": r})

    pr_curve_df = pd.DataFrame(pr_curve_rows)
    pr_curve_csv = f"model_pr_curve_{window_days}d.csv"
    pr_curve_df.to_csv(pr_curve_csv, index=False)
    print(f"  Saved -> {pr_curve_csv} (full test-set precision-recall curve, "
          f"for picking a non-default operating point)")

    return results


def build_adjacency_matrix(ward_ids: list[str]) -> torch.Tensor:
    print("\nBuilding normalized adjacency matrix from ward_adjacency_edges.csv...")
    edges = pd.read_csv(ADJACENCY_EDGES_CSV)
    idx = {w: i for i, w in enumerate(ward_ids)}
    n = len(ward_ids)
    A = np.zeros((n, n), dtype=np.float32)
    used = 0
    for _, row in edges.iterrows():
        w1, w2 = row["ward_uid_1"], row["ward_uid_2"]
        if w1 in idx and w2 in idx:
            i, j = idx[w1], idx[w2]
            weight = float(row["shared_boundary_km"])
            A[i, j] = weight
            A[j, i] = weight
            used += 1
    print(f"  {used}/{len(edges)} edges connect wards present in this run's sample.")
    A = A + np.eye(n, dtype=np.float32)
    deg = A.sum(axis=1)
    deg_inv_sqrt = np.zeros_like(deg)
    np.power(deg, -0.5, out=deg_inv_sqrt, where=deg > 0)
    D_inv_sqrt = np.diag(deg_inv_sqrt)
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    return torch.tensor(A_norm, dtype=torch.float32, device=DEVICE)


def make_sequences(ward_df: pd.DataFrame, feature_cols: list[str], window_days: int = 1):
    """
    window_days=1 (default) reproduces the original exact-next-day label
    exactly: labels[end:end+1].max() == labels[end].
    window_days>1: label = 1 if ANY of the window_days days starting at
    `end` (the day right after the LOOKBACK_DAYS input window) is a flood
    day -- "will this region flood at some point in the next N days",
    same semantics as build_flood_in_window_target used for LR/RF/HGB.
    This only changes the LABEL; the INPUT window (feats[start:end]) still
    only ever uses days before `end`, so no feature-side leakage either way.
    """
    feats = ward_df[feature_cols].to_numpy(dtype=np.float32)
    labels = ward_df["region_day_flooded_broadcast"].to_numpy(dtype=np.float32)
    years = ward_df["year"].to_numpy()
    X, y, yrs, starts = [], [], [], []
    for start in range(0, len(ward_df) - LOOKBACK_DAYS - window_days, STRIDE_DAYS):
        end = start + LOOKBACK_DAYS
        X.append(feats[start:end])
        y.append(labels[end:end + window_days].max())
        yrs.append(years[end])
        starts.append(start)
    if not X:
        return None, None, None, None
    return np.stack(X), np.array(y), np.array(yrs), np.array(starts)


def date_positions_for_windows(ward_df: pd.DataFrame, starts: np.ndarray, date_to_pos: dict) -> np.ndarray:
    dates = pd.to_datetime(ward_df["date"])
    out = np.empty((len(starts), LOOKBACK_DAYS), dtype=np.int64)
    for i, s in enumerate(starts):
        window_dates = dates.iloc[s: s + LOOKBACK_DAYS]
        out[i] = [date_to_pos[pd.Timestamp(d)] for d in window_dates]
    return out


# ---------------------------------------------------------------------
# 7. MODELS
# ---------------------------------------------------------------------

class FloodLSTM(nn.Module):
    def __init__(self, n_dynamic, n_static, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(n_dynamic, hidden_size, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size + n_static, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x_seq, x_static):
        _, (h_n, _) = self.lstm(x_seq)
        combined = torch.cat([h_n[-1], x_static], dim=1)
        return self.head(combined).squeeze(-1)


class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, agg):
        return torch.relu(self.linear(agg))


def graph_mix_for_batch(X_wide_t, A_norm, t_idx_batch, w_idx_batch):
    gathered = X_wide_t[t_idx_batch]
    A_rows = A_norm[w_idx_batch]
    return torch.einsum("bn,blnd->bld", A_rows, gathered)


def train_sequence_model(df: pd.DataFrame, use_graph: bool, target_window_days: int = 1):
    label = "LSTM+GNN" if use_graph else "LSTM"
    task_suffix = "" if target_window_days == 1 else f" ({target_window_days}d window)"
    print("\n" + "=" * 60)
    print(f"Training {label}{task_suffix} (ward-day grain)")
    print("=" * 60)

    ward_ids = sorted(df["ward_uid"].unique())
    ward_idx_map = {w: i for i, w in enumerate(ward_ids)}
    n_dynamic = len(DYNAMIC_FEATURE_COLS)
    n_static = len(STATIC_FEATURE_COLS)

    graph_layer = A_norm = X_wide_t = date_to_pos = None
    if use_graph:
        A_norm = build_adjacency_matrix(ward_ids)
        graph_layer = GraphConv(n_dynamic, GRAPH_HIDDEN_SIZE).to(DEVICE)
        seq_input_dim = n_dynamic + GRAPH_HIDDEN_SIZE

        print("Building full ward x day dynamic-feature tensor for graph propagation...")
        wide = df.pivot_table(index="date", columns="ward_uid", values=DYNAMIC_FEATURE_COLS, aggfunc="first")
        wide = wide.sort_index()
        dates = pd.to_datetime(wide.index)
        date_to_pos = {d: i for i, d in enumerate(dates)}
        X_wide = np.zeros((len(dates), len(ward_ids), n_dynamic), dtype=np.float32)
        for fi, feat in enumerate(DYNAMIC_FEATURE_COLS):
            sub = wide[feat].reindex(columns=ward_ids)
            X_wide[:, :, fi] = sub.fillna(sub.median()).to_numpy()
        X_wide_t = torch.tensor(X_wide, device=DEVICE)
        print(f"  Wide tensor: {X_wide_t.shape[0]} days x {X_wide_t.shape[1]} wards x {X_wide_t.shape[2]} features.")
    else:
        seq_input_dim = n_dynamic

    model = FloodLSTM(seq_input_dim, n_static, HIDDEN_SIZE).to(DEVICE)
    params = list(model.parameters()) + (list(graph_layer.parameters()) if graph_layer else [])
    optimizer = torch.optim.Adam(params, lr=LEARNING_RATE)

    train_X, train_Xs, train_y, train_tidx, train_widx = [], [], [], [], []
    val_windows, eval_windows = [], []

    val_start_year = TEST_START_YEAR - VAL_YEARS_BEFORE_TEST

    for w in ward_ids:
        wdf = df[df["ward_uid"] == w].sort_values("date").reset_index(drop=True)
        if len(wdf) < LOOKBACK_DAYS + 2:
            continue
        X_dyn, y_lab, yrs, starts = make_sequences(wdf, DYNAMIC_FEATURE_COLS, window_days=target_window_days)
        if X_dyn is None:
            continue
        static_vals = wdf[STATIC_FEATURE_COLS].iloc[0].to_numpy(dtype=np.float32)
        X_static = np.tile(static_vals, (len(X_dyn), 1))

        if use_graph:
            t_idx = date_positions_for_windows(wdf, starts, date_to_pos)
            w_idx = np.full(len(X_dyn), ward_idx_map[w], dtype=np.int64)
        else:
            t_idx = w_idx = None

        # Three chronological slices, no overlap: TRAIN (gradient updates)
        # < val_start_year <= VAL (threshold selection only, never seen by
        # the optimizer) < TEST_START_YEAR <= TEST (final, untouched score).
        is_train = yrs < val_start_year
        is_val = (yrs >= val_start_year) & (yrs < TEST_START_YEAR)
        is_test = yrs >= TEST_START_YEAR

        if is_train.any():
            train_X.append(X_dyn[is_train]); train_Xs.append(X_static[is_train]); train_y.append(y_lab[is_train])
            if use_graph:
                train_tidx.append(t_idx[is_train]); train_widx.append(w_idx[is_train])
        if is_val.any():
            val_windows.append((
                w, wdf["region"].iloc[0], X_dyn[is_val], X_static[is_val], yrs[is_val],
                (t_idx[is_val] if use_graph else None), (w_idx[is_val] if use_graph else None),
                y_lab[is_val],
            ))
        if is_test.any():
            eval_windows.append((
                w, wdf["region"].iloc[0], X_dyn[is_test], X_static[is_test], yrs[is_test],
                (t_idx[is_test] if use_graph else None), (w_idx[is_test] if use_graph else None),
                y_lab[is_test],
            ))

    train_X = np.concatenate(train_X); train_Xs = np.concatenate(train_Xs); train_y = np.concatenate(train_y)
    if use_graph:
        train_tidx_arr = np.concatenate(train_tidx)
        train_widx_arr = np.concatenate(train_widx)
    print(f"Training sequences: {len(train_X)} (lookback={LOOKBACK_DAYS} days, stride={STRIDE_DAYS}) "
          f"across {len(ward_ids)} wards. Validation years: [{val_start_year}, {TEST_START_YEAR}) "
          f"held out from gradient updates, used only for threshold selection.")

    # Minority-class oversampling, TRAINING split only (val/test untouched).
    #
    # Plain LSTM: real SMOTE. Each (lookback x n_dynamic) window is
    # flattened together with its static features into one vector, SMOTE
    # interpolates new minority-class vectors in that flattened space, and
    # the result is reshaped back into (window, static) pairs. This is the
    # standard "flatten trick" for applying SMOTE to sequence data.
    #
    # LSTM+GNN: NOT true SMOTE. The graph-mixing step looks up each
    # training window's REAL date/ward position in a wide tensor
    # (X_wide_t) to gather neighboring wards' features -- a SMOTE-
    # synthesized (interpolated) window has no such real position, so it
    # cannot be graph-mixed. Instead this duplicates REAL minority-class
    # windows (with their real, valid t_idx/w_idx) up to the same target
    # ratio -- oversampling, not synthetic generation, because the
    # architecture requires every training row to correspond to an actual
    # day/ward.
    print(f"  Fit positive rate before oversampling: {train_y.mean():.4f}")
    if use_graph:
        pos_idx = np.where(train_y == 1)[0]
        neg_count = int((train_y == 0).sum())
        target_pos_count = int(SMOTE_MINORITY_RATIO * neg_count)
        n_to_add = max(target_pos_count - len(pos_idx), 0)
        if n_to_add > 0 and len(pos_idx) > 0:
            rng = np.random.RandomState(42)
            dup_idx = rng.choice(pos_idx, size=n_to_add, replace=True)
            train_X = np.concatenate([train_X, train_X[dup_idx]])
            train_Xs = np.concatenate([train_Xs, train_Xs[dup_idx]])
            train_y = np.concatenate([train_y, train_y[dup_idx]])
            train_tidx_arr = np.concatenate([train_tidx_arr, train_tidx_arr[dup_idx]])
            train_widx_arr = np.concatenate([train_widx_arr, train_widx_arr[dup_idx]])
        print(f"  Fit positive rate after real-sample oversampling "
              f"(target ratio={SMOTE_MINORITY_RATIO}, LSTM+GNN uses duplication "
              f"of real windows, not synthetic SMOTE -- see comment above): "
              f"{train_y.mean():.4f} ({len(train_y)} rows)")
    else:
        n_win, lookback, n_dyn = train_X.shape
        flat = np.concatenate([train_X.reshape(n_win, -1), train_Xs], axis=1)
        smote = SMOTE(sampling_strategy=SMOTE_MINORITY_RATIO, random_state=42)
        flat_res, train_y = smote.fit_resample(flat, train_y)
        n_static = train_Xs.shape[1]
        train_X = flat_res[:, :lookback * n_dyn].reshape(-1, lookback, n_dyn).astype(np.float32)
        train_Xs = flat_res[:, lookback * n_dyn:].astype(np.float32)
        print(f"  Fit positive rate after SMOTE (target ratio={SMOTE_MINORITY_RATIO}): "
              f"{train_y.mean():.4f} ({len(train_y)} rows)")

    # Class-imbalance fix #1: weight the positive (flooded) class in the
    # loss itself, ON TOP OF the oversampling above. Flood-days are ~3% of
    # rows before oversampling -- an unweighted BCEWithLogitsLoss lets
    # "always predict not-flooded" sit at a ~97% accuracy local optimum,
    # which is exactly the degenerate solution this model was previously
    # converging to (precision=0.011, recall=0.002). pos_weight is
    # recomputed AFTER oversampling, from the resulting (less imbalanced,
    # but still not 1:1) training distribution.
    n_pos = train_y.sum()
    n_neg = len(train_y) - n_pos
    pos_weight_value = float(n_neg / max(n_pos, 1))
    print(f"Training positive rate: {n_pos / len(train_y):.4f} -- using "
          f"pos_weight={pos_weight_value:.2f} in BCEWithLogitsLoss.")
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_value, device=DEVICE))

    train_X_t = torch.tensor(train_X, device=DEVICE)
    train_Xs_t = torch.tensor(train_Xs, device=DEVICE)
    train_y_t = torch.tensor(train_y, device=DEVICE)
    if use_graph:
        train_tidx_t = torch.tensor(train_tidx_arr, device=DEVICE)
        train_widx_t = torch.tensor(train_widx_arr, device=DEVICE)

    n = len(train_X_t)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        if use_graph:
            graph_layer.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        for start in range(0, n, BATCH_SIZE):
            batch_idx = perm[start:start + BATCH_SIZE]
            x_seq = train_X_t[batch_idx]
            x_static = train_Xs_t[batch_idx]
            y_batch = train_y_t[batch_idx]
            if use_graph:
                agg = graph_mix_for_batch(X_wide_t, A_norm, train_tidx_t[batch_idx], train_widx_t[batch_idx])
                graph_component = graph_layer(agg)
                x_seq_final = torch.cat([x_seq, graph_component], dim=-1)
            else:
                x_seq_final = x_seq
            optimizer.zero_grad()
            pred = model(x_seq_final, x_static)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_idx)
        print(f"  Epoch {epoch}/{EPOCHS} -- avg BCE loss: {total_loss / n:.4f}")

    model.eval()
    if use_graph:
        graph_layer.eval()

    def _run_inference(windows):
        pred_rows_, true_, prob_ = [], [], []
        with torch.no_grad():
            for w, region, X_dyn, X_static, yrs, t_idx, w_idx, y_true_lab in windows:
                x_seq = torch.tensor(X_dyn, device=DEVICE)
                x_static_t = torch.tensor(X_static, device=DEVICE)
                if use_graph:
                    t_idx_t = torch.tensor(t_idx, device=DEVICE)
                    w_idx_t = torch.tensor(w_idx, device=DEVICE)
                    agg = graph_mix_for_batch(X_wide_t, A_norm, t_idx_t, w_idx_t)
                    x_seq = torch.cat([x_seq, graph_layer(agg)], dim=-1)
                probs = torch.sigmoid(model(x_seq, x_static_t)).cpu().numpy()
                for p, y in zip(probs, yrs):
                    pred_rows_.append({"ward_uid": w, "region": region, "year": int(y), "predicted_prob": p})
                true_.append(y_true_lab)
                prob_.append(probs)
        return pred_rows_, true_, prob_

    # Class-imbalance fix #2: pick the decision threshold on VALIDATION
    # data (years before TEST_START_YEAR, never used for gradient
    # updates), not a blind 0.5 -- with a ~3% positive rate, 0.5 is not a
    # meaningful cutoff and was exactly what produced the near-zero
    # precision/recall previously. Threshold is the one maximizing F1 on
    # the validation precision-recall curve, then applied AS-IS to the
    # test set -- test labels are never used to choose it.
    _, val_true_list, val_prob_list = _run_inference(val_windows)
    val_true = np.concatenate(val_true_list).astype(int)
    val_prob = np.concatenate(val_prob_list)
    precisions, recalls, thresholds = precision_recall_curve(val_true, val_prob)
    f1_scores = np.where(
        (precisions[:-1] + recalls[:-1]) > 0,
        2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-12),
        0.0,
    )
    if len(thresholds) > 0 and f1_scores.max() > 0:
        best_idx = f1_scores.argmax()
        decision_threshold = float(thresholds[best_idx])
        print(f"  Validation-selected decision threshold (max F1={f1_scores[best_idx]:.3f} "
              f"at threshold={decision_threshold:.3f}, N={len(val_true)} val ward-days) "
              f"-- replaces blind 0.5 cutoff.")
    else:
        decision_threshold = 0.5
        print("  Validation set had no positive predictions to tune a threshold from -- "
              "falling back to 0.5.")

    pred_rows, daily_true_list, daily_prob_list = _run_inference(eval_windows)

    # ---- Daily classification metrics ----
    # window=1: this model's NATIVE task ("did this region flood on this
    # day?"). window>1: "will this region flood within the next N days?",
    # same reframing (and same reasoning -- see FLOOD_WINDOW_DAYS_LIST)
    # applied to LR/RF/HGB. Scored at the validation-selected threshold.
    daily_true = np.concatenate(daily_true_list).astype(int)
    daily_prob_concat = np.concatenate(daily_prob_list)
    daily_pred = (daily_prob_concat >= decision_threshold).astype(int)
    daily_cm = confusion_matrix(daily_true, daily_pred, labels=[0, 1])
    task_label = (
        "flooded today? (native), daily ward-day grain" if target_window_days == 1
        else f"flood in next {target_window_days} day(s), daily ward-day grain"
    )
    pr_auc = average_precision_score(daily_true, daily_prob_concat)
    daily_clf = {
        "n": len(daily_true),
        "accuracy": accuracy_score(daily_true, daily_pred),
        "precision": precision_score(daily_true, daily_pred, zero_division=0),
        "recall": recall_score(daily_true, daily_pred, zero_division=0),
        "f1": f1_score(daily_true, daily_pred, zero_division=0),
        "confusion_matrix": daily_cm,
        "task": task_label,
        "pr_auc": pr_auc,
    }
    print(f"\n  Daily classification metrics ({task_label}, held-out ward-days, "
          f"threshold={decision_threshold:.3f}, PR-AUC={pr_auc:.3f}):")
    print_confusion_matrix(f"  {label}{task_suffix} (daily grain)", daily_clf)

    if target_window_days != 1:
        # Aggregating "will it flood in the next N days" probabilities
        # into an annual flood-DAY count doesn't have valid semantics --
        # overlapping N-day windows would each "vote" for the same real
        # flood day, inflating the sum well past the true day count. The
        # region-year regression comparison stays exclusively on the
        # window_days=1 (native) LSTM/LSTM+GNN run.
        return None, daily_clf

    pred_df = pd.DataFrame(pred_rows)
    pred_annual = (
        pred_df.groupby(["ward_uid", "region", "year"])["predicted_prob"]
        .sum().mul(STRIDE_DAYS).reset_index(name="predicted_flood_days")
    )
    return pred_annual, daily_clf


# ---------------------------------------------------------------------
# 8. EVALUATE (region-year headline, ward-level reference)
# ---------------------------------------------------------------------

def evaluate_against_common_target(pred_annual: pd.DataFrame, threshold: float):
    actual = load_corrected_annual_labels().rename(columns={COMMON_TARGET: "actual_lookup"})
    merged = pred_annual.merge(actual, on=["region", "year"], how="left")
    merged["actual_lookup"] = merged["actual_lookup"].fillna(0)

    rows = merged.rename(columns={"actual_lookup": "actual", "predicted_flood_days": "predicted"})[
        ["region", "ward_uid", "year", "actual", "predicted"]
    ]
    metrics, collapsed = compute_regionyear_metrics(rows)
    print(f"  [Region-year, TRUE N={len(collapsed)}] -> {metrics}  <- headline")

    clf = classification_metrics_from_regionyear(collapsed, threshold)
    print_confusion_matrix("  (this model, region-year classification)", clf)

    return metrics, rows, clf


# ---------------------------------------------------------------------
# 9. OUTPUT
# ---------------------------------------------------------------------

def save_comparison(results: dict, suffix: str = ""):
    rows = [{"model": name, **metrics} for name, metrics in results.items()]
    comp_df = pd.DataFrame(rows)
    filename = f"model_comparison_metrics{suffix}.csv"
    comp_df.to_csv(filename, index=False)
    print(f"\nSaved -> {filename}")
    print(comp_df.to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    colors = ["#999999" if m == "Naive baseline" else "#378ADD" for m in comp_df["model"]]
    for ax, metric in zip(axes, ["MAE", "RMSE", "NSE"]):
        ax.bar(comp_df["model"], comp_df[metric], color=colors)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=30)
        if metric == "NSE":
            ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    fig.suptitle(f"Flood model comparison -- true region-year grain, corrected labels "
                 f"(grey = naive baseline, the floor to beat)")
    fig.tight_layout()
    chart_name = f"model_comparison_chart{suffix}.png"
    fig.savefig(chart_name, dpi=150)
    print(f"Saved -> {chart_name}")


def per_year_breakdown(all_model_rows: dict) -> pd.DataFrame:
    out_rows = []
    for model_name, rows in all_model_rows.items():
        if rows.empty:
            continue
        rows = rows.copy()
        rows["abs_err"] = (rows["actual"] - rows["predicted"]).abs()
        total_abs_err = rows["abs_err"].sum()
        for year, sub in rows.groupby("year"):
            out_rows.append({
                "model": model_name, "year": int(year), "n_rows": len(sub),
                "actual_mean": sub["actual"].mean(), "actual_max": sub["actual"].max(),
                "predicted_mean": sub["predicted"].mean(),
                "MAE": np.mean(np.abs(sub["actual"] - sub["predicted"])),
                "share_of_total_abs_error": (
                    sub["abs_err"].sum() / total_abs_err if total_abs_err > 0 else np.nan
                ),
                # NOTE: per-year NSE deliberately omitted -- at ward-level
                # grain a single year has only ~5 distinct actual values
                # (one per region, duplicated across wards), so its local
                # variance can collapse toward zero and blow NSE up to
                # meaningless magnitudes. MAE/mean columns stay reliable;
                # use compute_regionyear_metrics for a trustworthy NSE.
            })
    breakdown = pd.DataFrame(out_rows).sort_values(["model", "year"]).reset_index(drop=True)
    breakdown.to_csv("model_comparison_by_year.csv", index=False)
    print(f"\nSaved -> model_comparison_by_year.csv")
    print(breakdown.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return breakdown


# ---------------------------------------------------------------------
# 9b. CLASSIFICATION METRICS OUTPUT
# ---------------------------------------------------------------------

def save_classification_comparison(regionyear_clf: dict, daily_clf: dict, threshold: float):
    """
    regionyear_clf: {model_name: clf_dict} for all 5 models, "high flood
                     year" classification at true region-year grain.
    daily_clf:      {model_name: clf_dict} for all models trained directly
                     on daily rows (LR, RF, LSTM, LSTM+GNN), their native
                     daily flooded/not-flooded classification.
    """
    rows = []
    for name, clf in regionyear_clf.items():
        tn, fp, fn, tp = clf["confusion_matrix"].ravel()
        rows.append({
            "model": name, "task": f"high flood year (>= {threshold:.1f} days), region-year grain",
            "n": clf["n"], "accuracy": clf["accuracy"], "precision": clf["precision"],
            "recall": clf["recall"], "f1": clf["f1"], "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        })
    for name, clf in daily_clf.items():
        tn, fp, fn, tp = clf["confusion_matrix"].ravel()
        rows.append({
            "model": name, "task": clf.get("task", "flooded today? (native), daily ward-day grain"),
            "n": clf["n"], "accuracy": clf["accuracy"], "precision": clf["precision"],
            "recall": clf["recall"], "f1": clf["f1"], "TN": tn, "FP": fp, "FN": fn, "TP": tp,
            "pr_auc": clf.get("pr_auc", np.nan),
        })
    df = pd.DataFrame(rows)
    df.to_csv("model_classification_metrics.csv", index=False)
    print(f"\nSaved -> model_classification_metrics.csv")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nNOTE on 'accuracy' for the daily task: held-out days are heavily")
    print("imbalanced (most days have no flood), so a high accuracy alone can be")
    print("misleading -- e.g. a model that always predicts 'no flood' scores high")
    print("accuracy but zero recall. Precision/recall/F1 are the metrics that")
    print("actually reveal whether a model is doing anything useful here.")
    return df


# ---------------------------------------------------------------------
# 10. MAIN
# ---------------------------------------------------------------------

def main():
    if QUICK_TEST_MODE:
        print("!" * 60)
        print("QUICK_TEST_MODE = True -- fast pipeline validation run, NOT final numbers.")
        print("!" * 60 + "\n")

    (lr_metrics, rf_metrics, naive_metrics, lr_rows, rf_rows, naive_rows,
     threshold, lr_clf, rf_clf, naive_clf) = train_lr_rf()

    ward_day_df = build_ward_day_in_memory()

    window_daily_clf = {}
    for window_days in FLOOD_WINDOW_DAYS_LIST:
        window_daily_clf.update(train_daily_window_models(ward_day_df, window_days))

    lstm_preds, lstm_daily_clf = train_sequence_model(ward_day_df, use_graph=False)
    print("\nEvaluating LSTM against common annual target...")
    lstm_metrics, lstm_rows, lstm_clf = evaluate_against_common_target(lstm_preds, threshold)

    gnn_preds, gnn_daily_clf = train_sequence_model(ward_day_df, use_graph=True)
    print("\nEvaluating LSTM+GNN against common annual target...")
    gnn_metrics, gnn_rows, gnn_clf = evaluate_against_common_target(gnn_preds, threshold)

    # LSTM/LSTM+GNN "flood within next N days" variants -- same reframing
    # as LR/RF/HGB above, for the daily classification comparison only
    # (region-year regression numbers stay on the native run above -- see
    # LSTM_RUN_WINDOW_VARIANTS comment for why). Adds ~2x LSTM training
    # time on top of the native run above.
    lstm_window_daily_clf = {}
    if LSTM_RUN_WINDOW_VARIANTS:
        for window_days in FLOOD_WINDOW_DAYS_LIST:
            _, lstm_w_clf = train_sequence_model(ward_day_df, use_graph=False, target_window_days=window_days)
            lstm_window_daily_clf[f"LSTM ({window_days}d)"] = lstm_w_clf
            _, gnn_w_clf = train_sequence_model(ward_day_df, use_graph=True, target_window_days=window_days)
            lstm_window_daily_clf[f"LSTM+GNN ({window_days}d)"] = gnn_w_clf

    results = {
        "Naive baseline": naive_metrics,
        "Linear Regression": lr_metrics,
        "Random Forest": rf_metrics,
        "LSTM": lstm_metrics,
        "LSTM+GNN": gnn_metrics,
    }
    save_comparison(results)

    all_model_rows = {
        "Naive baseline": naive_rows, "Linear Regression": lr_rows,
        "Random Forest": rf_rows, "LSTM": lstm_rows, "LSTM+GNN": gnn_rows,
    }
    per_year_breakdown(all_model_rows)

    regionyear_clf = {
        "Naive baseline": naive_clf, "Linear Regression": lr_clf,
        "Random Forest": rf_clf, "LSTM": lstm_clf, "LSTM+GNN": gnn_clf,
    }
    daily_clf = {
        **window_daily_clf,
        "LSTM": lstm_daily_clf, "LSTM+GNN": gnn_daily_clf,
        **lstm_window_daily_clf,
    }
    save_classification_comparison(regionyear_clf, daily_clf, threshold)

    print("\n" + "=" * 60)
    print("READING THESE NUMBERS")
    print("=" * 60)
    print("1. 'Naive baseline' predicts each region's own training-period mean --")
    print("   it is the floor every real model needs to beat. If a model's NSE")
    print("   is below the baseline's, it is adding negative value.")
    print("2. All regression metrics are TRUE region-year grain (N ~= 30 test rows")
    print("   across 5 regions), not ward-year (which inflates N ~60x via duplication).")
    print("3. 2020's flood-days are corrected via a data-driven per-event")
    print("   duration cap (14 days, the natural break in the real event-length")
    print("   distribution), not an arbitrary year-specific patch -- see")
    print("   build_corrected_flood_labels.py docstring for the evidence.")
    print("4. LR/RF features were narrowed from 92 to whichever pass the")
    print("   within-region-signal check above -- see the printed selection.")
    print("5. Classification metrics (accuracy/precision/recall/F1/confusion")
    print("   matrix) come in two flavors: 'high flood year' at region-year grain")
    print("   (N~30, same threshold applied to every model, derived from the")
    print("   training data's own median) for all five models, and a daily-grain")
    print("   flooded/not-flooded task (N in the thousands, held-out ward-days).")
    print("   LR/RF/HistGradientBoosting/Stack predict 'flood within next N days'")
    print("   for each N in FLOOD_WINDOW_DAYS_LIST -- see train_daily_window_models")
    print("   for why this beat exact-day prediction. LSTM/LSTM+GNN keep the")
    print("   native exact-day task. PR-AUC is also reported per model -- more")
    print("   robust than a single F1 point since it doesn't depend on one threshold.")


if __name__ == "__main__":
    main()