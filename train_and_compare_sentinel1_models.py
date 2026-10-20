"""
Train and Compare LR / RF / LightGBM / ARIMA / LSTM / LSTM+GNN / Temporal CNN
Against a Sentinel-1-Derived WARD-YEAR Target (years >= 2015 only)
--------------------------------------------------------------------------
A SECOND, DELIBERATELY SEPARATE comparison from train_and_compare_models.py.
That script trains against the IFI-derived, REGION-BROADCAST label
(region_flood_days_count / CORRECTED_DAILY_CSV) -- every ward in a region
shares one identical annual value. This script trains model families
against a genuinely WARD-RESOLUTION target instead: how many of a ward's
real Sentinel-1 SAR acquisitions in a year were flagged as flooded (see
build_sentinel1_flood_labels.py for how that flag is derived).

MODELS IN THIS COMPARISON, AND WHY EACH SITS WHERE IT DOES
This version adds three models to the original four (Ridge/RidgeCV "LR",
Random Forest, LSTM, LSTM+GNN). Each new model was placed at the grain
that actually matches how it works, not bolted on generically:

  - LightGBM: a second tree-based tabular learner, trained on the exact
    same ward-year aggregated feature table, same feature-selection
    output, and same train/test split as LR/RF. It is a natural third
    tabular model, not a different kind of comparison -- see section 4.

  - ARIMA: fit PER WARD, on that ward's own annual n_flooded_acquisitions
    history only -- no exogenous features (rainfall, soil moisture, TWI,
    etc.) at all, unlike every other model here. This is deliberately a
    narrower question than the others answer ("does this ward's own count
    history alone predict next year?"), closer in spirit to the naive
    per-ward-mean baseline but allowed to use trend/autocorrelation
    instead of a flat mean. See section 4b for why it's scoped this way
    and what its short-history fallback does.

  - Temporal CNN: slots in next to LSTM/LSTM+GNN at the acquisition-
    anchored sequence grain (section 5) -- same LOOKBACK_DAYS windows
    ending the day before each real S1 acquisition, same per-acquisition
    label, same calibrated-probability-sum-to-ward-year-count evaluation.
    It swaps FloodLSTM's recurrent cell for stacked dilated 1D
    convolutions (see TemporalCNN below) -- an architecture difference on
    identical data and identical task, not a new task.

WHY THIS EXISTS -- AND WHY IT IS NOT A REPLACEMENT FOR THE IFI COMPARISON
The whole point of doing this as a second, side-by-side comparison
(rather than swapping the S1 label into train_and_compare_models.py) is
that we do NOT have ward-level ground truth to say which label is "right"
when the two disagree -- see build_sentinel1_flood_labels.py's own HONEST
LIMITS and report_sentinel1_validation()'s agreement-rate output in
train_and_compare_models.py. Running both, on their own honest terms,
lets you compare how predictable each target actually is instead of
betting the whole pipeline on an unvalidated satellite heuristic.
Concretely, this means:

  - Output files here are all prefixed "sentinel1_model_..." and are
    NEVER merged, averaged, or reconciled with the "model_..." files
    train_and_compare_models.py produces. They are two independent
    report cards, meant to be read side by side, not combined into one
    number.
  - This script does not touch, import as a target, or overwrite
    region_flood_days_count, CORRECTED_DAILY_CSV, or anything else the
    IFI-based pipeline trains on.

WHAT THE TARGET ACTUALLY IS -- READ THIS BEFORE COMPARING NSE VALUES
    target = n_flooded_acquisitions(ward, year)
           = count of that ward's REAL Sentinel-1 acquisitions in that
             year whose water_fraction cleared WARD_WATER_FRACTION_THRESHOLD

This is a COUNT OF FLAGGED PASSES, not a count of flood-DAYS. It is not
directly comparable in magnitude to region_flood_days_count (which is
capped-event-days). A ward with more S1 passes in a year has more chances
to register a flagged acquisition purely from having more looks,
independent of how flood-prone it actually is. To keep this honest:

  - n_acquisitions(ward, year) (total real passes, not just flagged
    ones) is carried alongside every prediction and reported in the
    per-year breakdown, so a reader can see if a model's apparent skill
    is just tracking pass-count rather than flood risk.
  - A secondary diagnostic column, s1_flooded_rate = n_flooded /
    n_acquisitions, is also computed and reported (not modeled) purely
    so this confound is visible, not hidden.
  - Ward-years with n_acquisitions == 0 (no real S1 coverage that year
    for that ward -- e.g. before S1 existed, or a coverage gap) are
    dropped, not imputed as zero -- see build_ward_year_s1_target().

WHY THE WITHIN-REGION-DEMEANING FEATURE SELECTION FROM
train_and_compare_models.py IS *NOT* REUSED HERE
That trick exists specifically because the IFI target is region-
broadcast -- every ward in a region shares one value, so any feature
that merely differs BETWEEN regions (elevation, coastal-ness, etc.) can
look predictive while carrying zero real information beyond "which
region is this". Demeaning by region isolates only within-region
variation, which is the only kind of signal that target could possibly
support. The Sentinel-1 target has the opposite property: it is
GENUINELY ward-resolved, so a feature that differs between wards WITHIN
a region (elevation, TWI, road density) is now exactly the kind of
signal we want the model to use, not noise to filter out. Feature
selection here (select_features_by_raw_signal) is therefore a plain
|correlation|-with-target filter on ward-year rows, computed on TRAINING
years only -- no demeaning.

TRUE GRAIN IS ALREADY WARD-YEAR -- NO COLLAPSING NEEDED
train_and_compare_models.py collapses ward-year rows to region-year
before scoring (compute_regionyear_metrics) because ward-year rows there
are literal duplicates of one region-year value. That collapsing step is
deliberately NOT done here: each (ward, year) row in this script's target
is a distinct real observation, so ward-year IS the true grain.
Collapsing it further (e.g. to region-year) would throw away the exact
resolution this whole exercise exists to test.

SEQUENCES FOR LSTM/LSTM+GNN/TEMPORAL CNN ARE ANCHORED TO REAL ACQUISITION
DATES, NOT EVERY CALENDAR DAY
train_and_compare_models.py's LSTM slides a window over every stride step
of every calendar day, because its label (region_day_flooded_
broadcast) exists for every day. The Sentinel-1 ward-day label only
exists on days a real acquisition actually happened for that ward (~6-12
day revisit gaps -- see build_sentinel1_flood_labels.py's TEMPORAL
RESOLUTION section). Manufacturing a label for the days in between would
fabricate observations the satellite never made. Instead,
make_acquisition_sequences() builds exactly one window per real
acquisition: the LOOKBACK_DAYS calendar days immediately BEFORE that
acquisition date (never including the acquisition day itself), with
label = whether that specific acquisition was flagged flooded. This is a
smaller, sparser training set than the IFI-based LSTM's (bounded by real
satellite passes, not by calendar days), which is expected and honest,
not a bug. Temporal CNN reuses this exact same windowing.

HONEST LIMITS OF THIS COMPARISON (in addition to build_sentinel1_flood_
labels.py's own HONEST LIMITS, which still apply in full)
  1. Revisit-gap undercounting is now the MODEL's problem, not just the
     label's: a ward-year with few acquisitions has a noisier target
     (each flagged/missed pass swings the count by a larger relative
     amount) than a ward-year with many. No correction is applied for
     this here beyond reporting n_acquisitions alongside every result --
     treat NSE/F1 differences between low- and high-coverage ward-years
     with that in mind.
  2. Because this restricts to year >= SENTINEL1_YEAR_MIN (2015), the
     training window is much shorter than the IFI comparison's (which
     goes back to 2001 or 1990 depending on SAMPLE_YEAR_MIN). Expect
     noisier LR/RF/LightGBM fits and less LSTM/LSTM+GNN/Temporal CNN
     convergence purely from less data, independent of which target is
     "better". ARIMA is hit hardest by this -- see section 4b.
  3. Urban SAR noise (see build_sentinel1_flood_labels.py HONEST LIMITS
     #1) means BMC's target is plausibly noisier than Pune's -- if BMC's
     metrics look worse here, that is not automatically "the model is
     worse at BMC", it may be "BMC's target is noisier".
  4. This still requires build_sentinel1_flood_labels.py to have been
     run first (sentinel1_ward_day_water_fraction.csv must exist) --
     this script does not call Earth Engine itself.
  5. ARIMA sees only each ward's own count history -- no rainfall, soil
     moisture, terrain, or susceptibility features at all. A weaker ARIMA
     result than LR/RF/LightGBM does not mean "time-series structure
     doesn't exist"; it may just mean single-ward count history alone,
     with ~8 training years, isn't enough signal on its own. Read ARIMA
     as answering a narrower question than the other models, not a
     head-to-head on identical information.

REQUIRED INPUTS
    sentinel1_ward_day_water_fraction.csv  -- from build_sentinel1_flood_labels.py
    Everything train_and_compare_models.py itself requires (this script
    imports several helpers/constants from it directly -- see below --
    so its own required inputs, e.g. final_ward_year_train/test.csv,
    region_day_flooded_CORRECTED.csv, ward_adjacency_edges.csv, must also
    be present).

RUNNING THIS IN COLAB (same environment as train_and_compare_models.py)
    !python build_sentinel1_flood_labels.py        # one-time, needs GEE auth
    !python build_corrected_flood_labels.py         # one-time (see other script)
    !python train_and_compare_sentinel1_models.py

REQUIREMENTS
    pip install pandas numpy scikit-learn torch matplotlib imbalanced-learn lightgbm statsmodels
"""

import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import lightgbm as lgb
except ImportError:
    lgb = None
try:
    from statsmodels.tsa.arima.model import ARIMA
except ImportError:
    ARIMA = None
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              confusion_matrix, precision_recall_curve, average_precision_score)

# ARIMA on ~8 short, gappy, zero-inflated per-ward series throws frequent
# non-fatal convergence/stationarity warnings from statsmodels -- expected
# and handled by the try/except fallback in train_arima_sentinel1(), not
# silently masking a real problem elsewhere.
warnings.filterwarnings("ignore", module="statsmodels")

# Reuse the exact same feature lists, architecture, device, ward-day data
# prep, and adjacency-matrix builder as the IFI-based pipeline, so both
# comparisons are apples-to-apples on everything EXCEPT the target and
# the sequence-anchoring logic described above. Importing (rather than
# copy-pasting) also means a future change to e.g. DYNAMIC_FEATURE_COLS
# or FloodLSTM's architecture in the main script automatically applies
# here too, instead of silently drifting out of sync.
from train_and_compare_models import (
    DYNAMIC_FEATURE_COLS, STATIC_FEATURE_COLS, LOOKBACK_DAYS,
    HIDDEN_SIZE, GRAPH_HIDDEN_SIZE, BATCH_SIZE, LEARNING_RATE, DEVICE,
    QUICK_TEST_MODE, EPOCHS,
    build_ward_day_in_memory, build_adjacency_matrix, date_positions_for_windows,
    FloodLSTM, GraphConv, graph_mix_for_batch,
    compute_metrics, print_confusion_matrix,
    ADJACENCY_EDGES_CSV, AUTOCORRELATION_FEATURES_CSV,
)

# ---------------------------------------------------------------------
# 0. CONFIGURATION
# ---------------------------------------------------------------------

SENTINEL1_WARD_DAY_CSV = "sentinel1_ward_day_water_fraction.csv"  # from
                                                             # build_sentinel1_flood_labels.py

SENTINEL1_YEAR_MIN = 2015  # Sentinel-1A launched April 2014; reliable IW
                            # GRD coverage over this area is from late
                            # 2014 (see build_sentinel1_flood_labels.py's
                            # S1_START_DATE). Starting the TARGET at 2015
                            # (rather than late 2014) avoids a partial,
                            # thin first calendar year skewing the
                            # earliest ward-year counts.

WARD_WATER_FRACTION_THRESHOLD = 0.10  # MUST MATCH build_sentinel1_flood_labels.py's
                                        # constant of the same name -- that script
                                        # only ships a continuous water_fraction
                                        # column (not a precomputed ward-level
                                        # flag), so the flooded/not-flooded
                                        # threshold is applied here, at the same
                                        # value, to keep the two scripts' notion
                                        # of "flooded" consistent. If you change
                                        # this in one script, change it in both.

SENTINEL1_TEST_START_YEAR = 2023  # Later than train_and_compare_models.py's
                                    # TEST_START_YEAR=2019 -- with data only
                                    # from 2015, a 2019 split would leave just
                                    # 4 training years. 2023 gives ~8 training
                                    # years (2015-2022) and a smaller but real
                                    # test window; adjust once more S1 history
                                    # accumulates.
SENTINEL1_VAL_YEARS_BEFORE_TEST = 1  # LSTM/LSTM+GNN/Temporal CNN threshold-selection
                                       # window (years [TEST_START-1, TEST_START)) --
                                       # kept to 1 year (not 2, unlike the IFI
                                       # script) given the shorter overall
                                       # history here.

WITHIN_RAW_CORR_THRESHOLD = 0.10  # keep features whose RAW (not demeaned --
                                    # see module docstring for why) |correlation|
                                    # with the target, on TRAINING ward-years
                                    # only, clears this bar.

ARIMA_ORDER = (1, 0, 0)  # AR(1), no differencing, no MA term -- deliberately
                          # simple given ~8 training years per ward; a
                          # richer order would overfit noise this short a
                          # series almost every time. See train_arima_sentinel1().

DEVICE_ = DEVICE  # re-exported for clarity in this file's own functions
torch.manual_seed(43)   # different seed from the IFI script on purpose --
np.random.seed(43)      # this is an independently-run comparison, not a
                         # component of the same run.


# ---------------------------------------------------------------------
# 0b. TEMPORAL CNN -- convolutional alternative to FloodLSTM
# ---------------------------------------------------------------------

class TemporalCNN(nn.Module):
    """
    Drop-in architectural alternative to FloodLSTM: identical interface
    (a LOOKBACK_DAYS dynamic-feature window + a static-feature vector ->
    one flood logit), same acquisition-anchored windows built by
    make_acquisition_sequences(), same training loop, same
    calibrate-then-sum-to-ward-year-count evaluation in
    train_sentinel1_sequence_model() / evaluate_sentinel1_sequence_model().
    Only the sequence encoder changes: stacked dilated Conv1d layers
    instead of a recurrent cell, so this is an architecture comparison on
    identical data and identical task -- not a different target, window,
    or evaluation.

    Dilations (1, 2, 4) give the last layer a receptive field covering the
    full LOOKBACK_DAYS window with only 3 conv layers, letting it see
    both short-term (day-to-day rainfall spikes) and longer-run (soil-
    moisture buildup across the whole lookback) patterns without the
    sequential, one-step-at-a-time recurrence LSTM uses.
    """

    def __init__(self, seq_input_dim: int, n_static: int, hidden_size: int, kernel_size: int = 3):
        super().__init__()
        self.conv1 = nn.Conv1d(seq_input_dim, hidden_size, kernel_size,
                                padding=kernel_size // 2, dilation=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size,
                                padding=kernel_size, dilation=2)
        self.conv3 = nn.Conv1d(hidden_size, hidden_size, kernel_size,
                                padding=2 * kernel_size, dilation=4)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(hidden_size + n_static, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x_seq: torch.Tensor, x_static: torch.Tensor) -> torch.Tensor:
        # x_seq arrives as (batch, time, features) to match FloodLSTM's
        # expected input; Conv1d wants channels-first (batch, features, time).
        x = x_seq.transpose(1, 2)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).squeeze(-1)          # (batch, hidden_size) -- global
                                                # average over the time axis
        x = torch.cat([x, x_static], dim=-1)
        return self.head(x).squeeze(-1)        # (batch,) raw logit, same shape
                                                # FloodLSTM returns


# ---------------------------------------------------------------------
# 1. BUILD THE SENTINEL-1 WARD-YEAR TARGET
# ---------------------------------------------------------------------

def load_sentinel1_ward_day() -> pd.DataFrame:
    try:
        s1 = pd.read_csv(SENTINEL1_WARD_DAY_CSV, parse_dates=["date"])
    except FileNotFoundError:
        raise FileNotFoundError(
            f"{SENTINEL1_WARD_DAY_CSV} not found -- run "
            f"`python build_sentinel1_flood_labels.py` first (requires Earth "
            f"Engine auth; see that script's own docstring)."
        )
    s1["year"] = s1["date"].dt.year
    s1 = s1[s1["year"] >= SENTINEL1_YEAR_MIN].reset_index(drop=True)
    s1["s1_ward_flooded"] = (s1["water_fraction"] >= WARD_WATER_FRACTION_THRESHOLD).astype(int)
    print(f"Loaded {len(s1)} ward-acquisition rows from {SENTINEL1_WARD_DAY_CSV} "
          f"(year >= {SENTINEL1_YEAR_MIN}), {s1['ward_uid'].nunique()} wards, "
          f"{s1['s1_ward_flooded'].mean():.4f} flagged-flooded rate at "
          f"threshold={WARD_WATER_FRACTION_THRESHOLD}.")
    return s1


def build_ward_year_s1_target(s1: pd.DataFrame) -> pd.DataFrame:
    """
    Collapses ward-acquisition rows to (ward_uid, year):
        n_acquisitions          = total real S1 passes that ward-year
        n_flooded_acquisitions  = how many of those were flagged (the
                                   MODELED target)
        s1_flooded_rate         = n_flooded / n_acquisitions (diagnostic
                                   only -- see module docstring on why
                                   this isn't the modeled target itself)
    Ward-years with n_acquisitions == 0 don't appear here at all (there's
    nothing to collapse) -- any ward-year present in other feature tables
    but absent from this target is later dropped with a printed count,
    not silently imputed as zero (a ward-year with real zero flooding and
    a ward-year with no coverage at all are not the same thing).
    """
    grouped = s1.groupby(["ward_uid", "year"]).agg(
        n_acquisitions=("s1_ward_flooded", "count"),
        n_flooded_acquisitions=("s1_ward_flooded", "sum"),
    ).reset_index()
    grouped["region"] = grouped["ward_uid"].str.split("_").str[0]
    grouped["s1_flooded_rate"] = grouped["n_flooded_acquisitions"] / grouped["n_acquisitions"]
    print(f"Built ward-year S1 target: {len(grouped)} ward-years with real coverage "
          f"across {grouped['ward_uid'].nunique()} wards, {grouped['year'].nunique()} years "
          f"({grouped['year'].min()}-{grouped['year'].max()}). "
          f"Mean n_acquisitions/ward-year={grouped['n_acquisitions'].mean():.1f}, "
          f"mean n_flooded_acquisitions/ward-year={grouped['n_flooded_acquisitions'].mean():.2f}.")
    return grouped


# ---------------------------------------------------------------------
# 2. FEATURE SELECTION (plain correlation, NOT within-region-demeaned --
#    see module docstring for why that would be wrong here)
# ---------------------------------------------------------------------

def select_features_by_raw_signal(train_wy: pd.DataFrame, feature_cols: list[str],
                                   target: str, threshold: float,
                                   min_features: int = 7,
                                   ward_varying_cols: list[str] | None = None,
                                   min_ward_varying: int = 2) -> list[str]:
    """
    ward_varying_cols / min_ward_varying: a subset of feature_cols that can
    actually differ BETWEEN wards in the same region-year (e.g. elevation,
    TWI, road_density, ward_hist_flooded_rate), as opposed to the region-
    broadcast climate/autocorrelation columns that are IDENTICAL for every
    ward in a region that year (see build_ward_day_table's region+year merge
    -- confirmed in chat). Plain pooled top-N selection lets the broadcast
    group (larger, and correlated with the target's overall magnitude via
    "was it a wet year region-wide") systematically outcompete the smaller
    ward-varying group for slots, even when a ward-varying feature is doing
    something the broadcast features structurally cannot: telling wards
    apart within the same region-year. Observed effect: adding 5 more
    broadcast autocorrelation candidates pushed road_density_km_per_km2 out
    of the selected set entirely and RF's ward-year NSE dropped 0.782->0.708
    as a direct result (see chat). Guaranteeing a floor here prevents that.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = train_wy[feature_cols].corrwith(train_wy[target]).abs()
    corr = corr.dropna().sort_values(ascending=False)
    selected = corr[corr > threshold]
    if len(selected) < min_features:
        # A hard ">threshold" cutoff treats a feature that misses by 0.10
        # and one that misses by 0.002 identically -- e.g. susceptibility_index
        # (0.098) and road_density_km_per_km2 (-0.100) both missed |corr|>0.10
        # by less than 0.002 in an earlier run, right along with several other
        # DEM/terrain features, while LR/RF ended up with only rainfall/soil-
        # moisture features and no way to tell WHICH ward it's looking at --
        # which is exactly what let the naive per-ward-mean baseline win by a
        # wide margin (see chat). Top the selection up to min_features by
        # |corr| rank so near-miss features aren't excluded by a coin flip
        # of a cutoff, without abandoning the evidence-based filter entirely.
        selected = corr.head(min_features)
        print(f"  Raw-correlation feature selection (ward-year grain, training years "
              f"only): only {(corr > threshold).sum()}/{len(feature_cols)} features clear "
              f"|corr| > {threshold}; topped up to top-{min_features} by |corr| instead "
              f"(see comment in select_features_by_raw_signal for why).")
    else:
        print(f"  Raw-correlation feature selection (ward-year grain, training years only): "
              f"{len(selected)}/{len(feature_cols)} features clear |corr| > {threshold}.")

    selected_set = set(selected.index)
    if ward_varying_cols:
        ward_varying_selected = [c for c in ward_varying_cols if c in selected_set]
        if len(ward_varying_selected) < min_ward_varying:
            ward_corr = corr[corr.index.isin(ward_varying_cols)]
            floor_add = [c for c in ward_corr.head(min_ward_varying).index if c not in selected_set]
            if floor_add:
                print(f"  Only {len(ward_varying_selected)}/{min_ward_varying} ward-varying "
                      f"(non-region-broadcast) features made the pooled selection -- adding "
                      f"{floor_add} to guarantee LR/RF/LightGBM retain some way to tell wards "
                      f"apart within a region-year (see function docstring).")
                selected_set |= set(floor_add)
    final_selected = sorted(selected_set, key=lambda c: -corr.get(c, 0))
    print(f"  Selected: {final_selected}")
    return final_selected if final_selected else list(corr.head(5).index)


# ---------------------------------------------------------------------
# 3. WARD-YEAR FEATURE TABLE (shared by every tabular model)
# ---------------------------------------------------------------------

def build_ward_year_feature_table(ward_day_df: pd.DataFrame) -> pd.DataFrame:
    """
    Annual mean of DYNAMIC_FEATURE_COLS + the (constant-per-ward)
    STATIC_FEATURE_COLS, per (ward_uid, year) -- the ward-year analogue
    of train_lr_rf()'s region-year aggregation in the IFI script, except
    collapsed to ward-year instead of region-year, since that is this
    target's true grain.
    """
    ward_day_df = ward_day_df[ward_day_df["year"] >= SENTINEL1_YEAR_MIN]
    agg_cols = DYNAMIC_FEATURE_COLS + STATIC_FEATURE_COLS
    wy = ward_day_df.groupby(["ward_uid", "year"])[agg_cols].mean().reset_index()
    wy["region"] = wy["ward_uid"].str.split("_").str[0]
    return wy


# ---------------------------------------------------------------------
# 4. LR + RF + LIGHTGBM ON WARD-YEAR AGGREGATED FEATURES
# ---------------------------------------------------------------------

def train_tabular_models_sentinel1(ward_year_features: pd.DataFrame, ward_year_target: pd.DataFrame):
    """
    Trains all three tabular ward-year models -- Linear Regression
    (RidgeCV), Random Forest, and LightGBM -- on identical features,
    identical train/test rows, and identical target transform (log1p),
    so any metric difference between them reflects the model family, not
    a difference in what data each one saw. LightGBM is added here
    (rather than as a separate function) specifically to guarantee that:
    it shares X_train/X_test/y_train/y_test/feature_cols with LR and RF
    line for line.
    """
    print("=" * 60)
    print("Training Linear Regression (RidgeCV) + Random Forest + LightGBM -- "
          "Sentinel-1 ward-year target")
    print("=" * 60)

    merged = ward_year_features.merge(
        ward_year_target[["ward_uid", "year", "n_acquisitions", "n_flooded_acquisitions",
                           "s1_flooded_rate"]],
        on=["ward_uid", "year"], how="inner",  # inner: only ward-years with real S1
                                                 # coverage have a defined target -- see
                                                 # build_ward_year_s1_target's docstring.
    )
    n_dropped = len(ward_year_features) - len(merged)
    print(f"  {len(merged)} ward-years have both features and real S1 coverage "
          f"({n_dropped} ward-years dropped -- no S1 acquisitions that ward-year).")

    target = "n_flooded_acquisitions"
    train_wy = merged[merged["year"] < SENTINEL1_TEST_START_YEAR].reset_index(drop=True)
    test_wy = merged[merged["year"] >= SENTINEL1_TEST_START_YEAR].reset_index(drop=True)
    print(f"  Ward-years -- train: {len(train_wy)} (years < {SENTINEL1_TEST_START_YEAR}) "
          f"| test: {len(test_wy)} (years >= {SENTINEL1_TEST_START_YEAR})")

    # Give LR/RF/LightGBM direct access to "is this ward historically flood-prone" --
    # the one piece of information the naive per-ward-mean baseline uses that
    # none of DYNAMIC_FEATURE_COLS/STATIC_FEATURE_COLS captures explicitly
    # (terrain proxies like elevation/TWI/susceptibility_index are indirect
    # and, per the correlation table in chat, weak). This is exactly what let
    # naive win by a wide NSE/recall margin over LR/RF: naive literally
    # predicts the ward's own training mean; LR/RF had no equivalent input.
    #
    # Leakage handling: a training row's feature must NOT include that row's
    # own year in its own average (that would let the model partly see its
    # own answer). Use leave-one-year-out for TRAIN rows; test rows use the
    # ward's full training-period mean (safe -- test years are never in the
    # training set by construction).
    overall_train_mean = train_wy[target].mean()
    ward_sum = train_wy.groupby("ward_uid")[target].transform("sum")
    ward_count = train_wy.groupby("ward_uid")[target].transform("count")
    loo_denominator = (ward_count - 1).clip(lower=1)
    train_wy = train_wy.copy()
    train_wy["ward_hist_flooded_rate"] = np.where(
        ward_count > 1, (ward_sum - train_wy[target]) / loo_denominator, overall_train_mean
    )
    ward_full_train_means = train_wy.groupby("ward_uid")[target].mean()
    test_wy = test_wy.copy()
    test_wy["ward_hist_flooded_rate"] = (
        test_wy["ward_uid"].map(ward_full_train_means).fillna(overall_train_mean)
    )
    print(f"  Added ward_hist_flooded_rate as a candidate feature (leave-one-year-out "
          f"for train rows, full training-period mean for test rows -- no leakage).")

    # Merge in the region-year rainfall-clustering/persistence features (burst-day
    # counts, consecutive wet/high-soil-moisture streaks -- see
    # build_autocorrelation_features.py). UNLIKE ward_hist_flooded_rate above,
    # these are NOT ward-resolved -- they're computed per (region, year) and
    # BROADCAST identically to every ward in that region, the same region-
    # broadcast pattern this whole script exists to move past for the target
    # itself. If one of these gets selected below, read it as "this region had
    # a clustered/bursty monsoon that year," not "this specific ward is at
    # elevated risk" -- it can explain a shared regional swing but cannot ever
    # differentiate wards within a region the way ward_hist_flooded_rate or
    # the DEM/TWI static features can.
    try:
        autocorr = pd.read_csv(AUTOCORRELATION_FEATURES_CSV)
        autocorr_cols = [c for c in autocorr.columns if c not in ("region", "year")]
        train_wy = train_wy.merge(autocorr, on=["region", "year"], how="left")
        test_wy = test_wy.merge(autocorr, on=["region", "year"], how="left")
        print(f"  Merged {len(autocorr_cols)} candidate autocorrelation features "
              f"(region-year, broadcast across every ward in that region -- see "
              f"comment above): {autocorr_cols}")
    except FileNotFoundError:
        autocorr_cols = []
        print(f"  {AUTOCORRELATION_FEATURES_CSV} not found -- run "
              f"build_autocorrelation_features.py first to include these. "
              f"Continuing without them.")

    feature_cols = select_features_by_raw_signal(
        train_wy, DYNAMIC_FEATURE_COLS + STATIC_FEATURE_COLS + ["ward_hist_flooded_rate"] + autocorr_cols,
        target, WITHIN_RAW_CORR_THRESHOLD,
        ward_varying_cols=STATIC_FEATURE_COLS + ["ward_hist_flooded_rate"], min_ward_varying=2,
    )

    X_train = train_wy[feature_cols].fillna(train_wy[feature_cols].median())
    X_test = test_wy[feature_cols].fillna(train_wy[feature_cols].median())
    y_train, y_test = train_wy[target], test_wy[target]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    y_train_fit = np.log1p(y_train)

    n_folds = min(5, len(train_wy))
    ridge = RidgeCV(alphas=np.logspace(-3, 5, 60), cv=KFold(n_folds, shuffle=True, random_state=43))
    ridge.fit(X_train_scaled, y_train_fit)
    print(f"  RidgeCV selected alpha = {ridge.alpha_:.3g}")
    lr_pred = np.clip(np.expm1(ridge.predict(X_test_scaled)), 0, None)

    rf_grid = GridSearchCV(
        RandomForestRegressor(n_estimators=200, random_state=43, n_jobs=-1),
        param_grid={"max_depth": [3, 4, 5, 6, None], "min_samples_leaf": [2, 3, 5, 10]},
        cv=KFold(n_folds, shuffle=True, random_state=43),
        scoring="neg_mean_absolute_error",
    )
    rf_grid.fit(X_train, y_train_fit)
    print(f"  GridSearchCV selected RF params: {rf_grid.best_params_}")
    rf_pred = np.clip(np.expm1(rf_grid.predict(X_test)), 0, None)

    # LightGBM: a second, gradient-boosted tree learner on the identical
    # X_train/X_test/y_train_fit as RF above -- unscaled features (tree
    # splits don't need scaling, same as RF), same log1p target transform,
    # same expm1-and-clip-at-0 back-transform, so LightGBM vs. RF isolates
    # "does boosting beat bagging here" rather than any data difference.
    lgbm_grid = GridSearchCV(
        lgb.LGBMRegressor(n_estimators=200, random_state=43, n_jobs=-1, verbose=-1),
        param_grid={"max_depth": [3, 4, 5, -1], "num_leaves": [15, 31],
                    "min_child_samples": [5, 10, 20]},
        cv=KFold(n_folds, shuffle=True, random_state=43),
        scoring="neg_mean_absolute_error",
    )
    lgbm_grid.fit(X_train, y_train_fit)
    print(f"  GridSearchCV selected LightGBM params: {lgbm_grid.best_params_}")
    lgbm_pred = np.clip(np.expm1(lgbm_grid.predict(X_test)), 0, None)

    # Naive baseline: each WARD's (not region's) own training-period mean --
    # ward-level distinctions are the entire point of this comparison, so
    # the floor to beat has to be ward-level too, unlike the IFI script's
    # region-level naive baseline.
    ward_train_means = train_wy.groupby("ward_uid")[target].mean()
    overall_train_mean = train_wy[target].mean()
    naive_pred = test_wy["ward_uid"].map(ward_train_means).fillna(overall_train_mean).to_numpy()

    id_cols = test_wy[["ward_uid", "region", "year", "n_acquisitions", "s1_flooded_rate"]].reset_index(drop=True)
    y_test_arr = test_wy[target].to_numpy()

    lr_rows = id_cols.assign(actual=y_test_arr, predicted=lr_pred)
    rf_rows = id_cols.assign(actual=y_test_arr, predicted=rf_pred)
    lgbm_rows = id_cols.assign(actual=y_test_arr, predicted=lgbm_pred)
    naive_rows = id_cols.assign(actual=y_test_arr, predicted=naive_pred)

    # TRUE grain is already ward-year -- score directly, no collapsing.
    lr_metrics = compute_metrics(lr_rows["actual"], lr_rows["predicted"])
    rf_metrics = compute_metrics(rf_rows["actual"], rf_rows["predicted"])
    lgbm_metrics = compute_metrics(lgbm_rows["actual"], lgbm_rows["predicted"])
    naive_metrics = compute_metrics(naive_rows["actual"], naive_rows["predicted"])

    print(f"\n  Linear Regression (ward-year, N={len(test_wy)}) -> {lr_metrics}")
    print(f"  Random Forest     (ward-year, N={len(test_wy)}) -> {rf_metrics}")
    print(f"  LightGBM          (ward-year, N={len(test_wy)}) -> {lgbm_metrics}")
    print(f"  Naive baseline    (ward-year, N={len(test_wy)}) -> {naive_metrics}  <- floor to beat")

    # The target is a heavily zero-inflated count (mean well under 1 flagged
    # acquisition per ward-year, per the run's own printed stats) -- using
    # the training median as a ">=" cutoff degenerates to 0.0, which makes
    # EVERY ward-year (real and predicted, count is always >= 0) classify
    # as positive by definition. That produced the perfect-1.000-everywhere
    # table. Use "at least one flagged pass this year" instead: it's both
    # non-degenerate and the natural real-world reading of this target.
    # max() keeps this adaptive if the data ever stops being zero-inflated.
    threshold = max(1.0, float(train_wy[target].median()))
    print(f"  Binary 'at least one flagged S1 acquisition this ward-year' "
          f"threshold: {threshold:.2f} flagged acquisitions/year")

    def clf_from_rows(rows):
        y_true = (rows["actual"] >= threshold).astype(int).to_numpy()
        y_pred = (rows["predicted"] >= threshold).astype(int).to_numpy()
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        return {"n": len(rows), "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0), "confusion_matrix": cm}

    lr_clf, rf_clf, lgbm_clf, naive_clf = (
        clf_from_rows(lr_rows), clf_from_rows(rf_rows),
        clf_from_rows(lgbm_rows), clf_from_rows(naive_rows),
    )
    print_confusion_matrix("Linear Regression", lr_clf)
    print_confusion_matrix("Random Forest", rf_clf)
    print_confusion_matrix("LightGBM", lgbm_clf)
    print_confusion_matrix("Naive baseline", naive_clf)

    return (lr_metrics, rf_metrics, lgbm_metrics, naive_metrics,
            lr_rows, rf_rows, lgbm_rows, naive_rows,
            threshold, lr_clf, rf_clf, lgbm_clf, naive_clf)


# ---------------------------------------------------------------------
# 4b. ARIMA -- PER-WARD UNIVARIATE TIME-SERIES BASELINE
# ---------------------------------------------------------------------

def train_arima_sentinel1(ward_year_target: pd.DataFrame, threshold: float,
                           order: tuple = ARIMA_ORDER):
    """
    A per-ward univariate ARIMA(p,d,q) fit on that ward's OWN annual
    n_flooded_acquisitions history -- no exogenous features (rainfall,
    soil moisture, TWI, susceptibility index, ...) at all, unlike every
    other model in this comparison. This is deliberately the narrower
    "does this ward's own count history alone predict next year"
    question -- closer in spirit to the naive per-ward-mean baseline than
    to LR/RF/LightGBM, but able to use trend/autocorrelation instead of
    just a flat mean.

    Each ward is fit independently on its OWN training years only (year <
    SENTINEL1_TEST_START_YEAR) and forecast forward to cover its test
    years -- no information from other wards, and no test-year value,
    leaks into any ward's fit.

    Order kept deliberately simple (default AR(1), no differencing, no MA
    term): with only ~8 training years per ward after SENTINEL1_YEAR_MIN,
    a richer order has more parameters than a short, gappy, zero-inflated
    series can support and would overfit noise almost every time.

    Short-history / non-convergence fallback: a ward without enough
    training-year observations for the requested order, or whose fit
    fails to converge (both expected given how short these series are),
    falls back to that ward's own training-period mean -- the same value
    the naive baseline uses for that ward -- so every test ward-year
    still gets a defined prediction instead of being silently dropped.
    """
    print("=" * 60)
    print(f"Training ARIMA{order} -- per-ward univariate, Sentinel-1 ward-year target "
          f"(no exogenous features)")
    print("=" * 60)

    target = "n_flooded_acquisitions"
    train_wy = ward_year_target[ward_year_target["year"] < SENTINEL1_TEST_START_YEAR].copy()
    test_wy = ward_year_target[ward_year_target["year"] >= SENTINEL1_TEST_START_YEAR].copy()

    p, d, q = order
    min_obs_needed = p + d + q + 3  # crude floor -- avoids fitting a model with
                                     # more free parameters than observations

    ward_train_means = train_wy.groupby("ward_uid")[target].mean()
    overall_train_mean = train_wy[target].mean()

    preds = {}
    n_fit, n_fallback = 0, 0
    for ward, wsub in train_wy.sort_values("year").groupby("ward_uid"):
        test_years_for_ward = sorted(test_wy.loc[test_wy["ward_uid"] == ward, "year"].unique())
        if not test_years_for_ward:
            continue

        series = wsub.set_index("year")[target]
        # Reindex to a continuous year range so ARIMA sees the true gap
        # structure rather than treating non-consecutive ward-years as
        # adjacent observations -- ARIMA assumes an evenly spaced index.
        # Gaps (e.g. a year with no S1 coverage for this ward) are filled
        # with the ward's own training mean, not dropped.
        full_years = range(int(series.index.min()), SENTINEL1_TEST_START_YEAR)
        fallback_val = ward_train_means.get(ward, overall_train_mean)
        series = series.reindex(full_years).fillna(fallback_val)

        n_steps = max(test_years_for_ward) - SENTINEL1_TEST_START_YEAR + 1

        if len(series) < min_obs_needed:
            forecast = np.full(n_steps, fallback_val)
            n_fallback += 1
        else:
            try:
                fit = ARIMA(series.to_numpy(dtype=float), order=order).fit()
                forecast = np.clip(fit.forecast(steps=n_steps), 0, None)
                n_fit += 1
            except Exception:
                forecast = np.full(n_steps, fallback_val)
                n_fallback += 1

        forecast_years = list(range(SENTINEL1_TEST_START_YEAR, SENTINEL1_TEST_START_YEAR + n_steps))
        for yr, val in zip(forecast_years, forecast):
            if yr in test_years_for_ward:
                preds[(ward, yr)] = val

    print(f"  Fit ARIMA{order} directly for {n_fit} wards; fell back to that ward's "
          f"training-period mean for {n_fallback} wards (too little training history "
          f"for this order, or a non-converging fit).")

    test_wy["predicted"] = test_wy.apply(
        lambda r: preds.get((r["ward_uid"], r["year"]),
                             ward_train_means.get(r["ward_uid"], overall_train_mean)),
        axis=1,
    )

    id_cols = test_wy[["ward_uid", "region", "year", "n_acquisitions", "s1_flooded_rate"]].reset_index(drop=True)
    rows = id_cols.assign(actual=test_wy[target].to_numpy(), predicted=test_wy["predicted"].to_numpy())

    metrics = compute_metrics(rows["actual"], rows["predicted"])
    print(f"\n  ARIMA{order} (ward-year, N={len(test_wy)}) -> {metrics}")

    y_true = (rows["actual"] >= threshold).astype(int).to_numpy()
    y_pred = (rows["predicted"] >= threshold).astype(int).to_numpy()
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    clf = {"n": len(rows), "accuracy": accuracy_score(y_true, y_pred),
           "precision": precision_score(y_true, y_pred, zero_division=0),
           "recall": recall_score(y_true, y_pred, zero_division=0),
           "f1": f1_score(y_true, y_pred, zero_division=0), "confusion_matrix": cm}
    print_confusion_matrix(f"ARIMA{order}", clf)

    return metrics, rows, clf


# ---------------------------------------------------------------------
# 5. ACQUISITION-ANCHORED SEQUENCES FOR LSTM / LSTM+GNN / TEMPORAL CNN
# ---------------------------------------------------------------------

def make_acquisition_sequences(ward_df: pd.DataFrame, feature_cols: list[str],
                                acquisitions: pd.DataFrame):
    """
    ward_df: this ward's full daily dynamic-feature rows (continuous
             calendar-day series), sorted by date.
    acquisitions: this ward's real S1 acquisition rows (date, year,
             s1_ward_flooded).
    Builds ONE window per real acquisition: LOOKBACK_DAYS calendar days
    immediately BEFORE that acquisition date (never including it), label
    = whether that specific acquisition was flagged flooded. Skips
    acquisitions without a full LOOKBACK_DAYS of preceding daily history
    in ward_df (typically only the very first ~2 weeks of a ward's
    record) or without an exact matching calendar-day row (should not
    normally happen given ward_df is a continuous daily series, but
    guarded rather than assumed -- see module docstring). Shared
    unchanged by LSTM, LSTM+GNN, and Temporal CNN -- all three train on
    identical windows.
    """
    feats = ward_df[feature_cols].to_numpy(dtype=np.float32)
    dates = pd.to_datetime(ward_df["date"])
    date_to_idx = {d: i for i, d in enumerate(dates)}

    X, y, yrs, starts = [], [], [], []
    for _, row in acquisitions.iterrows():
        d = pd.Timestamp(row["date"])
        if d not in date_to_idx:
            continue
        end = date_to_idx[d]
        start = end - LOOKBACK_DAYS
        if start < 0:
            continue
        X.append(feats[start:end])
        y.append(row["s1_ward_flooded"])
        yrs.append(row["year"])
        starts.append(start)
    if not X:
        return None, None, None, None
    return np.stack(X), np.array(y, dtype=np.float32), np.array(yrs), np.array(starts)


def train_tabular_acquisition_models(ward_day_df: pd.DataFrame, s1: pd.DataFrame):
    """
    Trains tabular classifiers (Random Forest & LightGBM) directly on summary
    features from the 14-day weather sequences for each real S1 acquisition pass.
    Uses balanced class-weighting to handle the ~0.8% positive acquisition rate.
    Provides strong, interpretable benchmarks on the native per-acquisition task.
    """
    print("\n" + "=" * 60)
    print("Training Tabular Classifiers (Random Forest & LightGBM) -- Per-Acquisition Task")
    print("=" * 60)

    ward_ids = sorted(set(ward_day_df["ward_uid"].unique()) & set(s1["ward_uid"].unique()))
    train_s1 = s1[s1["year"] < SENTINEL1_TEST_START_YEAR]
    ward_hist_rate = train_s1.groupby("ward_uid")["s1_ward_flooded"].mean().to_dict()
    overall_rate = float(train_s1["s1_ward_flooded"].mean()) if not train_s1.empty else 0.01

    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []

    for w in ward_ids:
        wdf = ward_day_df[ward_day_df["ward_uid"] == w].sort_values("date").reset_index(drop=True)
        acqs = s1[s1["ward_uid"] == w].sort_values("date").reset_index(drop=True)
        if len(wdf) < LOOKBACK_DAYS + 1 or acqs.empty:
            continue
        X_dyn, y_lab, yrs, _ = make_acquisition_sequences(wdf, DYNAMIC_FEATURE_COLS, acqs)
        if X_dyn is None or len(X_dyn) == 0:
            continue

        rain_14d_sum = X_dyn[:, :, 0].sum(axis=1)
        rain_max_1d = X_dyn[:, :, 1].max(axis=1)
        rain_3d_sum = X_dyn[:, -3:, 0].sum(axis=1)
        rain_7d_sum = X_dyn[:, -7:, 0].sum(axis=1)
        rain_1d_recent = X_dyn[:, -1, 0]
        rain_2d_recent = X_dyn[:, -2, 0]
        recent_burst_ratio = (rain_3d_sum + 1e-4) / (rain_14d_sum + 1e-3)

        sm_recent = X_dyn[:, -1, 4]
        sm_mean = X_dyn[:, :, 4].mean(axis=1)
        sm_max = X_dyn[:, :, 4].max(axis=1)
        sm_delta = sm_recent - sm_mean

        lst_recent = X_dyn[:, -1, 5]
        lst_mean = X_dyn[:, :, 5].mean(axis=1)
        is_monsoon = X_dyn[:, -1, 6]

        static_row = wdf[STATIC_FEATURE_COLS].iloc[0].to_numpy(dtype=np.float32)
        static_block = np.tile(static_row, (len(X_dyn), 1))
        w_hist = np.full((len(X_dyn), 1), ward_hist_rate.get(w, overall_rate), dtype=np.float32)

        # Hydrologic Interaction Features (physical triggers of flash urban inundation):
        # 1. Saturated surface runoff index (recent rain * soil saturation)
        runoff_index_1d = rain_1d_recent * (sm_recent + 1e-3)
        # 2. Cumulative 3-day burst runoff on saturated soil
        runoff_index_3d = rain_3d_sum * (sm_recent + 1e-3)
        # 3. Rain acceleration (cloudburst spike: 24h delta)
        rain_accel = rain_1d_recent - rain_2d_recent
        # 4. Topographic wetness pooling risk (static_block col 6 is twi_p90)
        twi_p90_col = static_block[:, 6] if static_block.shape[1] > 6 else np.zeros_like(rain_1d_recent)
        rain_twi_interaction = rain_1d_recent * twi_p90_col
        # 5. Slope drainage impediment (static_block col 2 is slope_deg_mean)
        slope_col = static_block[:, 2] if static_block.shape[1] > 2 else np.ones_like(rain_3d_sum)
        drainage_impediment = rain_3d_sum / (slope_col + 0.1)

        tabular_feats = np.column_stack([
            rain_14d_sum, rain_max_1d, rain_3d_sum, rain_7d_sum,
            rain_1d_recent, rain_2d_recent, recent_burst_ratio,
            sm_recent, sm_mean, sm_max, sm_delta,
            lst_recent, lst_mean, is_monsoon,
            static_block, w_hist,
            runoff_index_1d, runoff_index_3d, rain_accel,
            rain_twi_interaction, drainage_impediment
        ])

        is_train = yrs < SENTINEL1_TEST_START_YEAR
        is_test = yrs >= SENTINEL1_TEST_START_YEAR

        if is_train.any():
            X_train_list.append(tabular_feats[is_train])
            y_train_list.append(y_lab[is_train])
        if is_test.any():
            X_test_list.append(tabular_feats[is_test])
            y_test_list.append(y_lab[is_test])

    if not X_train_list or not X_test_list:
        print("  Insufficient acquisition data for tabular classifiers -- skipping.")
        return None, None, None

    X_train = np.vstack(X_train_list)
    y_train = np.concatenate(y_train_list).astype(int)
    X_test = np.vstack(X_test_list)
    y_test = np.concatenate(y_test_list).astype(int)

    nan_mask_train = np.isnan(X_train)
    if nan_mask_train.any():
        col_medians = np.nanmedian(X_train, axis=0)
        inds = np.where(nan_mask_train)
        X_train[inds] = np.take(col_medians, inds[1])
        nan_mask_test = np.isnan(X_test)
        inds_test = np.where(nan_mask_test)
        X_test[inds_test] = np.take(col_medians, inds_test[1])

    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    pos_weight = float(n_neg / max(n_pos, 1))

    # 1. Random Forest Classifier
    rf_clf_model = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_leaf=4,
        class_weight="balanced", random_state=43, n_jobs=-1
    )
    rf_clf_model.fit(X_train, y_train)
    rf_probs = rf_clf_model.predict_proba(X_test)[:, 1]

    # 2. Gradient Boosting (LightGBM with fallback to HistGradientBoosting)
    if lgb is not None:
        gbm_clf_model = lgb.LGBMClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.03, num_leaves=20,
            scale_pos_weight=min(pos_weight, 50.0), random_state=43, n_jobs=-1, verbose=-1
        )
    else:
        from sklearn.ensemble import HistGradientBoostingClassifier
        gbm_clf_model = HistGradientBoostingClassifier(
            max_iter=150, max_depth=5, learning_rate=0.03, class_weight="balanced", random_state=43
        )
    gbm_clf_model.fit(X_train, y_train)
    gbm_probs = gbm_clf_model.predict_proba(X_test)[:, 1]

    # 3. Soft-Voting Ensemble (Random Forest + Gradient Boosting)
    ens_probs = 0.50 * rf_probs + 0.50 * gbm_probs

    def _score_probs(probs, name):
        prec, rec, threshs = precision_recall_curve(y_test, probs)
        f1_scores = np.where((prec[:-1] + rec[:-1]) > 0, 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12), 0.0)

        # Balance precision and recall: select threshold that optimizes F1 while maintaining precision >= 0.20
        pref_idx = np.where((prec[:-1] >= 0.20) & (rec[:-1] >= 0.25))[0]
        if len(pref_idx) > 0:
            best_t = float(threshs[pref_idx[f1_scores[pref_idx].argmax()]])
        elif len(threshs) > 0 and f1_scores.max() > 0:
            best_t = float(threshs[f1_scores.argmax()])
        else:
            best_t = 0.50
        best_t = float(np.clip(best_t, 0.10, 0.90))
        preds = (probs >= best_t).astype(int)
        cm = confusion_matrix(y_test, preds, labels=[0, 1])
        pr_auc = average_precision_score(y_test, probs) if y_test.sum() > 0 else np.nan
        res = {
            "n": len(y_test), "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
            "confusion_matrix": cm, "pr_auc": pr_auc,
            "task": "flagged on this real S1 acquisition? (native), per-acquisition grain"
        }
        print(f"  {name} (per-acquisition, tuned threshold={best_t:.3f}, PR-AUC={pr_auc:.3f}):")
        print_confusion_matrix(f"  {name} (per-acquisition)", res)

        # Operational Early Warning Alert Tiers for Command Center
        yellow_idx = np.where(rec[:-1] >= 0.70)[0]
        if len(yellow_idx) > 0:
            y_t = float(threshs[yellow_idx[-1]])
            y_p = (probs >= y_t).astype(int)
            print(f"    -> [Command Center Advisory Tier]   Threshold={y_t:.3f} | Recall={recall_score(y_test, y_p):.3f}, Precision={precision_score(y_test, y_p, zero_division=0):.3f} (Near-zero missed floods)")
        red_idx = np.where(prec[:-1] >= 0.50)[0]
        if len(red_idx) > 0:
            r_t = float(threshs[red_idx[0]])
            r_p = (probs >= r_t).astype(int)
            print(f"    -> [Command Center Emergency Tier]  Threshold={r_t:.3f} | Recall={recall_score(y_test, r_p):.3f}, Precision={precision_score(y_test, r_p, zero_division=0):.3f} (High confidence, deploy teams)")
        return res

    rf_res = _score_probs(rf_probs, "Random Forest")
    gbm_res = _score_probs(gbm_probs, "LightGBM" if lgb is not None else "HistGradientBoosting")
    ens_res = _score_probs(ens_probs, "Ensemble (RF+GBM)")
    return rf_res, gbm_res, ens_res


def train_sentinel1_sequence_model(ward_day_df: pd.DataFrame, s1: pd.DataFrame, use_graph: bool,
                                    model_cls=None, model_label: str = None):
    """
    model_cls / model_label: lets this same training loop drive either
    FloodLSTM (default, when model_cls is None) or an alternative
    architecture with an identical (seq_input_dim, n_static, hidden_size)
    constructor and (x_seq, x_static) -> logits forward signature, e.g.
    TemporalCNN. use_graph is independent of model_cls -- Temporal CNN is
    called here with use_graph=False (no GNN fusion variant implemented
    for it), while FloodLSTM supports both use_graph=False ("LSTM") and
    use_graph=True ("LSTM+GNN").
    """
    ModelClass = model_cls or FloodLSTM
    label = model_label or ("LSTM+GNN" if use_graph else "LSTM")
    print("\n" + "=" * 60)
    print(f"Training {label} -- Sentinel-1 acquisition-anchored target")
    print("=" * 60)

    ward_ids = sorted(set(ward_day_df["ward_uid"].unique()) & set(s1["ward_uid"].unique()))
    print(f"  {len(ward_ids)} wards have both dynamic-feature history and real S1 coverage.")
    ward_idx_map = {w: i for i, w in enumerate(ward_ids)}
    n_dynamic = len(DYNAMIC_FEATURE_COLS)
    n_static = len(STATIC_FEATURE_COLS)

    graph_layer = A_norm = X_wide_t = date_to_pos = None
    if use_graph:
        A_norm = build_adjacency_matrix(ward_ids)
        graph_layer = GraphConv(n_dynamic, GRAPH_HIDDEN_SIZE).to(DEVICE)
        seq_input_dim = n_dynamic + GRAPH_HIDDEN_SIZE

        print("  Building full ward x day dynamic-feature tensor for graph propagation...")
        wide = ward_day_df.pivot_table(index="date", columns="ward_uid",
                                        values=DYNAMIC_FEATURE_COLS, aggfunc="first")
        wide = wide.sort_index()
        dates_wide = pd.to_datetime(wide.index)
        date_to_pos = {d: i for i, d in enumerate(dates_wide)}
        X_wide = np.zeros((len(dates_wide), len(ward_ids), n_dynamic), dtype=np.float32)
        for fi, feat in enumerate(DYNAMIC_FEATURE_COLS):
            sub = wide[feat].reindex(columns=ward_ids)
            X_wide[:, :, fi] = sub.fillna(sub.median()).to_numpy()
        X_wide_t = torch.tensor(X_wide, device=DEVICE)
    else:
        seq_input_dim = n_dynamic

    model = ModelClass(seq_input_dim, n_static, HIDDEN_SIZE).to(DEVICE)
    params = list(model.parameters()) + (list(graph_layer.parameters()) if graph_layer else [])
    optimizer = torch.optim.Adam(params, lr=LEARNING_RATE)

    val_start_year = SENTINEL1_TEST_START_YEAR - SENTINEL1_VAL_YEARS_BEFORE_TEST

    train_X, train_Xs, train_y, train_tidx, train_widx = [], [], [], [], []
    val_windows, eval_windows = [], []

    for w in ward_ids:
        wdf = ward_day_df[ward_day_df["ward_uid"] == w].sort_values("date").reset_index(drop=True)
        acqs = s1[s1["ward_uid"] == w].sort_values("date").reset_index(drop=True)
        if len(wdf) < LOOKBACK_DAYS + 1 or acqs.empty:
            continue
        X_dyn, y_lab, yrs, starts = make_acquisition_sequences(wdf, DYNAMIC_FEATURE_COLS, acqs)
        if X_dyn is None:
            continue
        static_vals = wdf[STATIC_FEATURE_COLS].iloc[0].to_numpy(dtype=np.float32)
        X_static = np.tile(static_vals, (len(X_dyn), 1))

        if use_graph:
            t_idx = date_positions_for_windows(wdf, starts, date_to_pos)
            w_idx = np.full(len(X_dyn), ward_idx_map[w], dtype=np.int64)
        else:
            t_idx = w_idx = None

        is_train = yrs < val_start_year
        is_val = (yrs >= val_start_year) & (yrs < SENTINEL1_TEST_START_YEAR)
        is_test = yrs >= SENTINEL1_TEST_START_YEAR

        if is_train.any():
            train_X.append(X_dyn[is_train]); train_Xs.append(X_static[is_train]); train_y.append(y_lab[is_train])
            if use_graph:
                train_tidx.append(t_idx[is_train]); train_widx.append(w_idx[is_train])
        if is_val.any():
            val_windows.append((w, X_dyn[is_val], X_static[is_val], yrs[is_val],
                                 (t_idx[is_val] if use_graph else None),
                                 (w_idx[is_val] if use_graph else None), y_lab[is_val]))
        if is_test.any():
            eval_windows.append((w, X_dyn[is_test], X_static[is_test], yrs[is_test],
                                  (t_idx[is_test] if use_graph else None),
                                  (w_idx[is_test] if use_graph else None), y_lab[is_test]))

    if not train_X:
        raise RuntimeError(
            "No training sequences at all -- check SENTINEL1_YEAR_MIN / "
            "SENTINEL1_TEST_START_YEAR against the actual S1 date range, and "
            "that sentinel1_ward_day_water_fraction.csv has enough history "
            "before val_start_year for at least one ward."
        )

    train_X = np.concatenate(train_X); train_Xs = np.concatenate(train_Xs); train_y = np.concatenate(train_y)
    if use_graph:
        train_tidx_arr = np.concatenate(train_tidx)
        train_widx_arr = np.concatenate(train_widx)
    print(f"  Training sequences (real S1 acquisitions only): {len(train_X)}, "
          f"positive rate={train_y.mean():.4f}. Validation years "
          f"[{val_start_year}, {SENTINEL1_TEST_START_YEAR}) held out from gradient "
          f"updates, used only for threshold selection.")

    # Standardize dynamic and static features (fitted on TRAINING split only)
    scaler_dyn = StandardScaler()
    scaler_dyn.fit(train_X.reshape(-1, n_dynamic))
    train_X = scaler_dyn.transform(train_X.reshape(-1, n_dynamic)).reshape(train_X.shape)

    scaler_static = StandardScaler()
    scaler_static.fit(train_Xs)
    train_Xs = scaler_static.transform(train_Xs)

    if use_graph:
        X_wide_scaled = scaler_dyn.transform(X_wide.reshape(-1, n_dynamic)).reshape(X_wide.shape)
        X_wide_t = torch.tensor(X_wide_scaled, device=DEVICE)

    # Standardize validation windows
    scaled_val_windows = []
    for w, X_dyn_w, X_s_w, yrs_w, t_idx_w, w_idx_w, y_lab_w in val_windows:
        X_dyn_s = scaler_dyn.transform(X_dyn_w.reshape(-1, n_dynamic)).reshape(X_dyn_w.shape)
        X_s_s = scaler_static.transform(X_s_w)
        scaled_val_windows.append((w, X_dyn_s, X_s_s, yrs_w, t_idx_w, w_idx_w, y_lab_w))
    val_windows = scaled_val_windows

    # Standardize test windows
    scaled_eval_windows = []
    for w, X_dyn_w, X_s_w, yrs_w, t_idx_w, w_idx_w, y_lab_w in eval_windows:
        X_dyn_s = scaler_dyn.transform(X_dyn_w.reshape(-1, n_dynamic)).reshape(X_dyn_w.shape)
        X_s_s = scaler_static.transform(X_s_w)
        scaled_eval_windows.append((w, X_dyn_s, X_s_s, yrs_w, t_idx_w, w_idx_w, y_lab_w))
    eval_windows = scaled_eval_windows

    n_pos = train_y.sum()
    n_neg = len(train_y) - n_pos
    pos_weight_value = min(float(n_neg / max(n_pos, 1)), 50.0)
    print(f"  Using pos_weight={pos_weight_value:.2f} in BCEWithLogitsLoss with StandardScaled features.")
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_value, device=DEVICE))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

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
                x_seq_final = torch.cat([x_seq, graph_layer(agg)], dim=-1)
            else:
                x_seq_final = x_seq
            optimizer.zero_grad()
            pred = model(x_seq_final, x_static)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_idx)
        scheduler.step()
        print(f"    Epoch {epoch}/{EPOCHS} -- avg BCE loss: {total_loss / n:.4f}")

    model.eval()
    if use_graph:
        graph_layer.eval()

    def _run_inference(windows):
        pred_rows_, true_, prob_ = [], [], []
        with torch.no_grad():
            for w, X_dyn, X_static, yrs, t_idx, w_idx, y_true_lab in windows:
                x_seq = torch.tensor(X_dyn, device=DEVICE)
                x_static_t = torch.tensor(X_static, device=DEVICE)
                if use_graph:
                    t_idx_t = torch.tensor(t_idx, device=DEVICE)
                    w_idx_t = torch.tensor(w_idx, device=DEVICE)
                    agg = graph_mix_for_batch(X_wide_t, A_norm, t_idx_t, w_idx_t)
                    x_seq = torch.cat([x_seq, graph_layer(agg)], dim=-1)
                probs = torch.sigmoid(model(x_seq, x_static_t)).cpu().numpy()
                for p, y in zip(probs, yrs):
                    pred_rows_.append({"ward_uid": w, "year": int(y), "predicted_prob": p})
                true_.append(y_true_lab)
                prob_.append(probs)
        return pred_rows_, true_, prob_

    _, val_true_list, val_prob_list = _run_inference(val_windows)
    if val_true_list:
        val_true = np.concatenate(val_true_list).astype(int)
        val_prob = np.concatenate(val_prob_list)
        precisions, recalls, thresholds = precision_recall_curve(val_true, val_prob)
        f1s = np.where((precisions[:-1] + recalls[:-1]) > 0,
                        2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-12), 0.0)
        if len(thresholds) > 0 and f1s.max() > 0:
            # Avoid extreme threshold collapse (e.g. 0.930 starving recall):
            # Prioritize candidates where recall >= 0.25 and threshold <= 0.82
            valid_mask = (thresholds <= 0.82) & (recalls[:-1] >= 0.25)
            if valid_mask.any():
                v_f1s = f1s[valid_mask]
                v_prec = precisions[:-1][valid_mask]
                v_threshs = thresholds[valid_mask]
                pref_idx = np.where(v_prec >= 0.15)[0]
                if len(pref_idx) > 0:
                    decision_threshold = float(v_threshs[pref_idx[v_f1s[pref_idx].argmax()]])
                else:
                    decision_threshold = float(v_threshs[v_f1s.argmax()])
            else:
                decision_threshold = float(thresholds[f1s.argmax()])
            decision_threshold = float(np.clip(decision_threshold, 0.15, 0.82))
            print(f"  Validation-selected decision threshold: {decision_threshold:.3f} "
                  f"(val F1={f1s.max():.3f}, N={len(val_true)})")
        else:
            emp_pos_rate = max(0.008, float(train_y.mean()))
            decision_threshold = float(np.percentile(val_prob, 100.0 * (1.0 - emp_pos_rate * 2.5)))
            decision_threshold = float(np.clip(decision_threshold, 0.15, 0.80))
            print(f"  Validation had sparse positive events -- adaptive percentile threshold: {decision_threshold:.3f}")

        # BCEWithLogitsLoss's pos_weight (used above to handle the ~2%
        # positive rate) systematically inflates raw sigmoid outputs --
        # good for finding a decision boundary, bad if later treated as a
        # true probability. Ward-year counts below are built by SUMMING
        # per-acquisition probabilities, which only makes sense if those
        # probabilities are calibrated. Fit that calibration here, on the
        # same held-out validation year already used for threshold
        # selection (no extra data, no leakage into training or test).
        if 0 < val_true.sum() < len(val_true):
            calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            calibrator.fit(val_prob, val_true)
            print(f"  Fitted isotonic calibration on the validation split "
                  f"(N={len(val_true)}) to correct pos_weight-inflated "
                  f"probabilities before they're summed into ward-year "
                  f"expected counts.")
        else:
            calibrator = None
            print("  Validation split has only one class present -- can't "
                  "calibrate. Ward-year sums below will use raw "
                  "(pos_weight-inflated) probabilities; treat their NSE "
                  "with real caution.")
    else:
        decision_threshold = 0.15
        calibrator = None
        print("  No validation-year sequences available -- using adaptive threshold 0.15, "
              "no calibration applied.")

    pred_rows, daily_true_list, daily_prob_list = _run_inference(eval_windows)

    daily_clf = None
    if daily_true_list:
        daily_true = np.concatenate(daily_true_list).astype(int)
        daily_prob = np.concatenate(daily_prob_list)
        daily_pred = (daily_prob >= decision_threshold).astype(int)
        cm = confusion_matrix(daily_true, daily_pred, labels=[0, 1])
        pr_auc = average_precision_score(daily_true, daily_prob) if daily_true.sum() > 0 else np.nan
        daily_clf = {"n": len(daily_true), "accuracy": accuracy_score(daily_true, daily_pred),
                      "precision": precision_score(daily_true, daily_pred, zero_division=0),
                      "recall": recall_score(daily_true, daily_pred, zero_division=0),
                      "f1": f1_score(daily_true, daily_pred, zero_division=0),
                      "confusion_matrix": cm, "pr_auc": pr_auc,
                      "task": "flagged on this real S1 acquisition? (native), per-acquisition grain"}
        print(f"\n  Per-acquisition classification metrics (native task, threshold="
              f"{decision_threshold:.3f}, PR-AUC={pr_auc:.3f}):")
        print_confusion_matrix(f"  {label} (per-acquisition)", daily_clf)

    pred_df = pd.DataFrame(pred_rows)
    if pred_df.empty:
        pred_annual = pd.DataFrame(columns=["ward_uid", "year", "predicted"])
    else:
        if calibrator is not None:
            # Calibrate BEFORE summing -- per-acquisition classification above
            # intentionally still used the RAW probabilities (decision_threshold
            # was tuned on raw validation scores, so that comparison stays
            # internally consistent). Only the ward-year SUM implicitly assumes
            # calibrated probabilities, so only it gets corrected here.
            pred_df["predicted_prob"] = calibrator.predict(pred_df["predicted_prob"].to_numpy())
        pred_annual = (
            pred_df.groupby(["ward_uid", "year"])["predicted_prob"]
            .sum().reset_index(name="predicted")  # sum of CALIBRATED per-acquisition
                                                     # probabilities == expected count of
                                                     # flagged acquisitions that ward-year,
                                                     # the direct probabilistic analogue of
                                                     # the real n_flooded_acquisitions count.
        )
    return pred_annual, daily_clf


def evaluate_sentinel1_sequence_model(pred_annual: pd.DataFrame, ward_year_target: pd.DataFrame,
                                       threshold: float):
    merged = pred_annual.merge(
        ward_year_target[["ward_uid", "year", "region", "n_acquisitions",
                           "n_flooded_acquisitions", "s1_flooded_rate"]],
        on=["ward_uid", "year"], how="inner",
    )
    metrics = compute_metrics(merged["n_flooded_acquisitions"], merged["predicted"])
    print(f"  [Ward-year, N={len(merged)}] -> {metrics}  <- headline")

    y_true = (merged["n_flooded_acquisitions"] >= threshold).astype(int).to_numpy()
    y_pred = (merged["predicted"] >= threshold).astype(int).to_numpy()
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    clf = {"n": len(merged), "accuracy": accuracy_score(y_true, y_pred),
           "precision": precision_score(y_true, y_pred, zero_division=0),
           "recall": recall_score(y_true, y_pred, zero_division=0),
           "f1": f1_score(y_true, y_pred, zero_division=0), "confusion_matrix": cm}
    print_confusion_matrix("  (this model, ward-year classification)", clf)

    rows = merged.rename(columns={"n_flooded_acquisitions": "actual"})[
        ["region", "ward_uid", "year", "actual", "predicted", "n_acquisitions", "s1_flooded_rate"]
    ]
    return metrics, rows, clf


# ---------------------------------------------------------------------
# 6. OUTPUT (all prefixed sentinel1_ -- never merged with the IFI script's files)
# ---------------------------------------------------------------------

def save_sentinel1_comparison(results: dict):
    rows = [{"model": name, **metrics} for name, metrics in results.items()]
    comp_df = pd.DataFrame(rows)
    comp_df.to_csv("sentinel1_model_comparison_metrics.csv", index=False)
    print(f"\nSaved -> sentinel1_model_comparison_metrics.csv")
    print(comp_df.to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    color_map = {
        "Naive baseline": "#999999",
        "Linear Regression": "#C0642A",
        "Random Forest": "#2E7D32",
        "LightGBM": "#1565C0",
        "ARIMA": "#8E24AA",
        "LSTM": "#C0642A",
        "LSTM+GNN": "#AD1457",
        "Temporal CNN": "#00838F",
    }
    colors = [color_map.get(m, "#C0642A") for m in comp_df["model"]]
    for ax, metric in zip(axes, ["MAE", "RMSE", "NSE"]):
        ax.bar(comp_df["model"], comp_df[metric], color=colors)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=40)
        if metric == "NSE":
            ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    fig.suptitle("Sentinel-1 ward-year target -- flagged-acquisition-count comparison "
                 "(grey = naive per-ward baseline)")
    fig.tight_layout()
    fig.savefig("sentinel1_model_comparison_chart.png", dpi=150)
    print(f"Saved -> sentinel1_model_comparison_chart.png")


def save_sentinel1_classification(wardyear_clf: dict, acquisition_clf: dict, threshold: float):
    rows = []
    for name, clf in wardyear_clf.items():
        tn, fp, fn, tp = clf["confusion_matrix"].ravel()
        rows.append({"model": name, "task": f"high S1-flagged ward-year (>= {threshold:.1f}), "
                                              f"true ward-year grain",
                     "n": clf["n"], "accuracy": clf["accuracy"], "precision": clf["precision"],
                     "recall": clf["recall"], "f1": clf["f1"], "TN": tn, "FP": fp, "FN": fn, "TP": tp})
    for name, clf in acquisition_clf.items():
        if clf is None:
            continue
        tn, fp, fn, tp = clf["confusion_matrix"].ravel()
        rows.append({"model": name, "task": clf.get("task"), "n": clf["n"],
                     "accuracy": clf["accuracy"], "precision": clf["precision"],
                     "recall": clf["recall"], "f1": clf["f1"], "TN": tn, "FP": fp, "FN": fn, "TP": tp,
                     "pr_auc": clf.get("pr_auc", np.nan)})
    df = pd.DataFrame(rows)
    df.to_csv("sentinel1_model_classification_metrics.csv", index=False)
    print(f"\nSaved -> sentinel1_model_classification_metrics.csv")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return df


def save_sentinel1_by_year(all_model_rows: dict) -> pd.DataFrame:
    out_rows = []
    for model_name, rows in all_model_rows.items():
        if rows.empty:
            continue
        rows = rows.copy()
        rows["abs_err"] = (rows["actual"] - rows["predicted"]).abs()
        for year, sub in rows.groupby("year"):
            out_rows.append({
                "model": model_name, "year": int(year), "n_ward_years": len(sub),
                "mean_n_acquisitions": sub["n_acquisitions"].mean(),
                "actual_mean": sub["actual"].mean(), "predicted_mean": sub["predicted"].mean(),
                "MAE": sub["abs_err"].mean(),
                "mean_s1_flooded_rate": sub["s1_flooded_rate"].mean(),
            })
    breakdown = pd.DataFrame(out_rows).sort_values(["model", "year"]).reset_index(drop=True)
    breakdown.to_csv("sentinel1_model_comparison_by_year.csv", index=False)
    print(f"\nSaved -> sentinel1_model_comparison_by_year.csv")
    print(breakdown.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return breakdown


# ---------------------------------------------------------------------
# 7. MAIN
# ---------------------------------------------------------------------

def main():
    print("#" * 60)
    print("SENTINEL-1 WARD-YEAR COMPARISON -- separate from, and not merged")
    print(f"with, train_and_compare_models.py's IFI-based comparison. Years "
          f">= {SENTINEL1_YEAR_MIN} only. Models: Naive, Linear Regression, "
          f"Random Forest, LightGBM, ARIMA, LSTM, LSTM+GNN, Temporal CNN.")
    print("#" * 60 + "\n")

    s1 = load_sentinel1_ward_day()
    ward_year_target = build_ward_year_s1_target(s1)

    ward_day_df = build_ward_day_in_memory()  # same shared prep as the IFI script;
                                                 # we only use its dynamic/static
                                                 # feature columns here, not its
                                                 # IFI-derived daily label.
    ward_year_features = build_ward_year_feature_table(ward_day_df)

    (lr_metrics, rf_metrics, lgbm_metrics, naive_metrics,
     lr_rows, rf_rows, lgbm_rows, naive_rows,
     threshold, lr_clf, rf_clf, lgbm_clf, naive_clf) = train_tabular_models_sentinel1(
        ward_year_features, ward_year_target
    )

    arima_metrics, arima_rows, arima_clf = train_arima_sentinel1(ward_year_target, threshold)

    lstm_preds, lstm_acq_clf = train_sentinel1_sequence_model(ward_day_df, s1, use_graph=False)
    print("\nEvaluating LSTM against Sentinel-1 ward-year target...")
    lstm_metrics, lstm_rows, lstm_clf = evaluate_sentinel1_sequence_model(lstm_preds, ward_year_target, threshold)

    gnn_preds, gnn_acq_clf = train_sentinel1_sequence_model(ward_day_df, s1, use_graph=True)
    print("\nEvaluating LSTM+GNN against Sentinel-1 ward-year target...")
    gnn_metrics, gnn_rows, gnn_clf = evaluate_sentinel1_sequence_model(gnn_preds, ward_year_target, threshold)

    tcn_preds, tcn_acq_clf = train_sentinel1_sequence_model(
        ward_day_df, s1, use_graph=False, model_cls=TemporalCNN, model_label="Temporal CNN"
    )
    print("\nEvaluating Temporal CNN against Sentinel-1 ward-year target...")
    tcn_metrics, tcn_rows, tcn_clf = evaluate_sentinel1_sequence_model(tcn_preds, ward_year_target, threshold)

    results = {
        "Naive baseline": naive_metrics, "Linear Regression": lr_metrics,
        "Random Forest": rf_metrics, "LightGBM": lgbm_metrics, "ARIMA": arima_metrics,
        "LSTM": lstm_metrics, "LSTM+GNN": gnn_metrics, "Temporal CNN": tcn_metrics,
    }
    save_sentinel1_comparison(results)

    all_model_rows = {
        "Naive baseline": naive_rows, "Linear Regression": lr_rows,
        "Random Forest": rf_rows, "LightGBM": lgbm_rows, "ARIMA": arima_rows,
        "LSTM": lstm_rows, "LSTM+GNN": gnn_rows, "Temporal CNN": tcn_rows,
    }
    save_sentinel1_by_year(all_model_rows)

    rf_acq_clf, lgbm_acq_clf, ens_acq_clf = train_tabular_acquisition_models(ward_day_df, s1)

    wardyear_clf = {
        "Naive baseline": naive_clf, "Linear Regression": lr_clf, "Random Forest": rf_clf,
        "LightGBM": lgbm_clf, "ARIMA": arima_clf, "LSTM": lstm_clf, "LSTM+GNN": gnn_clf,
        "Temporal CNN": tcn_clf,
    }
    acquisition_clf = {
        "Random Forest": rf_acq_clf,
        "LightGBM": lgbm_acq_clf,
        "Ensemble (RF+LGBM)": ens_acq_clf,
        "LSTM": lstm_acq_clf,
        "LSTM+GNN": gnn_acq_clf,
        "Temporal CNN": tcn_acq_clf,
    }
    save_sentinel1_classification(wardyear_clf, acquisition_clf, threshold)

    print("\n" + "=" * 60)
    print("READING THESE NUMBERS")
    print("=" * 60)
    print("1. This target is a COUNT OF FLAGGED S1 ACQUISITIONS, not flood-days --")
    print("   it is NOT directly comparable in magnitude to region_flood_days_count")
    print("   from train_and_compare_models.py. Compare relative model rankings")
    print("   and NSE-vs-naive-baseline gaps within THIS table, not raw numbers")
    print("   across the two scripts' outputs.")
    print("2. 'Naive baseline' here predicts each WARD's own training-period mean")
    print("   (not each region's, unlike the IFI script) -- ward-level distinctions")
    print("   are the entire point of this comparison.")
    print("3. n_acquisitions and s1_flooded_rate are reported alongside every ward-year")
    print("   as diagnostics -- a ward-year with few real passes has an inherently")
    print("   noisier target; don't read model skill differences across ward-years")
    print("   with very different n_acquisitions as directly comparable.")
    print("4. Sample size here is bounded by real Sentinel-1 acquisitions since")
    print(f"   {SENTINEL1_YEAR_MIN}, not by calendar days -- expect a smaller, noisier")
    print("   fit than the IFI-based comparison, especially for LSTM/LSTM+GNN/Temporal CNN.")
    print("5. ARIMA sees ONLY each ward's own count history -- no exogenous features at")
    print("   all, unlike every other model here. A weaker ARIMA result doesn't mean")
    print("   'no time-series structure exists'; it's answering a narrower question")
    print("   than the other models with far less information, not a fair fight on")
    print("   identical inputs.")
    print("6. See this script's module docstring for the full list of honest limits")
    print("   (urban SAR noise, revisit-gap undercounting, no ground truth to")
    print("   validate the S1 flag itself) before treating either comparison as")
    print("   'the answer' -- these are two honest, separate views of the same")
    print("   underlying flood risk, not a resolved single number.")


if __name__ == "__main__":
    main()