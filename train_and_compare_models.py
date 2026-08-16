"""
Train and Compare All Four Flood Models -- LR, RF, LSTM, LSTM+GNN
--------------------------------------------------------------------------
Answers the actual question from build_final_training_dataset.py's NEXT
STEPS: train all four models and compare them with real numbers, not
just build their input tables.

RUNNING THIS IN COLAB
    !git clone https://github.com/NikitaMulam2005/AI-Digital-Twin.git
    %cd AI-Digital-Twin
    !pip install -q scikit-learn torch matplotlib

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
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler

from build_ward_day_dataset import build_ward_day_table, add_sequence_features

# ---------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------

FINAL_WARD_YEAR_TRAIN_CSV = "final_ward_year_train.csv"
FINAL_WARD_YEAR_TEST_CSV = "final_ward_year_test.csv"
ADJACENCY_EDGES_CSV = "ward_adjacency_edges.csv"
CORRECTED_ANNUAL_CSV = "region_flood_days_count_CORRECTED.csv"   # from build_corrected_flood_labels.py
CORRECTED_DAILY_CSV = "region_day_flooded_CORRECTED.csv"          # from build_corrected_flood_labels.py

COMMON_TARGET = "region_flood_days_count"
TEST_START_YEAR = 2019

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
]
STATIC_FEATURE_COLS = [
    "susceptibility_index", "elevation_m_mean", "slope_deg_mean",
    "road_density_km_per_km2", "is_coastal_region",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


def apply_corrected_annual_labels(df: pd.DataFrame) -> pd.DataFrame:
    corrected = load_corrected_annual_labels()
    df = df.drop(columns=[COMMON_TARGET], errors="ignore").merge(
        corrected, on=["region", "year"], how="left"
    )
    df[COMMON_TARGET] = df[COMMON_TARGET].fillna(0)
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

    lr_metrics, _ = compute_regionyear_metrics(lr_rows)
    rf_metrics, _ = compute_regionyear_metrics(rf_rows)
    naive_metrics, _ = compute_regionyear_metrics(naive_rows)

    print(f"\nLinear Regression (region-year, TRUE N={len(test_ry)}) -> {lr_metrics}")
    print(f"Random Forest     (region-year, TRUE N={len(test_ry)}) -> {rf_metrics}")
    print(f"Naive baseline    (region-year, TRUE N={len(test_ry)}) -> {naive_metrics}  <- floor to beat")

    return lr_metrics, rf_metrics, naive_metrics, lr_rows, rf_rows, naive_rows


# ---------------------------------------------------------------------
# 6. SHARED DATA PREP FOR LSTM / LSTM+GNN
# ---------------------------------------------------------------------

def build_ward_day_in_memory() -> pd.DataFrame:
    print("\nBuilding ward x day table in memory (reusing "
          "build_ward_day_dataset.py's functions, no multi-GB CSV write)...")
    df = build_ward_day_table()
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

    for c in DYNAMIC_FEATURE_COLS + STATIC_FEATURE_COLS:
        if c not in df.columns:
            raise KeyError(f"Expected column '{c}' missing from ward-day table.")
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[DYNAMIC_FEATURE_COLS + STATIC_FEATURE_COLS] = df[
        DYNAMIC_FEATURE_COLS + STATIC_FEATURE_COLS
    ].fillna(df[DYNAMIC_FEATURE_COLS + STATIC_FEATURE_COLS].median())

    print(f"Ward-day table ready: {len(df)} rows, {df['ward_uid'].nunique()} wards, "
          f"{df['date'].min().date()} to {df['date'].max().date()}.")
    return df


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


def make_sequences(ward_df: pd.DataFrame, feature_cols: list[str]):
    feats = ward_df[feature_cols].to_numpy(dtype=np.float32)
    labels = ward_df["region_day_flooded_broadcast"].to_numpy(dtype=np.float32)
    years = ward_df["year"].to_numpy()
    X, y, yrs, starts = [], [], [], []
    for start in range(0, len(ward_df) - LOOKBACK_DAYS - 1, STRIDE_DAYS):
        end = start + LOOKBACK_DAYS
        X.append(feats[start:end])
        y.append(labels[end])
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


def train_sequence_model(df: pd.DataFrame, use_graph: bool):
    label = "LSTM+GNN" if use_graph else "LSTM"
    print("\n" + "=" * 60)
    print(f"Training {label} (ward-day grain)")
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
    loss_fn = nn.BCEWithLogitsLoss()

    train_X, train_Xs, train_y, train_tidx, train_widx = [], [], [], [], []
    eval_windows = []

    for w in ward_ids:
        wdf = df[df["ward_uid"] == w].sort_values("date").reset_index(drop=True)
        if len(wdf) < LOOKBACK_DAYS + 2:
            continue
        X_dyn, y_lab, yrs, starts = make_sequences(wdf, DYNAMIC_FEATURE_COLS)
        if X_dyn is None:
            continue
        static_vals = wdf[STATIC_FEATURE_COLS].iloc[0].to_numpy(dtype=np.float32)
        X_static = np.tile(static_vals, (len(X_dyn), 1))

        if use_graph:
            t_idx = date_positions_for_windows(wdf, starts, date_to_pos)
            w_idx = np.full(len(X_dyn), ward_idx_map[w], dtype=np.int64)
        else:
            t_idx = w_idx = None

        is_train = yrs < TEST_START_YEAR
        if is_train.any():
            train_X.append(X_dyn[is_train]); train_Xs.append(X_static[is_train]); train_y.append(y_lab[is_train])
            if use_graph:
                train_tidx.append(t_idx[is_train]); train_widx.append(w_idx[is_train])
        if (~is_train).any():
            eval_windows.append((
                w, wdf["region"].iloc[0], X_dyn[~is_train], X_static[~is_train], yrs[~is_train],
                (t_idx[~is_train] if use_graph else None), (w_idx[~is_train] if use_graph else None),
            ))

    train_X = np.concatenate(train_X); train_Xs = np.concatenate(train_Xs); train_y = np.concatenate(train_y)
    print(f"Training sequences: {len(train_X)} (lookback={LOOKBACK_DAYS} days, stride={STRIDE_DAYS}) "
          f"across {len(ward_ids)} wards.")

    train_X_t = torch.tensor(train_X, device=DEVICE)
    train_Xs_t = torch.tensor(train_Xs, device=DEVICE)
    train_y_t = torch.tensor(train_y, device=DEVICE)
    if use_graph:
        train_tidx_t = torch.tensor(np.concatenate(train_tidx), device=DEVICE)
        train_widx_t = torch.tensor(np.concatenate(train_widx), device=DEVICE)

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
    pred_rows = []
    with torch.no_grad():
        for w, region, X_dyn, X_static, yrs, t_idx, w_idx in eval_windows:
            x_seq = torch.tensor(X_dyn, device=DEVICE)
            x_static_t = torch.tensor(X_static, device=DEVICE)
            if use_graph:
                t_idx_t = torch.tensor(t_idx, device=DEVICE)
                w_idx_t = torch.tensor(w_idx, device=DEVICE)
                agg = graph_mix_for_batch(X_wide_t, A_norm, t_idx_t, w_idx_t)
                x_seq = torch.cat([x_seq, graph_layer(agg)], dim=-1)
            probs = torch.sigmoid(model(x_seq, x_static_t)).cpu().numpy()
            for p, y in zip(probs, yrs):
                pred_rows.append({"ward_uid": w, "region": region, "year": int(y), "predicted_prob": p})

    pred_df = pd.DataFrame(pred_rows)
    pred_annual = (
        pred_df.groupby(["ward_uid", "region", "year"])["predicted_prob"]
        .sum().mul(STRIDE_DAYS).reset_index(name="predicted_flood_days")
    )
    return pred_annual


# ---------------------------------------------------------------------
# 8. EVALUATE (region-year headline, ward-level reference)
# ---------------------------------------------------------------------

def evaluate_against_common_target(pred_annual: pd.DataFrame):
    actual = load_corrected_annual_labels().rename(columns={COMMON_TARGET: "actual_lookup"})
    merged = pred_annual.merge(actual, on=["region", "year"], how="left")
    merged["actual_lookup"] = merged["actual_lookup"].fillna(0)

    rows = merged.rename(columns={"actual_lookup": "actual", "predicted_flood_days": "predicted"})[
        ["region", "ward_uid", "year", "actual", "predicted"]
    ]
    metrics, collapsed = compute_regionyear_metrics(rows)
    print(f"  [Region-year, TRUE N={len(collapsed)}] -> {metrics}  <- headline")
    return metrics, rows


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
# 10. MAIN
# ---------------------------------------------------------------------

def main():
    if QUICK_TEST_MODE:
        print("!" * 60)
        print("QUICK_TEST_MODE = True -- fast pipeline validation run, NOT final numbers.")
        print("!" * 60 + "\n")

    lr_metrics, rf_metrics, naive_metrics, lr_rows, rf_rows, naive_rows = train_lr_rf()

    ward_day_df = build_ward_day_in_memory()

    lstm_preds = train_sequence_model(ward_day_df, use_graph=False)
    print("\nEvaluating LSTM against common annual target...")
    lstm_metrics, lstm_rows = evaluate_against_common_target(lstm_preds)

    gnn_preds = train_sequence_model(ward_day_df, use_graph=True)
    print("\nEvaluating LSTM+GNN against common annual target...")
    gnn_metrics, gnn_rows = evaluate_against_common_target(gnn_preds)

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

    print("\n" + "=" * 60)
    print("READING THESE NUMBERS")
    print("=" * 60)
    print("1. 'Naive baseline' predicts each region's own training-period mean --")
    print("   it is the floor every real model needs to beat. If a model's NSE")
    print("   is below the baseline's, it is adding negative value.")
    print("2. All metrics are TRUE region-year grain (N ~= 30 test rows across 5")
    print("   regions), not ward-year (which inflates N ~60x via duplication).")
    print("3. 2020's flood-days are corrected via a data-driven per-event")
    print("   duration cap (14 days, the natural break in the real event-length")
    print("   distribution), not an arbitrary year-specific patch -- see")
    print("   build_corrected_flood_labels.py docstring for the evidence.")
    print("4. LR/RF features were narrowed from 92 to whichever pass the")
    print("   within-region-signal check above -- see the printed selection.")


if __name__ == "__main__":
    main()