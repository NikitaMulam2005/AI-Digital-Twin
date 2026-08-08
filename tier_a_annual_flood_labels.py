"""
Tier A (annual) -- Merge IFI Flood Labels into ward_year_master.csv
--------------------------------------------------------------------------
Your ward_year_master.csv is a ward x YEAR panel (one row per ward per
year, 2001-2024) -- but it currently has zero flood/waterlogging label
columns (per your own merge script's docstring: "waterlogging is still
skipped"). This script closes that gap at the grain your file actually
uses: annual, broadcast per region.

This is DIFFERENT from (and complements, doesn't replace) the daily
tier_a_region_hazard_model.py script -- that one stays useful for
anything needing date-level resolution (e.g. an LSTM learning how risk
builds up day-by-day). This script instead answers "did this region
have a reported flood event in this particular YEAR, and how bad was
it", which is the right grain to join onto your existing annual panel
for baseline models (Linear Regression / Random Forest) that consume
ward_year_master.csv directly.

SAME LABEL CAVEATS AS THE DAILY SCRIPT APPLY:
  - IFI's real ceiling is DISTRICT, not ward. Thane district events get
    broadcast to Thane, KDMC, AND NMMC alike (can't be told apart from
    this source). Raigad (Panvel) is dropped -- too broad a proxy, and
    Panvel isn't a region in your other pipeline files anyway.
  - IFI captures IMD-reported/damaging floods, not every minor
    waterlogging incident -- a "0" here means "no reported flood found",
    not "provably no waterlogging happened".

REQUIREMENTS
    pip install pandas numpy

HOW TO USE
1. Point IFI_RAW_CSV and WARD_MASTER_CSV at your real files
2. Run: python tier_a_annual_flood_labels.py
3. Output: ward_year_master_with_flood_labels.csv -- your original file
   plus five new columns per row:
     - region_flood_event_count : # distinct IFI events matched this
       region in this year
     - region_flood_days_count  : total days across those events
       (overlapping events on the same day counted once)
     - region_any_flood_year    : 1 if event_count > 0, else 0
     - region_max_severity      : highest Severity value among matched
       events this year (raw IFI text/value, NaN if none)
     - region_total_fatalities  : sum of Human fatality across matched
       events this year (0 if none reported, NaN if no events at all)
"""

import re
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------

IFI_RAW_CSV = "India_Flood_Inventory_v3.csv"
WARD_MASTER_CSV = "ward_year_master.csv"
OUTPUT_CSV = "ward_year_master_with_flood_labels.csv"

REGIONS = ["BMC", "Thane", "KDMC", "NMMC", "Pune"]

# Same district->region mapping as the daily script, for consistency.
DISTRICT_TO_REGIONS = {
    "mumbai": ["BMC"],
    "mumbai suburban": ["BMC"],
    "thane": ["Thane", "KDMC", "NMMC"],
    "pune": ["Pune"],
}


# ---------------------------------------------------------------------
# 2. PARSE IFI EVENTS -> REGION x YEAR SUMMARY
# ---------------------------------------------------------------------

def load_ifi_region_year_summary(ifi_csv: str, regions: list[str]) -> pd.DataFrame:
    print(f"Loading IFI events from {ifi_csv} ...")
    df = pd.read_csv(ifi_csv)
    df["Start Date"] = pd.to_datetime(df["Start Date"], format="%d-%m-%Y %H:%M", errors="coerce")
    df["End Date"] = pd.to_datetime(df["End Date"], format="%d-%m-%Y %H:%M", errors="coerce")
    df = df.dropna(subset=["Start Date", "Districts"])
    print(f"  {len(df)} events have a parseable Start Date and non-empty Districts.")

    day_rows = []
    for _, ev in df.iterrows():
        end = ev["End Date"] if pd.notna(ev["End Date"]) else ev["Start Date"]
        if end < ev["Start Date"]:
            end = ev["Start Date"]
        districts = [d.strip().lower() for d in re.split(r"[;,]", str(ev["Districts"])) if d.strip()]

        matched_regions = set()
        for d in districts:
            if d in DISTRICT_TO_REGIONS:
                matched_regions.update(DISTRICT_TO_REGIONS[d])
        matched_regions &= set(regions)
        if not matched_regions:
            continue

        date_range = pd.date_range(ev["Start Date"], end, freq="D")
        for region in matched_regions:
            for d in date_range:
                day_rows.append({
                    "region": region, "date": d, "year": d.year,
                    "event_uei": ev.get("UEI"),
                    "severity": ev.get("Severity"),
                    "fatalities": pd.to_numeric(ev.get("Human fatality"), errors="coerce"),
                })

    if not day_rows:
        print("  WARNING: no events matched any target region/year -- "
              "output will have all-zero flood labels. Check "
              "DISTRICT_TO_REGIONS against your IFI file's Districts "
              "values if this is unexpected.")
        return pd.DataFrame(columns=[
            "region", "year", "region_flood_event_count",
            "region_flood_days_count", "region_any_flood_year",
            "region_max_severity", "region_total_fatalities",
        ])

    day_df = pd.DataFrame(day_rows).drop_duplicates(subset=["region", "date", "event_uei"])
    print(f"  Matched {day_df['event_uei'].nunique()} unique events across "
          f"{day_df['region'].nunique()} regions.")

    # Per (region, year): count distinct events, count distinct
    # flood-days (de-duplicated across overlapping events), take max
    # severity, sum fatalities per distinct event (not per day, so a
    # 7-day event's fatality count isn't multiplied by 7).
    event_year = day_df.drop_duplicates(subset=["region", "event_uei"])[
        ["region", "year", "event_uei", "severity", "fatalities"]
    ]
    summary = event_year.groupby(["region", "year"]).agg(
        region_flood_event_count=("event_uei", "nunique"),
        region_max_severity=("severity", lambda s: s.dropna().max() if s.notna().any() else np.nan),
        region_total_fatalities=("fatalities", "sum"),
    ).reset_index()

    days_count = day_df.drop_duplicates(subset=["region", "date"]).groupby(
        ["region", "year"]
    ).size().reset_index(name="region_flood_days_count")

    summary = summary.merge(days_count, on=["region", "year"], how="left")
    summary["region_any_flood_year"] = 1
    print(f"  Built region-year summary: {len(summary)} (region, year) "
          f"rows with at least one matched event.")
    return summary


# ---------------------------------------------------------------------
# 3. MERGE INTO ward_year_master.csv
# ---------------------------------------------------------------------

def merge_into_master(master_csv: str, summary: pd.DataFrame, out_csv: str):
    print(f"\nLoading {master_csv} ...")
    master = pd.read_csv(master_csv)
    print(f"  {len(master)} rows, {len(master.columns)} columns before merge.")

    if "region" not in master.columns or "year" not in master.columns:
        raise KeyError(
            f"Expected 'region' and 'year' columns in {master_csv}, found: "
            f"{list(master.columns)}"
        )

    merged = master.merge(summary, on=["region", "year"], how="left")

    fill_zero_cols = ["region_flood_event_count", "region_flood_days_count",
                       "region_any_flood_year"]
    for c in fill_zero_cols:
        merged[c] = merged[c].fillna(0).astype(int)
    # severity/fatalities stay NaN when there was no event -- filling
    # fatalities with 0 would be defensible (no event = no deaths), but
    # severity has no natural "zero", so both are left as NaN for years
    # with no matched flood, and fatalities filled to 0 only where an
    # event WAS matched but the field itself was blank in IFI.
    merged["region_total_fatalities"] = merged["region_total_fatalities"].where(
        merged["region_any_flood_year"] == 1, np.nan
    )

    n_flood_rows = (merged["region_any_flood_year"] == 1).sum()
    positive_rate = n_flood_rows / len(merged)
    print(f"  {n_flood_rows} of {len(merged)} ward-year rows now flagged "
          f"as a flood year for their region ({positive_rate:.1%}).")
    print("\n  Flood years by region:")
    print(merged[merged["region_any_flood_year"] == 1]
          .groupby("region")["year"].apply(lambda s: sorted(s.unique()))
          .to_string())

    if positive_rate > 0.6:
        print("\n" + "!" * 60)
        print("WARNING: region_any_flood_year is positive in "
              f"{positive_rate:.0%} of rows. At district-year grain, "
              "IMD's archive logs SOME flood-related report almost every "
              "monsoon across a district this large -- so 'did it "
              "flood this year' has barely any negative class left to "
              "learn from and is a poor binary classification target "
              "at this granularity.")
        print("RECOMMENDATION: use region_flood_event_count or "
              "region_flood_days_count as a REGRESSION target instead "
              "(predict flood MAGNITUDE for the year, not presence/"
              "absence) -- these do vary meaningfully year to year. "
              "Reserve true binary flood/no-flood classification for "
              "the daily-resolution tier_a_region_hazard_model.py "
              "script, where the negative class is genuinely large.")
        print("!" * 60)

    merged.to_csv(out_csv, index=False)
    print(f"\nSaved -> {out_csv} ({len(merged)} rows, {len(merged.columns)} columns)")


# ---------------------------------------------------------------------
# 4. MAIN
# ---------------------------------------------------------------------

def main():
    summary = load_ifi_region_year_summary(IFI_RAW_CSV, REGIONS)
    merge_into_master(WARD_MASTER_CSV, summary, OUTPUT_CSV)


if __name__ == "__main__":
    main()