"""
Corrected Flood Labels -- Data-Driven Event-Duration Cap
--------------------------------------------------------------------------
REPLACES the arbitrary "cap 2020 at 20 because that's just under 2005"
patch used in earlier script versions. That patch picked its ceiling to
match a desired outcome, not from evidence, and applied it only to 2020.

WHAT'S ACTUALLY WRONG, WITH EVIDENCE
Both region_flood_days_count (annual, used by LR/RF) and
region_day_flooded_broadcast (daily, used by LSTM/LSTM+GNN) are built by
expanding each IFI event's [Start Date, End Date] into one row per day,
per matched region, then counting/flagging every one of those days.
Across the entire 1990-2024 IFI file, 346 events match our 5 regions.
95% of them last <=7 days, 98% last <=14 days -- then there's a clean
gap in the distribution before 6 outlier events at 32, 32, 62, 62, 79,
93 days. Three of those six -- the 93-, 62-, and 62-day ones -- all fall
in 2020 and contribute 651 "flood-days" to our regions between them,
more than 8x any other single year in the 55-year file (next highest:
79, from one 1975 event). A single "flood event" spanning April to
June, or May to August, isn't a real continuous flood; this looks like
a seasonal/monsoon summary logged with event-level start/end dates
rather than genuine day-by-day flood reporting, and it distorts every
region it touches for that year.

THE FIX
Cap each individual EVENT's duration at EVENT_DURATION_CAP_DAYS (=14,
chosen as the 98th percentile / the natural gap in the real duration
distribution -- see analysis in chat) BEFORE expanding to daily rows.
This is applied uniformly to every event, in every year, in every
region -- not a year-specific patch. Years/regions with only ordinary
short events are completely unaffected (their events are all under the
cap already). Only the 6 outlier events lose their excess tail.

This produces two corrected label tables:
  - CORRECTED_ANNUAL_CSV: region, year, region_flood_days_count
    (replaces the same-named column in final_ward_year_train/test.csv)
  - CORRECTED_DAILY_CSV: region, date, region_day_flooded_broadcast
    (replaces the same-named column build_ward_day_dataset.py produces)

HOW TO USE
Run this once (`python build_corrected_flood_labels.py`) after cloning
the repo. It only needs India_Flood_Inventory_v3.csv as input. Then
train_and_compare_models.py (the corrected version) merges these two
files in automatically, overriding the raw uncapped columns.
"""

import re
import numpy as np
import pandas as pd

IFI_RAW_CSV = "India_Flood_Inventory_v3.csv"
CORRECTED_ANNUAL_CSV = "region_flood_days_count_CORRECTED.csv"
CORRECTED_DAILY_CSV = "region_day_flooded_CORRECTED.csv"

REGIONS = ["BMC", "Thane", "KDMC", "NMMC", "Pune"]
DISTRICT_TO_REGIONS = {
    "mumbai": ["BMC"],
    "mumbai suburban": ["BMC"],
    "thane": ["Thane", "KDMC", "NMMC"],
    "pune": ["Pune"],
}

# Data-driven, not outcome-driven: p98 of the real event-duration
# distribution across all 346 matched events, 1990-2024, and it sits
# right in the natural gap before the 6 outlier events (32-93 days).
# See this script's docstring / chat analysis for the full distribution.
EVENT_DURATION_CAP_DAYS = 14

START_YEAR = 2001
END_YEAR = 2024


def build_corrected_labels():
    print(f"Loading raw IFI events from {IFI_RAW_CSV} ...")
    df = pd.read_csv(IFI_RAW_CSV)
    df["Start Date"] = pd.to_datetime(df["Start Date"], format="%d-%m-%Y %H:%M", errors="coerce")
    df["End Date"] = pd.to_datetime(df["End Date"], format="%d-%m-%Y %H:%M", errors="coerce")
    df = df.dropna(subset=["Start Date", "Districts"])

    day_rows = []
    n_matched_events = 0
    n_capped = 0
    for _, ev in df.iterrows():
        end = ev["End Date"] if pd.notna(ev["End Date"]) else ev["Start Date"]
        if end < ev["Start Date"]:
            end = ev["Start Date"]

        districts = [d.strip().lower() for d in re.split(r"[;,]", str(ev["Districts"])) if d.strip()]
        matched = set()
        for d in districts:
            if d in DISTRICT_TO_REGIONS:
                matched.update(DISTRICT_TO_REGIONS[d])
        matched &= set(REGIONS)
        if not matched:
            continue

        n_matched_events += 1
        capped_end = min(end, ev["Start Date"] + pd.Timedelta(days=EVENT_DURATION_CAP_DAYS - 1))
        if capped_end < end:
            n_capped += 1

        for region in matched:
            for d in pd.date_range(ev["Start Date"], capped_end, freq="D"):
                day_rows.append({"region": region, "date": d, "event_uei": ev["UEI"]})

    print(f"  {n_matched_events} events matched our 5 regions; {n_capped} of those "
          f"exceeded the {EVENT_DURATION_CAP_DAYS}-day cap and were truncated "
          f"(uniformly, across all years -- not just 2020).")

    day_df = pd.DataFrame(day_rows).drop_duplicates(subset=["region", "date", "event_uei"])
    daily = day_df.drop_duplicates(subset=["region", "date"])[["region", "date"]].copy()
    daily["region_day_flooded_broadcast"] = 1
    daily["year"] = daily["date"].dt.year
    daily = daily[(daily["year"] >= START_YEAR) & (daily["year"] <= END_YEAR)]

    annual = (
        daily.groupby(["region", "year"]).size().reset_index(name="region_flood_days_count")
    )

    print(f"\nCorrected annual flood-days, 2019-2024 (replaces the raw/uncapped column):")
    print(annual[annual.year.between(2019, 2024)]
          .pivot(index="year", columns="region", values="region_flood_days_count")
          .fillna(0).astype(int).to_string())

    daily[["region", "date", "region_day_flooded_broadcast"]].to_csv(CORRECTED_DAILY_CSV, index=False)
    annual.to_csv(CORRECTED_ANNUAL_CSV, index=False)
    print(f"\nSaved -> {CORRECTED_DAILY_CSV} ({len(daily)} rows)")
    print(f"Saved -> {CORRECTED_ANNUAL_CSV} ({len(annual)} rows)")
    return daily, annual


if __name__ == "__main__":
    build_corrected_labels()