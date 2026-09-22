"""Builds the working tables from a local Yelp Open Dataset download.

Never commits or copies the Yelp data anywhere in the repo -- reads directly
from a local directory (outside git) and writes only to data/raw/ and
data/processed/ (both gitignored). See ../../data/README.md.

Filtering scope is Pennsylvania restaurants only: a contained geography
with strong signal density and enough local review volume for recommenders.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

BUSINESS_PARQUET = PROCESSED_DIR / "business.parquet"
# Post city/category filter, pre 5-core.
REVIEWS_ALL_PARQUET = PROCESSED_DIR / "reviews_all.parquet"
# Post recency + 5-core + integer-encoded.
REVIEWS_CLEAN_PARQUET = PROCESSED_DIR / "reviews_clean.parquet"

RECENCY_CUTOFF = pd.Timestamp("2015-01-01")
K_CORE = 5


def filter_business(source_dir: Path) -> tuple[pd.DataFrame, set[str]]:
    biz = pd.read_json(source_dir / "yelp_academic_dataset_business.json", lines=True)
    is_restaurant = biz["categories"].str.contains("restaurants", case=False, na=False)
    biz_pa = biz.loc[(biz["state"] == "PA") & is_restaurant].copy()
    for col in ("attributes", "hours"):
        biz_pa[col] = biz_pa[col].apply(lambda x: json.dumps(x) if isinstance(x, dict) else None)
    return biz_pa[
        [
            "business_id", "name", "city", "state", "latitude", "longitude",
            "stars", "review_count", "is_open", "categories", "attributes", "hours",
        ]
    ], set(biz_pa["business_id"])


def filter_reviews(source_dir: Path, keep_ids: set[str]) -> pd.DataFrame:
    """Streams review.json line by line (too big to load whole) and keeps only
    reviews on the filtered business set. A regex pre-check on business_id
    skips a full json.loads() on lines that clearly aren't a match."""
    biz_id_re = re.compile(rb'"business_id":"([^"]+)"')
    keep_fields = (
        "review_id",
        "user_id",
        "business_id",
        "stars",
        "useful",
        "funny",
        "cool",
        "date",
    )
    review_path = source_dir / "yelp_academic_dataset_review.json"
    kept = []
    with open(review_path, "rb") as f:
        for line in f:
            m = biz_id_re.search(line)
            if m is None or m.group(1).decode("ascii") not in keep_ids:
                continue
            rec = json.loads(line)
            kept.append({k: rec[k] for k in keep_fields})
    return pd.DataFrame.from_records(kept, columns=list(keep_fields))


def k_core_filter(df: pd.DataFrame, k: int = K_CORE) -> pd.DataFrame:
    prev_len = -1
    while len(df) != prev_len:
        prev_len = len(df)
        user_counts = df["user_id"].value_counts()
        df = df.loc[df["user_id"].isin(user_counts[user_counts >= k].index)]
        item_counts = df["business_id"].value_counts()
        df = df.loc[df["business_id"].isin(item_counts[item_counts >= k].index)]
    return df


def build(source_dir: Path) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Filtering business.json for PA restaurants from {source_dir} ...")
    business, keep_ids = filter_business(source_dir)
    business.to_parquet(BUSINESS_PARQUET, index=False)
    print(f"  {len(business):,} restaurants")

    print("Streaming review.json for matching business_ids (this is the slow step) ...")
    reviews = filter_reviews(source_dir, keep_ids)
    reviews.to_parquet(REVIEWS_ALL_PARQUET, index=False)
    print(f"  {len(reviews):,} reviews, {reviews['user_id'].nunique():,} unique users")

    reviews["date"] = pd.to_datetime(reviews["date"])
    recent = reviews.loc[reviews["date"] >= RECENCY_CUTOFF].copy()
    clean = k_core_filter(recent, K_CORE)

    user_ids = clean["user_id"].unique()
    item_ids = clean["business_id"].unique()
    clean["user_idx"] = clean["user_id"].map({u: i for i, u in enumerate(user_ids)}).astype("int32")
    clean["item_idx"] = clean["business_id"].map(
        {b: i for i, b in enumerate(item_ids)}
    ).astype("int32")
    clean["stars"] = clean["stars"].astype("float32")
    clean = clean.reset_index(drop=True)
    clean.to_parquet(REVIEWS_CLEAN_PARQUET, index=False)

    print(
        f"After {RECENCY_CUTOFF.date()}+ recency and {K_CORE}-core filtering: "
        f"{len(clean):,} reviews, {clean['user_id'].nunique():,} users, "
        f"{clean['business_id'].nunique():,} restaurants"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Directory containing the raw Yelp *.json files (default: data/raw/)",
    )
    args = parser.parse_args()

    if REVIEWS_CLEAN_PARQUET.exists():
        print(f"{REVIEWS_CLEAN_PARQUET} already exists, skipping. Delete it to rebuild.")
        return

    if not (args.source_dir / "yelp_academic_dataset_business.json").exists():
        raise SystemExit(
            f"No yelp_academic_dataset_business.json found in {args.source_dir}. "
            "See data/README.md for how to obtain the Yelp Open Dataset, then pass "
            "--source-dir /path/to/it (or place it under data/raw/)."
        )

    build(args.source_dir)


if __name__ == "__main__":
    main()
