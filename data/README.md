# Data

**Source:** [Yelp Open Dataset](https://business.yelp.com/data/resources/open-dataset/) — business, review, user, tip, and check-in data, academic/research use.

**License:** Yelp's own Dataset License / Terms of Use — **not** a redistributable open license (not CC BY, not public domain). Yelp explicitly prohibits redistributing the dataset itself, so **no Yelp data is committed to this repository, in any form, at any stage** (raw JSON or any derived/processed table).

**How to get it yourself:**
1. Go to https://business.yelp.com/data/resources/open-dataset/ and accept Yelp's Dataset License to download the archive (`yelp_academic_dataset_business.json`, `..._review.json`, `..._user.json`, `..._tip.json`, `..._checkin.json`).
2. Extract the JSON files to a local directory outside this repo (e.g. `~/yelp_dataset/`) — do **not** place them under `data/` here, since `data/raw/` and `data/processed/` are both gitignored but still local disk, and this keeps the real dataset unambiguously outside any git-tracked tree.
3. Run `make data SOURCE_DIR=/path/to/yelp_dataset` (defaults to `data/raw/` if the files are already there). This filters to Pennsylvania restaurants, builds the working tables, and writes them to `data/processed/` (gitignored) as Parquet.

**Fixture (committed, <1 MB, no real Yelp data):** `data/fixtures/{business_sample,reviews_sample}.parquet` — fully synthetic data (seed `20260101`) with the same column schema as the real working tables (`business_id`, `stars`, `review_count`, `categories`, ... / `review_id`, `user_id`, `business_id`, `stars`, `date`, ...), sized for fast tests and CI, not for meaningful model metrics.

**Scope of the real dataset used:** filtered to Pennsylvania restaurants (`state == 'PA'` and `categories` containing "Restaurants"), overwhelmingly the Philadelphia metro in practice — 1,100,250 reviews, 312,427 unique users, 12,641 unique businesses before the recency/5-core filtering `make data` also applies.
