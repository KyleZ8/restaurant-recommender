[![CI](https://github.com/KyleZ8/restaurant-recommender/actions/workflows/ci.yml/badge.svg)](https://github.com/KyleZ8/restaurant-recommender/actions/workflows/ci.yml)

# Restaurant Recommender

## Headline Decision

Ship **matrix factorization for warm users only**, with a **popularity fallback for new users and new restaurants**, and validate it online before broad rollout. On the executed modeling sample, matrix factorization has the best RMSE (**1.4020**) and highest catalog coverage (**1.44%**), but hit rate@10 is **0.0%** for every offline model, so this should launch as a guarded A/B test rather than a blanket replacement.

## Model Comparison

| Model | RMSE | Hit rate@10 | Catalog coverage | Decision |
|---|---:|---:|---:|---|
| Matrix factorization | 1.4020 | 0.0% | 1.44% | Ship to users with enough history, behind an experiment. |
| Popularity | 1.4046 | 0.0% | 0.73% | Use as cold-start fallback. |
| Item-based CF | 1.4617 | 0.0% | 1.10% | Do not ship; weaker accuracy without enough discovery lift. |

![Model comparison](reports/figures/model_comparison.png)

## Segment Breakdown

| Segment | Best model | RMSE | Hit rate@10 | Coverage |
|---|---|---:|---:|---:|
| Casual users | Popularity | 1.4766 | 0.0% | 0.34% |
| Power users | Matrix factorization | 1.3723 | 0.0% | 1.42% |
| High-popularity restaurants | Matrix factorization | 1.3104 | 0.0% | 0.87% |
| Low-popularity restaurants | Popularity | 1.4223 | 0.0% | 0.07% |
| Medium-popularity restaurants | Popularity | 1.5375 | 0.0% | 0.51% |

![Segment hit rate](reports/figures/segment_hit_rate.png)

Cold-start is the failure mode: for new restaurants, popularity RMSE is **1.2398** versus **1.4503** for the personalized models. For warm users, matrix factorization improves RMSE to **1.2983** versus **1.3928** for popularity.

## Popularity Bias

![Popularity bias](reports/figures/popularity_bias.png)

Matrix factorization sends **19.6%** of recommendation slots to the top 10% most-reviewed restaurants, compared with **17.4%** for popularity and item-based CF. That concentration is a guardrail risk: the model improves warm-user RMSE, but it may still narrow discovery toward already-visible restaurants.

## How to Get the Data and Run

The Yelp Open Dataset is **not included** in this repository. Yelp's dataset terms do not allow redistribution, so this repo commits only a tiny synthetic fixture for tests.

1. Download the Yelp Open Dataset from Yelp after accepting its terms.
2. Extract the JSON files outside the repo, for example `~/yelp_dataset/`.
3. Run:

```bash
make setup
make data SOURCE_DIR=/path/to/yelp_dataset
make notebooks
make test
```

`make data` filters to Pennsylvania restaurants and writes gitignored Parquet files under `data/processed/`. `make notebooks` executes the data, model, and segment-evaluation notebooks against those local files. CI runs only fixture tests.

## Limitations

- Offline hit rate@10 is zero on the deterministic 20k-review modeling sample, so the ship decision depends on online validation, not offline ranking confidence.
- The recommender uses explicit star ratings; it does not include session context, distance, open hours, cuisine filters, or text/image features.
- New users and new restaurants need a fallback because collaborative signals are unavailable or sparse.
- The real Yelp data and processed Parquet files are intentionally gitignored and must never be committed.
