"""RMSE, hit rate@10, and catalog coverage.

RMSE alone can't tell you whether a model would actually surface a good
top-10 list, or whether it only ever recommends the same popular handful --
hit rate and coverage answer those two questions respectively.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def catalog_coverage(
    recommendations: pd.DataFrame,
    n_items: int,
    item_col: str = "item_idx",
) -> float:
    """Share of catalog appearing at least once in a recommendation list."""
    if n_items <= 0:
        raise ValueError("n_items must be positive")
    return float(recommendations[item_col].nunique() / n_items)


def user_level_hit_rate(
    recommendations: pd.DataFrame,
    user_col: str = "user_idx",
    hit_col: str = "is_test_hit",
) -> float:
    """Share of evaluated users with at least one hit in their recommendation list."""
    if recommendations.empty:
        return 0.0
    user_hits = recommendations.groupby(user_col)[hit_col].max()
    return float(user_hits.mean())


def topk_hit_rate_and_coverage(
    score_all_items_fn,
    train: pd.DataFrame,
    test: pd.DataFrame,
    n_items: int,
    k: int = 10,
    n_eval_users: int = 300,
    seed: int = 20260101,
) -> dict:
    """For a sample of test users, ranks the full catalog (minus items already
    rated in train) and takes the top-k. Hit rate = share of sampled users
    whose top-k contains at least one restaurant they actually reviewed in
    the test period. Coverage = share of the catalog that appears in *any*
    sampled user's top-k -- evaluated on the same user sample for both
    metrics, so a run is one consistent pass over the catalog per user.
    """
    rng = np.random.default_rng(seed)

    train_items_by_user = train.groupby("user_idx")["item_idx"].apply(set)
    test_items_by_user = test.groupby("user_idx")["item_idx"].apply(set)

    eval_users = test_items_by_user.index.to_numpy()
    if len(eval_users) > n_eval_users:
        eval_users = rng.choice(eval_users, size=n_eval_users, replace=False)

    hits = 0
    recommended_items: set[int] = set()
    for user_idx in eval_users:
        scores = score_all_items_fn(int(user_idx), n_items).copy()
        already_seen = train_items_by_user.get(user_idx, set())
        if already_seen:
            scores[list(already_seen)] = -np.inf

        actual_k = min(k, n_items)
        top_k = np.argpartition(-scores, actual_k - 1)[:actual_k]
        top_k = top_k[np.argsort(-scores[top_k])]

        recommended_items.update(top_k.tolist())
        if test_items_by_user[user_idx] & set(top_k.tolist()):
            hits += 1

    return {
        "n_eval_users": len(eval_users),
        f"hit_rate_at_{k}": hits / len(eval_users),
        "coverage": len(recommended_items) / n_items,
    }
