from __future__ import annotations

import numpy as np
import pandas as pd

from recsys.evaluate import catalog_coverage, rmse, topk_hit_rate_and_coverage, user_level_hit_rate


def test_rmse_known_answer() -> None:
    y_true = np.array([5.0, 3.0, 1.0])
    y_pred = np.array([4.0, 3.0, 3.0])

    assert round(rmse(y_true, y_pred), 4) == 1.291


def test_hit_rate_at_k_and_coverage_known_answer() -> None:
    train = pd.DataFrame(
        {
            "user_idx": [0, 1],
            "item_idx": [0, 1],
        }
    )
    test = pd.DataFrame(
        {
            "user_idx": [0, 1],
            "item_idx": [2, 3],
        }
    )
    scores = {
        0: np.array([0.0, 0.2, 0.9, 0.1]),
        1: np.array([0.1, 0.0, 0.8, 0.7]),
    }

    result = topk_hit_rate_and_coverage(
        lambda user_idx, n_items: scores[user_idx],
        train,
        test,
        n_items=4,
        k=2,
        n_eval_users=10,
    )

    assert result["hit_rate_at_2"] == 1.0
    assert result["coverage"] == 0.75


def test_catalog_coverage_known_answer() -> None:
    recs = pd.DataFrame({"item_idx": [1, 2, 2, 4]})

    assert catalog_coverage(recs, n_items=10) == 0.3


def test_user_level_hit_rate_is_a_proportion_not_row_rate() -> None:
    recs = pd.DataFrame(
        {
            "user_idx": [0, 0, 0, 1, 1, 1],
            "is_test_hit": [1, 1, 0, 0, 0, 0],
        }
    )

    assert user_level_hit_rate(recs) == 0.5
