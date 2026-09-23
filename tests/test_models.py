from __future__ import annotations

import numpy as np
import pandas as pd

from recsys.models import ItemBasedCF, MatrixFactorizationSGD


def _toy_ratings() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_idx": [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
            "item_idx": [0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3],
            "stars": [5.0, 5.0, 1.0, 4.0, 5.0, 1.0, 1.0, 5.0, 5.0, 2.0, 4.0, 5.0],
        }
    )


def test_item_based_cf_uses_neighbor_signal_not_bias_only() -> None:
    train = _toy_ratings()
    model = ItemBasedCF(k_neighbors=2, shrinkage=0.0).fit(train, n_users=4, n_items=4)

    users = np.array([0, 1, 2, 3])
    items = np.array([3, 2, 1, 0])
    cf_predictions = model.predict(users, items)

    global_mean = train["stars"].mean()
    user_bias = train.groupby("user_idx")["stars"].mean() - global_mean
    item_bias = train.groupby("item_idx")["stars"].mean() - global_mean
    bias_only = np.clip(
        global_mean
        + user_bias.reindex(users).fillna(0).to_numpy()
        + item_bias.reindex(items).fillna(0).to_numpy(),
        1.0,
        5.0,
    )

    assert model.neighbors_.shape == (4, 2)
    assert not np.allclose(cf_predictions, bias_only)


def test_matrix_factorization_sgd_loss_decreases() -> None:
    train = _toy_ratings()
    model = MatrixFactorizationSGD(
        n_users=4,
        n_items=4,
        n_factors=3,
        n_epochs=8,
        learning_rate=0.04,
        regularization=0.01,
        seed=7,
    ).fit(train)

    assert len(model.loss_history_) == 8
    assert model.loss_history_[-1] < model.loss_history_[0]
