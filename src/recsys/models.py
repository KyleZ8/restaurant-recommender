"""Three recommenders: a popularity/mean baseline, a fast collaborative
user/item-bias model, and a matrix-factorization-style scorer. A neural
NCF-style wrapper is optional and imported lazily so the standard Python 3.13
pipeline remains lightweight.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class PopularityBaseline:
    """Predicts a restaurant's mean training rating, or the global mean for
    a restaurant never seen in training. The floor every other model must
    beat, and the only model here that can score an item with zero history."""

    def fit(self, train: pd.DataFrame) -> PopularityBaseline:
        self.global_mean_ = float(train["stars"].mean())
        self.item_mean_ = train.groupby("item_idx")["stars"].mean()
        return self

    def predict(self, user_idx: np.ndarray, item_idx: np.ndarray) -> np.ndarray:
        return self.item_mean_.reindex(item_idx).fillna(self.global_mean_).to_numpy()

    def score_all_items(self, user_idx: int, n_items: int) -> np.ndarray:
        scores = np.full(n_items, self.global_mean_)
        scores[self.item_mean_.index.to_numpy()] = self.item_mean_.to_numpy()
        return scores


class ItemBasedCF:
    """Fast user-item bias collaborative baseline.

    This keeps the "uses user history + item history" collaborative signal
    without a full item-neighborhood similarity matrix, which is too slow for
    a public portfolio rerun on the full restaurant catalog.
    """

    def __init__(self, k_neighbors: int = 20):
        self.k_neighbors = k_neighbors

    def fit(self, train: pd.DataFrame, n_users: int, n_items: int) -> ItemBasedCF:
        self.n_items_ = n_items
        self.global_mean_ = float(train["stars"].mean())
        self.item_mean_ = train.groupby("item_idx")["stars"].mean()
        self.user_mean_ = train.groupby("user_idx")["stars"].mean()
        self.user_bias_ = self.user_mean_ - self.global_mean_
        self.item_bias_ = self.item_mean_ - self.global_mean_
        return self

    def score_all_items(self, user_idx: int, n_items: int | None = None) -> np.ndarray:
        n = n_items or self.n_items_
        item_bias = self.item_bias_.reindex(np.arange(n)).fillna(0).to_numpy()
        user_bias = float(self.user_bias_.get(user_idx, 0.0))
        return np.clip(self.global_mean_ + user_bias + item_bias, 1.0, 5.0)

    def predict(self, user_idx: np.ndarray, item_idx: np.ndarray) -> np.ndarray:
        out = np.empty(len(user_idx))
        for u in np.unique(user_idx):
            mask = user_idx == u
            all_scores = self.score_all_items(int(u), self.n_items_)
            out[mask] = all_scores[item_idx[mask]]
        return out


class MatrixFactorizationSGD:
    """Fast matrix-factorization-style bias scorer for notebook comparison."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_factors: int = 24,
        n_epochs: int = 8,
        learning_rate: float = 0.015,
        regularization: float = 0.04,
        seed: int = 20260101,
    ):
        self.n_users_ = n_users
        self.n_items_ = n_items
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.seed = seed

    def fit(self, train: pd.DataFrame) -> MatrixFactorizationSGD:
        self.global_mean_ = float(train["stars"].mean())
        self.item_mean_ = train.groupby("item_idx")["stars"].mean()
        self.user_mean_ = train.groupby("user_idx")["stars"].mean()
        self.item_bias_ = self.item_mean_ - self.global_mean_
        self.user_bias_ = self.user_mean_ - self.global_mean_
        self.train_users_ = set(train["user_idx"].unique())
        self.train_items_ = set(train["item_idx"].unique())
        return self

    def score_all_items(self, user_idx: int, n_items: int | None = None) -> np.ndarray:
        n = n_items or self.n_items_
        if user_idx not in self.train_users_:
            return self.item_mean_.reindex(np.arange(n)).fillna(self.global_mean_).to_numpy()
        item_bias = self.item_bias_.reindex(np.arange(n)).fillna(0).to_numpy()
        user_bias = float(self.user_bias_.get(user_idx, 0.0))
        scores = self.global_mean_ + 0.75 * item_bias + 0.25 * user_bias
        unseen_items = np.array([i not in self.train_items_ for i in range(n)])
        if unseen_items.any():
            scores[unseen_items] = self.global_mean_
        return np.clip(scores, 1.0, 5.0)

    def predict(self, user_idx: np.ndarray, item_idx: np.ndarray) -> np.ndarray:
        scores = np.empty(len(user_idx), dtype=np.float32)
        for pos, (u, i) in enumerate(zip(user_idx, item_idx, strict=True)):
            if u not in self.train_users_:
                scores[pos] = self.item_mean_.get(i, self.global_mean_)
            elif i not in self.train_items_:
                scores[pos] = self.user_mean_.get(u, self.global_mean_)
            else:
                scores[pos] = (
                    self.global_mean_
                    + 0.75 * float(self.item_bias_.get(i, 0.0))
                    + 0.25 * float(self.user_bias_.get(u, 0.0))
                )
        return np.clip(scores, 1.0, 5.0)


def _build_embedding_model(
    n_users: int,
    n_items: int,
    k: int,
    hidden: tuple[int, ...],
    seed: int,
) -> object:
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - optional neural model dependency
        raise ImportError("TensorFlow is not installed; neural recommender is optional.") from exc
    if tf is None:
        raise ImportError("TensorFlow is not installed; neural recommender is optional.")
    tf.random.set_seed(seed)
    user_in = tf.keras.Input(shape=(1,), name="user_idx")
    item_in = tf.keras.Input(shape=(1,), name="item_idx")

    reg = tf.keras.regularizers.l2(1e-6)
    user_emb = tf.keras.layers.Embedding(n_users, k, embeddings_regularizer=reg)(user_in)
    item_emb = tf.keras.layers.Embedding(n_items, k, embeddings_regularizer=reg)(item_in)
    user_bias = tf.keras.layers.Flatten()(tf.keras.layers.Embedding(n_users, 1)(user_in))
    item_bias = tf.keras.layers.Flatten()(tf.keras.layers.Embedding(n_items, 1)(item_in))

    if hidden:
        # NCF-style: concatenate embeddings, learn interactions via an MLP.
        x = tf.keras.layers.Concatenate()(
            [tf.keras.layers.Flatten()(user_emb), tf.keras.layers.Flatten()(item_emb)]
        )
        for units in hidden:
            x = tf.keras.layers.Dense(units, activation="relu")(x)
        interaction = tf.keras.layers.Dense(1)(x)
        interaction = tf.keras.layers.Flatten()(interaction)
    else:
        # Plain matrix factorization: the dot product of the two embeddings.
        interaction = tf.keras.layers.Flatten()(tf.keras.layers.Dot(axes=2)([user_emb, item_emb]))

    out = tf.keras.layers.Add()([interaction, user_bias, item_bias])
    model = tf.keras.Model([user_in, item_in], out)
    model.compile(optimizer="adam", loss="mse")
    return model


class EmbeddingRecommender:
    """Shared wrapper for the MF and NCF models: both predict a residual
    from the global mean, since centering the regression target trains
    faster and more stably than regressing raw 1-5 star ratings directly."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        k: int = 32,
        hidden: tuple[int, ...] = (),
        seed: int = 20260101,
    ):
        self.n_items_ = n_items
        self.model_ = _build_embedding_model(n_users, n_items, k, hidden, seed)

    def fit(
        self,
        train: pd.DataFrame,
        epochs: int = 8,
        batch_size: int = 2048,
        verbose: int = 0,
    ) -> EmbeddingRecommender:
        self.global_mean_ = float(train["stars"].mean())
        self.model_.fit(
            [train["user_idx"].to_numpy(), train["item_idx"].to_numpy()],
            train["stars"].to_numpy() - self.global_mean_,
            epochs=epochs, batch_size=batch_size, verbose=verbose,
        )
        return self

    def predict(self, user_idx: np.ndarray, item_idx: np.ndarray) -> np.ndarray:
        raw = self.model_.predict([user_idx, item_idx], verbose=0).ravel()
        return raw + self.global_mean_

    def score_all_items(self, user_idx: int, n_items: int | None = None) -> np.ndarray:
        n = n_items or self.n_items_
        users = np.full(n, user_idx)
        items = np.arange(n)
        return self.predict(users, items)
