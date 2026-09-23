"""Recommendation models used in the case study.

The portfolio comparison intentionally stays small enough to rerun locally:
a popularity baseline, an item-item collaborative filter, and explicit-feedback
matrix factorization trained with SGD. A neural embedding wrapper remains
optional and is imported lazily.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


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
    """Item-item collaborative filter with adjusted cosine neighborhoods."""

    def __init__(self, k_neighbors: int = 30, shrinkage: float = 25.0):
        self.k_neighbors = k_neighbors
        self.shrinkage = shrinkage

    def fit(self, train: pd.DataFrame, n_users: int, n_items: int) -> ItemBasedCF:
        if train.empty:
            raise ValueError("ItemBasedCF requires at least one training row")

        self.n_items_ = n_items
        self.global_mean_ = float(train["stars"].mean())
        item_mean_series = train.groupby("item_idx")["stars"].mean()
        self.item_mean_ = np.full(n_items, self.global_mean_, dtype=np.float32)
        item_mean_index = item_mean_series.index.to_numpy(dtype=np.int32)
        self.item_mean_[item_mean_index] = item_mean_series.to_numpy(dtype=np.float32)
        user_mean = train.groupby("user_idx")["stars"].mean()

        users = train["user_idx"].to_numpy(dtype=np.int32)
        items = train["item_idx"].to_numpy(dtype=np.int32)
        ratings = train["stars"].to_numpy(dtype=np.float32)
        self.user_item_ = csr_matrix((ratings, (users, items)), shape=(n_users, n_items))

        user_mean_values = user_mean.reindex(users).to_numpy(dtype=np.float32)
        centered = ratings - user_mean_values
        item_user = csr_matrix((centered, (items, users)), shape=(n_items, n_users))
        item_user_binary = csr_matrix(
            (np.ones_like(ratings), (items, users)), shape=(n_items, n_users)
        )

        row_norms = np.sqrt(item_user.multiply(item_user).sum(axis=1)).A1
        safe_norms = row_norms.copy()
        safe_norms[safe_norms == 0] = 1.0
        normalized = item_user.multiply(1.0 / safe_norms[:, None]).tocsr()

        k = min(self.k_neighbors + 1, n_items)
        nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute", n_jobs=-1)
        nn.fit(normalized)
        distances, indices = nn.kneighbors(normalized, return_distance=True)

        neighbors = np.full((n_items, self.k_neighbors), -1, dtype=np.int32)
        similarities = np.zeros((n_items, self.k_neighbors), dtype=np.float32)
        for item_idx in range(n_items):
            write_pos = 0
            for neighbor_idx, distance in zip(indices[item_idx], distances[item_idx], strict=True):
                if neighbor_idx == item_idx or write_pos >= self.k_neighbors:
                    continue
                sim = max(0.0, 1.0 - float(distance))
                if sim <= 0.0:
                    continue
                overlap = float(
                    item_user_binary[item_idx].multiply(item_user_binary[neighbor_idx]).sum()
                )
                shrunk = sim * overlap / (overlap + self.shrinkage)
                if shrunk <= 0.0:
                    continue
                neighbors[item_idx, write_pos] = int(neighbor_idx)
                similarities[item_idx, write_pos] = shrunk
                write_pos += 1

        self.neighbors_ = neighbors
        self.similarities_ = similarities
        return self

    def score_all_items(self, user_idx: int, n_items: int | None = None) -> np.ndarray:
        n = n_items or self.n_items_
        if user_idx >= self.user_item_.shape[0]:
            return self.item_mean_[:n].copy()

        user_row = self.user_item_.getrow(user_idx)
        if user_row.nnz == 0:
            return self.item_mean_[:n].copy()

        centered_ratings = np.zeros(self.n_items_, dtype=np.float32)
        centered_ratings[user_row.indices] = user_row.data - self.item_mean_[user_row.indices]
        rated_mask = np.zeros(self.n_items_, dtype=bool)
        rated_mask[user_row.indices] = True

        neighbors = self.neighbors_[:n]
        sims = self.similarities_[:n]
        safe_neighbors = np.where(neighbors >= 0, neighbors, 0)
        neighbor_values = centered_ratings[safe_neighbors]
        usable = (neighbors >= 0) & rated_mask[safe_neighbors] & (sims > 0)

        numerator = (sims * neighbor_values * usable).sum(axis=1)
        denominator = (np.abs(sims) * usable).sum(axis=1)
        scores = self.item_mean_[:n].copy()
        has_neighbors = denominator > 0
        scores[has_neighbors] = scores[has_neighbors] + numerator[has_neighbors] / denominator[
            has_neighbors
        ]
        return np.clip(scores, 1.0, 5.0)

    def predict(self, user_idx: np.ndarray, item_idx: np.ndarray) -> np.ndarray:
        out = np.empty(len(user_idx))
        for u in np.unique(user_idx):
            mask = user_idx == u
            all_scores = self.score_all_items(int(u), self.n_items_)
            out[mask] = all_scores[item_idx[mask]]
        return out


class MatrixFactorizationSGD:
    """Explicit-feedback matrix factorization trained with SGD."""

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
        if train.empty:
            raise ValueError("MatrixFactorizationSGD requires at least one training row")

        rng = np.random.default_rng(self.seed)
        self.global_mean_ = float(train["stars"].mean())
        self.item_mean_ = train.groupby("item_idx")["stars"].mean()
        self.user_mean_ = train.groupby("user_idx")["stars"].mean()
        self.user_factors_ = rng.normal(0.0, 0.05, size=(self.n_users_, self.n_factors)).astype(
            np.float32
        )
        self.item_factors_ = rng.normal(0.0, 0.05, size=(self.n_items_, self.n_factors)).astype(
            np.float32
        )
        self.user_bias_ = np.zeros(self.n_users_, dtype=np.float32)
        self.item_bias_ = np.zeros(self.n_items_, dtype=np.float32)

        users = train["user_idx"].to_numpy(dtype=np.int32)
        items = train["item_idx"].to_numpy(dtype=np.int32)
        ratings = train["stars"].to_numpy(dtype=np.float32)
        lr = self.learning_rate
        reg = self.regularization
        self.loss_history_: list[float] = []

        for _epoch in range(self.n_epochs):
            order = rng.permutation(len(train))
            squared_error = 0.0
            for pos in order:
                u = users[pos]
                i = items[pos]
                rating = ratings[pos]

                user_vec = self.user_factors_[u].copy()
                item_vec = self.item_factors_[i].copy()
                pred = self.global_mean_ + self.user_bias_[u] + self.item_bias_[i] + float(
                    np.dot(user_vec, item_vec)
                )
                err = rating - pred
                squared_error += err * err

                self.user_bias_[u] += lr * (err - reg * self.user_bias_[u])
                self.item_bias_[i] += lr * (err - reg * self.item_bias_[i])
                self.user_factors_[u] += lr * (err * item_vec - reg * user_vec)
                self.item_factors_[i] += lr * (err * user_vec - reg * item_vec)

            self.loss_history_.append(float(np.sqrt(squared_error / len(train))))

        self.train_users_ = set(users.tolist())
        self.train_items_ = set(items.tolist())
        return self

    def score_all_items(self, user_idx: int, n_items: int | None = None) -> np.ndarray:
        n = n_items or self.n_items_
        if user_idx not in self.train_users_:
            return self.item_mean_.reindex(np.arange(n)).fillna(self.global_mean_).to_numpy()
        item_factors = self.item_factors_[:n]
        scores = (
            self.global_mean_
            + self.user_bias_[user_idx]
            + self.item_bias_[:n]
            + item_factors @ self.user_factors_[user_idx]
        )
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
                    + float(self.user_bias_[u])
                    + float(self.item_bias_[i])
                    + float(np.dot(self.user_factors_[u], self.item_factors_[i]))
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
