"""Time-based train/test split.

A random split lets a model train on reviews that happened *after* a test
review it's being scored on -- leaking future information no real system
would have at serving time. Splitting on a global date cutoff instead means
every training row genuinely precedes every test row.
"""

from __future__ import annotations

import pandas as pd

TEST_FRACTION = 0.2


def time_based_split(
    df: pd.DataFrame,
    test_fraction: float = TEST_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = df["date"].quantile(1 - test_fraction)
    train = df.loc[df["date"] < cutoff].copy()
    test = df.loc[df["date"] >= cutoff].copy()
    return train, test
