from __future__ import annotations

import pandas as pd

from recsys.split import time_based_split


def test_time_based_split_has_no_future_rows_in_train() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"]
            ),
            "stars": [1, 2, 3, 4, 5],
        }
    )

    train, test = time_based_split(df, test_fraction=0.4)

    assert train["date"].max() < test["date"].min()
    assert train["date"].tolist() == pd.to_datetime(
        ["2020-01-01", "2020-01-02", "2020-01-03"]
    ).tolist()
    assert test["date"].tolist() == pd.to_datetime(
        ["2020-01-04", "2020-01-05"]
    ).tolist()
