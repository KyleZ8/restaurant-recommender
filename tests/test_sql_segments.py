from __future__ import annotations

from pathlib import Path

from recsys.engine import build_connection


def test_segment_sql_on_synthetic_fixture() -> None:
    con = build_connection(
        business_path=Path("data/fixtures/business_sample.parquet"),
        reviews_path=Path("data/fixtures/reviews_sample.parquet"),
    )

    user_summary = con.sql(
        """
        SELECT user_segment, n_users, total_reviews
        FROM user_segment_summary
        ORDER BY user_segment
        """
    ).df()
    restaurant_summary = con.sql(
        """
        SELECT popularity_tier, n_restaurants
        FROM restaurant_segment_summary
        ORDER BY popularity_tier
        """
    ).df()

    assert set(user_summary["user_segment"]) == {"casual", "new", "power"}
    assert user_summary["n_users"].sum() == 131
    assert user_summary["total_reviews"].sum() == 1200
    assert user_summary.set_index("user_segment").loc["new", "n_users"] == 40
    assert user_summary.set_index("user_segment").loc["casual", "n_users"] == 56
    assert user_summary.set_index("user_segment").loc["power", "n_users"] == 35
    assert set(restaurant_summary["popularity_tier"]) == {"high", "low", "medium"}
    assert restaurant_summary["n_restaurants"].sum() == 40
