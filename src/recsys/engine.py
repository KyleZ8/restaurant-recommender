"""DuckDB connection builder: registers the processed tables, then runs the
segmentation SQL files (sql/*.sql) in order.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

from recsys.build_data import BUSINESS_PARQUET, PROJECT_ROOT, REVIEWS_ALL_PARQUET

SQL_DIR = PROJECT_ROOT / "sql"
SQL_FILES = (
    "01_user_segments.sql",
    "02_restaurant_segments.sql",
)


def build_connection(
    business_path: Path = BUSINESS_PARQUET, reviews_path: Path = REVIEWS_ALL_PARQUET
) -> duckdb.DuckDBPyConnection:
    """Registers `business` and `reviews_all` from the given paths and runs
    every SQL file in sql/ in order, building the segment views."""
    con = duckdb.connect()
    con.sql(f"CREATE VIEW business AS SELECT * FROM read_parquet('{business_path.as_posix()}')")
    con.sql(f"CREATE VIEW reviews_all AS SELECT * FROM read_parquet('{reviews_path.as_posix()}')")

    for filename in SQL_FILES:
        con.sql((SQL_DIR / filename).read_text())

    return con


if __name__ == "__main__":
    connection = build_connection()
    print(connection.sql("SELECT * FROM user_segment_summary").df())
    print(connection.sql("SELECT * FROM restaurant_segment_summary").df())
