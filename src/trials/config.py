"""Paths, label definition and the start-year split."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = Path(os.environ.get("TRIALS_RAW_DIR", REPO_ROOT / "data" / "raw" / "studies"))
LAKE_DIR = Path(os.environ.get("TRIALS_LAKE_DIR", REPO_ROOT / "data" / "lakehouse"))

POSITIVE_STATUSES = ("TERMINATED", "WITHDRAWN", "SUSPENDED")
NEGATIVE_STATUSES = ("COMPLETED",)

# Split by the year the trial started so no later trial informs an earlier one.
# Recent start years are left out: trials that started recently and are already
# closed are mostly short or stopped ones, which would skew the test set.
TRAIN_YEARS = (2000, 2014)
VALID_YEARS = (2015, 2016)
TEST_YEARS = (2017, 2018)

SEED = 42


def table_path(layer: str, name: str) -> str:
    return str(LAKE_DIR / layer / name)


def write_table(df, layer: str, name: str) -> None:
    df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(table_path(layer, name))


def read_table(spark, layer: str, name: str):
    return spark.read.format("delta").load(table_path(layer, name))
