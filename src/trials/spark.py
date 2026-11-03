"""Local Spark session with Delta Lake. On Databricks the active session is reused."""

from __future__ import annotations

import os

from pyspark.sql import SparkSession


def get_spark(app_name: str = "trial-termination") -> SparkSession:
    active = SparkSession.getActiveSession()
    if active is not None:
        return active

    from delta import configure_spark_with_delta_pip

    builder = (
        SparkSession.builder.appName(app_name)
        .master(os.environ.get("SPARK_MASTER", "local[4]"))
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.sql.shuffle.partitions", os.environ.get("SPARK_SHUFFLE_PARTITIONS", "16"))
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.driver.memory", os.environ.get("SPARK_DRIVER_MEMORY", "5g"))
        .config("spark.ui.enabled", "false")
        # constraint propagation blows up planning time on the wide, CASE-heavy
        # cast projections (hundreds of nullable columns per claim file)
        .config("spark.sql.constraintPropagation.enabled", "false")
    )
    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark
