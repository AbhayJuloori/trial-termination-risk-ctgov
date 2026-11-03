"""Build bronze, silver and the feature table, with count checks and a data profile.

    python -m trials.run
"""

from __future__ import annotations

import gzip
import json
import sys
import time

from pyspark.sql import functions as F

from trials.config import LAKE_DIR, RAW_DIR
from trials.pipeline import build_all, label_and_split
from trials.spark import get_spark


def count_raw_lines() -> int:
    return sum(sum(1 for _ in gzip.open(p, "rt")) for p in sorted(RAW_DIR.glob("page_*.jsonl.gz")))


def main() -> int:
    t0 = time.time()
    spark = get_spark()
    bronze, silver, quarantine, feats = build_all(spark)
    raw = count_raw_lines()
    nb, ns, nq = bronze.count(), silver.count(), quarantine.count()
    dupes = nb - nq - ns
    labeled = label_and_split(silver)
    profile = {
        "raw_lines": raw, "bronze": nb, "silver": ns, "quarantine": nq, "duplicates_dropped": dupes,
        "status_counts": {r["overall_status"]: r["count"] for r in silver.groupBy("overall_status").count().collect()},
        "study_types": {r["study_type"]: r["count"] for r in silver.groupBy("study_type").count().collect()},
        "labeled_closed_interventional": labeled.count(),
        "labeled_by_split": {str(r["split"]): {"n": r["n"], "positive_rate": r["pos"]} for r in
                             labeled.groupBy("split").agg(F.count("*").alias("n"), F.avg("label").alias("pos")).collect()},
        "positive_rate_by_start_year": {r["start_year"]: [r["n"], r["pos"]] for r in
                                        labeled.filter("start_year BETWEEN 1999 AND 2025").groupBy("start_year")
                                        .agg(F.count("*").alias("n"), F.avg("label").alias("pos")).orderBy("start_year").collect()},
        "enrollment_type_closed": {r["enrollment_type"]: r["count"] for r in labeled.groupBy("enrollment_type").count().collect()},
        "features_rows": feats.count(), "n_feature_columns": len(feats.columns) - 5,
    }
    checks = [
        {"check": "raw lines = bronze rows", "left": raw, "right": nb, "passed": raw == nb},
        {"check": "bronze = silver + quarantine + duplicates", "left": nb, "right": ns + nq + dupes, "passed": dupes >= 0},
        {"check": "features rows = labeled rows", "left": profile["labeled_closed_interventional"], "right": profile["features_rows"],
         "passed": profile["labeled_closed_interventional"] == profile["features_rows"]},
    ]
    reports = LAKE_DIR / "_reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "profile.json").write_text(json.dumps(profile, indent=1, default=float))
    (reports / "checks.json").write_text(json.dumps(checks, indent=1))
    print(json.dumps(profile, indent=1, default=float))
    print(f"checks: {sum(c['passed'] for c in checks)}/{len(checks)} passed; {time.time() - t0:.0f}s")
    return 0 if all(c["passed"] for c in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
