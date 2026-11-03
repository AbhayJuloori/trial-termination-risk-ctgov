"""Label, split, feature and leakage tests on a hand-built set of study records."""

from __future__ import annotations

from trials.train import CATEGORICAL, DROP


def test_bronze_keeps_every_line_and_silver_dedupes(layers):
    assert layers["bronze"].count() == 10
    assert layers["silver"].count() == 8  # duplicate NCT00000001 and invalid id removed
    assert [r["nct_id"] for r in layers["quarantine"].collect()] == ["BAD"]


def test_label_population(layers):
    f = layers["features"]
    assert set(f) == {"NCT00000001", "NCT00000002", "NCT00000003", "NCT00000004", "NCT00000007", "NCT00000008"}
    assert {k: v["label"] for k, v in f.items()} == {
        "NCT00000001": 0, "NCT00000002": 1, "NCT00000003": 1, "NCT00000004": 1, "NCT00000007": 0, "NCT00000008": 0}


def test_start_year_split(layers):
    f = layers["features"]
    assert f["NCT00000001"]["split"] == "train"
    assert f["NCT00000002"]["split"] == "train"  # yyyy-MM start date parsed
    assert f["NCT00000003"]["split"] == "valid"
    assert f["NCT00000004"]["split"] == "test"
    assert f["NCT00000007"]["split"] is None


def test_design_and_site_features(layers):
    r = layers["features"]["NCT00000001"]
    assert r["phase"] == "PHASE2" and r["n_masked_roles"] == 2 and r["has_placebo_arm"] == 1
    assert r["n_sites"] == 3 and r["n_countries"] == 2 and r["has_us_site"] == 1
    assert r["iv_drug"] == 1 and r["iv_device"] == 0 and r["mesh_neoplasms"] == 1
    assert r["min_age_years"] == 18 and abs(r["max_age_years"] - 0.5) < 1e-9
    assert r["days_submit_to_start"] == 104
    r8 = layers["features"]["NCT00000008"]
    assert r8["phase"] == "PHASE1/PHASE2" and r8["n_sites"] == 0 and r8["iv_procedure"] == 1


def test_eligibility_counts(layers):
    r = layers["features"]["NCT00000001"]
    assert (r["elig_n_inclusion"], r["elig_n_exclusion"], r["elig_has_exclusion_section"]) == (2, 1, 1)
    assert layers["features"]["NCT00000008"]["elig_chars"] == 0


def test_no_post_outcome_fields_are_features(layers):
    banned = {"enrollment_count", "enrollment_type", "why_stopped", "overall_status", "completion_date"}
    model_inputs = set(layers["feature_cols"]) - set(DROP)
    assert not banned & model_inputs
    assert set(CATEGORICAL) <= model_inputs
