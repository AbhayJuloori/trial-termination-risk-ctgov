"""Model plumbing on synthetic data: categorical prep, LightGBM fit, threshold and metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from trials.train import CATEGORICAL, fit_lgbm, fit_logreg, metrics, prepare, youden_threshold, _lr_predict


def _frame(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.choice(["A", "B", None], n) for c in CATEGORICAL})
    df["n_sites"] = rng.poisson(5, n)
    df["elig_chars"] = rng.normal(1500, 400, n)
    logit = 0.5 + 1.5 * (df.phase == "A") - 0.4 * df.n_sites
    df["label"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df["nct_id"], df["split"], df["start_year"] = [f"NCT{i:08d}" for i in range(n)], "train", 2010
    return df


def test_prepare_marks_categoricals_and_drops_ids():
    df, feats = prepare(_frame())
    assert all(str(df[c].dtype) == "category" for c in CATEGORICAL)
    assert "label" not in feats and "nct_id" not in feats and "start_year" not in feats


def test_models_learn_signal():
    tr, feats = prepare(_frame(seed=1))
    va, _ = prepare(_frame(seed=2))
    m = fit_lgbm(tr[feats], tr.label, va[feats], va.label, {"learning_rate": 0.1, "num_leaves": 7, "min_child_samples": 20})
    p = m.predict_proba(va[feats])[:, 1]
    res = metrics(va.label, p, youden_threshold(va.label, p))
    assert res["auc"] > 0.65 and 0 < res["balanced_accuracy"] <= 1
    lr = fit_logreg(tr, feats)
    assert metrics(va.label, _lr_predict(lr, va, feats), 0.5)["auc"] > 0.65
