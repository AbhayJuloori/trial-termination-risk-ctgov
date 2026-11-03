"""Train and evaluate early-termination models on the start-year split, tracked in MLflow.

Models: a logistic regression baseline and LightGBM (class-weighted). LightGBM
settings are chosen on the validation years; the test years are scored once.
The best model by validation AUC is registered and gets the `champion` alias.

    python -m trials.train
"""

from __future__ import annotations

import json
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from trials.config import LAKE_DIR, REPO_ROOT, SEED, read_table

CATEGORICAL = ["phase", "allocation", "intervention_model", "primary_purpose", "masking", "sponsor_class",
               "responsible_party", "sex"]
DROP = ["nct_id", "label", "split", "start_year"]
MODEL_NAME = "trial-termination-lgbm"


def load_features(spark=None) -> pd.DataFrame:
    if spark is None:
        from trials.spark import get_spark

        spark = get_spark()
    return read_table(spark, "gold", "features").toPandas()


def prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    for c in CATEGORICAL:
        df[c] = df[c].fillna("MISSING").astype("category")
    feats = [c for c in df.columns if c not in DROP]
    return df, feats


def youden_threshold(y, p) -> float:
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


def metrics(y, p, thr) -> dict:
    return {"auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "balanced_accuracy": balanced_accuracy_score(y, (p >= thr).astype(int)),
            "positive_rate": float(np.mean(y)), "n": int(len(y))}


def fit_lgbm(Xtr, ytr, Xva, yva, params: dict):
    pos = ytr.mean()
    model = lgb.LGBMClassifier(
        n_estimators=2000, learning_rate=params["learning_rate"], num_leaves=params["num_leaves"],
        min_child_samples=params["min_child_samples"], subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        scale_pos_weight=(1 - pos) / pos, random_state=SEED, verbose=-1,
    )
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="auc",
              callbacks=[lgb.early_stopping(100, verbose=False)])
    return model


def fit_logreg(train: pd.DataFrame, feats: list[str]):
    num = [c for c in feats if c not in CATEGORICAL]
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50), CATEGORICAL),
    ])
    pipe = make_pipeline(pre, LogisticRegression(max_iter=2000, class_weight="balanced"))
    X = train[feats].copy()
    for c in CATEGORICAL:
        X[c] = X[c].astype(str)
    pipe.fit(X, train["label"])
    return pipe


def _lr_predict(pipe, df, feats):
    X = df[feats].copy()
    for c in CATEGORICAL:
        X[c] = X[c].astype(str)
    return pipe.predict_proba(X)[:, 1]


GRID = [
    {"learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 50},
    {"learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 100},
    {"learning_rate": 0.03, "num_leaves": 127, "min_child_samples": 200},
]


def main() -> int:
    import mlflow
    import mlflow.lightgbm
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient

    df, feats = prepare(load_features())
    tr, va, te = (df[df.split == s] for s in ("train", "valid", "test"))
    print({s: (len(d), round(d.label.mean(), 3)) for s, d in [("train", tr), ("valid", va), ("test", te)]}, flush=True)

    mlflow.set_tracking_uri(f"sqlite:///{REPO_ROOT / 'mlflow.db'}")
    mlflow.set_experiment("trial-termination")
    results = {}

    with mlflow.start_run(run_name="logreg_baseline") as run:
        lr = fit_logreg(tr, feats)
        pva, pte = _lr_predict(lr, va, feats), _lr_predict(lr, te, feats)
        thr = youden_threshold(va.label, pva)
        m = {"valid_" + k: v for k, v in metrics(va.label, pva, thr).items()} | {"test_" + k: v for k, v in metrics(te.label, pte, thr).items()}
        mlflow.log_params({"model": "logistic_regression", "class_weight": "balanced", "n_features": len(feats)})
        mlflow.log_metrics(m)
        mlflow.sklearn.log_model(lr, "model")
        results["logreg"] = {"run_id": run.info.run_id, **m}
        print("logreg", {k: round(v, 4) for k, v in m.items() if "auc" in k or "balanced" in k}, flush=True)

    best = None
    for i, params in enumerate(GRID):
        with mlflow.start_run(run_name=f"lgbm_{i}") as run:
            model = fit_lgbm(tr[feats], tr.label, va[feats], va.label, params)
            pva, pte = model.predict_proba(va[feats])[:, 1], model.predict_proba(te[feats])[:, 1]
            thr = youden_threshold(va.label, pva)
            m = {"valid_" + k: v for k, v in metrics(va.label, pva, thr).items()} | {"test_" + k: v for k, v in metrics(te.label, pte, thr).items()}
            mlflow.log_params({"model": "lightgbm", **params, "best_iteration": model.best_iteration_,
                               "n_features": len(feats), "threshold_youden_valid": thr})
            mlflow.log_metrics(m)
            mlflow.lightgbm.log_model(model.booster_, "model")
            results[f"lgbm_{i}"] = {"run_id": run.info.run_id, "params": params, "threshold": thr, **m}
            print(f"lgbm_{i}", {k: round(v, 4) for k, v in m.items() if "auc" in k or "balanced" in k}, flush=True)
            if best is None or m["valid_auc"] > results[best]["valid_auc"]:
                best = f"lgbm_{i}"
                best_model, best_pte = model, pte

    # register the best LightGBM by validation AUC as champion
    client = MlflowClient()
    mv = mlflow.register_model(f"runs:/{results[best]['run_id']}/model", MODEL_NAME)
    client.set_registered_model_alias(MODEL_NAME, "champion", mv.version)
    results["champion"] = {"name": best, "version": mv.version}

    imp = pd.Series(best_model.booster_.feature_importance("gain"), index=feats).sort_values(ascending=False)
    te = te.assign(p=best_pte)
    by_year = {int(y): roc_auc_score(g.label, g.p) for y, g in te.groupby("start_year")}
    by_phase = {str(ph): {"auc": roc_auc_score(g.label, g.p), "n": len(g)} for ph, g in te.groupby("phase", observed=True)
                if g.label.nunique() == 2 and len(g) >= 300}
    out = {"results": results, "importance_gain": (imp / imp.sum()).head(25).to_dict(), "test_auc_by_year": by_year,
           "test_auc_by_phase": by_phase,
           "splits": {s: {"n": int(len(d)), "positive_rate": float(d.label.mean())} for s, d in [("train", tr), ("valid", va), ("test", te)]}}
    reports = LAKE_DIR / "_reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "train_results.json").write_text(json.dumps(out, indent=1, default=float))
    te[["nct_id", "label", "p", "start_year", "phase"]].to_parquet(reports / "test_predictions.parquet", index=False)
    print(json.dumps({"champion": results["champion"], "test_auc": results[best]["test_auc"]}, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
