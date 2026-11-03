"""Charts and docs/assets/kpis.json from the run and training reports.

    python -m trials.report
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

from trials.config import LAKE_DIR, REPO_ROOT  # noqa: E402

ASSETS = REPO_ROOT / "docs" / "assets"
REPORTS = LAKE_DIR / "_reports"
BLUE, ORANGE, GREY = "#2f6fb0", "#d9822b", "#9aa0a6"


def _style(ax, title):
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=9)


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(ASSETS / name, dpi=150)
    plt.close(fig)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    prof = json.loads((REPORTS / "profile.json").read_text())
    res = json.loads((REPORTS / "train_results.json").read_text())
    pred = pd.read_parquet(REPORTS / "test_predictions.parquet")
    y, p = pred.label.values, pred.p.values

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.8))
    fpr, tpr, _ = roc_curve(y, p)
    champ = res["results"][res["results"]["champion"]["name"]]
    a1.plot(fpr, tpr, color=BLUE, label=f"LightGBM (AUC {champ['test_auc']:.3f})")
    a1.plot([0, 1], [0, 1], color=GREY, ls="--", lw=1)
    a1.set_xlabel("False positive rate", fontsize=9)
    a1.set_ylabel("True positive rate", fontsize=9)
    a1.legend(frameon=False, fontsize=8, loc="lower right")
    _style(a1, "ROC, test years")
    prec, rec, _ = precision_recall_curve(y, p)
    a2.plot(rec, prec, color=ORANGE, label=f"PR AUC {champ['test_pr_auc']:.3f}")
    a2.axhline(y.mean(), color=GREY, ls="--", lw=1, label=f"base rate {y.mean():.3f}")
    a2.set_xlabel("Recall", fontsize=9)
    a2.set_ylabel("Precision", fontsize=9)
    a2.legend(frameon=False, fontsize=8)
    _style(a2, "Precision-recall, test years")
    _save(fig, "roc_pr.png")

    imp = pd.Series(res["importance_gain"]).head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.barh(imp.index, imp.values, color=BLUE)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    _style(ax, "LightGBM feature importance (share of gain)")
    _save(fig, "importance.png")

    yr = prof["positive_rate_by_start_year"]
    years = sorted(int(k) for k in yr)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.bar([str(x) for x in years], [yr[str(x)][1] for x in years], color=GREY)
    for lo, hi, c in [(2000, 2014, BLUE), (2015, 2016, ORANGE), (2017, 2018, "#3c8d5a")]:
        idx = [i for i, x in enumerate(years) if lo <= x <= hi]
        if idx:
            ax.bar([str(years[i]) for i in idx], [yr[str(years[i])][1] for i in idx], color=c)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    _style(ax, "Share of closed trials that stopped early, by start year (blue train, orange valid, green test)")
    _save(fig, "rate_by_year.png")

    frac, mean_p = calibration_curve(y, p, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    ax.plot(mean_p, frac, marker="o", color=BLUE)
    ax.plot([0, 1], [0, 1], color=GREY, ls="--", lw=1)
    ax.set_xlabel("Mean predicted score", fontsize=9)
    ax.set_ylabel("Observed early-stop rate", fontsize=9)
    _style(ax, "Calibration, test years")
    _save(fig, "calibration.png")

    # top-decile lift for the page
    order = np.argsort(-p)
    top = y[order[: len(y) // 10]].mean()
    kpis = {"profile": prof, "results": res, "top_decile_rate": float(top), "test_base_rate": float(y.mean())}
    (ASSETS / "kpis.json").write_text(json.dumps(kpis, indent=1, default=float))
    print("report written")


if __name__ == "__main__":
    main()
