# Clinical trial early-termination risk from ClinicalTrials.gov

A model that flags interventional clinical trials likely to end early (terminated,
withdrawn or suspended) from what is known when a trial is registered. Every study record in
ClinicalTrials.gov (609,811 via API v2) goes through PySpark and Delta Lake bronze, silver
and feature steps; LightGBM is trained on trials that started 2000 to 2014, tuned on 2015 to
2016 and scored once on 2017 to 2018, with runs and a registered champion in MLflow.

**Demo page:** https://abhayjuloori.github.io/trial-termination-risk-ctgov/

## Problem

About one in seven closed interventional trials stops early, usually for slow accrual,
funding or sponsor decisions. A sponsor or CRO that could rank trials by that risk at
registration could plan sites and enrollment support where they are most needed. The hard
part is honesty: the registry rewrites several fields after a trial stops, and random
splits mix registry eras.

## Data

[ClinicalTrials.gov API v2](https://clinicaltrials.gov/data-api/api) (public domain), pulled
2026-11-03 with `scripts/download_studies.py`: 609,811 study records, protocol modules only
(no results, no free-text descriptions), saved as gzipped JSON lines.

| Step | Rows |
|---|---|
| Bronze (raw JSON per study) | 609,811 |
| Silver (typed, valid NCT id, de-duplicated) | 609,811 |
| Closed interventional trials with a start date | 304,100 |
| Train: started 2000-2014 | 131,272 (14.2% stopped early) |
| Valid: started 2015-2016 | 31,552 (14.8%) |
| Test: started 2017-2018 | 33,622 (15.6%) |

Label: `overallStatus` TERMINATED, WITHDRAWN or SUSPENDED = 1, COMPLETED = 0.

## Method

- **Bronze** keeps each study's raw JSON with its NCT id and source page. **Silver**
  parses it against an explicit schema into one typed row per study. **Features** are built in
  PySpark: phase, allocation, intervention model, primary purpose, masking, sponsor class,
  collaborators, DMC and FDA flags, condition and keyword counts, 16 MeSH disease groups,
  arm and intervention types, outcome counts, eligibility criteria length and inclusion /
  exclusion bullet counts, age limits, sex, healthy volunteers, site and country counts, US
  site, start month and days from registration to start (61 features).
- **Leakage controls.** Left out because they are written or rewritten after a trial stops:
  - *Enrollment count.* API v2 replaces the planned number with the actual one once a trial
    closes; 94% of closed trials here carry an ACTUAL count (withdrawn trials show 0).
  - *Start date type.* A withdrawn trial never starts, so its start date stays ESTIMATED. An
    early run included it and the logistic baseline reached 0.756 test AUC largely from
    that flag; removing it dropped the baseline to 0.719.
  - Completion and last-update dates, `whyStopped`, and results. A test fails if any of these
    reach the feature table.
- **Split by start year**, so every test trial started after all training trials. Trials
  started 2019 or later are not used: those already closed are skewed toward short or
  stopped studies.
- **Models.** Class-weighted logistic regression baseline and LightGBM (class weights, three
  settings, early stopping on the validation years). Threshold for balanced accuracy is
  chosen on validation (Youden). Each run logs parameters, metrics and the model to MLflow;
  the LightGBM with the best validation AUC is registered as `trial-termination-lgbm` with the
  `champion` alias.

## Results

Scored once on trials that started 2017-2018.

| Model | Valid AUC | Test AUC | Test PR AUC | Test balanced accuracy |
|---|---|---|---|---|
| Logistic regression | 0.721 | 0.719 | 0.313 | 65.8% |
| LightGBM, 31 leaves | 0.736 | 0.741 | 0.365 | 68.1% |
| LightGBM, 63 leaves | 0.739 | 0.746 | 0.386 | 68.1% |
| **LightGBM, 127 leaves (champion)** | **0.749** | **0.757** | **0.402** | **68.7%** |

- Published models on the full registry report about 0.73 to 0.75 AUC and about 67%
  balanced accuracy (Elkin and Zhu 2021). 0.757 is at the top of that range; nothing near the
  0.80+ that would suggest leakage on general registry data.
- The top score decile stops early at 44.3% against a 15.6% base rate (2.8x).
- Test AUC by start year: 0.752 (2017), 0.763 (2018). By phase: weakest for phase 2 (0.677)
  and phase 3 (0.691), strongest for phase 1 (0.763) and phase 4 (0.771).
- Largest features by gain: days from registration to start (23.5%), healthy volunteers
  (18.1%), behavioral intervention (8.2%), US site (7.6%), site count (5.9%).
- **Sensitivity.** Days from registration to start can be inflated by withdrawn trials,
  whose start date is only planned. Retrained without withdrawn trials (terminated or
  suspended vs completed), the champion settings reach 0.730 test AUC, so the signal is not
  only the withdrawn trials.

## Limitations

- Site count and design fields come from the latest registry version, not the record at
  registration; the version history API was not used.
- Withdrawn and terminated trials are one label although their causes differ.
- Class weights push scores up; use them as a ranking unless recalibrated
  (see the calibration chart on the demo page).
- Runs locally on Spark 3.5 and Delta 3.2 (pipeline about 1 minute, training a few minutes),
  not on a Databricks workspace.

## How to run

Needs Python 3.10 to 3.12 and Java 17.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python scripts/download_studies.py           # ~610 API pages, about an hour, resumable
python -m trials.run                          # bronze -> silver -> features, with count checks
python -m trials.train                        # baseline + LightGBM, MLflow, champion alias
TRIALS_VARIANT=no_withdrawn python -m trials.train   # sensitivity run
python -m trials.report                       # charts + docs/assets/kpis.json
mlflow ui --backend-store-uri sqlite:///mlflow.db
SPARK_MASTER=local[2] pytest -q
```

## Layout

```
src/trials/   config, pipeline (bronze, silver, features, label and split), run, train, report
scripts/      API download
tests/        hand-built study records: labels, split, features, leakage guard; model plumbing
docs/         demo page
```
