import gzip
import json
import os
import sys

import pytest

os.environ.setdefault("PYSPARK_PYTHON", sys.executable)


def study(nct, status, start, study_type="INTERVENTIONAL", phases=("PHASE2",), sponsor="INDUSTRY",
          criteria="Inclusion Criteria:\n\n* Age 18+\n* Diagnosis of X\n\nExclusion Criteria:\n\n* Pregnancy",
          interventions=("DRUG",), arms=("EXPERIMENTAL", "PLACEBO_COMPARATOR"), countries=("United States", "Canada", "United States"),
          min_age="18 Years", max_age="6 Months", enrollment=(120, "ACTUAL"), why=None, ancestors=("Neoplasms",)):
    return {
        "protocolSection": {
            "identificationModule": {"nctId": nct},
            "statusModule": {"overallStatus": status, "startDateStruct": {"date": start, "type": "ACTUAL"},
                             "studyFirstSubmitDate": "2009-12-01", "whyStopped": why},
            "sponsorCollaboratorsModule": {"leadSponsor": {"name": "Acme", "class": sponsor},
                                           "collaborators": [{"name": "Uni", "class": "OTHER"}],
                                           "responsibleParty": {"type": "SPONSOR"}},
            "oversightModule": {"oversightHasDmc": True},
            "conditionsModule": {"conditions": ["Cancer"], "keywords": ["a", "b"]},
            "designModule": {"studyType": study_type, "phases": list(phases),
                             "designInfo": {"allocation": "RANDOMIZED", "interventionModel": "PARALLEL",
                                            "primaryPurpose": "TREATMENT",
                                            "maskingInfo": {"masking": "DOUBLE", "whoMasked": ["PARTICIPANT", "INVESTIGATOR"]}},
                             "enrollmentInfo": {"count": enrollment[0], "type": enrollment[1]}},
            "armsInterventionsModule": {"armGroups": [{"label": a, "type": a} for a in arms],
                                        "interventions": [{"type": t, "name": t} for t in interventions]},
            "outcomesModule": {"primaryOutcomes": [{"measure": "m1"}], "secondaryOutcomes": [{"measure": "m2"}, {"measure": "m3"}]},
            "eligibilityModule": {"eligibilityCriteria": criteria, "healthyVolunteers": False, "sex": "ALL",
                                  "minimumAge": min_age, "maximumAge": max_age},
            "contactsLocationsModule": {"locations": [{"country": c, "facility": f"site{i}"} for i, c in enumerate(countries)]},
        },
        "derivedSection": {"conditionBrowseModule": {"ancestors": [{"term": a} for a in ancestors]}},
    }


STUDIES = [
    study("NCT00000001", "COMPLETED", "2010-03-15"),
    study("NCT00000002", "TERMINATED", "2012-06", why="Slow accrual", enrollment=(7, "ACTUAL")),
    study("NCT00000003", "WITHDRAWN", "2015-01-01", enrollment=(0, "ACTUAL")),
    study("NCT00000004", "SUSPENDED", "2017-05-05"),
    study("NCT00000005", "RECRUITING", "2018-01-01"),  # open: excluded
    study("NCT00000006", "COMPLETED", "2016-02-02", study_type="OBSERVATIONAL"),  # excluded
    study("NCT00000007", "COMPLETED", "2019-09-09"),  # outside split years
    study("NCT00000008", "COMPLETED", "2013-01-01", phases=("PHASE1", "PHASE2"), sponsor="OTHER", criteria=None,
          interventions=("DEVICE", "PROCEDURE"), countries=()),
    study("NCT00000001", "COMPLETED", "2010-03-15"),  # duplicate record
]


@pytest.fixture(scope="session")
def raw_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("studies")
    with gzip.open(d / "page_00000.jsonl.gz", "wt") as fh:
        for s in STUDIES:
            fh.write(json.dumps(s) + "\n")
        fh.write(json.dumps({"protocolSection": {"identificationModule": {"nctId": "BAD"}}}) + "\n")
    return d


@pytest.fixture(scope="session")
def spark():
    from trials.spark import get_spark

    s = get_spark("trials-tests")
    yield s
    s.stop()


@pytest.fixture(scope="session")
def layers(spark, raw_dir):
    from trials import pipeline as pl

    bronze = pl.build_bronze(spark, raw_dir, write=False)
    silver, quarantine = pl.build_silver(spark, bronze, write=False)
    labeled = pl.label_and_split(silver)
    feats = pl.build_features(labeled)
    return {"bronze": bronze, "silver": silver, "quarantine": quarantine, "labeled": labeled,
            "features": {r["nct_id"]: r.asDict() for r in feats.collect()}, "feature_cols": feats.columns}
