"""Bronze (raw study JSON), silver (typed study table) and the feature table.

Only fields known when a trial is registered or starts are turned into
features. Excluded on purpose, because they are filled in or rewritten after a
trial stops: enrollment count (API v2 replaces the planned number with the
actual one once a study closes), the start date type (a withdrawn trial never
starts, so its start date stays ESTIMATED), completion dates, last update
dates, whyStopped, and any results.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

from trials.config import NEGATIVE_STATUSES, POSITIVE_STATUSES, RAW_DIR, TEST_YEARS, TRAIN_YEARS, VALID_YEARS, read_table, write_table

S, A = T.StringType(), T.ArrayType


def _struct(**fields) -> T.StructType:
    return T.StructType([T.StructField(k, v) for k, v in fields.items()])


DATE = _struct(date=S, type=S)
STUDY_SCHEMA = _struct(
    protocolSection=_struct(
        identificationModule=_struct(nctId=S, organization=_struct(fullName=S, **{"class": S})),
        statusModule=_struct(overallStatus=S, startDateStruct=DATE, studyFirstSubmitDate=S, whyStopped=S,
                             completionDateStruct=DATE, lastUpdateSubmitDate=S),
        sponsorCollaboratorsModule=_struct(
            leadSponsor=_struct(name=S, **{"class": S}),
            collaborators=A(_struct(name=S, **{"class": S})),
            responsibleParty=_struct(type=S),
        ),
        oversightModule=_struct(oversightHasDmc=T.BooleanType(), isFdaRegulatedDrug=T.BooleanType(),
                                isFdaRegulatedDevice=T.BooleanType()),
        conditionsModule=_struct(conditions=A(S), keywords=A(S)),
        designModule=_struct(
            studyType=S, phases=A(S),
            designInfo=_struct(allocation=S, interventionModel=S, primaryPurpose=S,
                               maskingInfo=_struct(masking=S, whoMasked=A(S))),
            enrollmentInfo=_struct(count=T.LongType(), type=S),
        ),
        armsInterventionsModule=_struct(armGroups=A(_struct(label=S, type=S)),
                                        interventions=A(_struct(type=S, name=S))),
        outcomesModule=_struct(primaryOutcomes=A(_struct(measure=S)), secondaryOutcomes=A(_struct(measure=S))),
        eligibilityModule=_struct(eligibilityCriteria=S, healthyVolunteers=T.BooleanType(), sex=S,
                                  minimumAge=S, maximumAge=S, stdAges=A(S)),
        contactsLocationsModule=_struct(locations=A(_struct(country=S, facility=S))),
    ),
    derivedSection=_struct(conditionBrowseModule=_struct(ancestors=A(_struct(term=S)))),
)

INTERVENTION_TYPES = ["DRUG", "DEVICE", "BIOLOGICAL", "PROCEDURE", "BEHAVIORAL", "RADIATION",
                      "DIETARY_SUPPLEMENT", "GENETIC", "COMBINATION_PRODUCT", "DIAGNOSTIC_TEST", "OTHER"]
# MeSH tree tops that appear in derivedSection ancestors
MESH_GROUPS = ["Neoplasms", "Cardiovascular Diseases", "Nervous System Diseases", "Infections",
               "Respiratory Tract Diseases", "Digestive System Diseases", "Mental Disorders",
               "Nutritional and Metabolic Diseases", "Endocrine System Diseases", "Immune System Diseases",
               "Musculoskeletal Diseases", "Skin and Connective Tissue Diseases", "Urogenital Diseases",
               "Eye Diseases", "Pregnancy Complications", "Wounds and Injuries"]


# ---- bronze -----------------------------------------------------------------


def build_bronze(spark: SparkSession, raw_dir=RAW_DIR, write: bool = True) -> DataFrame:
    ts = datetime.now(timezone.utc).replace(microsecond=0)
    df = (
        spark.read.text(str(raw_dir / "page_*.jsonl.gz"))
        .withColumnRenamed("value", "study_json")
        .withColumn("nct_id", F.get_json_object("study_json", "$.protocolSection.identificationModule.nctId"))
        .withColumn("_source_file", F.regexp_extract(F.input_file_name(), r"(page_\d+\.jsonl\.gz)", 1))
        .withColumn("_ingested_at", F.lit(ts).cast("timestamp"))
    )
    if write:
        write_table(df, "bronze", "studies")
        df = read_table(spark, "bronze", "studies")
    return df


# ---- silver -----------------------------------------------------------------


def _date(c: Column) -> Column:
    # registry dates are yyyy-MM-dd or yyyy-MM
    return F.coalesce(F.to_date(c, "yyyy-MM-dd"), F.to_date(c, "yyyy-MM"))


def _age_years(c: Column) -> Column:
    num = F.regexp_extract(c, r"^(\d+(?:\.\d+)?)", 1).cast("double")
    unit = F.lower(F.regexp_extract(c, r"([A-Za-z]+)", 1))
    return (F.when(unit.startswith("year"), num).when(unit.startswith("month"), num / 12)
            .when(unit.startswith("week"), num / 52).when(unit.startswith("day"), num / 365)
            .when(unit.startswith("hour") | unit.startswith("minute"), num / 8760))


def build_silver(spark: SparkSession, bronze: DataFrame, write: bool = True) -> tuple[DataFrame, DataFrame]:
    p = F.from_json("study_json", STUDY_SCHEMA).alias("s")
    df = bronze.select("nct_id", p)
    ps, ds = "s.protocolSection", "s.derivedSection"
    st, dm, sp = f"{ps}.statusModule", f"{ps}.designModule", f"{ps}.sponsorCollaboratorsModule"
    elig = f"{ps}.eligibilityModule"
    silver = df.select(
        "nct_id",
        F.col(f"{st}.overallStatus").alias("overall_status"),
        _date(F.col(f"{st}.startDateStruct.date")).alias("start_date"),
        F.col(f"{st}.startDateStruct.type").alias("start_date_type"),
        _date(F.col(f"{st}.studyFirstSubmitDate")).alias("first_submit_date"),
        F.col(f"{st}.whyStopped").alias("why_stopped"),  # kept for audit, never a feature
        F.col(f"{dm}.studyType").alias("study_type"),
        F.col(f"{dm}.phases").alias("phases"),
        F.col(f"{dm}.designInfo.allocation").alias("allocation"),
        F.col(f"{dm}.designInfo.interventionModel").alias("intervention_model"),
        F.col(f"{dm}.designInfo.primaryPurpose").alias("primary_purpose"),
        F.col(f"{dm}.designInfo.maskingInfo.masking").alias("masking"),
        F.size(F.coalesce(F.col(f"{dm}.designInfo.maskingInfo.whoMasked"), F.array())).alias("n_masked_roles"),
        F.col(f"{dm}.enrollmentInfo.count").alias("enrollment_count"),  # audit only: actual once closed
        F.col(f"{dm}.enrollmentInfo.type").alias("enrollment_type"),
        F.col(f"{sp}.leadSponsor.class").alias("sponsor_class"),
        F.col(f"{sp}.leadSponsor.name").alias("sponsor_name"),
        F.size(F.coalesce(F.col(f"{sp}.collaborators"), F.array())).alias("n_collaborators"),
        F.col(f"{sp}.responsibleParty.type").alias("responsible_party"),
        F.col(f"{ps}.oversightModule.oversightHasDmc").alias("has_dmc"),
        F.col(f"{ps}.oversightModule.isFdaRegulatedDrug").alias("fda_drug"),
        F.col(f"{ps}.oversightModule.isFdaRegulatedDevice").alias("fda_device"),
        F.coalesce(F.col(f"{ps}.conditionsModule.conditions"), F.array()).alias("conditions"),
        F.size(F.coalesce(F.col(f"{ps}.conditionsModule.keywords"), F.array())).alias("n_keywords"),
        F.transform(F.coalesce(F.col(f"{ds}.conditionBrowseModule.ancestors"), F.array()), lambda x: x["term"]).alias("mesh_ancestors"),
        F.transform(F.coalesce(F.col(f"{ps}.armsInterventionsModule.armGroups"), F.array()), lambda x: x["type"]).alias("arm_types"),
        F.transform(F.coalesce(F.col(f"{ps}.armsInterventionsModule.interventions"), F.array()), lambda x: x["type"]).alias("intervention_types"),
        F.size(F.coalesce(F.col(f"{ps}.outcomesModule.primaryOutcomes"), F.array())).alias("n_primary_outcomes"),
        F.size(F.coalesce(F.col(f"{ps}.outcomesModule.secondaryOutcomes"), F.array())).alias("n_secondary_outcomes"),
        F.col(f"{elig}.eligibilityCriteria").alias("eligibility_criteria"),
        F.col(f"{elig}.healthyVolunteers").alias("healthy_volunteers"),
        F.col(f"{elig}.sex").alias("sex"),
        _age_years(F.col(f"{elig}.minimumAge")).alias("min_age_years"),
        _age_years(F.col(f"{elig}.maximumAge")).alias("max_age_years"),
        F.transform(F.coalesce(F.col(f"{ps}.contactsLocationsModule.locations"), F.array()), lambda x: x["country"]).alias("site_countries"),
    )
    bad = silver.filter(F.col("nct_id").isNull() | ~F.col("nct_id").rlike(r"^NCT\d{8}$"))
    silver = silver.filter(F.col("nct_id").rlike(r"^NCT\d{8}$")).dropDuplicates(["nct_id"])
    quarantine = bad.select("nct_id", F.lit("invalid_nct_id").alias("failed_rule"))
    if write:
        write_table(silver, "silver", "studies")
        write_table(quarantine, "silver", "_quarantine")
        silver, quarantine = read_table(spark, "silver", "studies"), read_table(spark, "silver", "_quarantine")
    return silver, quarantine


# ---- features ---------------------------------------------------------------


def label_and_split(silver: DataFrame) -> DataFrame:
    """Closed interventional trials with a known start year, labeled and assigned to a split."""
    y = F.year("start_date")
    split = (F.when(y.between(*TRAIN_YEARS), "train").when(y.between(*VALID_YEARS), "valid")
             .when(y.between(*TEST_YEARS), "test"))
    return (
        silver.filter((F.col("study_type") == "INTERVENTIONAL")
                      & F.col("overall_status").isin(*POSITIVE_STATUSES, *NEGATIVE_STATUSES)
                      & F.col("start_date").isNotNull())
        .withColumn("label", F.col("overall_status").isin(*POSITIVE_STATUSES).cast("int"))
        .withColumn("start_year", y)
        .withColumn("split", split)
    )


def _phase(c: Column) -> Column:
    j = F.array_join(F.array_sort(c), "/")
    return F.when(F.size(c) == 0, "NONE").otherwise(j)


def eligibility_features(c: Column) -> list[Column]:
    text = F.coalesce(c, F.lit(""))
    lower = F.lower(text)
    excl_at = F.locate("exclusion criteria", lower)
    incl = F.when(excl_at > 0, F.substring(text, 1, 10**6).substr(F.lit(1), excl_at - 1)).otherwise(text)
    excl = F.when(excl_at > 0, text.substr(excl_at, F.lit(10**6))).otherwise(F.lit(""))
    bullets = lambda s: F.size(F.split(s, r"\n\s*(?:[*\-•]|\d+[.)])\s+")) - 1  # noqa: E731
    return [
        F.length(text).alias("elig_chars"),
        bullets(incl).alias("elig_n_inclusion"),
        bullets(excl).alias("elig_n_exclusion"),
        (excl_at > 0).cast("int").alias("elig_has_exclusion_section"),
    ]


def build_features(labeled: DataFrame) -> DataFrame:
    it = F.col("intervention_types")
    feats = labeled.select(
        "nct_id", "label", "split", "start_year", "overall_status",
        F.month("start_date").alias("start_month"),
        F.datediff("start_date", "first_submit_date").alias("days_submit_to_start"),
        _phase(F.col("phases")).alias("phase"),
        "allocation", "intervention_model", "primary_purpose", "masking", "n_masked_roles",
        "sponsor_class", "n_collaborators", "responsible_party",
        F.col("has_dmc").cast("int").alias("has_dmc"),
        F.col("fda_drug").cast("int").alias("fda_drug"),
        F.col("fda_device").cast("int").alias("fda_device"),
        F.size("conditions").alias("n_conditions"), "n_keywords",
        F.size("arm_types").alias("n_arms"),
        F.array_contains("arm_types", "PLACEBO_COMPARATOR").cast("int").alias("has_placebo_arm"),
        F.array_contains("arm_types", "ACTIVE_COMPARATOR").cast("int").alias("has_active_comparator"),
        F.size(it).alias("n_interventions"),
        *[F.array_contains(it, t).cast("int").alias(f"iv_{t.lower()}") for t in INTERVENTION_TYPES],
        *[F.array_contains("mesh_ancestors", g).cast("int").alias("mesh_" + g.split()[0].lower()) for g in MESH_GROUPS],
        "n_primary_outcomes", "n_secondary_outcomes",
        *eligibility_features(F.col("eligibility_criteria")),
        F.col("healthy_volunteers").cast("int").alias("healthy_volunteers"),
        "sex", "min_age_years", "max_age_years",
        F.size("site_countries").alias("n_sites"),
        F.size(F.array_distinct("site_countries")).alias("n_countries"),
        F.array_contains("site_countries", "United States").cast("int").alias("has_us_site"),
    )
    return feats


def build_all(spark: SparkSession, write: bool = True):
    bronze = build_bronze(spark, write=write)
    silver, quarantine = build_silver(spark, bronze, write=write)
    labeled = label_and_split(silver)
    feats = build_features(labeled)
    if write:
        write_table(feats, "gold", "features")
        feats = read_table(spark, "gold", "features")
    return bronze, silver, quarantine, feats
