"""Loaders for the public datasets behind the reference suites.

Rules every loader follows:
  * Fail loudly with a message that says what to do (never silently substitute synthetic data).
  * Return a RawData with the sensitive attribute declared, via prepare_frame().
  * Say what modelling choices were made (which column is "sensitive", how it was binarised,
    whether it stays in the feature set) in the DatasetSpec note.
Which of these have been run end-to-end against the real source is recorded in `verified`.
"""
from __future__ import annotations

import io
import urllib.request
from typing import Callable, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..data.loaders import RawData, adult_from_frame, prepare_frame
from .spec import DatasetUnavailable, cache_dir

COMPAS_URL = ("https://raw.githubusercontent.com/propublica/compas-analysis/master/"
              "compas-scores-two-years.csv")
HEART_URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/"
             "processed.cleveland.data")
MEPS_H181_URL = "https://meps.ahrq.gov/mepsweb/data_files/pufs/h181ssp.zip"
STUDENT_URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
OULAD_URL = ("https://archive.ics.uci.edu/static/public/349/"
             "open+university+learning+analytics+dataset.zip")
HMDA_CSV_API = "https://ffiec.cfpb.gov/v2/data-browser-api/view/csv"


def _fetch(url: str, filename: str, what: str, timeout: int = 300) -> bytes:
    """Bytes of `url`, downloaded once into the cache as `filename`. Any failure becomes a
    DatasetUnavailable that names the URL and where to put a manual download."""
    cached = cache_dir() / filename
    if cached.exists():
        return cached.read_bytes()
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            data = r.read()
    except Exception as exc:
        raise DatasetUnavailable(
            f"{what}: could not fetch {url} ({type(exc).__name__}). Download it manually and "
            f"save it as {cached}") from exc
    cached.write_bytes(data)
    return data


def _codes_as_categories(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """Integer survey/clinical codes (1.0, 2.0, -1.0 ...) -> category strings, so prepare_frame
    one-hot encodes them instead of treating 'region 4' as twice 'region 2'. NaN stays NaN."""
    df = df.copy()
    for c in cols:
        v = df[c]
        as_int = pd.to_numeric(v, errors="coerce")
        df[c] = np.where(v.isna(), None,
                         np.where(as_int.notna(), as_int.round().astype("Int64").astype(str), v.astype(str)))
    return df


# --------------------------------------------------------------------------- justice
def compas_from_frame(df: pd.DataFrame) -> RawData:
    """ProPublica's COMPAS two-year recidivism data, cleaned the way ProPublica's own analysis
    did (screened within 30 days of arrest, known recidivism outcome, ordinary charge degree,
    a score on file). Target: re-offended within two years. Sensitive: race, with the three tiny
    groups (Asian, Native American, Other) merged into 'Other'. The COMPAS risk score itself
    (decile_score, score_text) is NOT a feature -- it is the existing tool's *output*, and using
    it would leak. Race is excluded from the features (the real tool did not take it as input);
    proxies for it can still be present, which is exactly what the audit looks for."""
    d = df[(df["days_b_screening_arrest"].abs() <= 30) & (df["is_recid"] != -1)
           & (df["c_charge_degree"] != "O") & (df["score_text"] != "N/A")].copy()
    d["race"] = d["race"].where(d["race"].isin(["African-American", "Caucasian", "Hispanic"]), "Other")
    keep = ["age", "sex", "juv_fel_count", "juv_misd_count", "juv_other_count", "priors_count",
            "c_charge_degree", "race", "two_year_recid"]
    return prepare_frame(d[keep], target="two_year_recid", sensitive="race",
                         name="compas_recidivism", exclude_sensitive=True)


def load_compas() -> RawData:
    data = _fetch(COMPAS_URL, "compas-scores-two-years.csv", "compas", timeout=60)
    return compas_from_frame(pd.read_csv(io.BytesIO(data)))


# --------------------------------------------------------------------------- lending / income
def load_german_credit() -> RawData:
    from ..data.loaders import load_german_credit_raw
    return load_german_credit_raw()


def load_adult_strict() -> RawData:
    """UCI Adult from OpenML. Unlike load_adult(), no synthetic fallback: a reference suite
    must not quietly report results for fake data."""
    from sklearn.datasets import fetch_openml
    return adult_from_frame(fetch_openml("adult", version=2, as_frame=True).frame)


def _fairlearn_dataset(fetch_name: str, name: str, sensitive_candidates: Sequence[str],
                       **prep) -> RawData:
    """Adapt a `fairlearn.datasets.fetch_*` loader. Column names are matched
    case-insensitively against `sensitive_candidates`; if none match, the error lists the
    real columns so the spec can be corrected in one edit."""
    try:
        from fairlearn import datasets as fd
        bunch = getattr(fd, fetch_name)(as_frame=True)
    except Exception as exc:
        raise DatasetUnavailable(
            f"{name}: fairlearn.datasets.{fetch_name}() failed ({type(exc).__name__}: {exc}); "
            f"it downloads from OpenML on first use, so it needs network access once") from exc
    df = bunch.data.copy()
    real_columns = list(df.columns)                    # before our own target column is added
    df["__target__"] = bunch.target
    lowered = {str(c).lower(): c for c in real_columns}
    col = next((lowered[c.lower()] for c in sensitive_candidates if c.lower() in lowered), None)
    if col is None:
        raise DatasetUnavailable(
            f"{name}: none of {list(sensitive_candidates)} is a column of fairlearn's {fetch_name} "
            f"data; columns are {real_columns}. Update sensitive_candidates in "
            f"ethikit/domains/sources.py")
    return prepare_frame(df, target="__target__", sensitive=col, name=name, **prep)


def load_credit_card_default() -> RawData:
    return _fairlearn_dataset("fetch_credit_card", "credit_card_default", ["SEX", "x2", "X2"])


def load_bank_marketing() -> RawData:
    return _fairlearn_dataset("fetch_bank_marketing", "bank_marketing", ["age", "V1", "x1"])


# HMDA: only what the lender knew when it decided. Everything else in the public file is either
# set *because of* the decision (pricing, purchaser, denial reasons, "not applicable" codes that
# appear only on denials -- AUS, credit-score model, preapproval, HOEPA, payable-to) or is a
# protected characteristic (race, ethnicity, sex, age: ECOA forbids deciding on them).
HMDA_NUMERIC = ["loan_amount", "property_value", "income", "loan_term", "tract_population",
                "tract_minority_population_percent", "ffiec_msa_md_median_family_income",
                "tract_to_msa_income_percentage", "tract_owner_occupied_units",
                "tract_one_to_four_family_homes", "tract_median_age_of_housing_units"]
HMDA_CATEGORICAL = ["loan_type", "lien_status", "occupancy_type", "construction_method",
                    "total_units", "conforming_loan_limit", "derived_dwelling_category",
                    "debt_to_income_ratio", "county_code"]


def hmda_from_frame(df: pd.DataFrame, name: str = "hmda") -> RawData:
    """Public HMDA loan-level records -> originated (1) vs denied (0) home-purchase applications.
    Sensitive: derived_race, restricted to applications where it is 'White' or 'Black or
    African American' (other groups and 'Race Not Available' are dropped -- a modelling choice,
    stated in the spec). Race, ethnicity, sex and age are all excluded from the features, and so
    is every field that is filled in only after (or because of) the decision; see HMDA_NUMERIC /
    HMDA_CATEGORICAL for the whitelist. The census-tract minority share stays in: it is a known
    proxy for race, which is what the audit is there to find."""
    need = ["action_taken", "derived_race"] + HMDA_NUMERIC + HMDA_CATEGORICAL
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise DatasetUnavailable(f"{name}: HMDA file lacks expected columns {missing}; columns are "
                                 f"{list(df.columns)}")
    d = df[df["action_taken"].isin([1, 3]) & df["derived_race"].isin(
        ["White", "Black or African American"])].copy()
    d["race"] = d["derived_race"].map({"White": "White", "Black or African American": "Black"})
    d["approved"] = (d["action_taken"] == 1).astype(int)
    for c in HMDA_NUMERIC:
        d[c] = pd.to_numeric(d[c], errors="coerce")          # "Exempt" / "NA" -> NaN -> median
    d = _codes_as_categories(d, HMDA_CATEGORICAL)
    return prepare_frame(d[HMDA_NUMERIC + HMDA_CATEGORICAL + ["race", "approved"]], target="approved",
                         sensitive="race", name=name, exclude_sensitive=True)


def hmda_loader(state: str, year: int = 2023) -> Callable[[], RawData]:
    """One state's home-purchase applications that were either originated or denied, from the
    CFPB/FFIEC HMDA Data Browser API (no account needed). One state keeps the download small."""
    def load() -> RawData:
        url = (f"{HMDA_CSV_API}?states={state}&years={year}&actions_taken=1,3&loan_purposes=1")
        data = _fetch(url, f"hmda_{state}_{year}_purchase.csv", f"hmda_{state}_{year}", timeout=600)
        df = pd.read_csv(io.BytesIO(data), low_memory=False)
        return hmda_from_frame(df, f"hmda_{state}_{year}")
    return load


# --------------------------------------------------------------------------- health
def load_diabetes_hospital() -> RawData:
    return _fairlearn_dataset("fetch_diabetes_hospital", "diabetes_hospital_readmission",
                              ["race", "gender"])


HEART_COLUMNS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang",
                 "oldpeak", "slope", "ca", "thal", "num"]


def heart_from_frame(df: pd.DataFrame) -> RawData:
    """UCI Heart Disease, Cleveland Clinic file (the 303-row one the literature uses).
    Target: any angiographic disease (num > 0). Sensitive: sex. Sex stays a feature -- it is a
    recognised clinical risk factor -- so the audit asks how much the model leans on it.
    Chest-pain type, resting ECG, ST slope and thal are codes, so they are one-hot encoded."""
    d = df.copy()
    d.columns = HEART_COLUMNS
    d["sex"] = d["sex"].map({1: "Male", 0: "Female", 1.0: "Male", 0.0: "Female"})
    d["disease"] = (d["num"] > 0).astype(int)
    d = _codes_as_categories(d, ["cp", "restecg", "slope", "thal"])
    return prepare_frame(d.drop(columns=["num"]), target="disease", sensitive="sex",
                         name="heart_disease_cleveland")


def load_heart_disease() -> RawData:
    data = _fetch(HEART_URL, "heart-disease-processed.cleveland.data", "heart_disease")
    return heart_from_frame(pd.read_csv(io.BytesIO(data), header=None, na_values="?"))


# MEPS 2015 full-year consolidated file (HC-181), panel 19, prepared exactly as IBM AIF360's
# MEPSDataset19 (aif360/datasets/meps_dataset_panel19_fy2015.py), which yields 15,830 rows.
MEPS_RENAME = {"FTSTU53X": "FTSTU", "ACTDTY53": "ACTDTY", "HONRDC53": "HONRDC",
               "RTHLTH53": "RTHLTH", "MNHLTH53": "MNHLTH", "CHBRON53": "CHBRON",
               "JTPAIN53": "JTPAIN", "PREGNT53": "PREGNT", "WLKLIM53": "WLKLIM",
               "ACTLIM53": "ACTLIM", "SOCLIM53": "SOCLIM", "COGLIM53": "COGLIM", "EMPST53": "EMPST",
               "REGION53": "REGION", "MARRY53X": "MARRY", "AGE53X": "AGE", "POVCAT15": "POVCAT",
               "INSCOV15": "INSCOV"}
MEPS_CATEGORICAL = ["REGION", "SEX", "MARRY", "FTSTU", "ACTDTY", "HONRDC", "RTHLTH", "MNHLTH",
                    "HIBPDX", "CHDDX", "ANGIDX", "MIDX", "OHRTDX", "STRKDX", "EMPHDX", "CHBRON",
                    "CHOLDX", "CANCERDX", "DIABDX", "JTPAIN", "ARTHDX", "ARTHTYPE", "ASTHDX",
                    "ADHDADDX", "PREGNT", "WLKLIM", "ACTLIM", "SOCLIM", "COGLIM", "DFHEAR42",
                    "DFSEE42", "ADSMOK42", "PHQ242", "EMPST", "POVCAT", "INSCOV"]
MEPS_NUMERIC = ["AGE", "PCS42", "MCS42", "K6SUM42"]
MEPS_FILTER = ["FTSTU", "ACTDTY", "HONRDC", "RTHLTH", "MNHLTH", "HIBPDX", "CHDDX", "ANGIDX",
               "EDUCYR", "HIDEG", "MIDX", "OHRTDX", "STRKDX", "EMPHDX", "CHBRON", "CHOLDX",
               "CANCERDX", "DIABDX", "JTPAIN", "ARTHDX", "ARTHTYPE", "ASTHDX", "ADHDADDX", "PREGNT",
               "WLKLIM", "ACTLIM", "SOCLIM", "COGLIM", "DFHEAR42", "DFSEE42", "ADSMOK42", "PHQ242",
               "EMPST", "POVCAT", "INSCOV"]
MEPS_VISITS = ["OBTOTV15", "OPTOTV15", "ERTOT15", "IPNGTD15", "HHTOTD15"]
_MEPS_RAW_NAME = {v: k for k, v in MEPS_RENAME.items()}
MEPS_RAW_COLUMNS = (["PANEL", "HISPANX", "RACEV2X"]
                    + sorted({_MEPS_RAW_NAME.get(c, c)
                              for c in MEPS_CATEGORICAL + MEPS_NUMERIC + MEPS_FILTER})
                    + MEPS_VISITS)


def meps_from_frame(df: pd.DataFrame) -> RawData:
    """AIF360's MEPS panel-19 recipe: race = 'White' (non-Hispanic white) vs 'Non-White';
    drop rows with invalid codes (< -1, or < 0 for region/age/marital status/asthma);
    target = high utilisation, i.e. >= 10 office, outpatient, ER, inpatient-night and home-health
    visits in 2015. Unlike AIF360, race is excluded from the features, and the survey weight
    (PERWT15F) is not used, so results describe the sample, not the US population."""
    d = df.copy()
    d["RACE"] = np.where((d["HISPANX"] == 2) & (d["RACEV2X"] == 1), "White", "Non-White")
    d = d[d["PANEL"] == 19].rename(columns=MEPS_RENAME)
    d = d[(d["REGION"] >= 0) & (d["AGE"] >= 0) & (d["MARRY"] >= 0) & (d["ASTHDX"] >= 0)]
    d = d[(d[MEPS_FILTER] >= -1).all(axis=1)]
    d["UTILIZATION"] = (d[MEPS_VISITS].sum(axis=1) >= 10).astype(int)
    d = _codes_as_categories(d, MEPS_CATEGORICAL)
    return prepare_frame(d[MEPS_CATEGORICAL + MEPS_NUMERIC + ["RACE", "UTILIZATION"]],
                         target="UTILIZATION", sensitive="RACE", name="meps_panel19_2015",
                         exclude_sensitive=True)


def load_meps_panel19() -> RawData:
    """Downloads AHRQ's HC-181 SAS transport file (13 MB zipped, ~300 MB unpacked, read in
    chunks, only the ~50 columns the recipe uses). No account is needed, but using the file
    means accepting AHRQ's data use agreement: statistical analysis only, no attempt to identify
    anyone, cite AHRQ/MEPS."""
    import zipfile
    data = _fetch(MEPS_H181_URL, "meps_h181ssp.zip", "meps_panel19", timeout=600)
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z, z.open("h181.ssp") as fh:
            parts = [chunk[[c for c in MEPS_RAW_COLUMNS if c in chunk.columns]]
                     for chunk in pd.read_sas(fh, format="xport", chunksize=5000)]
    except Exception as exc:
        raise DatasetUnavailable(
            f"meps_panel19: could not read h181.ssp from the cached zip ({type(exc).__name__}: "
            f"{exc}); delete {cache_dir() / 'meps_h181ssp.zip'} and run again") from exc
    df = pd.concat(parts, ignore_index=True)
    missing = [c for c in MEPS_RAW_COLUMNS if c not in df.columns]
    if missing:
        raise DatasetUnavailable(f"meps_panel19: HC-181 lacks expected columns {missing}")
    return meps_from_frame(df)


# --------------------------------------------------------------------------- education
def student_from_frame(df: pd.DataFrame) -> RawData:
    """UCI Student Performance, Portuguese-language course (649 students, two Portuguese
    secondary schools, 2005-06). Target: pass, final grade G3 >= 10 of 20. The first- and
    second-period grades G1/G2 are dropped: with them the task is near-trivial, and an
    early-warning model would not have them yet. Sensitive: sex, excluded from the features."""
    d = df.copy()
    d["sex"] = d["sex"].map({"F": "Female", "M": "Male"})
    d["passed"] = (d["G3"] >= 10).astype(int)
    return prepare_frame(d.drop(columns=["G1", "G2", "G3"]), target="passed", sensitive="sex",
                         name="student_performance_portuguese", exclude_sensitive=True)


def load_student_performance() -> RawData:
    import zipfile
    data = _fetch(STUDENT_URL, "uci-student-performance.zip", "student_performance")
    with zipfile.ZipFile(io.BytesIO(data)) as outer:
        with zipfile.ZipFile(io.BytesIO(outer.read("student.zip"))) as inner:
            df = pd.read_csv(io.BytesIO(inner.read("student-por.csv")), sep=";")
    return student_from_frame(df)


def oulad_from_frame(df: pd.DataFrame) -> RawData:
    """Open University Learning Analytics Dataset, studentInfo table (one row per student per
    module presentation, 2013-14). Withdrawn registrations are dropped (withdrawal is not a
    result); target: Pass or Distinction (1) vs Fail (0). Sensitive: gender, excluded from the
    features; disability, IMD deprivation band and age band stay in as features."""
    d = df[df["final_result"] != "Withdrawn"].copy()
    d["passed"] = d["final_result"].isin(["Pass", "Distinction"]).astype(int)
    d["gender"] = d["gender"].map({"F": "Female", "M": "Male"})
    d["imd_band"] = d["imd_band"].replace({"10-20": "10-20%", "?": "unknown"}).fillna("unknown")
    return prepare_frame(d.drop(columns=["id_student", "final_result"]), target="passed",
                         sensitive="gender", name="oulad", exclude_sensitive=True)


def load_oulad() -> RawData:
    """The UCI zip is 47 MB (mostly the click-stream table); only studentInfo.csv (3.5 MB) is
    kept in the cache."""
    import zipfile
    cached = cache_dir() / "oulad-studentInfo.csv"
    if not cached.exists():
        try:
            with urllib.request.urlopen(OULAD_URL, timeout=600) as r:
                data = r.read()
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                cached.write_bytes(z.read("studentInfo.csv"))
        except Exception as exc:
            raise DatasetUnavailable(
                f"oulad: could not fetch {OULAD_URL} ({type(exc).__name__}). Download it, take "
                f"studentInfo.csv out of the zip and save it as {cached}") from exc
    return oulad_from_frame(pd.read_csv(cached))


# --------------------------------------------------------------------------- census (folktables ACS)
ACS_TASKS = ["ACSIncome", "ACSEmployment", "ACSPublicCoverage", "ACSMobility", "ACSTravelTime"]


def acs_from_pums(df: pd.DataFrame, task_name: str, name: str) -> RawData:
    """Turn one state's ACS PUMS table into a RawData for a folktables task. The label comes
    from folktables' own task definition (its population filter and target transform); the
    sensitive attribute is SEX (relabelled Male/Female). Features are the task's integer-coded
    ACS columns used as numbers, as in folktables' own baselines (occupation/place codes are
    therefore crude -- see the domain caveats)."""
    import folktables
    task = getattr(folktables, task_name)
    X, y, _ = task.df_to_pandas(df)
    X = X.copy()
    if "SEX" in X.columns:
        X["SEX"] = X["SEX"].map({1: "Male", 2: "Female"}).fillna("Other")
        sensitive = "SEX"
    else:
        sensitive = "RAC1P"
    X["__target__"] = np.asarray(y).astype(int).ravel()
    return prepare_frame(X, target="__target__", sensitive=sensitive, name=name)


def acs_loader(task_name: str, state: str, year: str = "2018") -> Callable[[], RawData]:
    def load() -> RawData:
        try:
            from folktables import ACSDataSource
        except ImportError as exc:
            raise DatasetUnavailable("census suite needs `pip install folktables`") from exc
        try:
            src = ACSDataSource(survey_year=year, horizon="1-Year", survey="person",
                                root_dir=str(cache_dir() / "acs"))
            df = src.get_data(states=[state], download=True)
        except Exception as exc:
            raise DatasetUnavailable(
                f"{task_name}/{state}: could not download ACS PUMS {year} ({type(exc).__name__}: "
                f"{exc}). It is fetched once from census.gov (tens of MB per state) and cached "
                f"under {cache_dir() / 'acs'}") from exc
        return acs_from_pums(df, task_name, f"{task_name}_{state}_{year}")
    return load
