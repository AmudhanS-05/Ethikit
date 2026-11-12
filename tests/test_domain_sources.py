"""Adapters for the HMDA, MEPS, UCI Heart Disease, UCI Student Performance and OULAD loaders.

The offline tests feed each adapter a small frame shaped like the real file (the numbers mean
nothing) and check the modelling choices the registry promises: which rows are kept, what the
target is, which attribute is sensitive, and which columns must never reach the features.
The `network` tests fetch the small real files and skip themselves when offline."""
import io
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import pytest

from ethikit.core.types import TaskType
from ethikit.domains import DatasetUnavailable, get_domains
from ethikit.domains import sources as S
from ethikit.domains.spec import cache_dir

R = np.random.default_rng(0)


def _ids(domain):
    return [s.id for s in get_domains()[domain].datasets]


def test_new_datasets_are_registered_where_promised():
    assert "hmda_DE_2023" in _ids("lending")
    assert {"meps_panel19", "heart_disease"} <= set(_ids("health"))
    assert _ids("education") == ["student_performance", "oulad"]
    assert get_domains()["education"].strength == "small"
    assert get_domains()["hiring"].strength == "gap"              # still an explicit gap


# --------------------------------------------------------------------------- HMDA
def fake_hmda(n=600):
    df = pd.DataFrame({c: R.integers(1, 1000, n) for c in S.HMDA_NUMERIC})
    for c in S.HMDA_CATEGORICAL:
        df[c] = R.choice(["1", "2", "3"], n)
    df["property_value"] = R.choice(["Exempt", "250000", "400000"], n)        # strings in the CSV
    df["debt_to_income_ratio"] = R.choice(["20%-<30%", "36", ">60%", np.nan], n)
    df["action_taken"] = R.choice([1, 3, 4], n)
    df["derived_race"] = R.choice(["White", "Black or African American", "Asian",
                                   "Race Not Available"], n)
    # decision-time leaks and protected characteristics that must not become features
    df["interest_rate"] = np.where(df["action_taken"] == 1, 6.5, np.nan)
    df["denial_reason-1"] = np.where(df["action_taken"] == 3, 1, 10)
    df["purchaser_type"] = np.where(df["action_taken"] == 3, 0, 1)
    df["derived_sex"] = R.choice(["Male", "Female"], n)
    df["applicant_age"] = R.choice(["25-34", "35-44"], n)
    return df


def test_hmda_keeps_originated_vs_denied_for_white_and_black_applicants_only():
    df = fake_hmda()
    raw = S.hmda_from_frame(df, "hmda_fake")
    kept = df[df.action_taken.isin([1, 3])
              & df.derived_race.isin(["White", "Black or African American"])]
    assert len(raw.y) == len(kept) and raw.task == TaskType.CLASSIFICATION
    assert set(raw.sensitive) == {"White", "Black"}
    assert raw.y.mean() == pytest.approx((kept.action_taken == 1).mean())    # 1 = approved


def test_hmda_features_are_the_decision_time_whitelist_only():
    raw = S.hmda_from_frame(fake_hmda(), "hmda_fake")
    banned = ("race", "ethnicity", "sex", "age", "interest", "denial", "purchaser", "action")
    leaked = [f for f in raw.feature_names
              if any(b in f.lower() for b in banned) and not f.startswith("tract_")]
    assert not leaked, leaked
    assert "tract_minority_population_percent" in raw.feature_names   # the proxy stays, on purpose


def test_hmda_schema_change_names_the_missing_columns():
    with pytest.raises(DatasetUnavailable, match="lacks expected columns.*income"):
        S.hmda_from_frame(fake_hmda().drop(columns=["income"]))


# --------------------------------------------------------------------------- MEPS
def fake_meps(n=800):
    cols = set(S.MEPS_RAW_COLUMNS)
    df = pd.DataFrame({c: R.choice([1.0, 2.0], n) for c in cols})
    df["PANEL"] = R.choice([19.0, 20.0], n)
    df["AGE53X"] = R.integers(-1, 85, n).astype(float)
    df["REGION53"] = R.choice([-1.0, 1.0, 2.0, 3.0, 4.0], n)
    df["EDUCYR"] = R.choice([-9.0, 12.0, 16.0], n)                    # -9 must be filtered out
    for c in ["PCS42", "MCS42"]:
        df[c] = R.normal(50, 10, n)
    df["K6SUM42"] = R.integers(0, 24, n).astype(float)
    for c in S.MEPS_VISITS:
        df[c] = R.integers(0, 6, n).astype(float)
    df["HISPANX"] = R.choice([1.0, 2.0], n)
    df["RACEV2X"] = R.choice([1.0, 2.0, 3.0], n)
    return df


def test_meps_follows_the_aif360_panel19_recipe():
    df = fake_meps()
    raw = S.meps_from_frame(df)
    d = df[df.PANEL == 19]
    d = d[(d.REGION53 >= 0) & (d.AGE53X >= 0) & (d.MARRY53X >= 0) & (d.ASTHDX >= 0) & (d.EDUCYR >= -1)]
    assert len(raw.y) == len(d)
    white = (d.HISPANX == 2) & (d.RACEV2X == 1)
    assert (raw.sensitive == "White").sum() == white.sum() and set(raw.sensitive) == {"White", "Non-White"}
    assert raw.y.mean() == pytest.approx((d[S.MEPS_VISITS].sum(axis=1) >= 10).mean())
    assert not any(f.startswith("RACE") or "PERWT" in f for f in raw.feature_names)
    assert any(f.startswith("REGION=") for f in raw.feature_names)       # codes are one-hot, not numbers


def test_meps_reports_a_corrupt_cached_file_with_the_fix(monkeypatch):
    (cache_dir() / "meps_h181ssp.zip").write_bytes(b"not a zip")
    with pytest.raises(DatasetUnavailable, match="delete .*meps_h181ssp.zip"):
        S.load_meps_panel19()


# --------------------------------------------------------------------------- Heart disease
def test_heart_disease_target_is_any_disease_and_sex_stays_a_feature():
    n = 200
    df = pd.DataFrame(np.column_stack([R.integers(30, 77, n), R.integers(0, 2, n),
                                       R.integers(1, 5, n), R.integers(90, 200, n),
                                       R.integers(120, 560, n), R.integers(0, 2, n),
                                       R.integers(0, 3, n), R.integers(70, 200, n),
                                       R.integers(0, 2, n), R.random(n) * 6,
                                       R.integers(1, 4, n), R.integers(0, 4, n).astype(float),
                                       R.choice([3, 6, 7], n).astype(float),
                                       R.integers(0, 5, n)]))
    df.iloc[:3, 11] = np.nan                                          # '?' in the real file
    raw = S.heart_from_frame(df)
    assert len(raw.y) == n and set(raw.sensitive) == {"Male", "Female"}
    assert raw.y.mean() == pytest.approx((df[13] > 0).mean())
    assert "sex=Female" in raw.feature_names and any(f.startswith("cp=") for f in raw.feature_names)


# --------------------------------------------------------------------------- education
def fake_student(n=300):
    df = pd.DataFrame({"school": R.choice(["GP", "MS"], n), "sex": R.choice(["F", "M"], n),
                       "age": R.integers(15, 22, n), "studytime": R.integers(1, 5, n),
                       "failures": R.integers(0, 4, n), "absences": R.integers(0, 30, n),
                       "higher": R.choice(["yes", "no"], n)})
    for g in ["G1", "G2", "G3"]:
        df[g] = R.integers(0, 21, n)
    return df


def test_student_performance_drops_period_grades_and_sex():
    df = fake_student()
    raw = S.student_from_frame(df)
    assert raw.y.mean() == pytest.approx((df.G3 >= 10).mean())
    assert set(raw.sensitive) == {"Female", "Male"}
    assert not any(f.startswith(("G1", "G2", "G3", "sex")) for f in raw.feature_names)


def test_student_loader_reads_the_nested_zip_from_cache_without_network(monkeypatch):
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as z:
        z.writestr("student-por.csv", fake_student().to_csv(sep=";", index=False))
    outer = io.BytesIO()
    with zipfile.ZipFile(outer, "w") as z:
        z.writestr("student.zip", inner.getvalue())
    (cache_dir() / "uci-student-performance.zip").write_bytes(outer.getvalue())

    def no_network(*a, **k):
        raise AssertionError("must not download when the cache has the file")
    monkeypatch.setattr(urllib.request, "urlopen", no_network)
    assert len(S.load_student_performance().y) == 300


def test_oulad_drops_withdrawn_and_excludes_gender():
    n = 400
    df = pd.DataFrame({"code_module": R.choice(["AAA", "BBB"], n), "code_presentation": "2013J",
                       "id_student": np.arange(n), "gender": R.choice(["F", "M"], n),
                       "region": R.choice(["Wales", "London Region"], n),
                       "highest_education": R.choice(["A Level or Equivalent", "HE Qualification"], n),
                       "imd_band": R.choice(["0-10%", "10-20", "?", np.nan], n),
                       "age_band": R.choice(["0-35", "35-55"], n),
                       "num_of_prev_attempts": R.integers(0, 3, n), "studied_credits": 60,
                       "disability": R.choice(["N", "Y"], n),
                       "final_result": R.choice(["Pass", "Distinction", "Fail", "Withdrawn"], n)})
    raw = S.oulad_from_frame(df)
    kept = df[df.final_result != "Withdrawn"]
    assert len(raw.y) == len(kept)
    assert raw.y.mean() == pytest.approx(kept.final_result.isin(["Pass", "Distinction"]).mean())
    assert set(raw.sensitive) == {"Female", "Male"}
    assert not any(f.startswith(("gender", "id_student", "final_result")) for f in raw.feature_names)
    assert "imd_band=10-20%" in raw.feature_names and "imd_band=unknown" in raw.feature_names


# --------------------------------------------------------------------------- fetch failures
@pytest.mark.parametrize("loader", [S.load_heart_disease, S.load_student_performance,
                                    S.load_meps_panel19, S.load_oulad, S.hmda_loader("DE", 2023)])
def test_offline_fetch_fails_with_where_to_put_a_manual_download(monkeypatch, loader):
    def offline(*a, **k):
        raise OSError("no network")
    monkeypatch.setattr(urllib.request, "urlopen", offline)
    with pytest.raises(DatasetUnavailable, match=r"could not fetch .*(save it as|Download it)"):
        loader()


# --------------------------------------------------------------------------- live (small files)
def _live(load):
    try:
        return load()
    except DatasetUnavailable as exc:
        pytest.skip(f"offline: {exc}")


@pytest.mark.network
def test_heart_disease_live():
    raw = _live(S.load_heart_disease)
    assert len(raw.y) == 303 and (raw.sensitive == "Female").sum() == 97


@pytest.mark.network
def test_student_performance_live():
    raw = _live(S.load_student_performance)
    assert len(raw.y) == 649 and (raw.sensitive == "Female").sum() == 383


@pytest.mark.network
def test_hmda_live_small_state():
    raw = _live(S.hmda_loader("VT", 2023))          # ~2 MB: the smallest useful request
    assert len(raw.y) > 3000 and set(raw.sensitive) == {"White", "Black"}
    assert 0.6 < raw.y.mean() < 0.98
