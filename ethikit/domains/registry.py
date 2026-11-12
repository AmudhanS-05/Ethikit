"""The reference suites. `get_domains()` builds them fresh so options (e.g. which US states)
can vary. Suites are honest about their size: see Domain.strength.

Status of each loader (see DatasetSpec.verified):
  - COMPAS      : fetched and cleaned from the real file (6,172 rows, ProPublica's own count).
  - ACS census  : adapter checked on a PUMS-shaped fixture; the census.gov download itself is
                  done by folktables and has not been run from the build sandbox.
  - OpenML/fairlearn datasets (Adult, German Credit, credit-card default, bank marketing,
                  diabetes hospital): written against the documented schemas; first real run
                  happens on your machine and fails with a precise message if a column differs.
  - HMDA (DE 2023), MEPS panel 19, UCI Heart Disease, UCI Student Performance, OULAD: fetched
                  from the real sources and run through `ethikit domains run` on 2026-11-12
                  (11,570 / 15,830 / 303 / 649 / 22,437 rows; MEPS matches AIF360's count).
  - Considered and not shipped (see GUIDE.md section 7): NIJ Recidivism Challenge (official host
    gone, copies behind logins), LSAC bar passage (no licence; LSAC terms forbid redistribution).
"""
from __future__ import annotations

from typing import Dict, Sequence

from . import sources as S
from .spec import DatasetSpec, Domain

DEFAULT_STATES = ("CA", "TX", "NY")

_ACS_TITLES = {
    "ACSIncome": "Will income exceed $50k?",
    "ACSEmployment": "Is the person employed?",
    "ACSPublicCoverage": "Does the person have public health insurance?",
    "ACSMobility": "Did the person live at the same address a year ago?",
    "ACSTravelTime": "Is the commute longer than 20 minutes?",
}


def get_domains(states: Sequence[str] = DEFAULT_STATES) -> Dict[str, Domain]:
    census = [
        DatasetSpec(
            id=f"{task}_{st}_2018", title=f"{task} · {st} 2018 — {_ACS_TITLES[task]}",
            loader=S.acs_loader(task, st), verified=False,
            source="US Census ACS PUMS 2018 1-year, via folktables (Ding et al., 'Retiring Adult', NeurIPS 2021)",
            sensitive="sex (Male / Female)",
            note="Features are the task's integer-coded ACS columns used as numbers; sex stays a feature.")
        for st in states for task in S.ACS_TASKS
    ] + [DatasetSpec(
        id="adult_income", title="UCI Adult — will income exceed $50k? (1994 census)",
        loader=S.load_adult_strict, verified=False,
        source="UCI Adult via OpenML", sensitive="sex (Male / Female)",
        note="Included for comparability with older papers; the literature recommends the ACS tasks instead.")]

    return {d.id: d for d in [
        Domain("census", "Income, employment & benefits (US census)",
               "Population-scale tabular data about people: earnings, jobs, insurance, mobility, commuting.",
               census,
               ["US-only, 2018 survey data; a model that suits it may not suit another country or year.",
                "Sensitive attribute is sex only here; race (RAC1P) is available in the data but not audited by default.",
                "The 'Retiring Adult' authors argue Adult is a weak benchmark; ACS tasks are preferred."],
               "Public-sector or financial teams deciding on eligibility, outreach or income-related scoring."),
        Domain("lending", "Lending & credit",
               "Credit risk and marketing decisions about individuals.",
               [DatasetSpec("german_credit", "German Credit — good vs bad credit risk (1,000 rows)",
                            S.load_german_credit, "Statlog German Credit via OpenML (credit-g)",
                            "age >= 25 vs < 25", verified=False,
                            note="Small, dated (1994), single German bank; widely used but widely criticised."),
                DatasetSpec("credit_card_default", "Credit-card default (Taiwan, 2005)",
                            S.load_credit_card_default, "fairlearn.datasets.fetch_credit_card (UCI)",
                            "sex", verified=False),
                DatasetSpec("bank_marketing", "Bank marketing — term-deposit subscription",
                            S.load_bank_marketing, "fairlearn.datasets.fetch_bank_marketing (UCI)",
                            "age (median split)", verified=False,
                            note="A marketing outcome, not a credit decision; included for its sensitive age attribute."),
                DatasetSpec("hmda_DE_2023", "HMDA mortgages · Delaware 2023 — home-purchase application approved or denied",
                            S.hmda_loader("DE", 2023),
                            "CFPB / FFIEC HMDA Data Browser API (public loan-level data, privacy-modified by the CFPB)",
                            "race (White / Black, from derived_race)", verified=True,
                            note="Originated vs denied only; other races and 'not available' dropped. Race, ethnicity, "
                                 "sex and age are not features, nor is any field set after the decision. No credit "
                                 "score is in public HMDA, so the model misses the main underwriting variable.")],
               ["Four datasets: results are indicative, and significance tests have little power.",
                "Only HMDA is modern (2023) and real US lending data; the other three are old, small or not credit decisions.",
                "Public HMDA has no credit score and only banded DTI, so approval gaps partly reflect omitted variables, not "
                "only the model -- and a model that reproduces past approvals reproduces past lending patterns."],
               "Banks, lenders and fintechs choosing a starting model family for credit-style decisions."),
        Domain("health", "Health",
               "Clinical outcomes and care decisions.",
               [DatasetSpec("diabetes_hospital", "Diabetes hospital readmission within 30 days",
                            S.load_diabetes_hospital, "fairlearn.datasets.fetch_diabetes_hospital (UCI)",
                            "race", verified=False),
                DatasetSpec("meps_panel19", "MEPS 2015 panel 19 — high health-care utilisation (>= 10 visits)",
                            S.load_meps_panel19,
                            "AHRQ Medical Expenditure Panel Survey HC-181 (meps.ahrq.gov), prepared as AIF360's MEPSDataset19",
                            "race (non-Hispanic White / Non-White)", verified=True,
                            note="By downloading it you accept AHRQ's data use agreement: statistical analysis only, "
                                 "no attempt to identify anyone, cite AHRQ/MEPS. Race not a feature; survey weights "
                                 "not used, so results describe the sample, not the US population. Utilisation is a "
                                 "proxy for need that also reflects access to care."),
                DatasetSpec("heart_disease", "UCI Heart Disease — angiographic disease (Cleveland, 303 patients)",
                            S.load_heart_disease, "UCI Machine Learning Repository #45 (Janosi et al., 1989), CC BY 4.0",
                            "sex (Male / Female; 97 women)", verified=True,
                            note="Tiny and 1980s-era; sex stays a feature (a clinical risk factor). A diagnosis is not "
                                 "a benefit, so read error-rate gaps, not selection-rate parity.")],
               ["Three datasets: results are indicative, and significance tests have little power.",
                "Heart disease has ~300 rows: its per-group numbers come from a few dozen test patients.",
                "Clinical decisions need clinical validation that no public benchmark replaces."],
               "Hospitals and payers exploring readmission-style risk models."),
        Domain("justice", "Criminal justice",
               "Risk assessment in the justice system.",
               [DatasetSpec("compas", "COMPAS two-year recidivism (Broward County, FL)",
                            S.load_compas, "ProPublica compas-analysis (GitHub)",
                            "race (African-American / Caucasian / Hispanic / Other)", needs="network",
                            note="Cleaned as in ProPublica's analysis; race excluded from features; the COMPAS score itself is not used.")],
               ["One dataset: a starting point, not cross-dataset evidence.",
                "The very notion of risk scoring people in this setting is contested; an audit cannot settle it."],
               "Researchers and agencies studying recidivism-style risk tools."),
        Domain("education", "Education",
               "Predicting pass / fail for students (early-warning and support targeting).",
               [DatasetSpec("student_performance", "UCI Student Performance — pass the Portuguese course (649 students)",
                            S.load_student_performance, "UCI Machine Learning Repository #320 (Cortez & Silva, 2008), CC BY 4.0",
                            "sex (Female / Male)", verified=True,
                            note="Two Portuguese secondary schools, 2005-06. Period grades G1/G2 dropped (near-leaky); "
                                 "sex not a feature. The maths-course file is mostly the same students, so it is not added."),
                DatasetSpec("oulad", "Open University (OULAD) — pass a module (2013-14)",
                            S.load_oulad, "Open University Learning Analytics Dataset via UCI #349 (Kuzilek et al., 2017), CC BY 4.0",
                            "gender (Female / Male)", verified=True,
                            note="studentInfo table only; withdrawn registrations dropped. Gender not a feature; "
                                 "disability, deprivation band and age band are. A student can appear in several "
                                 "modules, so train and test rows are not fully independent.")],
               ["Two datasets: results are indicative, and significance tests have little power.",
                "Both are European (Portugal, UK distance learning) and a decade or more old.",
                "With demographic and enrolment fields only, models barely beat always predicting 'pass': "
                "differences between families here are mostly about fairness and calibration, not accuracy.",
                "A pass/fail prediction used to ration support can widen the gaps it measures; an audit cannot decide that use."],
               "Schools, universities and ed-tech teams exploring early-warning models."),
        Domain("hiring", "Hiring & HR",
               "Screening and ranking candidates.",
               [],
               ["No standard public benchmark was found, and a synthetic stand-in would only show the "
                "synthetic data's properties, so none is shipped.",
                "Use your own historical data: ethikit run --data candidates.csv --target hired --sensitive gender"],
               "HR teams: bring your own data."),
    ]}
