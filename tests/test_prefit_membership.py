"""A pre-trained model's training rows. ethikit does not refit a --model-file / uploaded model, so
it only knows which rows the model saw if the user marks them (--train-rows COLUMN). The contract:
 - without the column, every tool whose numbers depend on training-set membership
   (Evaluator.needs_membership) is skipped for that model, with a stated reason, and the model card
   says which rows were used and that the scores may be inflated;
 - with the column, every score is computed on the holdout rows only and the privacy attacks use
   the marked rows as members;
 - models ethikit fits itself are unaffected (it knows their training rows).
A new tool that needs membership must set needs_membership = True, or it fails here."""
import io
import json
import urllib.request

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from ethikit import cli, session
from ethikit.benchmark import CLASSIFIERS, run_benchmark, select_evaluators
from ethikit.benchmark.runner import MEMBERSHIP_SKIP_REASON
from ethikit.data import loaders
from ethikit.evaluators.base import all_evaluators, available_evaluators

MEMBERSHIP_TOOLS = {"membership_inference", "attribute_inference"}
PRIVACY = ["performance", "fairness", "membership_inference", "attribute_inference"]


@pytest.fixture(scope="module")
def marked(tmp_path_factory):
    """A CSV with a 'split' column, and a random forest (which memorises its rows) fitted on
    exactly the rows marked 'train'."""
    rng = np.random.default_rng(0)
    n = 900
    g = rng.integers(0, 2, n)
    x1 = rng.normal(0, 1, n) + 0.8 * g
    x2 = rng.normal(0, 1, n)
    y = ((x1 + 0.5 * x2 + rng.normal(0, 1, n)) > 0.4).astype(int)
    df = pd.DataFrame({"x1": x1, "x2": x2, "group": g, "y": y})
    train = rng.uniform(size=n) < 0.6
    df["split"] = np.where(train, "train", "holdout")
    model = RandomForestClassifier(n_estimators=40, random_state=0).fit(
        df.loc[train, ["x1", "x2", "group"]].to_numpy(), df.loc[train, "y"])
    d = tmp_path_factory.mktemp("prefit")
    df.to_csv(d / "with_split.csv", index=False)
    df.drop(columns="split").to_csv(d / "no_split.csv", index=False)
    joblib.dump(model, d / "rf.joblib")
    return {"dir": d, "df": df, "train": train, "model": model}


def _cfg(marked, split_column=None, tools=PRIVACY, zoo=False):
    data = {"source": "csv", "target": "y", "sensitive": "group",
            "path": str(marked["dir"] / ("with_split.csv" if split_column else "no_split.csv"))}
    if split_column:
        data["train_rows"] = split_column
    models = [{"name": "mine", "file": str(marked["dir"] / "rf.joblib")}]
    if zoo:
        models.insert(0, {"name": "logistic_regression"})
    return {"data": data, "split": {"scale": False}, "models": models, "tools": tools, "save": False}


# ---- parsing the column ------------------------------------------------------------------------
@pytest.mark.parametrize("values,expected", [
    (["train", "holdout", "Train ", "TEST"], [True, False, True, False]),
    ([1, 0, 1, 1], [True, False, True, True]),
    ([1.0, 0.0], [True, False]),
    ([True, False, False], [True, False, False]),
    (["yes", "no", "member", "non-member"], [True, False, True, False]),
])
def test_train_rows_spellings(values, expected):
    assert loaders.parse_train_rows(values, "c").tolist() == expected


@pytest.mark.parametrize("values,match", [
    (["train", "holdot"], "neither train"),          # a typo must not silently become 'holdout'
    ([1, 2, 0], "neither train"),
    (["train", None, "test"], "empty value"),
    (["train", "train"], "every row as train"),
    ([0, 0], "every row as holdout"),
])
def test_train_rows_rejects_anything_ambiguous(values, match):
    with pytest.raises(ValueError, match=match):
        loaders.parse_train_rows(values, "c")


def test_the_column_is_never_a_feature_and_stays_aligned(marked):
    df = marked["df"].copy()
    df.loc[[0, 1], "y"] = np.nan                     # dropped rows must drop their marks too
    raw = loaders.prepare_frame(df, "y", "group", train_rows="split")
    assert "split" not in raw.feature_names and len(raw.train_mask) == len(raw.y) == len(df) - 2
    assert raw.train_mask.tolist() == marked["train"][2:].tolist()
    with pytest.raises(ValueError, match="can't also be"):
        loaders.prepare_frame(df, "y", "group", train_rows="group")


# ---- the runner: skip when unknown -------------------------------------------------------------
def test_every_membership_tool_declares_it():
    assert {e.name for e in all_evaluators() if e.needs_membership} == MEMBERSHIP_TOOLS


@pytest.mark.parametrize("ev", [e for e in available_evaluators() if e.needs_membership],
                         ids=lambda e: e.name)
def test_contract_prefit_without_train_rows_skips_with_the_reason(ev, tiny_ds):
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(max_iter=300).fit(tiny_ds.X_train, tiny_ds.y_train)
    events = []
    df = run_benchmark(tiny_ds, {"zoo_lr": CLASSIFIERS["linear"]["logistic_regression"]},
                       prefit={"mine": m}, on_event=events.append, evaluators=[ev])
    skips = [e for e in events if e["type"] == "tool_skip"]
    assert [(e["model"], e["tool"]) for e in skips] == [("mine", ev.name)]
    assert skips[0]["reason"] == MEMBERSHIP_SKIP_REASON and skips[0]["membership"] is True
    assert "--train-rows" in skips[0]["reason"]
    assert set(df[df.evaluator == ev.name]["model"]) == {"zoo_lr"}   # ethikit's own model still runs
    assert df.attrs["membership_known"] == {"zoo_lr": True, "mine": False}
    assert df.attrs["skipped"]["mine"] == [{"name": ev.name, "principles": list(ev.principle_ids),
                                            "reason": MEMBERSHIP_SKIP_REASON}]
    assert "was not told which rows it was trained on" in df.attrs["rows_used"]["mine"]


def test_session_without_train_rows_skips_privacy_and_says_so(marked):
    p = session.prepare(_cfg(marked, zoo=True))
    assert p.dataset.split_info["source"] == "random"
    assert any("not told which rows" in n and "--train-rows" in n for n in p.notes)
    s = session.execute(p)["summary"]
    tools_for_mine = {r["evaluator"] for r in s["records"] if r["model"] == "mine"}
    assert tools_for_mine == {"performance", "fairness"}
    assert {t["name"] for t in s["skipped"]["mine"]} == MEMBERSHIP_TOOLS
    assert s["skipped"]["logistic_regression"] == []
    assert s["membership_known"] == {"logistic_regression": True, "mine": False}


# ---- the runner: real members when marked ------------------------------------------------------
def test_with_train_rows_scores_use_holdout_and_attacks_use_real_members(marked):
    p = session.prepare(_cfg(marked, split_column="split"))
    train = marked["train"]
    assert p.dataset.split_info == {"source": "column", "column": "split"}
    assert len(p.dataset.y_train) == train.sum() and len(p.dataset.y_test) == (~train).sum()
    np.testing.assert_allclose(p.dataset.X_test, marked["df"].loc[~train, ["x1", "x2", "group"]].to_numpy())
    assert not any("not told" in n for n in p.notes)

    s = session.execute(p)["summary"]
    assert s["membership_known"] == {"mine": True} and s["skipped"]["mine"] == []
    rec = {r["metric"]: r for r in s["records"] if r["model"] == "mine"}
    # accuracy on holdout rows only = what the model gets on rows it never saw
    holdout = marked["df"].loc[~train]
    true_acc = (marked["model"].predict(holdout[["x1", "x2", "group"]].to_numpy()) == holdout["y"]).mean()
    assert rec["accuracy"]["value"] == pytest.approx(true_acc)
    # a memorising forest against its real members: the attack finds the leak
    assert rec["membership_inference_advantage"]["value"] > 0.15
    assert "marked" in s["rows_used"]["mine"] and "holdout" in s["rows_used"]["mine"]


def test_a_random_split_hides_the_leak_that_real_members_reveal(marked):
    """Why the skip exists: on a random split of the same rows, 'members' are a guess."""
    raw = loaders.load_csv(str(marked["dir"] / "no_split.csv"), "y", "group")
    ds = loaders.split_dataset(raw, seed=0, scale=False)
    guessed = run_benchmark(ds, {}, prefit={"mine": marked["model"]},
                            evaluators=select_evaluators(["membership_inference"]))
    assert guessed.empty            # skipped, not a falsely reassuring number
    real = session.execute(session.prepare(_cfg(marked, "split", ["membership_inference"])))["summary"]
    adv = [r["value"] for r in real["records"] if r["metric"] == "membership_inference_advantage"][0]
    ds.split_info = {"source": "column", "column": "pretend"}    # force the attack on guessed members
    forced = run_benchmark(ds, {}, prefit={"mine": marked["model"]},
                           evaluators=select_evaluators(["membership_inference"]))
    assert forced.loc[forced.metric == "membership_inference_advantage", "value"].iloc[0] < adv


def test_train_rows_needs_a_csv_and_refuses_cv(marked):
    with pytest.raises(ValueError, match="needs --data"):
        session.prepare({"data": {"source": "synthetic", "train_rows": "split"}, "save": False})
    cfg = _cfg(marked, "split")
    cfg["models"] = [{"name": "logistic_regression"}]
    cfg["split"]["cv"] = [3, 1]
    with pytest.raises(ValueError, match="can't be combined with --cv"):
        session.prepare(cfg)


# ---- what the user is told ---------------------------------------------------------------------
def test_model_card_says_which_rows_and_lists_the_skipped_tools(marked):
    from ethikit.report import model_card_html, model_card_json
    out = session.execute(session.prepare(_cfg(marked)))
    args = ("mine", "no_split", out["raw"], out["objective"], out["summary"]["pareto"])
    html = model_card_html(*args, sensitive_name="group")
    assert "Which rows were used" in html and "was not told which rows it was trained on" in html
    assert "Tools that did not run" in html and "membership_inference" in html
    assert "needs to know which rows the model was trained on" in html
    js = model_card_json(*args, sensitive_name="group")
    risks = " ".join(r["name"] for r in js["considerations"]["ethical_considerations"])
    assert "membership_inference" in risks and "attribute_inference" in risks
    slices = {m["slice"] for m in js["quantitative_analysis"]["performance_metrics"]}
    assert all("training rows are unknown" in s for s in slices)

    out2 = session.execute(session.prepare(_cfg(marked, "split")))
    html2 = model_card_html("mine", "with_split", out2["raw"], out2["objective"], out2["summary"]["pareto"])
    assert "You marked" in html2 and "column &#x27;split&#x27;" in html2
    assert "needs to know which rows" not in html2


def test_cli_train_rows_flag(marked, tmp_path, capsys):
    d = marked["dir"]
    rc = cli.main(["run", "--data", str(d / "with_split.csv"), "--train-rows", "split", "--target", "y",
                   "--sensitive", "group", "--model-file", f"mine={d / 'rf.joblib'}", "--no-scale",
                   "--tools", "performance,membership_inference", "--no-save", "--out", str(tmp_path)])
    assert rc == 0
    res = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert res["config"]["data"]["train_rows"] == "split" and res["membership_known"] == {"mine": True}
    assert any(r["evaluator"] == "membership_inference" for r in res["records"])

    capsys.readouterr()
    rc = cli.main(["run", "--data", str(d / "no_split.csv"), "--target", "y", "--sensitive", "group",
                   "--model-file", f"mine={d / 'rf.joblib'}", "--no-scale",
                   "--tools", "performance,membership_inference", "--no-save", "--out", str(tmp_path)])
    printed = capsys.readouterr().out
    assert rc == 0 and "membership_inference" in printed and "--train-rows" in printed   # skip shown, not only with -v


# ---- the console ------------------------------------------------------------------------------
def _post(url, data, ctype="application/json"):
    req = urllib.request.Request(url, data=data if isinstance(data, bytes) else json.dumps(data).encode(),
                                 method="POST", headers={"Content-Type": ctype})
    return urllib.request.urlopen(req)


def test_console_upload_suggests_the_column_and_runs_with_it(server, marked):
    buf = io.BytesIO()
    joblib.dump(marked["model"], buf)
    with _post(server + "/api/upload-model", buf.getvalue(), "application/octet-stream") as r:
        mid = json.loads(r.read())["upload_model_id"]
    csv = (marked["dir"] / "with_split.csv").read_bytes()
    with _post(server + "/api/upload?name=marked.csv", csv, "text/csv") as r:
        info = json.loads(r.read())
    assert info["suggest"]["train_rows"] == "split"

    def run(train_rows):
        data = {"source": "upload", "upload_id": info["upload_id"], "name": "marked", "target": "y",
                "sensitive": "group"}
        if train_rows:
            data["train_rows"] = train_rows
        cfg = {"data": data, "split": {"scale": False}, "save": False,
               "models": [{"name": "mine", "upload_model_id": mid}], "tools": PRIVACY}
        with _post(server + "/api/run", cfg) as r:
            return [json.loads(line) for line in r if line.strip()]

    ev = run("split")
    assert ev[-1]["type"] == "result" and ev[-1]["membership_known"] == {"mine": True}
    assert not [e for e in ev if e["type"] == "tool_skip"]
    # a CSV with no marks (run() reads the rebound `info`): the pre-trained model's privacy
    # attacks are skipped, and the stream says which rows were used
    with _post(server + "/api/upload?name=plain.csv", (marked["dir"] / "no_split.csv").read_bytes(),
               "text/csv") as r:
        info = json.loads(r.read())
    assert info["suggest"]["train_rows"] is None
    ev = run(None)
    skips = {e["tool"]: e["reason"] for e in ev if e["type"] == "tool_skip"}
    assert set(skips) == MEMBERSHIP_TOOLS and all("--train-rows" in v for v in skips.values())
    fit = [e for e in ev if e["type"] == "model_fit_done"][0]
    assert fit["membership_known"] is False and "not told" in fit["rows_used"]
    assert ev[-1]["rows_used"]["mine"] == fit["rows_used"]
