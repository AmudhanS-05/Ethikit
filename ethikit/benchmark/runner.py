"""Benchmark runner: the train -> evaluate -> collect loop.

Given a dataset and a dict of model factories, it fits each model, builds an
EvalContext, runs every *available & applicable* evaluator, and returns a
tidy list of records (one row per model x metric) that the scoring and report
layers consume.

Two extras exist for the live console / CLI (ethikit/cli.py, ethikit/server.py):
  - `on_event(event_dict)` is called at every step (run start, model fit,
    each tool starting/finishing with its results), so a UI can animate the
    flow graph and stream a log while the sweep is still running.
  - every tool's intermediate artefacts (details["trace"]: per-group tables,
    attack histograms, surrogate-tree rules, the formula filled in with this
    run's numbers) are kept in `df.attrs["traces"][model][tool]`.
  - which rows each model was scored on (and which rows count as its training
    data) is stated in `df.attrs["rows_used"][model]`, and every tool that did
    not run for a model, with the reason, in `df.attrs["skipped"][model]`;
    `df.attrs["membership_known"][model]` says whether its training rows are known.

A pre-trained (`prefit`) model is not refit, so ethikit only knows its training
rows if the user marked them (Dataset.split_info["source"] == "column"). Without
that, the "train" split is just a random sample of the user's rows, and tools
whose numbers depend on training-set membership (Evaluator.needs_membership:
membership_inference, attribute_inference) are skipped with the reason instead
of reporting a member-vs-non-member gap between rows the model may never have seen.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from ..core.types import Dataset, EvalContext
from ..evaluators.base import Evaluator, available_evaluators
# importing the evaluators package registers all built-in evaluators
from .. import evaluators as _evaluators  # noqa: F401

Event = Dict[str, Any]


def _predict_all(model, ds: Dataset) -> EvalContext:
    y_pred = model.predict(ds.X_test)
    y_pred_train = model.predict(ds.X_train)
    y_proba = None
    y_proba_train = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(ds.X_test)
        except Exception:
            y_proba = None
        try:
            y_proba_train = model.predict_proba(ds.X_train)
        except Exception:
            y_proba_train = None
    return EvalContext(model=model, dataset=ds, y_pred=y_pred,
                       y_pred_train=y_pred_train, y_proba=y_proba,
                       y_proba_train=y_proba_train)


MEMBERSHIP_SKIP_REASON = (
    "needs to know which rows the model was trained on. This pre-trained model was not refit and "
    "no --train-rows column marked its training rows, so ethikit cannot tell which of your rows it "
    "saw: its 'train' and 'test' splits are random samples, each a mix of seen and unseen rows, and "
    "the attack would understate the leak (typically a falsely reassuring ~0). Mark the training "
    "rows (--train-rows COLUMN) to run it.")


def rows_used(dataset: Dataset, prefit: bool) -> str:
    """Plain-language statement of which rows a model was trained on and scored on."""
    n_tr, n_te = int(len(dataset.y_train)), int(len(dataset.y_test))
    info = dataset.split_info or {}
    if info.get("source") == "column":
        col = info.get("column")
        if prefit:
            return (f"Pre-trained model, not refit. You marked {n_tr} rows as its training data "
                    f"(column '{col}'). Every score is computed on the other {n_te} holdout rows; "
                    f"the privacy attacks use the {n_tr} marked rows as members and the holdout "
                    f"rows as non-members.")
        return (f"ethikit trained this model on the {n_tr} rows marked as training rows in column "
                f"'{col}' and scored it on the other {n_te} holdout rows.")
    how = (f"the test side of a random {1 - float(info['test_size']):.0%}/"
           f"{float(info['test_size']):.0%} split, seed {info['seed']}"
           if "test_size" in info else "the test side of one train/test split")
    if prefit:
        return (f"Pre-trained model, not refit, and ethikit was not told which rows it was trained "
                f"on. Scores are computed on {n_te} of your rows ({how}); if any of them were in "
                f"the model's training data, performance and fairness can look better than they "
                f"would on new data. Tools that need the training rows (membership_inference, "
                f"attribute_inference) did not run. To fix both, add a column marking the "
                f"training rows and pass --train-rows COLUMN.")
    return (f"ethikit trained this model on {n_tr} rows and scored it on the other {n_te} "
            f"held-out rows ({how}); the privacy attacks use the {n_tr} training rows as members.")


def jsonable(obj: Any) -> Any:
    """numpy/pandas scalars & arrays -> plain JSON types (for events and traces)."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if not np.isfinite(v) else v
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def select_evaluators(tools: Optional[Iterable[str]] = None,
                      tool_params: Optional[Dict[str, Dict[str, Any]]] = None) -> List[Evaluator]:
    """Available evaluators, optionally restricted to `tools` and with
    per-tool param overrides applied ({tool: {param: value}})."""
    evs = available_evaluators()
    if tools is not None:
        wanted = set(tools)
        unknown = wanted - {e.name for e in evs}
        if unknown:
            raise KeyError(f"unknown or unavailable tool(s): {', '.join(sorted(unknown))}")
        evs = [e for e in evs if e.name in wanted]
    tool_params = tool_params or {}
    return [e.configured(**tool_params[e.name]) if e.name in tool_params else e for e in evs]


def run_benchmark(dataset: Dataset,
                  model_factories: Dict[str, Callable],
                  on_event: Optional[Callable[[Event], None]] = None,
                  evaluators: Optional[List[Evaluator]] = None,
                  prefit: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Fit every factory (or take an already-fitted model from `prefit`), run
    the evaluator battery, return tidy records.

    `prefit` maps model_name -> an already-trained estimator that is evaluated
    as-is (no .fit()) -- "evaluate *my* model on *my* data".
    """
    emit = on_event or (lambda e: None)
    records: List[dict] = []
    traces: Dict[str, Dict[str, Any]] = {}
    skipped: Dict[str, List[Dict[str, Any]]] = {}
    used: Dict[str, str] = {}
    known: Dict[str, bool] = {}
    evaluators = evaluators if evaluators is not None else available_evaluators()
    prefit = prefit or {}
    names = list(model_factories) + [m for m in prefit if m not in model_factories]

    emit({"type": "run_start", "dataset": dataset.name, "task": dataset.task.value,
          "n_train": int(len(dataset.y_train)), "n_test": int(len(dataset.y_test)),
          "n_features": int(np.asarray(dataset.X_test).shape[1]),
          "sensitive": dataset.sensitive_name, "models": names,
          "split": dataset.split_info,
          "tools": [{"name": e.name, "principles": list(e.principle_ids)} for e in evaluators]})

    for model_name in names:
        t0 = time.perf_counter()
        if model_name in prefit:
            model = prefit[model_name]
            emit({"type": "model_fit_start", "model": model_name, "prefit": True})
        else:
            emit({"type": "model_fit_start", "model": model_name, "prefit": False})
            model = model_factories[model_name]()
            model.fit(dataset.X_train, dataset.y_train)
        ctx = _predict_all(model, dataset)
        is_prefit = model_name in prefit
        ctx.membership_known = not is_prefit or (dataset.split_info or {}).get("source") == "column"
        used[model_name] = rows_used(dataset, is_prefit)
        known[model_name] = bool(ctx.membership_known)
        emit({"type": "model_fit_done", "model": model_name,
              "seconds": round(time.perf_counter() - t0, 3), "rows_used": used[model_name],
              "membership_known": ctx.membership_known})
        traces[model_name] = {}
        skipped[model_name] = []

        for ev in evaluators:
            reason = None
            if getattr(ev, "needs_membership", False) and not ctx.membership_known:
                reason = MEMBERSHIP_SKIP_REASON
            elif not ev.applies_to(ctx):
                reason = "not applicable to this task / dataset"
            if reason:
                skipped[model_name].append({"name": ev.name, "principles": list(ev.principle_ids),
                                            "reason": reason})
                emit({"type": "tool_skip", "model": model_name, "tool": ev.name,
                      "reason": reason, "membership": reason == MEMBERSHIP_SKIP_REASON})
                continue
            emit({"type": "tool_start", "model": model_name, "tool": ev.name,
                  "principles": list(ev.principle_ids)})
            t1 = time.perf_counter()
            try:
                results = ev.evaluate(ctx)
                new = [res.as_record(model_name) for res in results]
                records.extend(new)
                trace = next((r.details.get("trace") for r in results
                              if r.details.get("trace")), None)
                traces[model_name][ev.name] = jsonable({
                    "trace": trace, "formula": ev.formula, "limits": ev.limits,
                    "params": ev.params(),
                    "mechanisms": sorted({r.mechanism for r in results}),
                })
                emit({"type": "tool_done", "model": model_name, "tool": ev.name,
                      "seconds": round(time.perf_counter() - t1, 3),
                      "results": jsonable([{k: r[k] for k in
                                            ("principle", "metric", "value", "higher_is_better")}
                                           for r in new])})
            except Exception as exc:  # never let one tool break the sweep
                records.append({
                    "model": model_name, "evaluator": ev.name,
                    "principle": (ev.principle_ids or ["?"])[0], "target": "error",
                    "metric": f"ERROR:{type(exc).__name__}", "value": float("nan"),
                    "higher_is_better": True,
                    "mechanism": f"{ev.name} raised {type(exc).__name__}: {exc}",
                })
                emit({"type": "tool_error", "model": model_name, "tool": ev.name,
                      "error": f"{type(exc).__name__}: {exc}"})

    df = pd.DataFrame.from_records(records)
    df.attrs["traces"] = traces
    df.attrs["skipped"] = skipped
    df.attrs["rows_used"] = used
    df.attrs["membership_known"] = known
    emit({"type": "run_done", "n_records": len(records)})
    return df
