"""Adversarial robustness evaluator via the Adversarial Robustness Toolbox
(ART, https://github.com/Trusted-AI/adversarial-robustness-toolbox).

HANDOFF.md's Phase 3 plan: "add ART as a robustness evaluator (real
attacks)" to complement the existing RobustnessEvaluator's Gaussian-noise
proxy (ethikit/evaluators/robustness.py), which measures stability under
*random* perturbation but says nothing about a model's worst case under a
perturbation an adversary deliberately searches for.

This evaluator wraps the fitted model as a black-box label oracle (ART's
SklearnClassifier) and runs HopSkipJump (Chen et al. 2019) -- a
decision-based attack that only needs predict(), no gradients or model
internals, matching how an external attacker actually queries a deployed
model. Query/iteration budgets are kept deliberately small (see __init__
defaults): decision-based attacks are query-hungry, and tree ensembles in
particular can take 10-40s per sample at ART's own defaults -- confirmed by
hand-timing HopSkipJump against RandomForestClassifier/GradientBoosting in
this environment. The tightened budget keeps a full run_benchmark() sweep
over the sklearn model zoo in the tens of seconds to a couple of minutes,
at the cost of a somewhat weaker (but still real) attack.

Classification only -- HopSkipJump's decision-boundary search assumes a
discrete predicted label to flip, which doesn't translate to regression.
"""
from __future__ import annotations

import contextlib
from typing import List

import numpy as np

from ..core.types import EvalContext, EvalResult, TaskType
from ..principles import InspectionTarget
from .base import Evaluator, make_trace, register


@contextlib.contextmanager
def _seeded_numpy(seed: int):
    """Make ART's HopSkipJump reproducible. It draws from numpy's global RNG *and* builds an
    unseeded `np.random.RandomState()` internally, so our own `seed` never reached it. For the
    duration of the call: seed the global state and make `RandomState()` default to `seed`;
    both are restored afterwards so nothing else is affected. The stand-in is a subclass, not a
    function, so libraries the model calls meanwhile (LightGBM checks
    `isinstance(x, np.random.RandomState)`) still see a real type."""
    saved_state = np.random.get_state()
    original = np.random.RandomState

    class seeded(original):
        def __init__(self, s=None):
            super().__init__(seed if s is None else s)

    np.random.seed(seed)
    np.random.RandomState = seeded
    try:
        yield
    finally:
        np.random.RandomState = original
        np.random.set_state(saved_state)


def _has_art() -> bool:
    try:
        import art  # noqa: F401
        return True
    except Exception:
        return False


class _QueryBudgetExceeded(RuntimeError):
    """Raised inside ART's attack loop to stop a row that keeps querying the model."""


def _art_wrap(model, n_features: int, lo: float, hi: float, sklearn_wrapper):
    """ART's SklearnClassifier only accepts scikit-learn's own classes. XGBoost, LightGBM,
    PyTorch/Keras adapters and anything else sklearn-shaped go through BlackBoxClassifier,
    which only calls predict/predict_proba -- all HopSkipJump needs, since it is label-only."""
    try:
        return sklearn_wrapper(model=model, clip_values=(lo, hi))
    except TypeError:
        from art.estimators.classification import BlackBoxClassifier
        classes = np.asarray(getattr(model, "classes_", [0, 1]))

        def predict(x):
            x = np.asarray(x, dtype=float)
            if hasattr(model, "predict_proba"):
                return np.asarray(model.predict_proba(x), dtype=float)
            pred = np.asarray(model.predict(x))
            return (pred[:, None] == classes[None, :]).astype(float)   # one-hot by class order
        return BlackBoxClassifier(predict, input_shape=(n_features,), nb_classes=len(classes),
                                  clip_values=(lo, hi))


class ARTRobustnessEvaluator(Evaluator):
    name = "art_robustness"
    principle_ids = ["safety"]
    supports = [TaskType.CLASSIFICATION]
    mechanism = (
        "Runs a real black-box adversarial evasion attack (HopSkipJump, Chen "
        "et al. 2019, via the Adversarial Robustness Toolbox) against the "
        "fitted model treated as a label-only oracle -- no gradients or "
        "model internals are used, matching how an external attacker would "
        "actually query a deployed model. Wraps the model in "
        "art.estimators.classification.SklearnClassifier (or BlackBoxClassifier "
        "for non-scikit-learn models such as XGBoost, LightGBM or PyTorch), then for a small "
        "sample of correctly-classified test rows searches for the smallest "
        "L2 perturbation that flips the prediction, within a tight "
        "iteration/query budget kept low so a full model-zoo sweep finishes "
        "in reasonable time (see this module's docstring for measured "
        "per-model timings). Reports the fraction of attacked rows the "
        "search could NOT flip within budget (higher = more robust) and, "
        "among the ones it did flip, the median perturbation size needed "
        "(higher = the attacker needed a bigger, more detectable change -- "
        "also more robust). Complements the existing RobustnessEvaluator's "
        "random-noise stability check with a worst-case, adversarially-"
        "searched one."
    )
    install_hint = "pip install adversarial-robustness-toolbox"
    limits = ('A small sample of rows with a tight query budget (see tool parameters): it can miss vulnerabilities a patient attacker would find; L2 perturbations in the scaled feature space only.')
    formula = (
        "for each correctly-classified x: find x' = argmin ‖x' − x‖₂  s.t.  f(x') ≠ f(x)\n"
        "   (HopSkipJump: binary search to the decision boundary + gradient estimate\n"
        "    from label-only queries, within max_iter / max_eval budget)\n"
        "adversarial_robustness = 1 − (#rows flipped / #rows attacked)\n"
        "adversarial_perturbation_size = median over flipped rows of ‖x' − x‖₂ / √d"
    )

    def __init__(self, sample_size: int = 15, max_iter: int = 10, max_eval: int = 300,
                init_eval: int = 15, init_size: int = 15, seed: int = 0):
        self.sample_size = sample_size
        self.max_iter = max_iter
        self.max_eval = max_eval
        self.init_eval = init_eval
        self.init_size = init_size
        self.seed = seed

    def available(self) -> bool:
        return _has_art()

    def applies_to(self, ctx: EvalContext) -> bool:
        return ctx.dataset.task in self.supports and _has_art()

    def _generate(self, clf, X: np.ndarray, attack_cls):
        """Run HopSkipJump under a model-query budget; returns (adversarial rows, #rows that hit it).

        ART's step-size search (`while not success: epsilon /= 2`) has no exit if the point after
        its binary search is not adversarial -- it happens on the flat, piecewise-constant decision
        surfaces of boosted trees (seen with XGBoost) -- and then queries the model forever. A row
        normally needs about 12 queries per iteration, so a cap of 100 per iteration (+ the initial
        search) never touches a healthy attack. Rows are attacked one at a time, each with its own
        cap and the same seed, so one stuck row costs a second or two and cannot affect the
        others; a row that hits the cap keeps its original value (not flipped within budget)."""
        inner = clf.predict
        per_row = 100 * self.max_iter + 4 * self.init_size + 10
        out, stuck = [], 0
        for i in range(len(X)):
            used = [0]

            def predict(z, *a, **k):
                used[0] += 1
                if used[0] > per_row:
                    raise _QueryBudgetExceeded()
                return inner(z, *a, **k)
            clf.predict = predict
            try:
                atk = attack_cls(classifier=clf, max_iter=self.max_iter, max_eval=self.max_eval,
                                 init_eval=self.init_eval, init_size=self.init_size, verbose=False)
                with _seeded_numpy(self.seed):
                    out.append(atk.generate(x=X[i:i + 1]))
            except _QueryBudgetExceeded:
                out.append(X[i:i + 1].copy())
                stuck += 1
            finally:
                clf.predict = inner
        return np.vstack(out), stuck

    def evaluate(self, ctx: EvalContext) -> List[EvalResult]:
        from art.estimators.classification import SklearnClassifier
        from art.attacks.evasion import HopSkipJump

        ds = ctx.dataset
        Xte = np.asarray(ds.X_test, dtype=float)
        Xtr = np.asarray(ds.X_train, dtype=float)
        n_features = Xte.shape[1]
        base_pred = np.asarray(ctx.y_pred)
        y_true = np.asarray(ds.y_test)

        correct_idx = np.where(base_pred == y_true)[0]
        rng = np.random.default_rng(self.seed)
        n = min(self.sample_size, len(correct_idx))
        sel = rng.choice(correct_idx, size=n, replace=False) if n > 0 else np.array([], dtype=int)

        lo = float(min(Xtr.min(), Xte.min()))
        hi = float(max(Xtr.max(), Xte.max()))

        success_rate = 0.0
        median_norm = None
        n_flipped = 0
        per_row = []
        n_stuck = 0
        if n > 0:
            clf = _art_wrap(ctx.model, n_features, lo, hi, SklearnClassifier)
            adv, n_stuck = self._generate(clf, Xte[sel], HopSkipJump)
            adv_pred = ctx.model.predict(adv)
            flipped = adv_pred != base_pred[sel]
            success_rate = float(np.mean(flipped))
            n_flipped = int(np.sum(flipped))
            all_norms = np.linalg.norm(adv - Xte[sel], axis=1) / np.sqrt(n_features)
            per_row = [[int(i), bool(f), round(float(d), 4)]
                       for i, f, d in zip(sel, flipped, all_norms)]
            if n_flipped > 0:
                norms = np.linalg.norm(adv[flipped] - Xte[sel][flipped], axis=1) / np.sqrt(n_features)
                median_norm = float(np.median(norms))

        robustness = 1.0 - success_rate
        mechanism = (
            f"HopSkipJump L2 black-box attack against {n} correctly-classified test rows "
            f"(max_iter={self.max_iter}, max_eval={self.max_eval}, clip_values=({lo:.3f}, {hi:.3f})): "
            f"{success_rate * 100:.1f}% of attacked rows had their prediction flipped within budget "
            f"-> adversarial_robustness = 1 - success_rate = {robustness:.4f}."
            + (f" {n_stuck} row(s) stopped at the query budget (ART's step-size search stalled on "
               f"this model's flat decision surface) and count as not flipped."
               if n_stuck else "")
        )
        trace = make_trace(
            steps=[f"attack budget: {n} rows, max_iter={self.max_iter}, max_eval={self.max_eval}, "
                   f"init_eval={self.init_eval}, init_size={self.init_size} (raise with --tool-param art_robustness.<key>=...)",
                   f"adversarial_robustness = 1 − {n_flipped}/{n} = {robustness:.4f}"]
                  + ([f"perturbation_size = median(‖x'−x‖₂/√{n_features}) over flipped rows "
                      f"= {median_norm:.4f}"] if median_norm is not None else []),
            tables=[{"title": "Every attacked row (test-set index, did the attack flip it, "
                              "scaled L2 distance moved)",
                     "columns": ["row", "flipped", "‖x'−x‖₂/√d"], "rows": per_row}],
            charts=[{"title": "Distance the attacker had to move each row", "type": "bars",
                     "labels": [f"#{r[0]}" for r in per_row],
                     "series": [{"name": "distance", "values": [r[2] for r in per_row]}]}],
        )
        results = [
            EvalResult(self.name, "safety", InspectionTarget.BEHAVIOR,
                       "adversarial_robustness", robustness, True, mechanism,
                       {"n_attacked": int(n), "success_rate": round(success_rate, 4),
                        "n_flipped": n_flipped, "trace": trace}),
        ]
        if median_norm is not None:
            pert_mechanism = (
                f"Among the {n_flipped} of {n} attacked rows HopSkipJump successfully flipped, "
                f"median L2 perturbation norm (scaled by 1/sqrt(n_features)={1 / np.sqrt(n_features):.4f}) "
                f"needed = {median_norm:.4f}; larger means a bigger, more detectable change was required "
                f"to fool the model."
            )
            results.append(EvalResult(
                self.name, "safety", InspectionTarget.BEHAVIOR,
                "adversarial_perturbation_size", median_norm, True, pert_mechanism,
                {"n_flipped": n_flipped, "n_attacked": int(n), "trace": trace}))
        return results


register(ARTRobustnessEvaluator())
