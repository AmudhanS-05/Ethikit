"""Every model the zoo offers must fit, predict and give probabilities on ethikit's data,
since every evaluator talks to models only through that surface."""
import numpy as np
import pytest

from ethikit.benchmark.models import (
    DEFAULT_SELECTION, OPTIONAL_FAMILY, OPTIONAL_MODELS, default_params, family_of,
    flat_classifiers, flat_regressors,
)


@pytest.mark.parametrize("name", sorted(flat_classifiers()))
def test_every_classifier_fits_and_gives_probabilities(name, tiny_ds):
    model = flat_classifiers()[name]().fit(tiny_ds.X_train, tiny_ds.y_train)
    pred = model.predict(tiny_ds.X_test)
    proba = model.predict_proba(tiny_ds.X_test)
    assert len(pred) == len(tiny_ds.y_test)
    assert proba.shape == (len(tiny_ds.y_test), 2)
    assert np.allclose(proba.sum(axis=1), 1, atol=1e-5)
    assert isinstance(default_params(name), dict)        # the console's hyperparameter panel


@pytest.mark.parametrize("name", sorted(flat_regressors()))
def test_every_regressor_fits(name, tiny_ds):
    y = tiny_ds.X_train[:, 0] * 2.0
    model = flat_regressors()[name]().fit(tiny_ds.X_train, y)
    assert len(model.predict(tiny_ds.X_test)) == len(tiny_ds.y_test)


def test_art_robustness_runs_on_a_non_sklearn_model(tiny_ds):
    """ART's SklearnClassifier rejects XGBoost/LightGBM/torch adapters; they must still be attacked."""
    pytest.importorskip("art")
    from sklearn.linear_model import LogisticRegression
    from ethikit.benchmark.runner import _predict_all
    from ethikit.evaluators.art_robustness import ARTRobustnessEvaluator

    class Wrapped:                                       # sklearn-shaped, but not an sklearn class
        def __init__(self):
            self.inner = LogisticRegression(max_iter=1000).fit(tiny_ds.X_train, tiny_ds.y_train)
            self.classes_ = self.inner.classes_

        def predict(self, X):
            return self.inner.predict(X)

        def predict_proba(self, X):
            return self.inner.predict_proba(X)

    res = ARTRobustnessEvaluator(sample_size=4, max_iter=2, max_eval=50).evaluate(
        _predict_all(Wrapped(), tiny_ds))
    rob = next(r for r in res if r.metric == "adversarial_robustness")
    assert 0.0 <= rob.value <= 1.0


def test_art_query_budget_stops_a_stalled_attack():
    """ART's step-size search can query forever on flat (boosted-tree) surfaces; the budget
    must stop that row, keep it unchanged and count it, without touching the other rows."""
    from ethikit.evaluators.art_robustness import ARTRobustnessEvaluator

    class Clf:
        def predict(self, x, **_):
            return np.tile([1.0, 0.0], (len(x), 1))

    class StallsOnSecondRow:
        def __init__(self, classifier, **_):
            self.c = classifier

        def generate(self, x):
            if x[0, 0] == 2.0:
                while True:                              # what ART does on a stalled row
                    self.c.predict(x)
            return x + 0.5

    X = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    adv, stuck = ARTRobustnessEvaluator(max_iter=2)._generate(Clf(), X, StallsOnSecondRow)
    assert stuck == 1
    assert np.array_equal(adv[1], X[1]) and np.allclose(adv[[0, 2]], X[[0, 2]] + 0.5)


def test_optional_models_name_a_family_and_defaults_exist():
    assert set(OPTIONAL_MODELS) == set(OPTIONAL_FAMILY)
    for m in OPTIONAL_MODELS:
        if m in flat_classifiers():                      # installed: it sits in the family it claims
            assert family_of(m) == OPTIONAL_FAMILY[m]
    assert set(DEFAULT_SELECTION) <= set(flat_classifiers())
