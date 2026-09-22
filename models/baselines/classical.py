"""Classical and simple representation baselines for E1-a."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable, Iterable

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from ..shared.regressors import RidgeRegressor


@dataclass
class BaselineResult:
    prediction: np.ndarray
    selected_params: dict[str, float | int | str]
    train_prediction: np.ndarray | None = None


def normalized_mse(y_true: np.ndarray, y_pred: np.ndarray, scale: np.ndarray) -> float:
    return float(np.mean(((y_pred - y_true) / scale) ** 2))


def choose_candidate(
    candidates: Iterable[dict],
    folds: list[tuple[np.ndarray, np.ndarray]],
    y: np.ndarray,
    fit_predict: Callable[[dict, np.ndarray, np.ndarray], np.ndarray],
) -> dict:
    candidates = list(candidates)
    if not candidates:
        raise ValueError("At least one valid hyperparameter candidate is required")
    scale = y.std(axis=0)
    scale[scale < 1e-12] = 1.0
    scored = []
    for candidate in candidates:
        fold_scores = [
            normalized_mse(y[valid], fit_predict(candidate, train, valid), scale)
            for train, valid in folds
        ]
        scored.append((float(np.mean(fold_scores)), candidate))
    return min(scored, key=lambda item: item[0])[1]


def raw_ridge(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    alphas: tuple[float, ...],
) -> BaselineResult:
    candidates = [{"alpha": alpha} for alpha in alphas]

    def fit_predict(candidate, train, valid):
        return RidgeRegressor(candidate["alpha"]).fit(train_x[train], train_y[train]).predict(train_x[valid])

    best = choose_candidate(candidates, folds, train_y, fit_predict)
    model = RidgeRegressor(best["alpha"]).fit(train_x, train_y)
    return BaselineResult(model.predict(test_x), best, model.predict(train_x))


def pca_ridge(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    components: tuple[int, ...],
    alphas: tuple[float, ...],
    seed: int,
) -> BaselineResult:
    minimum_inner_train = min(len(train) for train, _ in folds)
    maximum = min(train_x.shape[0] - 1, minimum_inner_train - 1, train_x.shape[1])
    valid_components = sorted({min(value, maximum) for value in components if value > 0})
    candidates = [dict(n_components=n, alpha=alpha) for n, alpha in product(valid_components, alphas)]

    def fit_predict(candidate, train, valid):
        n = min(candidate["n_components"], len(train) - 1, train_x.shape[1])
        pca = PCA(n_components=n, random_state=seed).fit(train_x[train])
        model = RidgeRegressor(candidate["alpha"]).fit(pca.transform(train_x[train]), train_y[train])
        return model.predict(pca.transform(train_x[valid]))

    best = choose_candidate(candidates, folds, train_y, fit_predict)
    pca = PCA(n_components=best["n_components"], random_state=seed).fit(train_x)
    model = RidgeRegressor(best["alpha"]).fit(pca.transform(train_x), train_y)
    return BaselineResult(model.predict(pca.transform(test_x)), best, model.predict(pca.transform(train_x)))


def pls(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    components: tuple[int, ...],
    max_iter: int,
    tolerance: float,
) -> BaselineResult:
    minimum_inner_train = min(len(train) for train, _ in folds)
    maximum = min(train_x.shape[0] - 1, minimum_inner_train - 1, train_x.shape[1])
    candidates = [{"n_components": min(value, maximum)} for value in sorted(set(components)) if value > 0]

    def fit_predict(candidate, train, valid):
        n = min(candidate["n_components"], len(train) - 1, train_x.shape[1])
        model = PLSRegression(n_components=n, scale=True, max_iter=max_iter, tol=tolerance)
        return model.fit(train_x[train], train_y[train]).predict(train_x[valid])

    best = choose_candidate(candidates, folds, train_y, fit_predict)
    model = PLSRegression(
        n_components=best["n_components"], scale=True, max_iter=max_iter, tol=tolerance
    ).fit(train_x, train_y)
    return BaselineResult(model.predict(test_x), best, model.predict(train_x))


def rbf_svr(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    c_values: tuple[float, ...],
    epsilon_values: tuple[float, ...],
    gamma_values: tuple[str | float, ...],
) -> BaselineResult:
    candidates = [dict(C=c, epsilon=e, gamma=g) for c, e, g in product(c_values, epsilon_values, gamma_values)]

    def predict_targets(candidate, x_fit, y_fit, x_predict):
        mean, scale = y_fit.mean(axis=0), y_fit.std(axis=0)
        scale[scale < 1e-12] = 1.0
        outputs = []
        for column in range(y_fit.shape[1]):
            model = SVR(kernel="rbf", C=candidate["C"], epsilon=candidate["epsilon"], gamma=candidate["gamma"])
            model.fit(x_fit, (y_fit[:, column] - mean[column]) / scale[column])
            outputs.append(model.predict(x_predict) * scale[column] + mean[column])
        return np.column_stack(outputs)

    def fit_predict(candidate, train, valid):
        return predict_targets(candidate, train_x[train], train_y[train], train_x[valid])

    best = choose_candidate(candidates, folds, train_y, fit_predict)
    return BaselineResult(
        predict_targets(best, train_x, train_y, test_x),
        best,
        predict_targets(best, train_x, train_y, train_x),
    )


def random_forest(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    estimators: tuple[int, ...],
    max_depths: tuple[int | None, ...],
    min_samples_leaf: tuple[int, ...],
    max_features: tuple[str | float, ...],
    seed: int,
    jobs: int,
) -> BaselineResult:
    candidates = [
        dict(n_estimators=n, max_depth=depth, min_samples_leaf=leaf, max_features=features)
        for n, depth, leaf, features in product(estimators, max_depths, min_samples_leaf, max_features)
    ]

    def build(candidate):
        return RandomForestRegressor(
            **candidate, random_state=seed, n_jobs=jobs, criterion="squared_error"
        )

    def fit_predict(candidate, train, valid):
        return build(candidate).fit(train_x[train], train_y[train]).predict(train_x[valid])

    best = choose_candidate(candidates, folds, train_y, fit_predict)
    model = build(best).fit(train_x, train_y)
    return BaselineResult(model.predict(test_x), best, model.predict(train_x))
