"""Classical and simple representation baselines for E1-a."""

from __future__ import annotations

from itertools import product

import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVR

from ..shared.regressors import (
    BaselineResult, RidgeRegressor, choose_candidate, normalized_mse, pls, random_forest,
)


def raw_ridge(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    alphas: tuple[float, ...],
) -> BaselineResult:
    """在原始特征上调节岭回归惩罚并拟合最终模型。"""
    candidates = [{"alpha": alpha} for alpha in alphas]

    def fit_predict(candidate, train, valid):
        """用一个内层折评估岭回归候选参数。"""
        return RidgeRegressor(candidate["alpha"]).fit(train_x[train], train_y[train]).predict(train_x[valid])

    best = choose_candidate(candidates, folds, train_y, fit_predict)
    model = RidgeRegressor(best["alpha"]).fit(train_x, train_y)
    return BaselineResult(model.predict(test_x), best, model.predict(train_x), model)


def pca_ridge(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    components: tuple[int, ...],
    alphas: tuple[float, ...],
    seed: int,
) -> BaselineResult:
    """在内层折联合选择 PCA 维数和岭回归惩罚。"""
    minimum_inner_train = min(len(train) for train, _ in folds)
    maximum = min(train_x.shape[0] - 1, minimum_inner_train - 1, train_x.shape[1])
    valid_components = sorted({min(value, maximum) for value in components if value > 0})
    candidates = [dict(n_components=n, alpha=alpha) for n, alpha in product(valid_components, alphas)]

    def fit_predict(candidate, train, valid):
        """仅用内层训练样本拟合 PCA 与岭回归。"""
        n = min(candidate["n_components"], len(train) - 1, train_x.shape[1])
        pca = PCA(n_components=n, random_state=seed).fit(train_x[train])
        model = RidgeRegressor(candidate["alpha"]).fit(pca.transform(train_x[train]), train_y[train])
        return model.predict(pca.transform(train_x[valid]))

    best = choose_candidate(candidates, folds, train_y, fit_predict)
    pca = PCA(n_components=best["n_components"], random_state=seed).fit(train_x)
    model = RidgeRegressor(best["alpha"]).fit(pca.transform(train_x), train_y)
    return BaselineResult(model.predict(pca.transform(test_x)), best, model.predict(pca.transform(train_x)), (pca, model))


def rbf_svr(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    c_values: tuple[float, ...],
    epsilon_values: tuple[float, ...],
    gamma_values: tuple[str | float, ...],
) -> BaselineResult:
    """按目标拟合 RBF-SVR，并用内层折选择核参数。"""
    candidates = [dict(C=c, epsilon=e, gamma=g) for c, e, g in product(c_values, epsilon_values, gamma_values)]

    def predict_targets(candidate, x_fit, y_fit, x_predict):
        """逐目标标准化标签、拟合 SVR 并还原预测尺度。"""
        mean, scale = y_fit.mean(axis=0), y_fit.std(axis=0)
        scale[scale < 1e-12] = 1.0
        outputs = []
        for column in range(y_fit.shape[1]):
            model = SVR(kernel="rbf", C=candidate["C"], epsilon=candidate["epsilon"], gamma=candidate["gamma"])
            model.fit(x_fit, (y_fit[:, column] - mean[column]) / scale[column])
            outputs.append(model.predict(x_predict) * scale[column] + mean[column])
        return np.column_stack(outputs)

    def fit_predict(candidate, train, valid):
        """评估一个 SVR 候选参数在内层验证折的预测。"""
        return predict_targets(candidate, train_x[train], train_y[train], train_x[valid])

    best = choose_candidate(candidates, folds, train_y, fit_predict)
    return BaselineResult(
        predict_targets(best, train_x, train_y, test_x),
        best,
        predict_targets(best, train_x, train_y, train_x),
        None,
    )
