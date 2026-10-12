"""Common frozen-representation regressors."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable, Iterable

import numpy as np
try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # classical baselines do not require PyTorch
    torch = None
    class _NN:
        """在未安装 PyTorch 时提供最小的类型占位。"""

        class Module:
            """使经典回归器可独立导入的模块占位类。"""

            pass
    nn = _NN()


class RidgeRegressor:
    """Multi-target ridge with an unpenalized intercept."""

    def __init__(self, alpha: float = 1.0) -> None:
        """设置岭回归的 L2 惩罚强度。"""
        self.alpha = float(alpha)
        self.mean_x_: np.ndarray | None = None
        self.mean_y_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RidgeRegressor":
        """中心化特征与目标后求解多目标岭回归系数。"""
        x, y = np.asarray(x, float), np.asarray(y, float)
        if y.ndim == 1:
            y = y[:, None]
        self.mean_x_, self.mean_y_ = x.mean(0), y.mean(0)
        xc, yc = x - self.mean_x_, y - self.mean_y_
        gram = xc.T @ xc + self.alpha * np.eye(x.shape[1])
        self.coef_ = np.linalg.solve(gram, xc.T @ yc)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """对新样本预测并恢复训练目标均值。"""
        if self.coef_ is None or self.mean_x_ is None or self.mean_y_ is None:
            raise RuntimeError("Fit regressor before predict")
        return (np.asarray(x, float) - self.mean_x_) @ self.coef_ + self.mean_y_


@dataclass
class BaselineResult:
    """保存基线模型的预测、所选参数和可选已拟合模型。"""
    prediction: np.ndarray
    selected_params: dict[str, float | int | str]
    train_prediction: np.ndarray | None = None
    model: object | None = None


def normalized_mse(y_true: np.ndarray, y_pred: np.ndarray, scale: np.ndarray) -> float:
    """按目标标准差归一化后计算平均平方误差。"""
    return float(np.mean(((y_pred - y_true) / scale) ** 2))


def choose_candidate(
    candidates: Iterable[dict],
    folds: list[tuple[np.ndarray, np.ndarray]],
    y: np.ndarray,
    fit_predict: Callable[[dict, np.ndarray, np.ndarray], np.ndarray],
) -> dict:
    """按内层折平均归一化误差选择超参数。"""
    candidates = list(candidates)
    if not candidates:
        raise ValueError("At least one valid hyperparameter candidate is required")
    y_array = np.asarray(y)
    if y_array.ndim == 1:
        y_array = y_array[:, None]
    scale = y_array.std(axis=0)
    scale[scale < 1e-12] = 1.0
    scored = []
    for candidate in candidates:
        fold_scores = [
            normalized_mse(y_array[valid], np.asarray(fit_predict(candidate, train, valid)).reshape(-1, y_array.shape[1]), scale)
            for train, valid in folds
        ]
        scored.append((float(np.mean(fold_scores)), candidate))
    return min(scored, key=lambda item: item[0])[1]


PLS_COMPONENTS = (2, 4, 8, 16)
PLS_MAX_ITER = 500
PLS_TOLERANCE = 1e-6
RF_ESTIMATORS = (200,)
RF_MAX_DEPTHS = (None, 20)
RF_MIN_SAMPLES_LEAF = (1, 2)
RF_MAX_FEATURES = ("sqrt", 1.0)


def pls(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    components: tuple[int, ...] = PLS_COMPONENTS,
    max_iter: int = PLS_MAX_ITER,
    tolerance: float = PLS_TOLERANCE,
) -> BaselineResult:
    """选择潜变量个数并拟合 PLS 回归模型。"""
    from sklearn.cross_decomposition import PLSRegression

    minimum_inner_train = min(len(train) for train, _ in folds)
    maximum = min(train_x.shape[0] - 1, minimum_inner_train - 1, train_x.shape[1])
    candidates = [{"n_components": min(value, maximum)} for value in sorted(set(components)) if value > 0]

    def fit_predict(candidate, train, valid):
        """用当前内层训练折评估 PLS 候选维数。"""
        n = min(candidate["n_components"], len(train) - 1, train_x.shape[1])
        model = PLSRegression(n_components=n, scale=True, max_iter=max_iter, tol=tolerance)
        return model.fit(train_x[train], train_y[train]).predict(train_x[valid])

    best = choose_candidate(candidates, folds, train_y, fit_predict)
    model = PLSRegression(
        n_components=best["n_components"], scale=True, max_iter=max_iter, tol=tolerance
    ).fit(train_x, train_y)
    return BaselineResult(model.predict(test_x), best, model.predict(train_x), model)


def random_forest(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    estimators: tuple[int, ...] = RF_ESTIMATORS,
    max_depths: tuple[int | None, ...] = RF_MAX_DEPTHS,
    min_samples_leaf: tuple[int, ...] = RF_MIN_SAMPLES_LEAF,
    max_features: tuple[str | float, ...] = RF_MAX_FEATURES,
    seed: int = 42,
    jobs: int = -1,
) -> BaselineResult:
    """在内层折选择树参数并拟合随机森林。"""
    from sklearn.ensemble import RandomForestRegressor

    candidates = [
        dict(n_estimators=n, max_depth=depth, min_samples_leaf=leaf, max_features=features)
        for n, depth, leaf, features in product(estimators, max_depths, min_samples_leaf, max_features)
    ]

    def build(candidate):
        """用候选参数创建具有固定随机种子的森林。"""
        return RandomForestRegressor(
            **candidate, random_state=seed, n_jobs=jobs, criterion="squared_error"
        )

    def fit_predict(candidate, train, valid):
        """用当前内层折评估森林候选参数。"""
        target = np.asarray(train_y[train])
        if target.ndim == 2 and target.shape[1] == 1:
            target = target[:, 0]
        return build(candidate).fit(train_x[train], target).predict(train_x[valid])

    best = choose_candidate(candidates, folds, train_y, fit_predict)
    target = np.asarray(train_y)
    if target.ndim == 2 and target.shape[1] == 1:
        target = target[:, 0]
    model = build(best).fit(train_x, target)
    return BaselineResult(model.predict(test_x), best, model.predict(train_x), model)


MLP_HIDDEN_DIM = 64
MLP_DROPOUT = 0.1
MLP_EPOCHS = 100
MLP_LEARNING_RATE = 1e-3


def fit_fixed_representation_mlp(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    seed: int,
    batch_size: int | str = 64,
    scheduler_name: str = "warmup_cosine",
    scheduler_eta_min: float = 1e-6,
) -> tuple[np.ndarray, int, dict[str, object]]:
    """Fit the repository's fixed-embedding MLP head with fixed settings."""
    if torch is None:
        raise RuntimeError("PyTorch is required for the MLP regressor")
    from torch.utils.data import DataLoader, TensorDataset
    from .scheduling import build_step_scheduler

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    x_mean = train_x.mean(axis=0)
    x_scale = train_x.std(axis=0)
    x_scale[x_scale < 1e-12] = 1.0
    y_mean = float(np.mean(train_y))
    y_scale = float(np.std(train_y)) or 1.0
    x_normalized = (train_x - x_mean) / x_scale
    test_normalized = (test_x - x_mean) / x_scale
    y_normalized = (train_y - y_mean) / y_scale
    # 归一化统计量仅来自训练集，预测后再恢复目标的原始尺度。
    model = nn.Sequential(
        nn.Linear(train_x.shape[1], MLP_HIDDEN_DIM), nn.GELU(), nn.Dropout(MLP_DROPOUT),
        nn.Linear(MLP_HIDDEN_DIM, 1),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=MLP_LEARNING_RATE, weight_decay=1e-4)
    effective_batch_size = len(train_x) if batch_size == "full" else min(int(batch_size), len(train_x))
    loader = DataLoader(
        TensorDataset(
            torch.as_tensor(x_normalized, dtype=torch.float32),
            torch.as_tensor(y_normalized[:, None], dtype=torch.float32),
        ), batch_size=effective_batch_size, shuffle=True,
    )
    scheduler = build_step_scheduler(
        optimizer, scheduler_name, total_steps=len(loader) * MLP_EPOCHS,
        warmup_steps=len(loader), eta_min=scheduler_eta_min,
    )
    model.train()
    for _ in range(MLP_EPOCHS):
        for batch_x, batch_y in loader:
            loss = (model(batch_x.to(device)) - batch_y.to(device)).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
    model.eval()
    with torch.no_grad():
        predicted = model(torch.as_tensor(test_normalized, dtype=torch.float32, device=device))
    config = {
        "hidden_dim": MLP_HIDDEN_DIM, "dropout": MLP_DROPOUT,
        "epochs": MLP_EPOCHS, "learning_rate": MLP_LEARNING_RATE,
        "scheduler": scheduler_name, "scheduler_eta_min": scheduler_eta_min,
        "warmup_steps": len(loader), "scheduler_interval": "step",
        "final_learning_rate": float(optimizer.param_groups[0]["lr"]),
    }
    return predicted.cpu().numpy()[:, 0] * y_scale + y_mean, sum(p.numel() for p in model.parameters()), config


class EncoderMLPRegressor(nn.Module):
    """End-to-end spectral regressor with an unfrozen encoder and two linear layers."""

    def __init__(
        self,
        encoder: nn.Module,
        output_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        """连接可训练的光谱编码器与两层回归头。"""
        if torch is None:
            raise RuntimeError("PyTorch is required for EncoderMLPRegressor")
        super().__init__()
        if output_dim < 1 or hidden_dim < 1:
            raise ValueError("output_dim and hidden_dim must be positive")
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(encoder.embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, spectra: torch.Tensor) -> torch.Tensor:
        """从编码器池化表示预测所有目标。"""
        _, pooled = self.encoder(spectra, None)
        return self.head(pooled)
