"""
可视化模块 - 绘制预测结果和回测
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List


def plot_backtest(backtest_result: pd.DataFrame, target_cols: List[str], save_path: str = "./backtest_plot.png"):
    """绘制回测结果"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for i, col in enumerate(target_cols):
        ax = axes[i]
        ax.plot(backtest_result["date"], backtest_result[f"actual_{col}"], label=f"实际 {col}", color="blue", linewidth=1)
        ax.plot(backtest_result["date"], backtest_result[f"pred_{col}"], label=f"预测 {col}", color="red", linewidth=1, alpha=0.7)
        ax.set_title(f"{col.upper()} 预测 vs 实际", fontsize=12)
        ax.legend(loc="best", fontsize=10)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"回测图表已保存至: {save_path}")
    plt.show()


def plot_future_prediction(future_pred: pd.DataFrame, save_path: str = "./future_prediction.png"):
    """绘制未来预测"""
    fig, ax = plt.subplots(figsize=(12, 6))
    for col in future_pred.columns:
        ax.plot(range(1, len(future_pred) + 1), future_pred[col], marker="o", label=col.upper())

    ax.set_title("未来股价预测", fontsize=14)
    ax.set_xlabel("预测天数", fontsize=12)
    ax.set_ylabel("价格", fontsize=12)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"未来预测图表已保存至: {save_path}")
    plt.show()


def print_backtest_metrics(backtest_result: pd.DataFrame, target_cols: List[str]):
    """打印回测指标"""
    print("\n" + "=" * 50)
    print("📈 回测指标")
    print("=" * 50)

    for col in target_cols:
        actual = backtest_result[f"actual_{col}"].values
        pred = backtest_result[f"pred_{col}"].values
        mae = np.mean(np.abs(actual - pred))
        mse = np.mean((actual - pred) ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((actual - pred) / actual)) * 100

        print(f"\n{col.upper()}:")
        print(f"  MAE   = {mae:.4f}")
        print(f"  RMSE  = {rmse:.4f}")
        print(f"  MAPE  = {mape:.4f}%")

    print("=" * 50)
