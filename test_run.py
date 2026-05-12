"""
快速测试脚本 - 训练少量 epochs 并运行回测
"""
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 非 GUI 后端

# 临时调整配置
import config
config.MAX_EPOCHS = 30
config.PATIENCE = 5
config.HIDDEN_SIZE = 64
config.NUM_LAYERS = 2

from data_processor import download_data
from trainer import StockTrainer
from visualizer import print_backtest_metrics

FEATURE_COLS = [
    "open", "high", "low", "close", "volume",
    "MA5", "MA10", "MA20", "MA60",
    "EMA12", "EMA26", "MACD", "Signal_Line",
    "RSI", "BB_Middle", "BB_Upper", "BB_Lower",
    "K", "D", "J", "ATR", "Volatility"
]
TARGET_COLS = ["open", "high", "low", "close"]


def main():
    print("=" * 60)
    print("📈 LSTM 股票预测 - 快速测试")
    print("=" * 60)

    data_path = os.path.join(config.DATA_DIR, f"{config.DEFAULT_STOCK_CODE}.csv")

    # 1. 检查数据
    if not os.path.exists(data_path):
        print("\n📥 下载数据...")
        download_data()

    df = pd.read_csv(data_path)
    print(f"\n✅ 数据加载完成，共 {len(df)} 条记录")

    # 2. 初始化并训练
    print("\n🚀 开始训练...")
    trainer = StockTrainer(FEATURE_COLS, TARGET_COLS)
    train_loader, val_loader, test_loader, df = trainer.load_data(df)
    trainer.train(train_loader, val_loader)

    # 3. 测试集评估
    print("\n📊 测试集评估:")
    test_loss = trainer.evaluate(test_loader)
    print(f"Test Loss = {test_loss:.6f}")

    # 4. 回测
    print("\n⏳ 运行回测...")
    backtest_result = trainer.backtest(df)
    backtest_result.to_csv("./backtest_result.csv", index=False)
    print("✅ 回测结果已保存至 ./backtest_result.csv")
    print_backtest_metrics(backtest_result, TARGET_COLS)

    # 5. 预测未来
    print("\n🔮 预测未来 5 天...")
    future_pred = trainer.predict_future(df, days=5)
    future_pred.to_csv("./future_prediction.csv", index=True)
    print("\n📊 未来预测结果:")
    print(future_pred.round(2))

    print("\n" + "=" * 60)
    print("✅ 测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
