"""
主程序入口
用法:
    python main.py download       # 下载数据
    python main.py train          # 训练模型
    python main.py backtest       # 回测
    python main.py predict        # 预测未来
    python main.py all            # 全部执行
"""
import os
import sys
import pandas as pd
import config
from data_processor import download_data
from trainer import StockTrainer
from visualizer import plot_backtest, plot_future_prediction, print_backtest_metrics

FEATURE_COLS = [
    "open", "high", "low", "close", "volume",
    "MA5", "MA10", "MA20", "MA60",
    "EMA12", "EMA26", "MACD", "Signal_Line", "MACD_Histogram",
    "RSI", "BB_Middle", "BB_Upper", "BB_Lower", "BB_Width",
    "OBV", "K", "D", "J", "ATR", "Volatility"
]
TARGET_COLS = ["open", "high", "low", "close"]


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    command = sys.argv[1]
    data_path = os.path.join(config.DATA_DIR, f"{config.DEFAULT_STOCK_CODE}.csv")
    model_path = "./models/best_model.pth"

    # 检查数据是否存在
    if not os.path.exists(data_path):
        print("📥 数据不存在，开始下载...")
        download_data()

    df = pd.read_csv(data_path)
    trainer = StockTrainer(FEATURE_COLS, TARGET_COLS)

    if command == "download":
        download_data()
    elif command == "train":
        print("🚀 开始训练...")
        train_loader, val_loader, test_loader, df = trainer.load_data(df)
        trainer.train(train_loader, val_loader)

        print("\n📊 测试集评估:")
        test_loss = trainer.evaluate(test_loader)
        print(f"Test Loss = {test_loss:.6f}")
    elif command == "backtest":
        if not os.path.exists(model_path):
            print(f"❌ 模型不存在: {model_path}，请先运行 train")
            return
        print("⏳ 加载模型并回测...")
        trainer.load_model(model_path)
        backtest_result = trainer.backtest(df)
        backtest_result.to_csv("./backtest_result.csv", index=False)
        print("✅ 回测结果已保存至 ./backtest_result.csv")
        print_backtest_metrics(backtest_result, TARGET_COLS)
        plot_backtest(backtest_result, TARGET_COLS)
    elif command == "predict":
        if not os.path.exists(model_path):
            print(f"❌ 模型不存在: {model_path}，请先运行 train")
            return
        print("🔮 预测未来...")
        trainer.load_model(model_path)
        future_pred = trainer.predict_future(df)
        future_pred.to_csv("./future_prediction.csv", index=True)
        print("\n📊 未来预测结果:")
        print(future_pred)
        plot_future_prediction(future_pred)
    elif command == "all":
        print("🚀 完整流程: 训练 -> 回测 -> 预测")
        train_loader, val_loader, test_loader, df = trainer.load_data(df)
        trainer.train(train_loader, val_loader)

        test_loss = trainer.evaluate(test_loader)
        print(f"\n📊 测试集 Loss = {test_loss:.6f}")

        print("\n⏳ 开始回测...")
        backtest_result = trainer.backtest(df)
        backtest_result.to_csv("./backtest_result.csv", index=False)
        print_backtest_metrics(backtest_result, TARGET_COLS)
        plot_backtest(backtest_result, TARGET_COLS)

        print("\n🔮 预测未来...")
        future_pred = trainer.predict_future(df)
        future_pred.to_csv("./future_prediction.csv", index=True)
        print("\n📊 未来预测结果:")
        print(future_pred)
        plot_future_prediction(future_pred)
    else:
        print(f"未知命令: {command}")
        print(__doc__)


if __name__ == "__main__":
    main()
