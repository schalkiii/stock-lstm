"""
简化可运行版本 - 直接在内存中训练和测试
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============= 配置 =============
SEQ_LENGTH = 10
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2
LEARNING_RATE = 0.001
PATIENCE = 10
MAX_EPOCHS = 50
BATCH_SIZE = 32
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
PRED_DAYS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============= 模型定义 =============
class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

# ============= 主程序 =============
def main():
    print("=" * 60)
    print("📈 LSTM 股票预测 - 简化运行")
    print("=" * 60)

    # 1. 加载数据
    data_path = "./datasets/sh.000001.csv"
    df = pd.read_csv(data_path)
    
    # 确保数值列是数值类型
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    print(f"✅ 数据加载成功，共 {len(df)} 条记录")

    # 2. 计算简单技术指标
    df = df.copy()
    df["MA5"] = df["close"].rolling(window=5).mean()
    df["MA10"] = df["close"].rolling(window=10).mean()
    df["MA20"] = df["close"].rolling(window=20).mean()
    df["Return"] = df["close"].pct_change()
    df = df.dropna()
    print(f"✅ 技术指标计算完成，剩余 {len(df)} 条记录")

    # 3. 定义特征和目标
    feature_cols = ["open", "high", "low", "close", "volume", "MA5", "MA10", "MA20", "Return"]
    target_cols = ["open", "high", "low", "close"]

    # 4. 准备数据 - 只在训练集 fit scalers
    n_train = int(len(df) * TRAIN_RATIO)
    df_train = df.iloc[:n_train]
    
    scalers = {}
    df_scaled = df.copy()
    for col in feature_cols:
        scalers[col] = MinMaxScaler(feature_range=(0, 1))
        scalers[col].fit(df_train[col].values.reshape(-1, 1))
        df_scaled[col] = scalers[col].transform(df[col].values.reshape(-1, 1)).flatten()
    
    data = df_scaled[feature_cols].values
    X, y = [], []
    for i in range(len(data) - SEQ_LENGTH):
        X.append(data[i:i + SEQ_LENGTH])
        y.append(data[i + SEQ_LENGTH, [feature_cols.index(t) for t in target_cols]])
    X, y = np.array(X), np.array(y)

    # 5. 划分数据
    n = len(X)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)
    
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    print(f"✅ 数据划分: 训练 {len(X_train)}, 验证 {len(X_val)}, 测试 {len(X_test)}")

    # 6. 创建 DataLoader
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                             torch.tensor(y_train, dtype=torch.float32)),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                           torch.tensor(y_val, dtype=torch.float32)),
                            batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                           torch.tensor(y_test, dtype=torch.float32)),
                            batch_size=BATCH_SIZE, shuffle=False)

    # 7. 初始化模型
    model = StockLSTM(
        input_size=len(feature_cols),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=len(target_cols),
        dropout=DROPOUT
    ).to(DEVICE)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5, factor=0.5)

    # 8. 训练
    print("\n🚀 开始训练...")
    best_val_loss = float("inf")
    counter = 0
    best_model_state = None

    for epoch in range(MAX_EPOCHS):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            best_model_state = model.state_dict().copy()
            if epoch % 5 == 0:
                print(f"✅ Epoch {epoch:2d}: 最佳模型! Val Loss = {val_loss:.6f}")
        else:
            counter += 1
            if counter >= PATIENCE:
                print(f"⏹️  Epoch {epoch:2d}: 早停!")
                break
        
        if epoch % 10 == 0:
            print(f"📊 Epoch {epoch:2d} | Train Loss = {train_loss:.6f} | Val Loss = {val_loss:.6f}")

    print(f"\n🏁 训练完成! 最佳 Val Loss = {best_val_loss:.6f}")

    # 9. 加载最佳模型并测试
    model.load_state_dict(best_model_state)
    model.eval()

    test_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            test_loss += loss.item()
    test_loss /= len(test_loader)
    print(f"\n📊 测试集 Loss = {test_loss:.6f}")

    # 10. 回测
    print("\n⏳ 回测测试集...")
    model.eval()
    predictions, actuals = [], []

    with torch.no_grad():
        test_start_idx = val_end
        for i in range(test_start_idx, len(data) - SEQ_LENGTH):
            X_seq = data[i:i + SEQ_LENGTH]
            X_tensor = torch.tensor(X_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            pred_scaled = model(X_tensor).cpu().numpy()[0]
            predictions.append(pred_scaled)
            actuals.append(data[i + SEQ_LENGTH, [feature_cols.index(t) for t in target_cols]])

    # 反归一化
    pred_arr, actual_arr = np.array(predictions), np.array(actuals)
    pred_inv = pred_arr.copy()
    actual_inv = actual_arr.copy()
    for i, col in enumerate(target_cols):
        pred_inv[:, i] = scalers[col].inverse_transform(pred_arr[:, i].reshape(-1, 1)).flatten()
        actual_inv[:, i] = scalers[col].inverse_transform(actual_arr[:, i].reshape(-1, 1)).flatten()

    # 计算回测指标
    print("\n" + "=" * 60)
    print("📈 回测指标")
    print("=" * 60)
    for i, col in enumerate(target_cols):
        actual = actual_inv[:, i]
        pred = pred_inv[:, i]
        mae = np.mean(np.abs(actual - pred))
        rmse = np.sqrt(np.mean((actual - pred) ** 2))
        mape = np.mean(np.abs((actual - pred) / (actual + 1e-8))) * 100
        print(f"{col.upper():8s} | MAE = {mae:8.2f} | RMSE = {rmse:8.2f} | MAPE = {mape:6.2f}%")

    # 11. 可视化
    os.makedirs("./results", exist_ok=True)
    print("\n🎨 生成回测图表...")
    dates = df["date"].iloc[test_start_idx + SEQ_LENGTH: len(data)].values
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, col in enumerate(target_cols):
        ax = axes[i]
        ax.plot(range(len(actual_inv)), actual_inv[:, i], label="实际", color="blue", linewidth=1.5, alpha=0.8)
        ax.plot(range(len(pred_inv)), pred_inv[:, i], label="预测", color="red", linewidth=1.5, alpha=0.7)
        ax.set_title(f"{col.upper()} 预测 vs 实际", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("./results/backtest_plot.png", dpi=150)
    print(f"✅ 回测图表已保存到 ./results/backtest_plot.png")

    # 12. 预测未来
    print(f"\n🔮 预测未来 {PRED_DAYS} 天...")
    model.eval()
    current_seq = data[-SEQ_LENGTH:].copy()
    future_preds_scaled = []

    with torch.no_grad():
        for _ in range(PRED_DAYS):
            X_tensor = torch.tensor(current_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            pred_scaled = model(X_tensor).cpu().numpy()[0]
            future_preds_scaled.append(pred_scaled)
            new_step = current_seq[-1].copy()
            for i, col in enumerate(target_cols):
                new_step[feature_cols.index(col)] = pred_scaled[i]
            current_seq = np.vstack([current_seq[1:], new_step])

    # 反归一化
    future_preds = np.array(future_preds_scaled)
    for i, col in enumerate(target_cols):
        future_preds[:, i] = scalers[col].inverse_transform(future_preds[:, i].reshape(-1, 1)).flatten()
    
    future_df = pd.DataFrame(future_preds, columns=target_cols, index=[f"Day {i+1}" for i in range(PRED_DAYS)])
    print("\n📊 未来预测结果:")
    print(future_df.round(2))
    future_df.to_csv("./results/future_prediction.csv")
    print(f"✅ 预测结果已保存到 ./results/future_prediction.csv")

    print("\n" + "=" * 60)
    print("✅ 所有任务完成!")
    print("=" * 60)

if __name__ == "__main__":
    main()
