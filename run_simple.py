"""
LSTM 股票预测 - 完整可运行版本
修复: 梯度裁剪、中文图表、基线对比、结果保存
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
import matplotlib.dates as mdates

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

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
GRAD_CLIP = 5.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def calculate_indicators(df):
    df = df.copy()
    df["MA5"] = df["close"].rolling(window=5).mean()
    df["MA10"] = df["close"].rolling(window=10).mean()
    df["MA20"] = df["close"].rolling(window=20).mean()
    df["Return"] = df["close"].pct_change()
    return df.dropna()


def main():
    print("=" * 60)
    print("LSTM Stock Prediction")
    print("=" * 60)

    data_path = "./datasets/sh.000001.csv"
    df = pd.read_csv(data_path)

    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"Data loaded: {len(df)} records")

    df = calculate_indicators(df)
    print(f"After indicators: {len(df)} records")

    feature_cols = ["open", "high", "low", "close", "volume", "MA5", "MA10", "MA20", "Return"]
    target_cols = ["open", "high", "low", "close"]

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

    n = len(X)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    print(f"Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                             torch.tensor(y_train, dtype=torch.float32)),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                           torch.tensor(y_val, dtype=torch.float32)),
                            batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                           torch.tensor(y_test, dtype=torch.float32)),
                            batch_size=BATCH_SIZE, shuffle=False)

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

    print(f"\nTraining on {DEVICE}...")
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
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
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            counter += 1
            if counter >= PATIENCE:
                print(f"Early stop at epoch {epoch}")
                break

        if epoch % 10 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:3d} | Train={train_loss:.6f} | Val={val_loss:.6f} | Best={best_val_loss:.6f} | LR={lr:.6f}")

    print(f"\nTraining done! Best Val Loss = {best_val_loss:.6f}")

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
    print(f"Test Loss = {test_loss:.6f}")

    # ---- Backtest ----
    print("\nBacktesting on test set...")
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

    pred_arr, actual_arr = np.array(predictions), np.array(actuals)
    pred_inv = pred_arr.copy()
    actual_inv = actual_arr.copy()
    for i, col in enumerate(target_cols):
        pred_inv[:, i] = scalers[col].inverse_transform(pred_arr[:, i].reshape(-1, 1)).flatten()
        actual_inv[:, i] = scalers[col].inverse_transform(actual_arr[:, i].reshape(-1, 1)).flatten()

    # Naive baseline: predict today's price as tomorrow's
    naive_inv = actual_inv[:-1].copy()

    print("\n" + "=" * 70)
    print("Backtest Metrics (LSTM vs Naive Baseline)")
    print("=" * 70)
    header = f"{'Metric':<8} | {'Target':<8} | {'LSTM':>12} | {'Naive':>12} | {'Better?':>8}"
    print(header)
    print("-" * 70)
    for i, col in enumerate(target_cols):
        actual = actual_inv[1:, i]
        pred = pred_inv[1:, i]
        naive = naive_inv[:, i]

        mae_lstm = np.mean(np.abs(actual - pred))
        mae_naive = np.mean(np.abs(actual - naive))
        rmse_lstm = np.sqrt(np.mean((actual - pred) ** 2))
        rmse_naive = np.sqrt(np.mean((actual - naive) ** 2))
        mape_lstm = np.mean(np.abs((actual - pred) / (actual + 1e-8))) * 100
        mape_naive = np.mean(np.abs((actual - naive) / (actual + 1e-8))) * 100

        print(f"{'MAE':<8} | {col.upper():<8} | {mae_lstm:>12.2f} | {mae_naive:>12.2f} | {'YES' if mae_lstm < mae_naive else 'NO':>8}")
        print(f"{'RMSE':<8} | {col.upper():<8} | {rmse_lstm:>12.2f} | {rmse_naive:>12.2f} | {'YES' if rmse_lstm < rmse_naive else 'NO':>8}")
        print(f"{'MAPE':<8} | {col.upper():<8} | {mape_lstm:>11.2f}% | {mape_naive:>11.2f}% | {'YES' if mape_lstm < mape_naive else 'NO':>8}")
        if i < len(target_cols) - 1:
            print("-" * 70)

    # Save backtest results
    os.makedirs("./results", exist_ok=True)
    dates = df["date"].iloc[test_start_idx + SEQ_LENGTH: len(data)].values
    backtest_df = pd.DataFrame({
        "date": dates,
        **{f"actual_{col}": actual_inv[:, i] for i, col in enumerate(target_cols)},
        **{f"pred_{col}": pred_inv[:, i] for i, col in enumerate(target_cols)}
    })
    backtest_df.to_csv("./results/backtest_result.csv", index=False)
    print(f"\nBacktest results saved to ./results/backtest_result.csv")

    # Plot
    print("Generating plots...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, col in enumerate(target_cols):
        ax = axes[i]
        ax.plot(range(len(actual_inv)), actual_inv[:, i], label="Actual", color="blue", linewidth=1.5, alpha=0.8)
        ax.plot(range(len(pred_inv)), pred_inv[:, i], label="LSTM Pred", color="red", linewidth=1.5, alpha=0.7)
        ax.set_title(f"{col.upper()} Prediction vs Actual", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Trading Days")

    plt.tight_layout()
    plt.savefig("./results/backtest_plot.png", dpi=150)
    print(f"Plot saved to ./results/backtest_plot.png")

    # ---- Future Prediction ----
    print(f"\nPredicting next {PRED_DAYS} days...")
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

    future_preds = np.array(future_preds_scaled)
    for i, col in enumerate(target_cols):
        future_preds[:, i] = scalers[col].inverse_transform(future_preds[:, i].reshape(-1, 1)).flatten()

    future_df = pd.DataFrame(future_preds, columns=target_cols, index=[f"Day {i+1}" for i in range(PRED_DAYS)])
    print("\nFuture Predictions:")
    print(future_df.round(2))
    future_df.to_csv("./results/future_prediction.csv")

    print("\n" + "=" * 60)
    print("All done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
