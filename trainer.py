"""
训练器模块 - 训练、评估、预测
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, Tuple, List
import config
import model
from data_processor import calculate_technical_indicators, prepare_data, inverse_transform


class StockTrainer:
    def __init__(self, feature_cols: List[str], target_cols: List[str]):
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化模型
        self.net = model.StockLSTM(
            input_size=len(feature_cols),
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            output_size=len(target_cols),
            dropout=config.DROPOUT
        ).to(self.device)

        # 损失函数和优化器
        self.criterion = model.PriceConstraintLoss()
        self.optimizer = optim.AdamW(self.net.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, "min", patience=10, factor=0.5)

        self.scalers = None
        self.best_val_loss = float("inf")

    def load_data(self, df: pd.DataFrame) -> Tuple[DataLoader, DataLoader, DataLoader, pd.DataFrame]:
        """加载并预处理数据"""
        df = calculate_technical_indicators(df)

        # 只在训练集上 fit scalers (避免数据泄露)
        total_len = len(df)
        train_data_end = int(total_len * config.TRAIN_RATIO) + config.SEQ_LENGTH

        df_train = df.iloc[:train_data_end]
        self.scalers = {}
        for col in self.feature_cols:
            self.scalers[col] = MinMaxScaler(feature_range=(0, 1))
            self.scalers[col].fit(df_train[col].values.reshape(-1, 1))

        # 归一化所有数据
        df_full_scaled = df[self.feature_cols].copy()
        for col in self.feature_cols:
            df_full_scaled[col] = self.scalers[col].transform(df[col].values.reshape(-1, 1)).flatten()

        # 生成序列
        data = df_full_scaled.values
        X, y = [], []
        for i in range(len(data) - config.SEQ_LENGTH):
            X.append(data[i:i + config.SEQ_LENGTH])
            y.append(data[i + config.SEQ_LENGTH, [self.feature_cols.index(t) for t in self.target_cols]])
        X, y = np.array(X), np.array(y)

        # 划分数据集
        n = len(X)
        train_end = int(n * config.TRAIN_RATIO)
        val_end = train_end + int(n * config.VAL_RATIO)

        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]

        # 转换为 tensor
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

        # 创建 DataLoader
        train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor),
                                  batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor),
                                batch_size=config.BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor),
                                 batch_size=config.BATCH_SIZE, shuffle=False)

        return train_loader, val_loader, test_loader, df

    def train_epoch(self, loader: DataLoader) -> float:
        """训练一个 epoch"""
        self.net.train()
        total_loss = 0.0

        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

            self.optimizer.zero_grad()
            pred = self.net(X_batch)
            loss = self.criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), config.GRAD_CLIP)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    def evaluate(self, loader: DataLoader) -> float:
        """评估模型"""
        self.net.eval()
        total_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                pred = self.net(X_batch)
                loss = self.criterion(pred, y_batch)
                total_loss += loss.item()

        return total_loss / len(loader)

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """完整训练过程"""
        os.makedirs("./models", exist_ok=True)
        counter = 0
        best_path = "./models/best_model.pth"

        for epoch in range(config.MAX_EPOCHS):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.evaluate(val_loader)

            self.scheduler.step(val_loss)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                counter = 0
                torch.save({
                    "model_state_dict": self.net.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scalers": self.scalers,
                    "feature_cols": self.feature_cols,
                    "target_cols": self.target_cols
                }, best_path)
                print(f"✅ Epoch {epoch}: 新的最佳模型! Val Loss = {val_loss:.6f}")
            else:
                counter += 1
                if counter >= config.PATIENCE:
                    print(f"⏹️  Epoch {epoch}: 早停! Val Loss 已 {config.PATIENCE} 轮无改善")
                    break

            if epoch % 10 == 0:
                print(f"📊 Epoch {epoch:4d} | Train Loss = {train_loss:.6f} | Val Loss = {val_loss:.6f} | Best = {self.best_val_loss:.6f}")

        print(f"\n🏁 训练完成! 最佳 Val Loss = {self.best_val_loss:.6f}")
        self.load_model(best_path)

    def load_model(self, path: str):
        """加载保存的模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.net.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scalers = checkpoint["scalers"]
        self.feature_cols = checkpoint["feature_cols"]
        self.target_cols = checkpoint["target_cols"]
        self.net.to(self.device)

    def backtest(self, df: pd.DataFrame, start_idx: int = None) -> pd.DataFrame:
        """
        回测：从 start_idx 开始逐天预测，与真实值对比
        """
        if self.scalers is None:
            raise ValueError("请先加载模型!")

        df = calculate_technical_indicators(df)
        df_full_scaled = df[self.feature_cols].copy()
        for col in self.feature_cols:
            df_full_scaled[col] = self.scalers[col].transform(df[col].values.reshape(-1, 1)).flatten()
        data = df_full_scaled.values

        if start_idx is None:
            start_idx = int(len(data) * (config.TRAIN_RATIO + config.VAL_RATIO))

        self.net.eval()
        predictions = []
        actuals = []

        with torch.no_grad():
            for i in range(start_idx, len(data) - config.SEQ_LENGTH):
                # 取过去 SEQ_LENGTH 天作为输入
                X_seq = data[i:i + config.SEQ_LENGTH]
                X_tensor = torch.tensor(X_seq, dtype=torch.float32).unsqueeze(0).to(self.device)
                pred_scaled = self.net(X_tensor).cpu().numpy()[0]
                predictions.append(pred_scaled)
                actuals.append(data[i + config.SEQ_LENGTH, [self.feature_cols.index(t) for t in self.target_cols]])

        # 反归一化
        pred_arr = np.array(predictions)
        actual_arr = np.array(actuals)
        pred_inv = inverse_transform(self.scalers, pred_arr, self.target_cols)
        actual_inv = inverse_transform(self.scalers, actual_arr, self.target_cols)

        # 构建结果 DataFrame
        dates = df["date"].iloc[start_idx + config.SEQ_LENGTH: len(data)].values
        result = pd.DataFrame({
            "date": dates,
            **{f"pred_{col}": pred_inv[:, i] for i, col in enumerate(self.target_cols)},
            **{f"actual_{col}": actual_inv[:, i] for i, col in enumerate(self.target_cols)}
        })

        return result

    def predict_future(self, df: pd.DataFrame, days: int = config.PRED_DAYS) -> pd.DataFrame:
        """递归预测未来 N 天"""
        df = calculate_technical_indicators(df)
        df_scaled = df[self.feature_cols].copy()
        for col in self.feature_cols:
            df_scaled[col] = self.scalers[col].transform(df[col].values.reshape(-1, 1)).flatten()
        data = df_scaled.values

        current_seq = data[-config.SEQ_LENGTH:].copy()
        self.net.eval()
        predictions_scaled = []

        with torch.no_grad():
            for _ in range(days):
                X_tensor = torch.tensor(current_seq, dtype=torch.float32).unsqueeze(0).to(self.device)
                pred_scaled = self.net(X_tensor).cpu().numpy()[0]
                predictions_scaled.append(pred_scaled)

                # 更新序列：移除最旧的一天，添加新预测
                new_step = current_seq[-1].copy()
                for i, col in enumerate(self.target_cols):
                    col_idx = self.feature_cols.index(col)
                    new_step[col_idx] = pred_scaled[i]
                current_seq = np.vstack([current_seq[1:], new_step])

        pred_arr = np.array(predictions_scaled)
        pred_inv = inverse_transform(self.scalers, pred_arr, self.target_cols)

        result = pd.DataFrame(pred_inv, columns=self.target_cols)
        return result
