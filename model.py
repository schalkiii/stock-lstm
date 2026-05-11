"""
模型模块 - LSTM 网络定义
"""
import torch
import torch.nn as nn


class StockLSTM(nn.Module):
    """用于股票预测的 LSTM 模型"""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int, dropout: float = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM 层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LSTM 输出
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步
        last_step = lstm_out[:, -1, :]
        # 全连接
        out = self.fc(last_step)
        return out


class PriceConstraintLoss(nn.Module):
    """带价格逻辑约束的损失函数"""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred/target: (batch_size, 4) - [open, high, low, close]
        """
        mse_loss = self.mse(pred, target)

        # 约束: high >= open, high >= low, high >= close
        high_constraint = torch.mean(
            torch.relu(pred[:, 0] - pred[:, 1]) +  # open > high
            torch.relu(pred[:, 2] - pred[:, 1]) +  # low > high
            torch.relu(pred[:, 3] - pred[:, 1])    # close > high
        )

        # 约束: low <= open, low <= close
        low_constraint = torch.mean(
            torch.relu(pred[:, 2] - pred[:, 0]) +  # low > open
            torch.relu(pred[:, 2] - pred[:, 3])    # low > close
        )

        total_loss = mse_loss + 0.5 * (high_constraint + low_constraint)
        return total_loss
