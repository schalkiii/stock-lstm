"""
数据处理模块 - 包含数据下载、技术指标计算、归一化
"""
import numpy as np
import pandas as pd
import baostock as bs
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, Tuple, List
import os
import config


def download_data(code: str = config.DEFAULT_STOCK_CODE,
                  start_date: str = config.DEFAULT_START_DATE,
                  end_date: str = config.DEFAULT_END_DATE,
                  save_dir: str = config.DATA_DIR) -> pd.DataFrame:
    """从 baostock 下载股票历史数据"""
    os.makedirs(save_dir, exist_ok=True)

    lg = bs.login()
    rs = bs.query_history_k_data_plus(
        code,
        "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST",
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="2"
    )

    data_list = []
    while (rs.error_code == "0") & rs.next():
        data_list.append(rs.get_row_data())
    df = pd.DataFrame(data_list, columns=rs.fields)
    bs.logout()

    # 转换数值型列
    numeric_cols = ["open", "high", "low", "close", "preclose", "volume", "amount", "turn", "pctChg"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    save_path = os.path.join(save_dir, f"{code}.csv")
    df.to_csv(save_path, index=False)
    print(f"数据已下载并保存至: {save_path}")
    return df


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """计算技术指标"""
    df = df.copy()

    # 移动平均线 MA
    for window in config.MA_WINDOWS:
        df[f"MA{window}"] = df["close"].rolling(window=window).mean()

    # 指数移动平均线 EMA
    for window in config.EMA_WINDOWS:
        df[f"EMA{window}"] = df["close"].ewm(span=window, adjust=False).mean()

    # MACD
    df["EMA12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["EMA26"] = df["close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Histogram"] = df["MACD"] - df["Signal_Line"]

    # RSI (Wilder's 平滑方法)
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0))
    loss = (-delta.where(delta < 0, 0))
    avg_gain = gain.rolling(window=config.RSI_WINDOW).mean()
    avg_loss = loss.rolling(window=config.RSI_WINDOW).mean()
    for i in range(config.RSI_WINDOW, len(df)):
        avg_gain.iloc[i] = (avg_gain.iloc[i - 1] * (config.RSI_WINDOW - 1) + gain.iloc[i]) / config.RSI_WINDOW
        avg_loss.iloc[i] = (avg_loss.iloc[i - 1] * (config.RSI_WINDOW - 1) + loss.iloc[i]) / config.RSI_WINDOW
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df["BB_Middle"] = df["close"].rolling(window=config.BB_WINDOW).mean()
    df["BB_Std"] = df["close"].rolling(window=config.BB_WINDOW).std()
    df["BB_Upper"] = df["BB_Middle"] + (df["BB_Std"] * 2)
    df["BB_Lower"] = df["BB_Middle"] - (df["BB_Std"] * 2)
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Middle"]

    # OBV
    df["OBV"] = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()

    # KDJ
    low_min = df["low"].rolling(window=config.KDJ_WINDOW).min()
    high_max = df["high"].rolling(window=config.KDJ_WINDOW).max()
    df["RSV"] = (df["close"] - low_min) / (high_max - low_min) * 100
    df["K"] = df["RSV"].ewm(com=2, adjust=False).mean()
    df["D"] = df["K"].ewm(com=2, adjust=False).mean()
    df["J"] = 3 * df["K"] - 2 * df["D"]

    # ATR
    df["TR"] = np.maximum(
        df["high"] - df["low"],
        np.maximum(abs(df["high"] - df["close"].shift()), abs(df["low"] - df["close"].shift()))
    )
    df["ATR"] = df["TR"].rolling(window=14).mean()

    # 波动率
    df["Volatility"] = df["close"].rolling(window=20).std() / df["close"].rolling(window=20).mean()

    # 价格变化率
    df["Price_Return"] = df["close"].pct_change()

    df = df.dropna()
    return df


def prepare_data(df: pd.DataFrame,
                 feature_cols: List[str],
                 target_cols: List[str]) -> Tuple[Dict[str, MinMaxScaler], pd.DataFrame, np.ndarray, np.ndarray]:
    """
    准备训练数据
    返回: (scalers, df_scaled, X, y)
    """
    df = df[feature_cols].copy()

    # 为每列创建单独的 scaler，只在训练数据上 fit
    scalers = {}
    df_scaled = df.copy()
    for col in feature_cols:
        scalers[col] = MinMaxScaler(feature_range=(0, 1))
        df_scaled[col] = scalers[col].fit_transform(df[col].values.reshape(-1, 1)).flatten()

    # 生成序列
    data = df_scaled.values
    X, y = [], []
    for i in range(len(data) - config.SEQ_LENGTH):
        X.append(data[i:i + config.SEQ_LENGTH])
        y.append(data[i + config.SEQ_LENGTH, [feature_cols.index(t) for t in target_cols]])

    return scalers, df_scaled, np.array(X), np.array(y)


def inverse_transform(scalers: Dict[str, MinMaxScaler],
                      preds: np.ndarray,
                      target_cols: List[str]) -> np.ndarray:
    """反归一化预测结果"""
    result = preds.copy()
    for i, col in enumerate(target_cols):
        result[:, i] = scalers[col].inverse_transform(result[:, i].reshape(-1, 1)).flatten()
    return result
