# LSTM 股票预测模型

基于 LSTM 神经网络的股票价格预测项目，支持技术指标、回测和未来预测。

## 功能特点

- 数据下载：通过 baostock 获取股票历史数据
- 技术指标：计算 MA、EMA、MACD、RSI、BOLL、KDJ、ATR 等多种指标
- 数据预处理：规范化处理，避免数据泄露
- 模型训练：LSTM 网络，带价格逻辑约束的损失函数
- 回测系统：逐天预测并与真实值对比
- 未来预测：递归预测未来 N 天
- 可视化：回测图表、预测图表和指标展示

## 文件说明

| 文件 | 说明 |
|------|------|
| `config.py` | 配置文件，包含所有超参数和路径 |
| `data_processor.py` | 数据处理模块，下载、计算指标、归一化 |
| `model.py` | LSTM 模型定义和约束损失函数 |
| `trainer.py` | 训练器，负责训练、评估、回测、预测 |
| `visualizer.py` | 可视化模块，绘制图表 |
| `main.py` | 主程序入口 |
| `requirements.txt` | 依赖库列表 |

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

```bash
# 下载数据
python main.py download

# 训练模型
python main.py train

# 回测
python main.py backtest

# 预测未来
python main.py predict

# 完整流程（训练 -> 回测 -> 预测）
python main.py all
```

## 模型配置

可在 `config.py` 中调整以下参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| SEQ_LENGTH | 输入序列长度 | 15 |
| HIDDEN_SIZE | LSTM 隐藏层大小 | 128 |
| NUM_LAYERS | LSTM 层数 | 2 |
| LEARNING_RATE | 学习率 | 0.001 |
| PATIENCE | 早停耐心值 | 30 |
| MAX_EPOCHS | 最大训练轮数 | 1000 |
| PRED_DAYS | 预测未来天数 | 14 |

## 回测指标

- MAE (Mean Absolute Error): 平均绝对误差
- RMSE (Root Mean Square Error): 均方根误差
- MAPE (Mean Absolute Percentage Error): 平均绝对百分比误差

## 注意事项

⚠️ 本项目仅供学习和研究使用，不构成任何投资建议。股票市场存在风险，投资需谨慎！
