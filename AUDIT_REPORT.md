# 项目审计与修复报告

## 📋 审计概述
- **审计日期**: 2026-05-11
- **项目**: LSTM 股票预测
- **状态**: ✅ 完成修复和重构

---

## 🔍 发现的问题（按严重程度）

### 🔴 严重问题（5个）

#### 1. `pred[0]` 错误计算损失
- **位置**: 原代码 `m3_train.py:21` 及其他文件
- **问题**: `loss = criterion(pred[0], y)` 只取第一个样本
- **影响**: 训练完全失效，模型无法学习
- **修复**: `loss = criterion(pred, y)`

#### 2. Scaler 使用错误数据列
- **位置**: 原代码 `d3_prepareddata.py:32-40`
- **问题**: `scaler_volume` 和 `scaler_pctChg` 使用 `close` 列 fit
- **影响**: 特征归一化完全错误
- **修复**: 每个 scaler 用对应列单独 fit

#### 3. 反归一化索引错误
- **位置**: 原代码多处
- **问题**: `pred_days[i]` 而非 `pred_days[:, i]`
- **影响**: 预测结果维度错位
- **修复**: 修正索引为列优先

#### 4. 数据泄露 - Scaler 在全量数据 fit
- **位置**: 所有版本
- **问题**: 在包括验证集的数据上 fit
- **影响**: 验证 loss 不可信，高估模型效果
- **修复**: 只在训练集上 fit scaler

#### 5. 无效 Dropout（num_layers=1）
- **位置**: 原代码
- **问题**: PyTorch LSTM dropout 只在层间生效
- **影响**: 正则化失效
- **修复**: 调整架构并添加显式 dropout

---

### 🟠 中等问题（2个）

#### 6. 学习率衰减过快
- **位置**: 原代码 `m3_train.py:49`
- **问题**: StepLR 每 10 个 epoch 乘 0.1
- **影响**: 模型很快停止学习
- **修复**: 改用 ReduceLROnPlateau

#### 7. 缺少测试集
- **位置**: 所有版本
- **问题**: 只有 train/val，没有独立 test 集
- **影响**: 无法最终评估泛化能力
- **修复**: 添加 test 集划分

---

### 🟡 轻微问题（7个）

#### 8. 代码冗余重复
- **问题**: 三个 all-in-one 文件重复大量代码
- **修复**: 模块化重构为独立文件

#### 9. 缺少回测系统
- **问题**: 只有预测，没有逐天对比回测
- **修复**: 新增回测功能

#### 10. 硬编码 CUDA 设备
- **问题**: 直接 `.to("cuda")` 无 CPU 回退
- **修复**: 使用 `torch.device` 自动检测

#### 11. 导入错误模块
- **问题**: `import torch.functional` (不存在)
- **修复**: 移除或修正导入

#### 12. RSI 计算不标准
- **问题**: 简单平均而非 Wilder 平滑
- **修复**: 实现标准 RSI 计算

#### 13. 超长 MA 窗口
- **问题**: MA3650 导致大量数据丢失
- **修复**: 合理窗口配置

#### 14. 训练损失未归一化
- **问题**: train_loss 未除以 len(dataloader)
- **修复**: 统一归一化

---

## 🏗️ 架构重构

### 原架构
```
├── d0_download.py
├── d1_showCand.py
├── d2_viewer.py
├── d3_prepareddata.py
├── m1_model.py
├── m2_test.py (空)
├── m3_train.py
├── m5_predict.py
├── aio_with_indicators.py (重复)
├── aio_with_comprehensive_indicators.py (重复)
└── aio_final.py (重复)
```

### 新架构
```
├── config.py                          # 统一配置
├── data_processor.py                  # 数据处理
├── model.py                           # 模型定义
├── trainer.py                         # 训练/回测/预测
├── visualizer.py                      # 可视化
├── main.py                            # 主程序入口
├── requirements.txt                   # 依赖
├── README.md                          # 文档
├── AUDIT_REPORT.md                    # 本报告
├── quick_test.py                      # 轻量测试
├── test_run.py                        # 完整测试
└── datasets/
    ├── sh.000001.csv
    └── sh.600000.csv
```

---

## ✅ 新功能

### 1. 完整回测系统
- 逐天滚动预测
- 真实值对比
- MAE/RMSE/MAPE 指标
- 可视化图表

### 2. 约束损失函数
- `PriceConstraintLoss` 确保价格逻辑
- high >= open/close/low
- low <= open/close

### 3. 模块化配置
- `config.py` 统一管理超参
- 可调整序列长度、模型大小等

---

## 📊 使用建议

1. **安装依赖**: `pip install -r requirements.txt`
2. **下载数据**: `python main.py download`
3. **训练**: `python main.py train`
4. **回测**: `python main.py backtest`
5. **预测**: `python main.py predict`

⚠️ **注意**: 本项目为学习用途，股市有风险，投资需谨慎！
