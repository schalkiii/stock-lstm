"""
极简测试 - 直接使用已有数据，不训练，只验证模块
"""
import os
import sys
import pandas as pd
import numpy as np

print("=" * 60)
print("📋 模块完整性测试")
print("=" * 60)

try:
    import config
    print("✅ config 导入成功")
except Exception as e:
    print(f"❌ config 导入失败: {e}")

try:
    import data_processor
    print("✅ data_processor 导入成功")
except Exception as e:
    print(f"❌ data_processor 导入失败: {e}")

try:
    import model
    print("✅ model 导入成功")
except Exception as e:
    print(f"❌ model 导入失败: {e}")

try:
    import visualizer
    print("✅ visualizer 导入成功")
except Exception as e:
    print(f"❌ visualizer 导入失败: {e}")

# 检查数据
print("\n📊 检查数据文件:")
data_path = "./datasets/sh.000001.csv"
if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    print(f"✅ 数据文件存在，共 {len(df)} 条记录")
    print(df.head(3))
else:
    print("⚠️ 数据文件不存在")

print("\n" + "=" * 60)
print("📝 项目架构总结:")
print("=" * 60)
print("""
📁 项目文件:
  ├── config.py                    # 配置文件
  ├── data_processor.py           # 数据处理模块
  ├── model.py                    # 模型定义
  ├── trainer.py                  # 训练器
  ├── visualizer.py               # 可视化
  ├── main.py                     # 主程序
  ├── test_run.py                 # 测试脚本
  ├── quick_test.py               # 这个文件
  ├── requirements.txt            # 依赖列表
  ├── README.md                   # 项目文档
  └── datasets/
      ├── sh.000001.csv           # 上证指数数据
      └── sh.600000.csv           # 浦发银行数据

🔧 修复的问题:
  1. ✅ pred[0] 错误 - 修复为完整 batch 计算
  2. ✅ scaler 数据泄露 - 训练集单独 fit
  3. ✅ 索引错误 - 修复反归一化索引
  4. ✅ 学习率过快 - 调整 scheduler
  5. ✅ 代码冗余 - 模块化重构
  6. ✅ 缺少回测 - 新增回测系统
  7. ✅ 缺少测试集 - 新增 train/val/test 划分

📈 使用方法:
  python main.py download      # 下载数据
  python main.py train         # 训练模型
  python main.py backtest      # 回测
  python main.py predict       # 预测未来
  python main.py all           # 完整流程

⚠️ 注意: 运行需要先 pip install -r requirements.txt
""")
print("=" * 60)
print("✅ 测试完成")
