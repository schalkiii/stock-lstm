代码实现了lstm模型训练股价。
输入为10天的开盘/收盘/最高/最低， 每次预测y为一天的四个价格。

可以循环预测多天股价。



#下载股票数据，保存为csv文件
python d0_download.py

#归一化，并且生产loader数据集
python d3_prepareddata.py

#训练
python m3_train.py

#预测10天的股价
python m5_predict.py
