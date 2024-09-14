import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from tqdm.notebook import tqdm

class ImprovedNeuralNetwork(nn.Module):
    def __init__(self, num_input_features, hidden_size,num_output_features):
        super(ImprovedNeuralNetwork, self).__init__()
        self.lstm = nn.LSTM(num_input_features, hidden_size, num_layers=2, batch_first=True, dropout=0.1)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=2)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, num_output_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        
    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)
        attn_output, _ = self.attention(output, output, output)
        x = self.dropout(attn_output[:, -1, :])
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class NeuralNetwork(nn.Module):
    def __init__(self, num_feature, hidden_size):
        super(NeuralNetwork, self).__init__()
        self.lstm  = nn.LSTM(num_feature,hidden_size,num_layers=3,batch_first=True,dropout=0.1)
      # self.lstm  = nn.LSTM(num_feature,64,bidirectional=True,batch_first=True)
      # self.lstm  = nn.LSTM(num_feature,64,batch_first=True)
        self.fc    = nn.Linear(hidden_size,num_feature)
        self.dropout = nn.Dropout(p=0.2)  # 20%的dropout概率
        
    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)
      # x = self.dropout(hidden[-1])  # 在全连接层之前加dropout
        x = self.dropout(output[:, -1, :])  # 使用最后一个时间步的输出
        x = self.fc(x)
        return x



def calculate_technical_indicators(df):
    # Moving Averages
    for window in [5, 10, 20, 60, 120, 180, 240, 360, 480, 720, 1080, 3650]:
        df[f'MA{window}'] = df['close'].rolling(window=window).mean()

    # Exponential Moving Averages
    for window in [5, 10, 20, 60, 120]:
        df[f'EMA{window}'] = df['close'].ewm(span=window, adjust=False).mean()

    # 添加波动率指标
    df['volatility'] = df['close'].rolling(window=20).std() / df['close'].rolling(window=20).mean()
    
    # 添加成交量变化率
    df['volume_change'] = df['volume'].pct_change()
    
    # 添加价格动量指标
    df['momentum'] = df['close'] - df['close'].shift(10)
    

    # MACD
    df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df['20MA'] = df['close'].rolling(window=20).mean()
    df['20SD'] = df['close'].rolling(window=20).std()
    df['Upper_BB'] = df['20MA'] + (df['20SD'] * 2)
    df['Lower_BB'] = df['20MA'] - (df['20SD'] * 2)
    df['BB_Width'] = (df['Upper_BB'] - df['Lower_BB']) / df['20MA']

    # OBV (On-Balance Volume)
    df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()

    # KDJ
    low_min = df['low'].rolling(window=9).min()
    high_max = df['high'].rolling(window=9).max()
    df['RSV'] = (df['close'] - low_min) / (high_max - low_min) * 100
    df['K'] = df['RSV'].rolling(window=3).mean()
    df['D'] = df['K'].rolling(window=3).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']

    # AVL (Average Value Line)
    df['AVL'] = (df['high'] + df['low'] + df['close']) / 3

    # ATR (Average True Range)
    df['TR'] = np.maximum(df['high'] - df['low'], 
                          np.maximum(abs(df['high'] - df['close'].shift()), 
                                     abs(df['low'] - df['close'].shift())))
    df['ATR'] = df['TR'].rolling(window=14).mean()

    # CCI (Commodity Channel Index)
    df['TP'] = (df['high'] + df['low'] + df['close']) / 3
    df['SMA_TP'] = df['TP'].rolling(window=20).mean()
    # 使用 apply 方法计算滚动窗口内的均绝对偏差 (MAD)
    df['MAD'] = df['TP'].rolling(window=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=False)
    
    
    df['CCI'] = (df['TP'] - df['SMA_TP']) / (0.015 * df['MAD'])

    # Williams %R
    df['Williams_R'] = (df['high'].rolling(window=14).max() - df['close']) / \
                       (df['high'].rolling(window=14).max() - df['low'].rolling(window=14).min()) * -100

    # Stochastic Oscillator
    df['SO_K'] = (df['close'] - df['low'].rolling(window=14).min()) / \
                 (df['high'].rolling(window=14).max() - df['low'].rolling(window=14).min()) * 100
    df['SO_D'] = df['SO_K'].rolling(window=3).mean()

    # MFI (Money Flow Index)
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    money_flow = typical_price * df['volume']
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=14).sum()
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=14).sum()
    mfi_ratio = positive_flow / negative_flow
    df['MFI'] = 100 - (100 / (1 + mfi_ratio))

    # Chaikin Oscillator
    adl = ((2 * df['close'] - df['high'] - df['low']) / (df['high'] - df['low'])) * df['volume']
    adl = adl.cumsum()
    df['Chaikin_Oscillator'] = adl.ewm(span=3).mean() - adl.ewm(span=10).mean()

    # ROC (Rate of Change)
    df['ROC'] = df['close'].pct_change(periods=10) * 100


    return df

def create_scalers(df, columns, feature_range=(0, 1)):
    scalers = {}
    for column in columns:
        if column == "volume":
            scalers[column] = MinMaxScaler(feature_range=feature_range).fit(df[column].values.reshape(-1,1))
        else:
            scalers[column] = MinMaxScaler(feature_range=feature_range).fit(df[['open', 'high', 'low', 'close']].values.reshape(-1,1))
    return scalers

def apply_scalers(df, scalers):
    df_scaled = df.copy(deep=True)
    for column, scaler in scalers.items():
        df_scaled[column] = scaler.transform(df[column].values.reshape(-1,1))
    return df_scaled

def inverse_transform_predictions(pred_days, scalers, original_features):
    # 只对 original_features 进行逆缩放
    for i in range(0,len(original_features)):
        scaler = scalers[original_features[i]]
        pred_days[i] = scaler.inverse_transform(pred_days[i].reshape(-1, 1)).squeeze()
    return pred_days

def get_datasets(file_path, original_features, all_features, seq_len=21, batch_size=32, shuffle=False):
    df = pd.read_csv(file_path)
    
    # Calculate technical indicators
    df = calculate_technical_indicators(df)

    # Drop NaN values
    df.dropna(inplace=True)
    df= df[all_features]
    df.to_csv("./datasets/feature_origin.csv")

    scalers = create_scalers(df, all_features)

    df_scaled = apply_scalers(df, scalers) 
    df_scaled = df_scaled[all_features]
    df_scaled.to_csv("./datasets/feature_scaled.csv")

    data = df_scaled[all_features].values

    sequences = np.array([data[i:i + seq_len] for i in range(len(data) - seq_len + 1)])

    valid_set_size = int(np.round(0.2 * sequences.shape[0]))
    train_set_size = sequences.shape[0] - valid_set_size

    # 所有特征作为输入，但只预测原始特征
    x_train = sequences[:train_set_size, :-1, :]
    y_train = sequences[:train_set_size, -1, :len(original_features)]
    
    x_valid = sequences[train_set_size:, :-1, :]
    y_valid = sequences[train_set_size:, -1, :len(original_features)]

    x_train, y_train = torch.tensor(x_train).float(), torch.tensor(y_train).float()
    x_valid, y_valid = torch.tensor(x_valid).float(), torch.tensor(y_valid).float()

    train_dataset = TensorDataset(x_train, y_train)
    valid_dataset = TensorDataset(x_valid, y_valid)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_dataloader, valid_dataloader, sequences, scalers, all_features

def train(dataloader, model, optimizer, criterion, scheduler):
    model.train()
    epoch_loss = 0
    
    for x, y in dataloader:
        optimizer.zero_grad()
        x, y = x.to("cuda"), y.to("cuda")
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        epoch_loss += loss.item()
    
    scheduler.step()
    return epoch_loss / len(dataloader)


def evaluate(dataloader, model, criterion):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to("cuda"), y.to("cuda")
            pred = model(x)
            loss = criterion(pred, y)
            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)

class PricePredictionLoss(nn.Module):
    def __init__(self):
        super(PricePredictionLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        
    def forward(self, pred, target):
        mse_loss = self.mse_loss(pred, target)
        
        # 增加额外的约束条件，确保价格合理
        high_low_constraint = torch.mean(torch.relu(pred[:, 2] - pred[:, 1]))  # high >= low
        open_high_constraint = torch.mean(torch.relu(pred[:, 0] - pred[:, 1]))  # high >= open
        open_low_constraint = torch.mean(torch.relu(pred[:, 2] - pred[:, 0]))  # open >= low

        close_high_constraint = torch.mean(torch.relu(pred[:, 3] - pred[:, 1]))  # high >= close
        close_low_constraint = torch.mean(torch.relu(pred[:, 2] - pred[:, 3]))  # close >= low


        # 总损失：MSE损失 + 约束损失
        return mse_loss + 0.5 * (high_low_constraint + open_high_constraint + open_low_constraint + close_low_constraint + close_high_constraint)

def main():
    original_features = ['open', 'high', 'low', 'close', 'volume']

    all_features = original_features + [
        'MA5', 'MA10', 'MA20', 'MA60', 'MA120', 'MA180', 'MA360','MA480','MA720','MA1080','MA3650',
        'volatility', 'volume_change','momentum',
        'EMA5', 'EMA10', 'EMA20', 'EMA60', 'EMA120', 
        'EMA12', 'EMA26', 'MACD', 'Signal_Line', 'MACD_Histogram', 
        'RSI', '20MA', '20SD', 'Upper_BB', 'Lower_BB', 'BB_Width', 
        'OBV', 'RSV', 'K', 'D', 'J', 'AVL', 'TR', 'ATR', 'TP', 'SMA_TP', 'MAD', 'CCI', 
        'Williams_R', 'SO_K', 'SO_D', 'MFI', 'Chaikin_Oscillator', 'ROC'
    ]

    train_dataloader, valid_dataloader, _, _, all_features = get_datasets("./datasets/sh.000001.csv", original_features,all_features)

    model = ImprovedNeuralNetwork(len(all_features), 512,len(original_features)).to("cuda")

    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    criterion = PricePredictionLoss()

    patience = 100
    n_epochs = 10000
    best_valid_loss = float('inf')
    counter = 0

    for epoch in range(1, n_epochs + 1):
        train_loss = train(train_dataloader, model, optimizer, criterion, scheduler)
        valid_loss = evaluate(valid_dataloader, model, criterion)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            counter += 1

        if counter >= patience:
            print(f"Early stopping at epoch {epoch} due to no improvement")
            break

        print(f'Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {valid_loss:.4f}, Best Val Loss = {best_valid_loss:.4f}')

def predict():
    original_features = ['open', 'high', 'low', 'close', 'volume']
    all_features = original_features + [
        'MA5', 'MA10', 'MA20', 'MA60', 'MA120', 'MA180', 'MA360','MA480','MA720','MA1080','MA3650',
        'volatility', 'volume_change','momentum',
        'EMA5', 'EMA10', 'EMA20', 'EMA60', 'EMA120', 
        'EMA12', 'EMA26', 'MACD', 'Signal_Line', 'MACD_Histogram', 
        'RSI', '20MA', '20SD', 'Upper_BB', 'Lower_BB', 'BB_Width', 
        'OBV', 'RSV', 'K', 'D', 'J', 'AVL', 'TR', 'ATR', 'TP', 'SMA_TP', 'MAD', 'CCI', 
        'Williams_R', 'SO_K', 'SO_D', 'MFI', 'Chaikin_Oscillator', 'ROC'
    ]

    _, _, sequences, scalers, all_features = get_datasets("./datasets/sh.000001.csv", original_features, all_features)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 读取历史数据
    df_origin = pd.read_csv('./datasets/sh.000001.csv')
    df_origin = df_origin[original_features]

    model = ImprovedNeuralNetwork(len(all_features), 512,len(original_features)).to(device)
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()

    # 提取历史序列的最后一段作为预测输入
    last_sequence = sequences[-1:, 1:, :]
    last_sequence = torch.from_numpy(last_sequence).float().to(device)

    PRED_DAYS = 7
    predicted_sequences = []

    with torch.no_grad():
        for _ in range(PRED_DAYS):
            print(last_sequence)
            # 1. 预测origin feature
            pred_i = model(last_sequence)
            pred_i_np = pred_i.squeeze(0).cpu().numpy()  # 预测结果是origin feature的值

            # 2. 对预测结果进行inverse scaling，恢复到原始尺度
            pred_i_np = inverse_transform_predictions(pred_i_np, scalers, original_features)  # 逆缩放

            # 3. 将逆缩放后的预测结果和历史数据组合，形成新的DataFrame
           #print("df_origin")
           #print(df_origin)
            last_row = df_origin.iloc[-1].copy()  # 拿最后一行作为基础
           #df_origin = df_origin[original_features]
           #print("df_origin")
           #print(df_origin)
            for i in range(0,len(original_features)):
                last_row.iloc[i] = pred_i_np[i]  # 更新新的预测数据

            # 4. 将新预测行插入到历史数据中并重新计算技术指标
            df_origin = pd.concat([df_origin, pd.DataFrame([last_row])], ignore_index=True)
           #print("df_origin concat")
           #print(df_origin)
            df_origin_w_indi = calculate_technical_indicators(df_origin)  # 重新计算技术指标
            df_origin_w_indi = df_origin_w_indi[all_features]
           #print("df_origin_with_instruction")
           #print(df_origin_w_indi)

            df_scaled = apply_scalers(df_origin_w_indi, scalers)
           #print(df_scaled)
            # 5. 准备下一个预测的输入，将新的特征行拼接到输入序列中
            # 1. 将新计算的技术指标从 DataFrame 转换为 NumPy 数组，再转换为 Tensor
            new_features = df_scaled.iloc[-1].values  # 提取最后一行
            new_features = torch.tensor(new_features, dtype=torch.float32).to(device)  # 转换为Tensor并放入CUDA
            
            # 2. Reshape the tensor to match the input dimensions of last_sequence (1, 1, features)
            new_features = new_features.unsqueeze(0).unsqueeze(0)  # 变成 (1, 1, num_features)
            
            # 3. 将新的特征拼接到 last_sequence 的末尾 (time steps 维度拼接)
            last_sequence = torch.cat((last_sequence[:, 1:, :], new_features), dim=1)  # (batch_size, time_steps, features)
            
            # 记录预测结果
            predicted_sequences.append(pred_i_np)

    # 构建预测结果的DataFrame
    pred_days = np.array(predicted_sequences).reshape(PRED_DAYS, -1)
    df_pred = pd.DataFrame(data=pred_days, columns=original_features)
  
    print(df_pred)

    # 可视化预测结果
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1)
    plt.plot(df_pred['open'], label='Open')
    plt.plot(df_pred['close'], label='Close')
    plt.plot(df_pred['high'], label='High')
    plt.plot(df_pred['low'], label='Low')
    plt.legend()
    plt.title('Price Predictions')

    plt.subplot(2, 1, 2)
    plt.plot(df_pred['volume'], label='Volume')
    plt.legend()
    plt.title('Volume and Percent Change Predictions')

    plt.tight_layout()
    plt.show()

   #plt.figure(figsize=(12, 6))
   #plt.plot(df_pred['MA5'], label='MA5')
   #plt.plot(df_pred['MA10'], label='MA10')
   #plt.plot(df_pred['MA20'], label='MA20')
   #plt.legend()
   #plt.title('Moving Average Predictions')
   #plt.show()

   #plt.figure(figsize=(12, 6))
   #plt.plot(df_pred['MACD'], label='MACD')
   #plt.plot(df_pred['Signal_Line'], label='Signal Line')
   #plt.legend()
   #plt.title('MACD Predictions')
   #plt.show()

if __name__ == "__main__":
    main()
    predict()
