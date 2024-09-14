

# import requirement libraries and tools
import torch
import torch.optim as optim
import torch.nn as nn
import torch.functional as F
from m1_model import NeuralNetwork
from torchsummary import summary
from d3_prepareddata import get_datasets
import pandas as pd
import matplotlib.pyplot as plt
def predict():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load('saved_weights.pt').to(device)
    
    _, _, sequences, scaler_open, scaler_high, scaler_low, scaler_close,scaler_volume,scaler_pctChg = get_datasets()

    last_sequence = sequences[-1:, 1:, :]
    last_sequence = torch.from_numpy(last_sequence).float().to(device)

    PRED_DAYS = 14
    predicted_sequences = []

    model.eval()
    with torch.no_grad():
        for i in range(PRED_DAYS):
            pred_i = model(last_sequence)
            pred_i = pred_i.unsqueeze(1)  # 调整维度
            predicted_sequences.append(pred_i.squeeze(0).cpu().numpy())
            last_sequence = torch.cat((last_sequence[:, 1:, :], pred_i), dim=1)  # 更新序列

    pred_days = torch.tensor(predicted_sequences).reshape(PRED_DAYS, -1).numpy()
    
    # 分别对每个特征进行反归一化

    # 反归一化
    pred_days[:, 0] = scaler_open.inverse_transform(pred_days[:, 0].reshape(-1, 1)).squeeze()
    pred_days[:, 1] = scaler_high.inverse_transform(pred_days[:, 1].reshape(-1, 1)).squeeze()
    pred_days[:, 2] = scaler_low.inverse_transform(pred_days[:, 2].reshape(-1, 1)).squeeze()
    pred_days[:, 3] = scaler_close.inverse_transform(pred_days[:, 3].reshape(-1, 1)).squeeze()
    pred_days[:, 4] = scaler_volume.inverse_transform(pred_days[:, 4].reshape(-1, 1)).squeeze()
    pred_days[:, 5] = scaler_pctChg.inverse_transform(pred_days[:, 5].reshape(-1, 1)).squeeze()

    df_pred = pd.DataFrame(data=pred_days, columns=['open', 'high', 'low', 'close','volume','pctChg'])

    print(df_pred)
    plt.plot(df_pred['open'], label='Open Prices')
    plt.plot(df_pred['close'], label='Closing Prices')
    plt.plot(df_pred['high'], label='High Prices')
    plt.plot(df_pred['low'], label='Low Prices')
    plt.legend()
    plt.show()

predict()
