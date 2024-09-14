import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from tqdm.notebook import tqdm
from m1_model import NeuralNetwork

def calculate_technical_indicators(df):
    # Moving Averages
    df['MA5'] = df['close'].rolling(window=5).mean()
    df['MA10'] = df['close'].rolling(window=10).mean()
    df['MA20'] = df['close'].rolling(window=20).mean()
    
    # Long-term Moving Averages
    df['MA60'] = df['close'].rolling(window=60).mean()
    df['MA120'] = df['close'].rolling(window=120).mean()
    df['MA180'] = df['close'].rolling(window=180).mean()
    df['MA360'] = df['close'].rolling(window=360).mean()

    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

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

    # OBV (On-Balance Volume)
    df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()

    # KDJ
    low_min = df['low'].rolling(window=9).min()
    high_max = df['high'].rolling(window=9).max()
    df['RSV'] = (df['close'] - low_min) / (high_max - low_min) * 100
    df['K'] = df['RSV'].rolling(window=3).mean()
    df['D'] = df['K'].rolling(window=3).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']

    # Drop NaN values
    df.dropna(inplace=True)

    return df

def create_scalers(df, columns, feature_range=(0, 100)):
    scalers = {}
    for column in columns:
        scalers[column] = MinMaxScaler(feature_range=feature_range).fit(df[column].values.reshape(-1,1))
    return scalers

def apply_scalers(df, scalers):
    df_scaled = df.copy(deep=True)
    for column, scaler in scalers.items():
        df_scaled[column] = scaler.transform(df[column].values.reshape(-1,1))
    return df_scaled

def inverse_transform_predictions(pred_days, scalers):
    for i, (column, scaler) in enumerate(scalers.items()):
        pred_days[:, i] = scaler.inverse_transform(pred_days[:, i].reshape(-1, 1)).squeeze()
    return pred_days

def get_datasets(file_path, columns, seq_len=15, batch_size=32, shuffle=False):
    df = pd.read_csv(file_path)
    
    # Calculate technical indicators
    df = calculate_technical_indicators(df)
    
    # Add new indicators to columns list
    new_columns = columns + ['MA5', 'MA10', 'MA20', 'MA60', 'MA120', 'MA180', 'MA360', 
                             'MACD', 'Signal_Line', 'RSI', 'Upper_BB', 'Lower_BB', 
                             'OBV', 'K', 'D', 'J']
    
    scalers = create_scalers(df, new_columns)
    df_scaled = apply_scalers(df, scalers)
    df_scaled.to_csv("./datasets/features_with_indicators.csv")

    data = df_scaled[new_columns].values

    sequences = np.array([data[i:i + seq_len] for i in range(len(data) - seq_len + 1)])

    valid_set_size = int(np.round(0.2 * sequences.shape[0]))
    train_set_size = sequences.shape[0] - valid_set_size

    x_train, y_train = sequences[:train_set_size, :-1, :], sequences[:train_set_size, -1, :]
    x_valid, y_valid = sequences[train_set_size:, :-1, :], sequences[train_set_size:, -1, :]

    x_train, y_train = torch.tensor(x_train).float(), torch.tensor(y_train).float()
    x_valid, y_valid = torch.tensor(x_valid).float(), torch.tensor(y_valid).float()

    train_dataset = TensorDataset(x_train, y_train)
    valid_dataset = TensorDataset(x_valid, y_valid)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_dataloader, valid_dataloader, sequences, scalers, new_columns

def train(dataloader, model, optimizer, criterion):
    model.train()
    epoch_loss = 0

    for x, y in dataloader:
        optimizer.zero_grad()
        x, y = x.to("cuda"), y.to("cuda")
        pred = model(x)
        loss = criterion(pred[0], y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss

def evaluate(dataloader, model, criterion):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to("cuda"), y.to("cuda")
            pred = model(x)
            loss = criterion(pred[0], y)
            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)

def main():
    columns = ['open', 'high', 'low', 'close', 'volume', 'pctChg']
    train_dataloader, valid_dataloader, _, _, new_columns = get_datasets("./datasets/sh.000001.csv", columns)

    model = NeuralNetwork(len(new_columns), 256).to("cuda")
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = nn.MSELoss()

    patience = 20
    n_epochs = 50000
    best_valid_loss = float('inf')
    counter = 0

    for epoch in range(1, n_epochs + 1):
        train_loss = train(train_dataloader, model, optimizer, criterion)
        valid_loss = evaluate(valid_dataloader, model, criterion)

        scheduler.step()

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            counter = 0
            torch.save(model, 'saved_weights.pt')
        else:
            counter += 1

        if counter >= patience:
            print(f"Early stopping at epoch {epoch} due to no improvement")
            break

        print(f'Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {valid_loss:.4f}, Best Val Loss = {best_valid_loss:.4f}')

def predict():
    columns = ['open', 'high', 'low', 'close', 'volume', 'pctChg']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load('saved_weights.pt').to(device)

    _, _, sequences, scalers, new_columns = get_datasets("./datasets/sh.000001.csv", columns)

    last_sequence = sequences[-1:, 1:, :]
    last_sequence = torch.from_numpy(last_sequence).float().to(device)

    PRED_DAYS = 14
    predicted_sequences = []

    model.eval()
    with torch.no_grad():
        for _ in range(PRED_DAYS):
            pred_i = model(last_sequence)
            predicted_sequences.append(pred_i.squeeze(0).cpu().numpy())
            last_sequence = torch.cat((last_sequence[:, 1:, :], pred_i.unsqueeze(1)), dim=1)

    pred_days = np.array(predicted_sequences).reshape(PRED_DAYS, -1)
    pred_days = inverse_transform_predictions(pred_days, scalers)

    df_pred = pd.DataFrame(data=pred_days, columns=new_columns)

    print(df_pred)
    # Plot price predictions
    plt.figure(figsize=(12, 6))
    plt.plot(df_pred['open'], label='Open Prices')
    plt.plot(df_pred['close'], label='Closing Prices')
    plt.plot(df_pred['high'], label='High Prices')
    plt.plot(df_pred['low'], label='Low Prices')
    plt.legend()
    plt.title('Price Predictions')
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
