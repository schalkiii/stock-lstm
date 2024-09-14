# import requirement libraries and tools
import torch
import torch.optim as optim
import torch.nn as nn
from m1_model import NeuralNetwork
from d3_prepareddata import get_datasets


def train(dataloader, model, optimizer, mse):
    epoch_loss = 0
    model.train()

    for batch in dataloader:
        optimizer.zero_grad()
        x, y = batch
        x = x.to("cuda")
        y = y.to("cuda")

        pred = model(x)

        loss = mse(pred[0], y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss


def evaluate(dataloader, model, mse):
    epoch_loss = 0
    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            x = x.to("cuda")
            y = y.to("cuda")
            pred = model(x)
            loss = mse(pred[0], y)
            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def main():
    m = NeuralNetwork(6,256).to("cuda")

    optimizer = optim.Adam(m.parameters(), lr=0.0005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # 添加学习率调度器
    mse = nn.MSELoss()

    patience = 20
    n_epochs = 50000
    best_valid_loss = float('inf')
    train_dataloader, valid_dataloader, _, _, _, _, _, _, _ = get_datasets()
    counter = 0

    for epoch in range(1, n_epochs + 1):
        train_loss = train(train_dataloader, m, mse=mse, optimizer=optimizer)
        valid_loss = evaluate(valid_dataloader, m, mse=mse)

        # 调用学习率调度器
        scheduler.step()

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            counter = 0
            torch.save(m, 'saved_weights.pt')  # 保存模型权重
        else:
            counter += 1

        if counter >= patience:  # 如果连续没有进步的epoch数达到耐心值
            print(f"Early stopping at epoch {epoch} due to no improvement")
            break

        print(f'\tEpoch : {epoch} | ' + f'\tTrain Loss: {train_loss:.2f} | ' + f'\tVal Loss: {valid_loss:.4f}' + f'\tBest Loss: {best_valid_loss:.4f}')


main()
