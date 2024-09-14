from torch import nn
from torchsummary import summary


class NeuralNetwork(nn.Module):
    def __init__(self, num_feature, hidden_size):
        super(NeuralNetwork, self).__init__()
        self.lstm  = nn.LSTM(num_feature,hidden_size,num_layers=3,batch_first=True,dropout=0.2)
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



    
    
