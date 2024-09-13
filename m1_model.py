from torch import nn
from torchsummary import summary


class NeuralNetwork(nn.Module):
    def __init__(self, num_feature):
        super(NeuralNetwork, self).__init__()
        self.lstm  = nn.LSTM(num_feature,64,num_layers=2,batch_first=True,dropout=0.3)
      # self.lstm  = nn.LSTM(num_feature,64,bidirectional=True,batch_first=True)
      # self.lstm  = nn.LSTM(num_feature,64,batch_first=True)
        self.fc    = nn.Linear(64,num_feature)
        self.dropout = nn.Dropout(p=0.5)  # 50%的dropout概率
        
    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)
        x = self.dropout(hidden[-1])  # 在全连接层之前加dropout
        x = self.fc(x)
      # x = self.fc(output[:, -1, :])  # 使用最后一个时间步的输出
        return x



    
    
