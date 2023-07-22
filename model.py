import torch
import torch.nn as nn
import torch.nn.functional as F


class MyModel(nn.Module):
    def __init__(self, input_channels):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=16, stride=1, padding='same')
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=16, stride=1, padding='same')
        # self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=16, stride=1, padding='same')
        self.pool = nn.MaxPool1d(2, 2)
        self.fc1 = nn.Linear(in_features=1280, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        # print(x.shape)
        x = self.pool(F.relu(self.conv2(x)))
        # print(x.shape)
        # x = self.pool(F.relu(self.conv3(x)))
        # print(x.shape)
        x = torch.flatten(x, start_dim=1)
        # print(x.shape)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

# net = MyModel(6)
# tmp = torch.randn(10,6, 41)
# out = net(tmp)
# print('MyModel:', out.shape)



