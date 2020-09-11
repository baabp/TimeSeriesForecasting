import torch
import torch.nn as nn
from torch.autograd import Variable

class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, initial_normal=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=False)
        self.fc = nn.Linear(in_features=hidden_size,
                            out_features=num_classes)
        if initial_normal:
            nn.init.xavier_normal_(self.lstm.weight_ih_l0)
            nn.init.orthogonal_(self.lstm.weight_hh_l0)

    def forward(self, x):
        # h_0 = Variable(torch.zeros(
        #     self.num_layers, x.size(0), self.hidden_size)).to(device)
        #
        # c_0 = Variable(torch.zeros(
        #     self.num_layers, x.size(0), self.hidden_size)).to(device)
        #
        # # Propagate input through LSTM
        # ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        #
        # h_out = h_out.view(-1, self.hidden_size)
        #
        # out = self.fc(h_out).to(device)

        h, _ = self.lstm(x)
        out = self.fc(h[:, -1])

        return out