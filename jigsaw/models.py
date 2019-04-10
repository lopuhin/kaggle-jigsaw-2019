import torch
from torch import nn
from torch.nn import functional as F


class SimpleLSTM(nn.Module):
    def __init__(
            self,
            n_vocab: int,
            n_embed: int = 128,
            n_lstm: int = 128,
            n_linear: int = 128,
            ):
        super().__init__()
        self.embedding = nn.Embedding(n_vocab, n_embed)
        self.lstm = nn.LSTM(n_embed, n_lstm,
                            bidirectional=True, batch_first=True)
        self.linear = nn.Linear(2 * n_lstm, n_linear)
        self.linear_out = nn.Linear(n_linear, 1)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = torch.mean(x, dim=1)
        x = F.relu(self.linear(x))
        x = self.linear_out(x)
        return x
