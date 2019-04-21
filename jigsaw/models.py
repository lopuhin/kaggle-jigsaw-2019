import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class SimpleLSTM(nn.Module):
    def __init__(
            self, *,
            n_vocab: int,
            n_embed: int,
            n_out: int,
            n_lstm: int = 128,
            n_linear: int = 128,
            ):
        super().__init__()
        self.embedding = nn.Embedding(n_vocab, n_embed)
        # self.embedding_dropout = SpatialDropout(0.2)
        self.lstm = nn.LSTM(n_embed, n_lstm,
                            bidirectional=True, batch_first=True, num_layers=1)
        self.linear_1 = nn.Linear(2 * n_lstm, n_linear)
        self.linear_2 = nn.Linear(n_linear, n_linear)
        # self.dropout = nn.Dropout(p=0.5)
        self.linear_out = nn.Linear(n_linear, n_out)

    def forward(self, x, lengths):
        x = self.embedding(x)
        # x = self.embedding_dropout(x)
        x = pack_padded_sequence(x, lengths, batch_first=True)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, batch_first=True)
        x = masked_mean(x, lengths) + torch.max(x, dim=1)[0]
        x = F.relu(self.linear_1(x))
        # x = self.dropout(x)
        x = F.relu(self.linear_2(x))
        # x = self.dropout(x)
        x = self.linear_out(x)
        return x


def masked_mean(x, lengths):
    n_batch, n_seq = x.shape[:2]
    lengths = lengths.unsqueeze(0).t()
    mask = (torch.arange(0, n_seq).unsqueeze(0).expand((n_batch, n_seq))
            < lengths).float() / lengths.max(torch.tensor(1)).float()
    x = x * mask.to(x.device).unsqueeze(2)
    return torch.sum(x, dim=1)


def TinyLSTM(n_vocab: int, n_embed: int, n_out: int):
    return SimpleLSTM(n_vocab=n_vocab, n_embed=n_embed, n_out=n_out,
                      n_lstm=32, n_linear=32)


class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)    # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T)
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x


# TODO try it
class KernelLSTM(nn.Module):
    def __init__(
            self, *,
            n_vocab: int,
            n_embed: int,
            n_out: int,
            n_lstm: int = 128,
            ):
        """ Similar to
        https://www.kaggle.com/bminixhofer/simple-lstm-pytorch-version
        """
        super().__init__()
        self.embedding = nn.Embedding(n_vocab, n_embed)
        # self.embedding_dropout = SpatialDropout(0.2)
        self.lstm = nn.LSTM(
            n_embed, n_lstm, bidirectional=True, batch_first=True, num_layers=2)
        n_linear = 4 * n_lstm
        self.linear_1 = nn.Linear(n_linear, n_linear)
        self.linear_2 = nn.Linear(n_linear, n_linear)
        # self.dropout = nn.Dropout(p=0.5)
        self.linear_out = nn.Linear(n_linear, n_out)

    def forward(self, x, lengths):
        x = self.embedding(x)
        # x = self.embedding_dropout(x)
        x = pack_padded_sequence(x, lengths, batch_first=True)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, batch_first=True)
        x_mean = masked_mean(x)
        x_max, _ = torch.max(x, dim=1)
        x = torch.cat([x_mean, x_max], dim=1)
        x_h1 = F.relu(self.linear_1(x))
        x_h2 = F.relu(self.linear_2(x))
        x = x + x_h1 + x_h2
        x = self.linear_out(x)
        return x
