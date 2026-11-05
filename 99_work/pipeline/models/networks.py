"""Network architectures.

EALSTM        regional LSTM = the Kratzert et al. (2019) EA-LSTM used before (same parameters, same
              equations as papercode/ealstm.py), evaluated through cuDNN: the static input gate is
              expressed as an LSTM input gate whose weights see only the (time-tiled) statics and
              whose recurrent weights are zero. checks/check_ealstm.py asserts numerical equality.
LSTMIndiv     individual LSTM (run_LSTM.py Model): nn.LSTM on the 3 forcings + dropout + linear head.
TCN           temporal blocks as in main_tcn.py; with n_static > 0 (regional / adapted) the statics
              go through an MLP to an embedding that FiLM-modulates every residual block and is
              concatenated to the input of the output head (plan 3.4). n_static = 0 -> the plain
              TCNModel used for individual basins.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EALSTM(nn.Module):
    def __init__(self, input_size_dyn, input_size_stat, hidden_size, initial_forget_bias=5, dropout=0.0):
        super().__init__()
        self.input_size_dyn, self.input_size_stat, self.hidden_size = input_size_dyn, input_size_stat, hidden_size
        H = hidden_size
        # identical parameterisation to papercode/ealstm.py (gate order f, o, g)
        self.weight_ih = nn.Parameter(torch.empty(input_size_dyn, 3 * H))
        self.weight_hh = nn.Parameter(torch.empty(H, 3 * H))
        self.weight_sh = nn.Parameter(torch.empty(input_size_stat, H))
        self.bias = nn.Parameter(torch.empty(3 * H))
        self.bias_s = nn.Parameter(torch.empty(H))
        nn.init.orthogonal_(self.weight_ih)
        nn.init.orthogonal_(self.weight_sh)
        self.weight_hh.data = torch.eye(H).repeat(1, 3)
        nn.init.zeros_(self.bias)
        nn.init.zeros_(self.bias_s)
        if initial_forget_bias:
            self.bias.data[:H] = initial_forget_bias
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(H, 1)

    def _cudnn_weights(self):
        H, D, S = self.hidden_size, self.input_size_dyn, self.input_size_stat
        wf, wo, wg = self.weight_ih.t().split(H, 0)          # (H, D) each
        hf, ho, hg = self.weight_hh.t().split(H, 0)          # (H, H) each
        bf, bo, bg = self.bias.split(H)
        zD = wf.new_zeros(H, D)
        zS = wf.new_zeros(H, S)
        # torch gate order: i, f, g, o ; input = [x_dyn, x_static]
        w_ih = torch.cat([torch.cat([zD, self.weight_sh.t()], 1),
                          torch.cat([wf, zS], 1), torch.cat([wg, zS], 1), torch.cat([wo, zS], 1)], 0)
        w_hh = torch.cat([wf.new_zeros(H, H), hf, hg, ho], 0)
        b_ih = torch.cat([self.bias_s, bf, bg, bo])
        b_hh = torch.zeros_like(b_ih)
        return [w_ih.contiguous(), w_hh.contiguous(), b_ih, b_hh]

    def forward(self, x_d, x_s):
        B, T, _ = x_d.shape
        x = torch.cat([x_d, x_s.unsqueeze(1).expand(B, T, x_s.shape[1])], 2)
        h0 = x.new_zeros(1, B, self.hidden_size)
        out, _, _ = torch._VF.lstm(x, (h0, h0), self._cudnn_weights(), True, 1, 0.0, self.training, False, True)
        return self.fc(self.dropout(out[:, -1, :]))


class LSTMIndiv(nn.Module):
    def __init__(self, input_size_dyn, hidden_size, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size_dyn, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x_d, x_s=None):
        _, (h_n, _) = self.lstm(x_d)
        return self.fc(self.dropout(h_n[-1]))


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, padding=padding, dilation=dilation)
        self.net = nn.Sequential(self.conv1, Chomp1d(padding), nn.ReLU(), nn.Dropout(dropout),
                                 self.conv2, Chomp1d(padding), nn.ReLU(), nn.Dropout(dropout))
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        res = x if self.downsample is None else self.downsample(x)
        return F.relu(self.net(x) + res)


class TCN(nn.Module):
    def __init__(self, input_size_dyn, num_filters, num_levels, kernel_size, dropout,
                 n_static=0, static_embedding=32):
        super().__init__()
        chans = [num_filters] * num_levels
        self.blocks = nn.ModuleList(
            TemporalBlock(input_size_dyn if i == 0 else chans[i - 1], chans[i], kernel_size, 2 ** i, dropout)
            for i in range(num_levels))
        self.n_static = n_static
        head_in = chans[-1]
        if n_static:
            self.static_mlp = nn.Sequential(nn.Linear(n_static, 64), nn.ReLU(), nn.Linear(64, static_embedding), nn.ReLU())
            self.film = nn.ModuleList(nn.Linear(static_embedding, 2 * c) for c in chans)
            for f in self.film:            # start as the identity modulation
                nn.init.zeros_(f.weight)
                nn.init.zeros_(f.bias)
            head_in += static_embedding
        self.head = nn.Linear(head_in, 1)
        self.receptive_field = 1 + 2 * (kernel_size - 1) * (2 ** num_levels - 1)

    def forward(self, x_d, x_s=None):
        y = x_d.transpose(1, 2)
        emb = self.static_mlp(x_s) if self.n_static else None
        for i, blk in enumerate(self.blocks):
            y = blk(y)
            if emb is not None:
                gamma, beta = self.film[i](emb).unsqueeze(2).chunk(2, 1)
                y = y * (1 + gamma) + beta
        last = y[:, :, -1]
        if emb is not None:
            last = torch.cat([last, emb], 1)
        return self.head(last)


class NSELoss(nn.Module):
    """NSE* loss (Kratzert et al. 2019): mean over samples of (q_sim - q_obs)^2 / (q_std_basin + eps)^2,
    in physical units, q_std from the basin's TRAIN period -> every basin weighs equally."""
    name = "NSE*"

    def __init__(self, eps=0.1):
        super().__init__()
        self.eps = eps

    def forward(self, pred_z, obs_z, q_std):
        err = (pred_z - obs_z) * q_std            # back to physical units (mm/d)
        return torch.mean(err ** 2 / (q_std + self.eps) ** 2)
