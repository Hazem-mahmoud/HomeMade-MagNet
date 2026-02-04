
import torch
import torch.nn as nn
import numpy as np

try:
    from torch.nn.utils.parametrizations import weight_norm
except ImportError:
    from torch.nn.utils import weight_norm

# ==========================================
# Paderborn's TCN Architecture Components
# ==========================================

class Biased_Elu(nn.Module):
    def __init__(self):
        super().__init__()
        self.elu = nn.ELU()

    def forward(self, x):
        return self.elu(x) + 1

class SinusAct(nn.Module):
    def forward(self, x):
        return torch.sin(x)

class GeneralizedCosinusUnit(nn.Module):
    def forward(self, x):
        return torch.cos(x) * x

ACTIVATION_FUNCS = {
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "biased_elu": Biased_Elu,
    "sinus": SinusAct,
    "gcu": GeneralizedCosinusUnit,
}

class TemporalBlock(nn.Module):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        kernel_size,
        stride,
        dilation,
        residual=True,
        double_layered=True,
        dropout=0.0,
        act_func=None,
    ):
        super(TemporalBlock, self).__init__()
        padding = ((kernel_size - 1) // 2) * dilation

        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                padding_mode="circular",
            )
        )
        self.relu1 = ACTIVATION_FUNCS.get(act_func, nn.Identity)()
        self.dropout1 = nn.Dropout1d(dropout)
        if double_layered:
            self.relu2 = nn.Identity()
            self.conv2 = weight_norm(
                nn.Conv1d(
                    n_inputs,
                    n_outputs,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    padding_mode="circular",
                )
            )
            self.dropout2 = nn.Dropout1d(dropout)
            self.net = nn.Sequential(
                self.conv1,
                self.relu1,
                self.dropout1,
                self.conv2,
                self.relu2,
                self.dropout2,
            )
        else:
            self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1)
        self.relu = nn.ReLU()
        self.residual = residual
        if residual:
            self.downsample = (
                nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
            )
        else:
            self.downsample = None
        self.double_layered = double_layered
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        if self.double_layered:
            self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        if self.residual:
            res = x if self.downsample is None else self.downsample(x)
            y = torch.clip(
                out + res, -10, 10
            )
            y = self.relu(y)
        else:
            y = out
        return y


class TCNWithScalarsAsBias(nn.Module):
    def __init__(
        self,
        num_input_scalars,
        num_input_ts=1,
        tcn_layer_cfg=None,
        scalar_layer_cfg=None,
    ):
        super().__init__()
        self.num_input_ts = num_input_ts
        self.num_input_scalar = num_input_scalars
        tcn_layer_cfg = tcn_layer_cfg or {
            "f": [
                {"units": (num_input_scalars + 1), "act_func": "tanh"},
                {"units": 8, "act_func": "tanh"},
                {"units": 1},
            ]
        }
        scalar_layer_cfg = scalar_layer_cfg or {
            "f": [
                {"units": num_input_scalars, "act_func": "tanh"},
            ]
        }
        # build CNN layer path
        cnn_layers = []
        dilation_offset = tcn_layer_cfg.get("starting_dilation_rate", 2)  # >= 0
        for i, l_cfg in enumerate(tcn_layer_cfg["f"]):
            kernel_size = l_cfg.get("kernel_size", 9)
            dropout_rate = tcn_layer_cfg.get("dropout", 0.0)
            dilation_size = 2 ** (i + dilation_offset)
            if i == 0:
                in_channels = num_input_ts  
            else:
                in_channels = tcn_layer_cfg["f"][i - 1]["units"]
            cnn_layers += [
                TemporalBlock(
                    in_channels,
                    l_cfg["units"],
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    residual=tcn_layer_cfg.get("residual", False),
                    double_layered=tcn_layer_cfg.get("double_layered", False),
                    dropout=dropout_rate,
                    act_func=l_cfg.get("act_func", nn.Identity),
                ),
            ]
            if i == 0:
                self.ts_branch = cnn_layers.pop()
        self.upper_tcn = nn.Sequential(*cnn_layers)
        # build scalar NN path
        scalar_layers = []
        fan_in = num_input_scalars
        for i, l_cfg in enumerate(scalar_layer_cfg["f"]):
            scalar_layers.append(nn.Linear(fan_in, l_cfg["units"]))
            scalar_layers.append(ACTIVATION_FUNCS.get(l_cfg["act_func"], nn.Identity)())
            fan_in = l_cfg["units"]

        self.scalar_branch = nn.Sequential(*scalar_layers)

    def forward(self, x_ts, x_scalars):
        """x_ts has shape (#batch, #channels, #length)"""
        b_proc = self.ts_branch(x_ts)
        scalar_proc = self.scalar_branch(x_scalars)
        catted = torch.cat(
            [
                b_proc[:, : -scalar_proc.shape[1], :],
                b_proc[:, -scalar_proc.shape[1] :, :] + scalar_proc.unsqueeze(-1),
            ],
            dim=1,
        )
        y = self.upper_tcn(catted)
        y = y + x_ts[:, [0], :]
        y = y - y.mean(dim=-1).unsqueeze(-1)
        return y

class LossPredictor(nn.Module):
    def __init__(self, h_predictor):
        super().__init__()
        self.h_predictor = h_predictor
        self.post_processor = nn.Sequential(
            nn.Linear(self.h_predictor.num_input_scalar, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
            nn.Tanh()
        )

    def forward(self, x_ts, x_scalars, stats):
        # 1. Predict H-field (Normalized)
        h_pred = self.h_predictor(x_ts, x_scalars).permute(2, 0, 1) # (Seq, Batch, 1)

        # 2. De-normalize Physical Quantities for the Shoelace Formula
        # We need real units: B in Tesla, H in A/m, Freq in Hz
        
        # Recover Frequency (Hz)
        # x_scalars[:, 0] is Standardized Freq: (F - mu) / sigma
        f_norm = x_scalars[:, 0]
        freq_hz = (f_norm * stats['freq_std']) + stats['freq_mean']
        freq_hz = torch.abs(freq_hz) # Ensure positive

        # Recover B (Tesla)
        # Scaled B and H (using learnable/fixed limits to ensure valid range)
        # Note: Paderborn hardcoded these offsets (+5) to ensure positivity for log.
        b_lim = 0.5   # Approx max Tesla
        h_lim = 100.0 # Approx max A/m
        
        # x_ts is (Batch, Channels, Seq) = (Batch, 1, Seq)
        # We want (Seq, Batch, 1)
        scaled_b = x_ts.permute(2, 0, 1) * b_lim
        scaled_h = h_pred * h_lim
        
        # Shoelace
        term1 = scaled_b
        term2 = torch.roll(scaled_h, 1, dims=0) - torch.roll(scaled_h, -1, dims=0)
        area = 0.5 * torch.sum(term1 * term2, dim=0) # (Batch, 1)
        
        # 4. Compute Power Loss: Pv = Freq * Area * (Correction Factor)
        # Correction factor allows network to adjust for non-ideal shapes
        correction = 1.0 + 0.1 * self.post_processor(x_scalars)
        
        ploss_pred = freq_hz.unsqueeze(1) * torch.abs(area) * correction
        
        # Return Log(Pv) to match Log-Loss Target
        return torch.log(ploss_pred + 1e-6)

# ==========================================
# Validated Model Wrapper
# ==========================================

class CNNNetwork(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, output_dim=1, stats=None):
        super(CNNNetwork, self).__init__()
        
        # Save stats for de-normalization
        # Defaults if not provided (avoid crash, but won't be accurate)
        if stats is None:
            stats = {'freq_mean': 100000.0, 'freq_std': 50000.0}
            
        self.register_buffer('freq_mean', torch.tensor(stats.get('freq_mean', 100000.0)))
        self.register_buffer('freq_std', torch.tensor(stats.get('freq_std', 50000.0)))

        self.tcn = TCNWithScalarsAsBias(
            num_input_scalars=3,
            num_input_ts=input_dim
        )
        self.model = LossPredictor(self.tcn)

    def forward(self, b_seq, scalars):
        # b_seq: (Batch, Seq, 1) -> TCN wants (Batch, Channels, Seq)
        x_ts = b_seq.permute(0, 2, 1)
        
        stats = {'freq_mean': self.freq_mean, 'freq_std': self.freq_std}
        
        return self.model(x_ts, scalars, stats)
