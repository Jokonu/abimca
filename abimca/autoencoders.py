# Standard Libraries
import random

# Third Party Libraries
import numpy as np
import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self, dim_in, dim_bottleneck = 3):
        super(AutoEncoder, self).__init__()
        self.name = "AutoEncoder"
        self.encoder = nn.Sequential(
            nn.Linear(dim_in, dim_bottleneck),
            nn.LeakyReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(dim_bottleneck, dim_in),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        h = x
        x = self.decoder(x)
        return x, h


class GRUAutoEncoder(nn.Module):
    def __init__(self, dim_in, dim_bottleneck):
        super(GRUAutoEncoder, self).__init__()
        self.name = "GRUAutoEncoder"
        self.rnn = nn.GRU(
            input_size=dim_in, hidden_size=dim_in, num_layers=1, batch_first=True
        )
        self.rnn2 = nn.GRU(
            input_size=dim_in, hidden_size=dim_in, num_layers=1, batch_first=True
        )
        self.encoder = nn.Sequential(
            nn.Linear(dim_in, 50),
            nn.LeakyReLU(),
            nn.Linear(50, dim_bottleneck),
            nn.LeakyReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(dim_bottleneck, 50),
            nn.LeakyReLU(),
            nn.Linear(50, dim_in),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out, hidden = self.rnn(x)
        x = self.encoder(out)
        h = x
        x = self.decoder(x)
        out, x = self.rnn2(x)
        return out, h



class BidirectionalGruAutoEncoder(nn.Module):
    def __init__(self, dim_in, dim_bottleneck, seed: int = 42):
        super().__init__()
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.name = "BidirectionalGruAutoEncoder"
        self.dim_in = dim_in
        self.dim_bottleneck = dim_bottleneck
        self.batchnorm = nn.BatchNorm1d(dim_in)
        self.encoder = nn.GRU(
            input_size=dim_in,
            hidden_size=dim_bottleneck,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.decoder = nn.GRU(
            input_size=dim_bottleneck,
            hidden_size=dim_in,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x):
        output, h_n = self.encoder(x)
        output = (
            output[:, :, : self.dim_bottleneck] + output[:, :, self.dim_bottleneck :]
        ) / 2
        x_hat, hidden = self.decoder(output)
        x_hat = x_hat[:, :, : self.dim_in] + x_hat[:, :, self.dim_in :]
        h_n = h_n.permute(1, 0, 2)
        self.encoder.flatten_parameters()
        self.decoder.flatten_parameters()
        return x_hat, h_n


def set_optimizer(
    optimizer_type: str = "Adam",
    model_parameters=None,
    learning_rate: float = 1e-3,
    momentum: float = None,
    sgd_nesterov: bool = True,
):
    if optimizer_type == "Adam":
        return torch.optim.Adam(model_parameters, lr=learning_rate)
    elif optimizer_type == "AdamW":
        return torch.optim.AdamW(model_parameters, lr=learning_rate)
    elif optimizer_type == "SGD":
        return torch.optim.SGD(
            model_parameters, lr=learning_rate, momentum=momentum, nesterov=sgd_nesterov
        )
    elif optimizer_type == "NAdam":
        return torch.optim.NAdam(model_parameters, lr=learning_rate)
    elif optimizer_type == "RAdam":
        return torch.optim.RAdam(model_parameters, lr=learning_rate)


def setWeightInitialization(
    model, use_weight_init: str = "kaiming_normal", seed: int = 42
):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    def weights_init(m):
        if isinstance(m, nn.Linear):
            if use_weight_init == "uniform":
                nn.init.uniform_(m.weight)
            elif use_weight_init == "normal":
                nn.init.normal_(m.weight)
            elif use_weight_init == "constant":
                nn.init.constant_(m.weight, 0.5)
            elif use_weight_init == "ones":
                nn.init.ones_(m.weight)
            elif use_weight_init == "zeros":
                nn.init.zeros_(m.weight)
            elif use_weight_init == "eye":
                nn.init.eye_(m.weight)
            elif use_weight_init == "dirac":
                nn.init.dirac_(m.weight)
            elif use_weight_init == "xavier_uniform":
                nn.init.xavier_uniform_(m.weight)
            elif use_weight_init == "xavier_normal":
                nn.init.xavier_normal_(m.weight)
            elif use_weight_init == "kaiming_uniform":
                nn.init.kaiming_uniform_(m.weight, mode="fan_out")
            elif use_weight_init == "kaiming_normal":
                nn.init.kaiming_normal_(m.weight)
            elif use_weight_init == "orthogonal":
                nn.init.orthogonal_(m.weight)
            elif use_weight_init == "sparse":
                nn.init.sparse_(m.weight, sparsity=0.1)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    model.apply(weights_init)
    return model
