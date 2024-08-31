import torch
from torch import nn
from torch.nn import functional as F
from typing import Callable, Iterable


class MLP(nn.Module):
    n_layers: int
    shape: tuple[int]
    fcs: nn.ModuleList
    activation: Callable
    dropout: nn.ModuleList | None

    def __init__(
        self,
        shape: Iterable[int],
        activation: str | Callable = F.gelu,
        dropout: int = 0.0,
    ):
        super().__init__()

        self.n_layers = len(shape) - 1

        assert self.n_layers >= 1

        self._build_linear_layers(shape)
        self._add_nonlinearity(activation, dropout)

    def _build_linear_layers(self, shape: Iterable[int]) -> None:
        self.shape = tuple(shape)
        self.fcs = nn.ModuleList()
        for j in range(self.n_layers):
            self.fcs.append(nn.Linear(shape[j], shape[j + 1]))

    def _add_nonlinearity(self, activation: str | Callable, dropout: float) -> None:
        if isinstance(activation, Callable):
            self.activation = activation
        else:
            self.activation = _get_activation_function(activation)
        self.dropout = (
            nn.ModuleList([nn.Dropout(dropout) for _ in range(self.n_layers)])
            if dropout > 0.0
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, fc in enumerate(self.fcs):
            x = fc(x)
            if i < self.n_layers - 1:
                x = self.activation(x)
            if self.dropout is not None:
                x = self.dropout[i](x)
        return x


class CMLP(nn.Module):
    n_layers: int
    in_channels: int
    out_channels: int
    hidden_channels: int
    fcs: nn.ModuleList
    activation: Callable
    dropout: nn.ModuleList | None
    n_dim: int

    def __init__(
        self,
        in_channels: int,
        out_channels: int = None,
        hidden_channels: int = None,
        n_layers: int = 2,
        n_dim: int = 2,
        activation: str | Callable = F.gelu,
        dropout: float = 0.0,
    ):
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.hidden_channels = (
            in_channels if hidden_channels is None else hidden_channels
        )
        self.n_dim = n_dim
        super().__init__(
            [self.in_channels]
            + [self.hidden_channels] * (n_layers - 1)
            + [self.out_channels],
            activation=activation,
            dropout=dropout,
        )

    def _build_linear_layers(self, layers: Iterable[int]) -> None:
        Conv = getattr(nn, f"Conv{self.n_dim}d")
        self.fcs = nn.ModuleList()
        for i in range(self.n_layers):
            self.fcs.append(Conv(layers[i], layers[i + 1], 1))

    def _add_nonlinearity(self, activation: str | Callable, dropout: float) -> None:
        if isinstance(activation, Callable):
            self.activation = activation
        else:
            self.activation = _get_activation_function(activation)
        self.dropout = (
            nn.ModuleList([nn.Dropout(dropout) for _ in range(self.n_layers)])
            if dropout > 0.0
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, fc in enumerate(self.fcs):
            x = fc(x)
            if i < self.n_layers - 1:
                x = self.activation(x)
            if self.dropout is not None:
                x = self.dropout[i](x)
        return x


class MMLP(nn.Module):
    n_layers: int
    in_features: int
    hidden_features: int
    out_features: int
    fc_u: nn.Module
    fc_v: nn.Module
    fc_m: nn.ModuleList
    activation: Callable
    dropout: nn.ModuleList | None

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        n_layers: int,
        activation: str | Callable = F.gelu,
        dropout: int = 0.0,
    ):
        super().__init__()

        self.n_layers = n_layers
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

        self.fc_u = nn.Linear(self.in_features, self.hidden_features)
        self.fc_v = nn.Linear(self.in_features, self.hidden_features)
        self.fc_m = nn.ModuleList()
        self.fc_m.append(nn.Linear(in_features, hidden_features))
        for _ in range(self.n_layers - 2):
            self.fc_m.append(nn.Linear(hidden_features, hidden_features))
        self.fc_m.append(nn.Linear(hidden_features, out_features))

        self._set_activation(activation)
        self._set_dropout(dropout)

    def _set_activation(self, activation: str | Callable) -> None:
        if isinstance(activation, Callable):
            self.activation = activation
            return
        if activation == "sigmoid":
            self.activation = F.sigmoid
        elif activation == "tanh":
            self.activation = F.tanh
        elif activation == "relu":
            self.activation = F.relu
        elif activation == "relu6":
            self.activation = F.relu6
        elif activation == "gelu":
            self.activation = F.gelu
        elif activation == "selu":
            self.activation = F.selu
        else:
            raise ValueError(
                f"{activation} is not supported in MLP, try to provide a callable instead."
            )

    def _set_dropout(self, dropout: float) -> None:
        self.dropout = (
            nn.ModuleList([nn.Dropout(dropout) for _ in range(self.n_layers)])
            if dropout > 0.0
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.activation(self.fc_u(x))
        v = self.activation(self.fc_v(x))
        for i, fc in enumerate(self.fc_m):
            x = fc(x)
            if i < self.n_layers - 1:
                x = self.activation(x)
                x = (1 - x) * u + x * v
            if self.dropout is not None:
                x = self.dropout[i](x)

        return x
