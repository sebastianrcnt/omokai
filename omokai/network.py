from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .config import NetworkConfig


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int) -> None:
        super().__init__()
        hidden = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.fc(self.pool(x)).view(x.size(0), x.size(1), 1, 1)
        return x * weights


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, se_reduction: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels, se_reduction)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.se(x)
        return self.relu(x + residual)


class PolicyValueNet(nn.Module):
    def __init__(self, board_size: int, cfg: NetworkConfig) -> None:
        super().__init__()
        self.board_size = board_size
        self.action_size = board_size * board_size
        channels = cfg.channels
        self.stem = nn.Sequential(
            nn.Conv2d(cfg.input_planes, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.tower = nn.Sequential(*[ResidualBlock(channels, cfg.se_reduction) for _ in range(cfg.blocks)])
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2 * self.action_size, self.action_size),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(self.action_size, cfg.value_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.value_hidden, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.tower(x)
        policy_logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        return policy_logits, value


def clone_model(model: PolicyValueNet) -> PolicyValueNet:
    cloned = PolicyValueNet(model.board_size, NetworkConfig(
        input_planes=model.stem[0].in_channels,
        channels=model.stem[0].out_channels,
        blocks=len(model.tower),
        value_hidden=model.value_head[4].out_features,
        se_reduction=max(1, model.stem[0].out_channels // model.tower[0].se.fc[1].out_features),
    ))
    cloned.load_state_dict(model.state_dict())
    return cloned
