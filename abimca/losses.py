# Third Party Libraries
import torch


class RMSELoss(torch.nn.Module):
    def __init__(self, eps=1e-10, reduction="mean"):
        super().__init__()
        self.reduction = reduction
        self.mse = torch.nn.MSELoss(reduction=reduction)
        self.eps = eps

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, h: torch.Tensor = None
    ) -> torch.Tensor:
        loss = torch.sqrt(self.mse(x, y) + self.eps)
        loss = loss.mean(dim=2).mean(dim=1) if self.reduction == "none" else loss
        return loss


class SparseRMSELoss(torch.nn.Module):
    def __init__(
        self, eps=1e-10, reduction: str = "mean", penalty_factor: float = 5e-3
    ):
        super().__init__()
        self.rmse = RMSELoss(reduction=reduction, eps=eps)
        self.penalty_factor = penalty_factor

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, h: torch.Tensor
    ) -> torch.Tensor:
        penalty = torch.linalg.norm(h - 0.5, dim=(1, 2))
        loss = self.rmse(x, y) + (self.penalty_factor * penalty)
        return loss


class SparseMSELoss(torch.nn.Module):
    def __init__(
        self, eps=1e-10, reduction: str = "mean", penalty_factor: float = 5e-3
    ):
        super().__init__()
        self.reduction = reduction
        self.mse = torch.nn.MSELoss(reduction=reduction)
        self.eps = eps
        self.penalty_factor = penalty_factor

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, h: torch.Tensor
    ) -> torch.Tensor:
        penalty = torch.linalg.norm(h - 0.5, dim=(1, 2))
        loss = self.mse(x, y)
        loss = loss.mean(dim=2).mean(dim=1) if self.reduction == "none" else loss
        loss = loss + (self.penalty_factor * penalty)
        return loss
