"""Expert MLP and vectorized MoE gating layer."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class MyMLP(nn.Sequential):
    """Three-layer MLP expert: Linear → GELU → Linear.

    Weights are initialised so the block starts as an identity
    (output layer zeroed) to avoid disrupting the base model.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_size: int):
        super().__init__(
            nn.Linear(input_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, output_dim),
        )
        with torch.no_grad():
            # Small random weights on the input projection → stable training
            nn.init.normal_(self[0].weight, mean=0.0, std=0.01)
            nn.init.normal_(self[0].bias, mean=0.0, std=0.01)
            # Zero output projection → block starts as identity
            self[2].weight.fill_(0)
            self[2].bias.fill_(0)


class VectorizedExternallyGatedMoELayer(nn.Module):
    """Vectorized MoE layer with externally supplied gate indices.

    Each item in the batch is routed to exactly one expert via
    ``gate_indices``.  A gate index of ``-1`` means passthrough (the
    expert contributes nothing and the layer acts as an identity).
    """

    def __init__(self, input_dim: int, output_dim: int, num_experts: int):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            MyMLP(input_dim, output_dim, input_dim * 2) for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor, gate_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:            (batch, seq_len, input_dim)
            gate_indices: (batch,) — integer index per batch item, or -1 for passthrough
        Returns:
            x + selected_expert_output  — same shape as x
        """
        # Compute all expert outputs: list of (batch, seq_len, output_dim)
        expert_outputs = [expert(x) for expert in self.experts]
        # → (batch, num_experts, seq_len, output_dim)
        stacked = torch.stack(expert_outputs, dim=1)

        # Build one-hot mask; -1 entries get an all-zero row (passthrough)
        passthrough_mask = (gate_indices == -1)
        clamped = gate_indices.clamp(min=0)
        one_hot = F.one_hot(clamped, num_classes=self.num_experts)   # (batch, num_experts)
        one_hot[passthrough_mask] = 0

        # Broadcast mask over (seq_len, output_dim) dims
        mask = one_hot.unsqueeze(-1).unsqueeze(-1)                   # (batch, num_experts, 1, 1)
        final = (stacked * mask).sum(dim=1)                          # (batch, seq_len, output_dim)
        return x + final
