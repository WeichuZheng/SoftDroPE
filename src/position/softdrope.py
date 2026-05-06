import math
import torch
import torch.nn as nn
from typing import Optional, Tuple

from .rope import rotate_half as _rotate_half
from .cope import ClippedRoPE


class SoftDroPE(nn.Module):
    """
    SoftDroPE: Hybrid post-training calibration combining DroPE and CoPE.

    Two-stage approach:
    1. Stage 1: DroPE-style Recalibration - Remove RoPE, perform short recalibration
    2. Stage 2: CoPE Injection - Replace with soft-clipped RoPE (CoPE)

    Reference: SoftDroPE: Post-Training Calibration with Soft Frequency Regularization
               for Extreme Context Extension (2026)
    """

    def __init__(
        self,
        dim: int,
        base: int = 10000,
        max_seq_len: int = 2048,
        theta_cutoff: float = 1.0,
        stage: int = 2
    ):
        """
        Args:
            dim: Hidden dimension per head
            base: Base for inverse frequency computation
            max_seq_len: Maximum sequence length
            theta_cutoff: Cutoff threshold for CoPE soft clipping
            stage: Current stage (1 = DroPE only, 2 = CoPE injection)
        """
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.theta_cutoff = theta_cutoff
        self.stage = stage

        # CoPE module for soft frequency regularization
        self.cope = ClippedRoPE(dim, base, max_seq_len, theta_cutoff)

    def forward(self, seq_len: int, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return cos and sin for the given sequence length.
        In Stage 1 (DroPE), this returns identity (no position).
        In Stage 2 (CoPE), this returns soft-clipped positions.
        """
        if self.stage == 1:
            # DroPE stage: no position encoding
            return torch.ones(seq_len, self.dim, device=device), torch.zeros(seq_len, self.dim, device=device)
        else:
            # CoPE stage: soft-clipped position encoding
            return self.cope(seq_len, device)

    def set_stage(self, stage: int):
        """Set the current stage (1 = DroPE, 2 = CoPE)."""
        self.stage = stage

    def update_theta_cutoff(self, theta_cutoff: float):
        """Update theta_cutoff for CoPE."""
        self.theta_cutoff = theta_cutoff
        self.cope.theta_cutoff = theta_cutoff
        self.cope._set_cos_sin_cache(self.cope.max_seq_len)


def apply_softdrope_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply SoftDroPE (CoPE) position embedding to query and key tensors.

    Args:
        q: Query tensor [batch, num_heads, seq_len, head_dim]
        k: Key tensor [batch, num_heads, seq_len, head_dim]
        cos: Cosine component [seq_len, head_dim]
        sin: Sine component [seq_len, head_dim]

    Returns:
        Rotated q and k tensors
    """
    # Reshape for broadcasting: [1, 1, seq_len, head_dim] to match [batch, num_heads, seq_len, head_dim]
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
    sin = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]

    # Apply rotation (same as RoPE/CoPE)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)

    return q_embed, k_embed


class SoftDroPEModel(nn.Module):
    """
    Complete SoftDroPE model that can be integrated with attention.
    Supports both DroPE (Stage 1) and CoPE (Stage 2) modes.
    """

    def __init__(
        self,
        dim: int,
        base: int = 10000,
        max_seq_len: int = 2048,
        theta_cutoff: float = 1.0,
        initial_stage: int = 2
    ):
        super().__init__()
        self.softdrope = SoftDroPE(dim, base, max_seq_len, theta_cutoff, initial_stage)

    def forward(self, q: torch.Tensor, k: torch.Tensor, position_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply SoftDroPE to query and key.

        Args:
            q: Query tensor [batch, num_heads, seq_len, head_dim]
            k: Key tensor [batch, num_heads, seq_len, head_dim]
            position_ids: Optional position indices

        Returns:
            Rotated q and k
        """
        seq_len = q.shape[2]
        cos, sin = self.softdrope(seq_len, q.device)
        return apply_softdrope_pos_emb(q, k, cos, sin)

    def set_stage(self, stage: int):
        """Switch between DroPE (1) and CoPE (2) stages."""
        self.softdrope.set_stage(stage)

    def update_theta_cutoff(self, theta_cutoff: float):
        """Update theta_cutoff for CoPE."""
        self.softdrope.update_theta_cutoff(theta_cutoff)


# Factory function to create position encoder based on method
def create_position_encoder(
    method: str,
    dim: int,
    base: int = 10000,
    max_seq_len: int = 2048,
    **kwargs
) -> nn.Module:
    """
    Factory function to create position encoder.

    Args:
        method: One of 'rope', 'drope', 'cope', 'softdrope'
        dim: Hidden dimension per head
        base: Base for inverse frequency computation
        max_seq_len: Maximum sequence length
        **kwargs: Additional method-specific arguments

    Returns:
        Position encoder module
    """
    method = method.lower()

    if method == 'rope':
        from .rope import RoPEModel
        return RoPEModel(dim, base, max_seq_len)
    elif method == 'drope':
        from .drope import DroPEModel
        return DroPEModel(dim=dim, max_seq_len=max_seq_len)
    elif method in ['cope', 'clipped_rope']:
        from .cope import CoPEModel
        theta_cutoff = kwargs.get('theta_cutoff', 1.0)
        return CoPEModel(dim, base, max_seq_len, theta_cutoff)
    elif method == 'softdrope':
        theta_cutoff = kwargs.get('theta_cutoff', 1.0)
        stage = kwargs.get('stage', 2)
        return SoftDroPEModel(dim, base, max_seq_len, theta_cutoff, stage)
    else:
        raise ValueError(f"Unknown position encoder: {method}. Supported: rope, drope, cope, softdrope")


if __name__ == "__main__":
    # Test SoftDroPE
    softdrope = SoftDroPEModel(dim=64, max_seq_len=128, theta_cutoff=1.0, initial_stage=2)
    q = torch.randn(2, 4, 128, 64)  # batch, heads, seq, dim
    k = torch.randn(2, 4, 128, 64)
    q_rot, k_rot = softdrope(q, k)
    print(f"Input shape: q={q.shape}, k={k.shape}")
    print(f"Output shape: q_rot={q_rot.shape}, k_rot={k_rot.shape}")

    # Test stage switching
    softdrope.set_stage(1)  # DroPE mode
    print("SoftDroPE stage switching test passed!")

    # Test theta_cutoff update
    softdrope.update_theta_cutoff(0.5)
    print("SoftDroPE theta_cutoff update test passed!")

    # Test factory function
    encoder = create_position_encoder('softdrope', dim=64)
    print(f"Factory function test: {type(encoder).__name__}")

    print("SoftDroPE test passed!")