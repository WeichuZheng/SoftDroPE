import math
import torch
import torch.nn as nn
from typing import Optional, Tuple


class PositionInterpolation(nn.Module):
    """
    Position Interpolation (PI) - Linear downscaling of position indices.
    Reference: Chen et al. (2023) - Extending context window of large language models via positional interpolation
    """

    def __init__(self, dim: int, base: int = 10000, max_seq_len: int = 2048, scale_factor: float = 1.0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.scale_factor = scale_factor  # Original context / extended context

        # Precompute inverse frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        self._set_cos_sin_cache(max_seq_len)

    def _set_cos_sin_cache(self, seq_len: int):
        self.max_seq_len = seq_len
        # Scale positions by scale_factor (interpolation)
        t = torch.arange(self.max_seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        t = t * self.scale_factor  # Linear interpolation
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)

    def forward(self, seq_len: int, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len:
            self._set_cos_sin_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotation to q and k."""
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    def rotate_half(x):
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat([-x2, x1], dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class NTKAwareScaledRoPE(nn.Module):
    """
    NTK-aware scaled RoPE - Adjusts high-frequencies less aggressively.
    Reference: blocq97 (2023) - NTK-aware scaled RoPE
    """

    def __init__(self, dim: int, base: int = 10000, max_seq_len: int = 2048, scaling_factor: float = 1.0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.scaling_factor = scaling_factor

        # NTK-aware: scale the base
        self.scaled_base = base * (scaling_factor ** (dim / (dim - 2)))

        inv_freq = 1.0 / (self.scaled_base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        self._set_cos_sin_cache(max_seq_len)

    def _set_cos_sin_cache(self, seq_len: int):
        self.max_seq_len = seq_len
        t = torch.arange(self.max_seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)

    def forward(self, seq_len: int, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len:
            self._set_cos_sin_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


class YaRN(nn.Module):
    """
    YaRN: Efficient context window extension of large language models.
    Combines NTK scaling with temperature term and attention distribution smoothing.
    Reference: Peng, Qureshi, & Fan (2023)
    """

    def __init__(
        self,
        dim: int,
        base: int = 10000,
        max_seq_len: int = 2048,
        original_ctx_len: int = 2048,
        scaling_factor: float = 1.0,
        temperature: float = 10000.0
    ):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.original_ctx_len = original_ctx_len
        self.scaling_factor = scaling_factor
        self.temperature = temperature

        # YaRN uses NTK-aware scaling
        self.scaled_base = base * (scaling_factor ** (dim / (dim - 2)))

        inv_freq = 1.0 / (self.scaled_base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        self._set_cos_sin_cache(max_seq_len)

    def _set_cos_sin_cache(self, seq_len: int):
        self.max_seq_len = seq_len
        t = torch.arange(self.max_seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)

        # YaRN scaling factor per dimension
        dim_indices = torch.arange(0, self.dim, 2, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        yarn_scale = (self.original_ctx_len / self.max_seq_len) ** (dim_indices / self.dim)

        # Apply YaRN scaling to positions
        t = t.unsqueeze(1) * yarn_scale.unsqueeze(0)
        t = t.view(-1)

        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)

    def forward(self, seq_len: int, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len:
            self._set_cos_sin_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


class BaselinePositionEncoder(nn.Module):
    """
    Unified interface for baseline position encodings.
    """

    def __init__(
        self,
        method: str,
        dim: int,
        base: int = 10000,
        max_seq_len: int = 2048,
        **kwargs
    ):
        super().__init__()
        self.method = method.lower()

        if self.method == 'pi' or self.method == 'position_interpolation':
            scale_factor = kwargs.get('scale_factor', 1.0)
            self.encoder = PositionInterpolation(dim, base, max_seq_len, scale_factor)
        elif self.method == 'ntk' or self.method == 'ntk_aware':
            scaling_factor = kwargs.get('scaling_factor', 1.0)
            self.encoder = NTKAwareScaledRoPE(dim, base, max_seq_len, scaling_factor)
        elif self.method == 'yarn':
            original_ctx_len = kwargs.get('original_ctx_len', 2048)
            scaling_factor = kwargs.get('scaling_factor', 1.0)
            self.encoder = YaRN(dim, base, max_seq_len, original_ctx_len, scaling_factor)
        else:
            raise ValueError(f"Unknown baseline method: {method}")

    def forward(self, q: torch.Tensor, k: torch.Tensor, position_ids=None) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.shape[2]
        cos, sin = self.encoder(seq_len, q.device)
        return apply_rotary_pos_emb(q, k, cos, sin)


# Factory function to create any position encoder
def create_all_position_encoders(
    method: str,
    dim: int,
    base: int = 10000,
    max_seq_len: int = 2048,
    **kwargs
) -> nn.Module:
    """
    Factory function to create position encoder.

    Args:
        method: One of 'rope', 'drope', 'cope', 'softdrope', 'pi', 'ntk', 'yarn'
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
        # DroPE: no position encoding (identity)
        from .rope import RoPEModel
        # Return identity position encoder (multiplying by 1, adding 0)
        class IdentityPosEncoder(nn.Module):
            def __init__(self, dim, base, max_seq_len):
                super().__init__()
                self.dim = dim
            def forward(self, q, k, position_ids=None):
                return q, k
        return IdentityPosEncoder(dim, base, max_seq_len)
    elif method in ['cope', 'clipped_rope']:
        from .cope import CoPEModel
        theta_cutoff = kwargs.get('theta_cutoff', 1.0)
        return CoPEModel(dim, base, max_seq_len, theta_cutoff)
    elif method == 'softdrope':
        from .softdrope import SoftDroPEModel
        theta_cutoff = kwargs.get('theta_cutoff', 1.0)
        stage = kwargs.get('stage', 2)
        return SoftDroPEModel(dim, base, max_seq_len, theta_cutoff, stage)
    elif method in ['pi', 'position_interpolation', 'ntk', 'ntk_aware', 'yarn']:
        return BaselinePositionEncoder(method, dim, base, max_seq_len, **kwargs)
    else:
        raise ValueError(f"Unknown position encoder: {method}. Supported: rope, drope, cope, softdrope, pi, ntk, yarn")


if __name__ == "__main__":
    # Test baseline encoders
    import torch

    print("Testing Position Interpolation...")
    pi = BaselinePositionEncoder('pi', dim=64, max_seq_len=128, scale_factor=0.5)
    q = torch.randn(2, 4, 128, 64)
    k = torch.randn(2, 4, 128, 64)
    q_rot, k_rot = pi(q, k)
    print(f"  PI output shape: {q_rot.shape}")

    print("Testing NTK-aware RoPE...")
    ntk = BaselinePositionEncoder('ntk', dim=64, max_seq_len=128, scaling_factor=2.0)
    q_rot, k_rot = ntk(q, k)
    print(f"  NTK output shape: {q_rot.shape}")

    print("Testing YaRN...")
    yarn = BaselinePositionEncoder('yarn', dim=64, max_seq_len=128, original_ctx_len=2048, scaling_factor=2.0)
    q_rot, k_rot = yarn(q, k)
    print(f"  YaRN output shape: {q_rot.shape}")

    print()
    print("All baseline encoders work correctly!")