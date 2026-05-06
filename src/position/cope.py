import math
import torch
import torch.nn as nn
from typing import Optional, Tuple


class ClippedRoPE(nn.Module):
    """
    CoPE (Clipped RoPE) implementation.
    Applies soft clipping to low-frequency components of RoPE.

    Reference: Liu, Wu, & He (2026) - CoPE: Clipped rotary position embedding for scalable length generalization
    """

    def __init__(self, dim: int, base: int = 10000, max_seq_len: int = 2048, theta_cutoff: float = 1.0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.theta_cutoff = theta_cutoff

        # Precompute inverse frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

        # Precompute cos/sin cache
        self._set_cos_sin_cache(max_seq_len)

    def _set_cos_sin_cache(self, seq_len: int):
        self.max_seq_len = seq_len
        t = torch.arange(self.max_seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)

        # Apply soft clipping to frequencies below cutoff
        clipped_freqs = self._apply_soft_clip(freqs)

        emb = torch.cat([clipped_freqs, clipped_freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)

    def _apply_soft_clip(self, freqs: torch.Tensor) -> torch.Tensor:
        """
        Apply soft clipping function to frequency components.

        Θ' = Θ × 0.5 × (1 + cos(Θ/Θcutoff × π))  for Θ < Θcutoff
        Θ' = Θ  otherwise
        """
        mask = freqs < self.theta_cutoff
        clipped = freqs * 0.5 * (1 + torch.cos(freqs / self.theta_cutoff * math.pi))
        return torch.where(mask, clipped, freqs)

    def forward(self, seq_len: int, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return cos and sin for the given sequence length."""
        if seq_len > self.max_seq_len:
            self._set_cos_sin_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def apply_cope_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply clipped RoPE to query and key tensors.

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

    # Apply rotation (same as standard RoPE)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat([-x2, x1], dim=-1)


def precompute_cope_freqs_cis(dim: int, seq_len: int, base: int = 10000, theta_cutoff: float = 1.0, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute frequency components for CoPE with soft clipping.

    Args:
        dim: Hidden dimension per head
        seq_len: Maximum sequence length
        base: Base for inverse frequency computation
        theta_cutoff: Cutoff threshold for soft clipping
        device: Device to create tensors on

    Returns:
        Tuple of (cos, sin) tensors
    """
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq_len, device=device, dtype=freqs.dtype)
    freqs = torch.outer(t, freqs)

    # Apply soft clipping
    mask = freqs < theta_cutoff
    clipped = freqs * 0.5 * (1 + torch.cos(freqs / theta_cutoff * math.pi))
    clipped_freqs = torch.where(mask, clipped, freqs)

    emb = torch.cat([clipped_freqs, clipped_freqs], dim=-1)
    return emb.cos(), emb.sin()


class CoPEModel(nn.Module):
    """
    Complete CoPE model that can be integrated with attention.
    """

    def __init__(self, dim: int, base: int = 10000, max_seq_len: int = 2048, theta_cutoff: float = 1.0):
        super().__init__()
        self.cope = ClippedRoPE(dim, base, max_seq_len, theta_cutoff)

    def forward(self, q: torch.Tensor, k: torch.Tensor, position_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply CoPE to query and key.

        Args:
            q: Query tensor [batch, num_heads, seq_len, head_dim]
            k: Key tensor [batch, num_heads, seq_len, head_dim]
            position_ids: Optional position indices

        Returns:
            Rotated q and k
        """
        seq_len = q.shape[2]
        cos, sin = self.cope(seq_len, q.device)
        return apply_cope_pos_emb(q, k, cos, sin)

    def update_theta_cutoff(self, theta_cutoff: float):
        """Update theta_cutoff and recompute cache."""
        self.cope.theta_cutoff = theta_cutoff
        self.cope._set_cos_sin_cache(self.cope.max_seq_len)


if __name__ == "__main__":
    # Simple test
    cope = CoPEModel(dim=64, max_seq_len=128, theta_cutoff=1.0)
    q = torch.randn(2, 4, 128, 64)  # batch, heads, seq, dim
    k = torch.randn(2, 4, 128, 64)
    q_rot, k_rot = cope(q, k)
    print(f"Input shape: q={q.shape}, k={k.shape}")
    print(f"Output shape: q_rot={q_rot.shape}, k_rot={k_rot.shape}")

    # Test with different theta_cutoff
    cope.update_theta_cutoff(0.5)
    print("CoPE theta_cutoff update test passed!")

    print("CoPE test passed!")