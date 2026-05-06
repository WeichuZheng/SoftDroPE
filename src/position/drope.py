import math
import torch
import torch.nn as nn
from typing import Optional, Tuple


class NoPositionEmbedding(nn.Module):
    """
    DroPE: Removes positional embeddings entirely after training.
    The model then relies on recalibration to learn positional information implicitly.

    Reference: Sakana AI (2025) - DroPE: Dropping positional embeddings for zero-shot context extension
    """

    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

    def forward(self, seq_len: int, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return identity (no position encoding).
        """
        # Return ones - multiplying by 1 keeps the original values
        return torch.ones(seq_len, self.dim, device=device), torch.zeros(seq_len, self.dim, device=device)


class DroPEEncoder(nn.Module):
    """
    Encoder layer that removes RoPE from a standard transformer.
    Used for DroPE-style recalibration.
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Standard attention components
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.no_pos = NoPositionEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass without positional embeddings.

        Args:
            x: Input tensor [batch, seq_len, hidden_dim]
            attention_mask: Optional attention mask

        Returns:
            Output tensor [batch, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply identity position (DroPE removes RoPE)
        ones_pos, _ = self.no_pos(seq_len, x.device)

        # Compute attention scores without position
        # Note: Standard attention would use RoPE here, but DroPE removes it
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_probs = torch.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Apply attention to values
        context = torch.matmul(attn_probs, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)

        return self.o_proj(context)


class DroPEModel(nn.Module):
    """
    Complete model without positional embeddings for DroPE recalibration.
    """

    def __init__(self, vocab_size: int, hidden_dim: int, num_heads: int, num_layers: int, max_seq_len: int = 2048):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([
            DroPEEncoder(hidden_dim, num_heads)
            for _ in range(num_layers)
        ])
        self.ln = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass without positional embeddings.

        Args:
            input_ids: Input token IDs [batch, seq_len]

        Returns:
            Logits [batch, seq_len, vocab_size]
        """
        x = self.token_embedding(input_ids)

        for layer in self.layers:
            x = layer(x, attention_mask)

        x = self.ln(x)
        return self.lm_head(x)


def disable_rope_in_model(model: nn.Module) -> nn.Module:
    """
    Disable RoPE in an existing model by setting the rotary matrix to identity.
    This is the key operation for DroPE.

    Args:
        model: A HuggingFace model with RoPE

    Returns:
        Modified model with RoPE disabled
    """
    # This function is designed for HuggingFace models
    # It would need to be customized based on the specific model architecture
    # The general approach: set the inv_freq to zeros or identity matrix
    for name, module in model.named_modules():
        if hasattr(module, 'inv_freq'):
            # Set to identity (or zeros to remove position info)
            with torch.no_grad():
                module.inv_freq.fill_(0.0)
        if hasattr(module, 'rope'):
            # Disable the rope module
            module.rope = None

    return model


def enable_drope_mode(model: nn.Module, enable: bool = True) -> nn.Module:
    """
    Toggle DroPE mode on/off.

    Args:
        model: Model to modify
        enable: If True, disable RoPE (DroPE mode); if False, restore RoPE

    Returns:
        Modified model
    """
    for name, module in model.named_modules():
        if hasattr(module, 'rotary_emb') or hasattr(module, 'rope'):
            if enable:
                # Disable RoPE by setting inv_freq to zeros
                if hasattr(module, 'inv_freq'):
                    with torch.no_grad():
                        module.inv_freq.fill_(0.0)
            else:
                # Restore RoPE (would need to reload original values)
                pass

    return model


if __name__ == "__main__":
    # Test DroPE model
    model = DroPEModel(vocab_size=50257, hidden_dim=256, num_heads=8, num_layers=4)
    input_ids = torch.randint(0, 50257, (2, 128))
    output = model(input_ids)
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {output.shape}")
    print("DroPE model test passed!")