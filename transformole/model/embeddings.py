import torch
from torch import nn, Tensor
import math

class PositionalEncoding(nn.Module):
    """Transformer positional encoding module"""

    def __init__(self, dim_model: int, max_len: int, device: torch.device):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1).to(device)
        div_term = torch.exp(torch.arange(0, dim_model, 2) * (-math.log(10000.0) / dim_model)).to(device)
        pe = torch.zeros(1, max_len, dim_model).to(device)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[:, :x.size(1)]

class RoPE(nn.Module):
    """
    Enhanced positional encoding combining multiple approaches

    Features:
    - Rotary Position Embedding (RoPE)
    - Learnable frequency components
    - Dynamic gating mechanism
    - Relative position biases

    Args:
        dim_model: Dimension of the model embeddings
        max_len: Maximum sequence length to handle
        num_heads: Number of attention heads (for relative position)
    """

    def __init__(self, dim_model: int, max_len: int, num_heads: int, device: torch.device):
        super().__init__()
        self.attn_weights = None
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.device = device
        # Rotary Position Embedding (RoPE) components
        self.rope_freq = nn.Parameter(torch.randn(num_heads, dim_model // num_heads // 2)).to(device)
        nn.init.normal_(self.rope_freq, mean=math.log(max_len) / 2, std=0.02)

        # Learnable sinusoidal components
        self.freq_weights = nn.Parameter(torch.randn(dim_model)).to(device)
        self.phase_shift = nn.Parameter(torch.randn(dim_model)).to(device)

        # Relative position bias table
        self.rel_pos_bias = nn.Embedding(2 * max_len + 1, num_heads).to(device)

        # Dynamic gating mechanism
        self.gate_net = nn.Sequential(
            nn.Linear(dim_model, 4 * dim_model, device=device),
            nn.SiLU(),
            nn.Linear(4 * dim_model, dim_model, device=device),
            nn.Sigmoid()
        )

    def _apply_rope(self, x: Tensor) -> Tensor:
        """Apply rotary position embedding to input tensor"""
        batch_size, seq_len, _ = x.size()

        # Reshape for rotary transformation
        x_flat = x.view(batch_size, seq_len, self.num_heads, -1)
        x_rot = x_flat.permute(0, 2, 1, 3)  # [B, H, T, D]

        # Create rotation matrix
        position = torch.arange(seq_len, device=x.device).view(1, 1, seq_len)
        freq = torch.exp(self.rope_freq.view(1, self.num_heads, 1, -1) * position)
        cos = torch.cos(freq)
        sin = torch.sin(freq)

        # Apply rotation
        x1, x2 = x_rot.chunk(2, dim=-1)
        rotated = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
        return rotated.permute(0, 2, 1, 3).reshape_as(x)

    def _get_rel_pos_bias(self, seq_len: int) -> Tensor:
        """Generate relative position bias matrix"""
        context_pos = torch.arange(seq_len, device=self.rel_pos_bias.weight.device)[:, None]
        memory_pos = torch.arange(seq_len, device=self.rel_pos_bias.weight.device)[None, :]
        relative_pos = memory_pos - context_pos + seq_len  # Shift to positive indices
        return self.rel_pos_bias(relative_pos).permute(2, 0, 1)  # [H, T, T]

    def forward(self, x: Tensor) -> Tensor:
        """Enhanced position-aware transformation

        Returns:
            Position-augmented tensor with shape [B, T, D]
        """
        seq_len = x.size(1)

        # Base rotary encoding
        rope_out = self._apply_rope(x)

        # Learnable frequency modulation
        position = torch.arange(seq_len, device=x.device).float()
        freq_enc = torch.sin(position[:, None] * self.freq_weights + self.phase_shift)
        freq_out = x * freq_enc[None, :, :]

        # Dynamic gating
        gate = self.gate_net(x)
        combined = gate * rope_out + (1 - gate) * freq_out

        # Add relative position biases to attention
        if hasattr(self, 'attn_weights'):  # For attention integration
            self.attn_weights += self._get_rel_pos_bias(seq_len)

        return combined

    def integrate_with_attention(self, attn_weights: Tensor) -> Tensor:
        """Integrate relative position biases with attention matrix"""
        self.attn_weights = attn_weights
        return attn_weights
