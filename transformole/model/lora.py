from torch import nn, Tensor
import torch
import math
import torch.nn.functional as F
from typing import Optional, Tuple


class LoRALinear(nn.Module):
    """Low-Rank Adaptation linear layer with dynamic rank management"""

    def __init__(
            self,
            in_features: int,
            out_features: int,
            rank: int,
            alpha: int,
            device: torch.device,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, device=device)
        self.lora_A = nn.Parameter(torch.empty(rank, in_features)).to(device)
        self.lora_B = nn.Parameter(torch.empty(out_features, rank)).to(device)
        self.scaling = alpha / rank

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.linear.weight.requires_grad_(False)
        if self.linear.bias is not None:
            self.linear.bias.requires_grad_(False)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling


class LoRAAttention(nn.Module):
    """LoRA-enhanced self-attention implementation"""

    def __init__(
            self,
            dim_model: int,
            num_head: int,
            rank: int,
            alpha: int,
            device: torch.device
    ):
        super().__init__()
        self.embed_dim = dim_model
        self.num_heads = num_head
        self.head_dim = dim_model // num_head

        # LoRA projections
        self.q_proj = LoRALinear(dim_model, dim_model, rank, alpha, device=device)
        self.k_proj = LoRALinear(dim_model, dim_model, rank, alpha, device=device)
        self.v_proj = LoRALinear(dim_model, dim_model, rank, alpha, device=device)
        self.out_proj = LoRALinear(dim_model, dim_model, rank, alpha, device=device)

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            attn_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        # Shared projections for decoder-only self-attention
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Reshape for multi-head attention
        q = q.view(*q.shape[:2], self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(*k.shape[:2], self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(*v.shape[:2], self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            attn_weights += attn_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        output = torch.matmul(attn_weights, v)

        # Combine heads and project
        output = output.transpose(1, 2).contiguous().view(*query.shape[:2], self.embed_dim)
        return self.out_proj(output), attn_weights