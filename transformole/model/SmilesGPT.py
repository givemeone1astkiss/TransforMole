import os
import torch
import csv
import math
import pytorch_lightning as pl
from torch import nn, Tensor
from typing import Tuple, Optional

from ..config import GEN_PATH
from torch.nn import functional as F

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


class DecoderOnlyLayer(nn.Module):
    """Decoder-only transformer layer without cross-attention

    Features:
    - Self-attention with LoRA support
    - Position-wise feedforward network
    - Pre-LayerNorm configuration
    - Residual connections
    - Dynamic LoRA integration

    Args:
        dim_model: Dimension of model embeddings
        num_head: Number of attention heads
        dim_feedforward: Dimension of FFN hidden layer
        dropout: Dropout probability (default: 0.1)
        use_lora: Enable LoRA adaptation
        lora_rank: LoRA projection rank
        lora_alpha: LoRA scaling factor
    """

    def __init__(
            self,
            dim_model: int,
            num_head: int,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
            use_lora: bool = False,
            lora_rank: int = 8,
            lora_alpha: int = 16,
            device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)
        self.dropout = nn.Dropout(dropout)

        # Self-attention components
        self.self_attn = self._create_attention(
            dim_model=dim_model, num_head=num_head, use_lora=use_lora, rank=lora_rank, alpha=lora_alpha, device=device
        )

        # Feedforward components
        self.ffn = self._create_ffn(
            dim_model=dim_model, dim_feedforward=dim_feedforward, use_lora=use_lora, rank=lora_rank, alpha=lora_alpha, device=device
        )

    def _create_attention(
            self,
            dim_model: int,
            num_head: int,
            use_lora: bool,
            rank: int,
            alpha: int,
            device: torch.device
    ) -> nn.Module:
        """Create self-attention mechanism with LoRA options"""
        return nn.MultiheadAttention(
            embed_dim=dim_model,
            num_heads=num_head,
            batch_first=True,
            device=device
        ) if not use_lora else self.LoRAAttention(
            dim_model, num_head, rank, alpha, device=device
        )

    @staticmethod
    def _create_ffn(
            dim_model: int,
            dim_feedforward: int,
            use_lora: bool,
            rank: int,
            alpha: int,
            device: torch.device
    ) -> nn.Sequential:
        """Create position-wise FFN with LoRA options"""
        if not use_lora:
            return nn.Sequential(
                nn.Linear(dim_model, dim_feedforward),
                nn.GELU(),
                nn.Linear(dim_feedforward, dim_model),
                nn.Dropout(0.1)
            )
        else:
            return nn.Sequential(
                LoRALinear(in_features=dim_model, out_features=dim_feedforward, rank=rank, alpha=alpha, device=device),
                nn.GELU(),
                LoRALinear(in_features=dim_feedforward, out_features=dim_model, rank=rank, alpha=alpha, device=device),
                nn.Dropout(0.1)
            )

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

    def forward(
            self,
            x: Tensor,
            attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass with residual connections"""
        # Self-attention branch
        attn_out, _ = self.self_attn(
            query=self.norm1(x),
            key=self.norm1(x),
            value=self.norm1(x),
            attn_mask=attn_mask
        )
        x = x + self.dropout(attn_out)

        # Feedforward branch
        ffn_out = self.ffn(self.norm2(x))
        return x + self.dropout(ffn_out)

class TransforMole(pl.LightningModule):
    """
    Molecular Transformer model for SMILES generation.
    """

    def __init__(
            self,
            vocab_size: int,
            dim_model: int = 256,
            num_head: int = 8,
            num_layers: int = 6,
            dim_feedforward: int = 1024,
            lr: float = 1e-4,
            use_lora: bool = False,
            lora_rank: int = 8,
            lora_alpha: int = 16,
            pad_idx: int = 0,
            use_RePE: bool = False,
            RoPE_num_head: int = 8,
            max_len: int = 100,
            device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ):
        """
        Initialize the TransforMole model.

        Attributes:
        vocab_size: Size of the token vocabulary
        dim_model: Model dimension
        num_head: Number of attention heads
        num_layers: Number of transformer layers
        dim_feedforward: Feedforward dimension
        lr: Learning rate
        use_lora: Enable LoRA adaptation
        lora_rank: LoRA projection rank
        lora_alpha: LoRA scaling factor
        pad_idx: Padding token index
        """
        super().__init__()
        self.save_hyperparameters()
        self.max_len = max_len
        self._device = device
        self.embedding = nn.Embedding(vocab_size, dim_model, padding_idx=pad_idx, device=self.device)
        if use_RePE:
            self.pos_encoder = RoPE(dim_model, max_len=self.max_len, num_heads=RoPE_num_head,device=self.device)
        else:
            self.pos_encoder = PositionalEncoding(dim_model, max_len=self.max_len, device=self.device)

        self.transformer = nn.ModuleList([
            DecoderOnlyLayer(
                dim_model=dim_model,
                num_head=num_head,
                dim_feedforward=dim_feedforward,
                use_lora=use_lora,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                device=self.device
            ) for _ in range(num_layers)
        ])
        self._init_lora(use_lora, lora_rank, lora_alpha)
        self.fc_out = nn.Linear(dim_model, vocab_size, device=self.device)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)

    def _init_lora(self, use_lora: bool, rank: int, alpha: int) -> None:
        """Initialize LoRA parameters in decoder layers

        Args:
            use_lora: Enable LoRA adaptation mode
            rank: LoRA projection rank
            alpha: LoRA scaling factor
        """
        if not use_lora:
            return

        for layer in self.layers:
            # Initialize self-attention LoRA parameters
            if isinstance(layer.self_attn, DecoderOnlyLayer.LoRAAttention):
                attn = layer.self_attn
                for proj in [attn.q_proj, attn.k_proj, attn.v_proj, attn.out_proj]:
                    nn.init.kaiming_uniform_(proj.lora_A, a=math.sqrt(5))
                    nn.init.zeros_(proj.lora_B)
                    proj.scaling = alpha / rank
                    proj.linear.weight.requires_grad_(False)

            # Initialize FFN LoRA parameters
            ffn_linears = [layer.ffn[0], layer.ffn[2]]  # First and third layers are Linear
            for linear in ffn_linears:
                if isinstance(linear, LoRALinear):
                    nn.init.kaiming_uniform_(linear.lora_A, a=math.sqrt(5))
                    nn.init.zeros_(linear.lora_B)
                    linear.scaling = alpha / rank
                    linear.linear.weight.requires_grad_(False)

    def _create_mask(self, sz: int) -> Tensor:
        """Generate causal attention mask"""
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1).to(self.device)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Forward pass with attention handling"""
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        seq_len = input_ids.size(1)
        src_mask = self._create_mask(seq_len)

        embedded = self.embedding(input_ids)
        encoded = self.pos_encoder(embedded)

        for layer in self.transformer:
            encoded = layer(encoded, src_mask)

        return self.fc_out(encoded)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Tensor:
        """
        Training step with masked attention
        :param batch: Input batch, tuple of input_ids and attention_mask.
        :param batch_idx: The index of the batch.
        :return: Loss value.
        """
        input_ids = batch[0]
        attention_mask = batch[1]

        outputs = self(input_ids, attention_mask)
        shift_labels = input_ids[..., 1:].contiguous()
        shift_logits = outputs[..., :-1, :].contiguous()

        loss = self.loss_fn(
            shift_logits.view(-1, self.hparams.vocab_size),
            shift_labels.view(-1)
        )
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Tensor:
        """
        Validation step with masked attention
        :param batch: Input batch, tuple of input_ids and attention_mask.
        :param batch_idx: The index of the batch.
        :return: Loss value.
        """
        input_ids = batch[0]
        attention_mask = batch[1]

        outputs = self(input_ids, attention_mask)
        shift_labels = input_ids[..., 1:].contiguous()
        shift_logits = outputs[..., :-1, :].contiguous()

        loss = self.loss_fn(
            shift_logits.view(-1, self.hparams.vocab_size),
            shift_labels.view(-1)
        )
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Tensor:
        """
        Test step with masked attention
        :param batch: Input batch, tuple of input_ids and attention_mask.
        :param batch_idx: The index of the batch.
        :return: Loss value.
        """
        input_ids = batch[0]
        attention_mask = batch[1]

        outputs = self(input_ids, attention_mask)
        shift_labels = input_ids[..., 1:].contiguous()
        shift_logits = outputs[..., :-1, :].contiguous()

        loss = self.loss_fn(
            shift_logits.view(-1, self.hparams.vocab_size),
            shift_labels.view(-1)
        )
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizer with LoRA parameter filtering"""
        params = self.parameters() if not self.hparams.use_lora else [
            p for n, p in self.named_parameters() if 'lora_' in n
        ]
        return torch.optim.Adam(params, lr=self.hparams.lr)

    @torch.no_grad()
    def generate(
            self,
            num_samples: int = 100,
            max_length: int = 100,
            start_token: int = 1,
            end_token: int = 2,
            output_dir: str = GEN_PATH
    ) -> None:
        """Generate molecules and save as CSV"""
        self.eval()
        os.makedirs(output_dir, exist_ok=True)

        generated = []
        for _ in range(num_samples):
            tokens = [[start_token]]
            for _ in range(max_length):
                inputs = torch.tensor(tokens).to(self.device)
                logits = self(inputs, torch.ones_like(inputs).bool())
                next_token = logits[0, -1].argmax().item()

                if next_token == end_token:
                    break
                tokens[0].append(next_token)

            generated.append(tokens[0][1:])  # Remove start token

        with open(f"{output_dir}/generated.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(["SMILES"])
            for i,seq in enumerate(generated):
                writer.writerow([f'SMILES_{i}', seq])


# Full Workflow Example
if __name__ == "__main__":
    pass