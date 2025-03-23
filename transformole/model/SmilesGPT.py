import os
import csv
import pytorch_lightning as pl
import yaml
from ..config import OUTPUT_PATH
from .lora import *
from .embeddings import *

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

    import json

    @torch.no_grad()
    def generate(
            self,
            num_samples: int,
            max_length: int,
            output_dir: str = OUTPUT_PATH,
            vocab_path: str = None
    ) -> None:
        """Generate molecules and save as CSV"""
        self.eval()
        os.makedirs(output_dir, exist_ok=True)

        # Load vocabulary
        with open(vocab_path, 'r') as f:
            vocab = yaml.safe_load(f)
        idx_to_token = {v: k for k, v in vocab.items()}

        # Extract start and end token indices
        start_token = vocab['<BOS>']
        end_token = vocab['<EOS>']

        generated = []
        for _ in range(num_samples):
            tokens = [[start_token]]
            for _ in range(max_length):
                inputs = torch.tensor(tokens).to(self.device)
                logits = self(inputs, torch.ones_like(inputs).bool())
                probs = torch.softmax(logits[0, -1], dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()

                if next_token == end_token:
                    break
                tokens[0].append(next_token)

            generated.append(tokens[0][1:])  # Remove start token

        # Decode tokens to SMILES
        decoded_smiles = [''.join([idx_to_token[idx] for idx in seq]) for seq in generated]

        # Write to CSV
        with open(f"{output_dir}/generated.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(["ID","SMILES"])
            for i, smiles in enumerate(decoded_smiles):
                writer.writerow([f'SMILES_{i}', smiles])
