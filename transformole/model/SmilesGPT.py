import os
import csv
from typing import final, List
import pytorch_lightning as pl
import yaml
from ..config import OUTPUT_PATH
from .lora import *
from .embeddings import *
from tqdm import tqdm

class DecoderOnlyLayer(nn.Module):
    """Decoder-only transformer layer without cross-attention

    Features:
    - Self-attention with LoRA support
    - Position-wise feedforward network
    - Pre-LayerNorm configuration
    - Residual connections

    Args:
        dim_model: Dimension of model embeddings
        num_head: Number of attention heads
        dim_feedforward: Dimension of FFN hidden layer
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(
            self,
            dim_model: int,
            num_head: int,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
            device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)
        self.dropout = nn.Dropout(dropout)

        # Self-attention components
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim_model,
            num_heads=num_head,
            batch_first=True,
            device=device
        )

        # Feedforward components
        self.ffn = nn.Sequential(
                nn.Linear(dim_model, dim_feedforward),
                nn.GELU(),
                nn.Linear(dim_feedforward, dim_model),
                nn.Dropout(0.1)
            )

    def forward(
            self,
            x: Tensor,
            attn_mask: Optional[Tensor] = None,
            key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass with residual connections"""
        # Self-attention branch
        attn_out, _ = self.self_attn(
            query=self.norm1(x),
            key=self.norm1(x),
            value=self.norm1(x),
            attn_mask=attn_mask,
            key_padding_mask = key_padding_mask
        )
        x = x + self.dropout(attn_out)

        # Feedforward branch
        ffn_out = self.ffn(self.norm2(x))
        return x + self.dropout(ffn_out)

@final
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
                device=self.device
            ) for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(dim_model, vocab_size, device=self.device)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)

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
            encoded = layer(encoded, attn_mask=src_mask, key_padding_mask=attention_mask.to(self.device))
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
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch= True)
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
        self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch= True)
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
        self.log("test_loss", loss, prog_bar=True, on_step=True, on_epoch= True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizer with LoRA parameter filtering"""
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    @staticmethod
    def _sample_molecules(model, device, start_token, end_token, unk_token, max_length, num_samples: int,
                          batch_size: int) -> List[List[int]]:
        generated = []
        num_batches = (num_samples + batch_size - 1) // batch_size  # Calculate the number of batches
        for _ in tqdm(range(num_batches), desc='Generating molecules'):
            current_batch_size = min(batch_size, num_samples - len(generated))
            tokens = torch.full((current_batch_size, 1), start_token, dtype=torch.long, device=device)
            for _ in range(max_length):
                logits = model(tokens, torch.ones_like(tokens).float())
                probs = torch.softmax(logits[:, -1, :], dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze()

                # Ensure the first token after the start token is not <EOS>
                if tokens.size(1) == 1:
                    while (next_tokens == end_token).any() or (next_tokens == unk_token).any():
                        next_tokens = torch.multinomial(probs, num_samples=1).squeeze()

                next_tokens[next_tokens == unk_token] = torch.multinomial(probs[next_tokens == unk_token],
                                                                          num_samples=1).squeeze()
                tokens = torch.cat([tokens, next_tokens.unsqueeze(-1)], dim=-1)
                if (next_tokens == end_token).all():
                    break
            for i in range(current_batch_size):
                generated.append(tokens[i, 1:].tolist())  # Remove start token
        return generated

    @torch.no_grad()
    def generate(
            self,
            num_samples: int,
            max_length: int,
            output_dir: str = f"{OUTPUT_PATH}generated",
            vocab_path: str = f"{OUTPUT_PATH}vocab/vocab.yaml",
            batch_size: int = 32
    ) -> str:
        """Generate molecules and save as CSV using batch sampling"""
        self.eval()
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Load vocabulary
        with open(vocab_path, 'r') as f:
            vocab = yaml.safe_load(f)
        idx_to_token = {v: k for k, v in vocab.items()}

        # Extract start and end token indices
        start_token = vocab['<BOS>']
        end_token = vocab['<EOS>']
        unk_token = vocab['<UNK>']

        # Sample molecules in batches
        generated = self._sample_molecules(self, self.device, start_token, end_token, unk_token, max_length,
                                           num_samples, batch_size)

        # Decode tokens to SMILES
        decoded_smiles = []
        for seq in generated:
            smiles = []
            for idx in seq:
                if idx == end_token:
                    break
                if idx != unk_token:
                    smiles.append(idx_to_token[idx])
            decoded_smiles.append(''.join(smiles))

        # Find the largest existing file number
        existing_files = os.listdir(output_dir)
        file_numbers = [int(f[7:-4]) for f in existing_files if f.startswith("smiles_") and f.endswith(".csv")]
        largest_existing_number = max(file_numbers, default=-1)

        # Write to CSV
        output_file = f"{output_dir}/smiles_{largest_existing_number + 1}.csv"
        with open(output_file, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "SMILES"])
            for i, smiles in enumerate(decoded_smiles):
                writer.writerow([f'SMILES_{i}', smiles])
        return output_file