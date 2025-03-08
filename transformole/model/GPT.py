import torch
import pytorch_lightning as pl
from transformers import GPT2Config, GPT2LMHeadModel
from ..utils.data import SmilesDataset, SmilesDataModule


class SmilesGPT(pl.LightningModule):
    def __init__(self, vocab_size=50257, d_model=768, n_layer=12, n_head=12):
        super().__init__()
        self.config = GPT2Config(
            vocab_size=vocab_size,
            n_embd=d_model,
            n_layer=n_layer,
            n_head=n_head
        )
        self.model = GPT2LMHeadModel(self.config)
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask=attention_mask)

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = input_ids.clone()
        labels[labels == self.model.config.pad_token_id] = -100

        outputs = self(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        loss = self.loss_fn(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = input_ids.clone()
        labels[labels == self.model.config.pad_token_id] = -100

        outputs = self(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        loss = self.loss_fn(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )
        self.log('val_loss', loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4)


def pretrain():
    # 初始化组件
    dm = SmilesDataModule(batch_size=64)
    model = SmilesGPT(vocab_size=len(dm.tokenizer))

    # 训练配置
    trainer = pl.Trainer(
        max_epochs=10,
        gpus=1 if torch.cuda.is_available() else 0,
        accelerator='auto',
        accumulate_grad_batches=2,
        gradient_clip_val=1.0
    )

    # 开始训练
    trainer.fit(model, dm)


if __name__ == '__main__':
    pretrain()