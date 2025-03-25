from transformole.model import *
from transformole.utils import *
from torch.utils.data import DataLoader, TensorDataset
from transformole.utils.train import *
import pytorch_lightning as pl

if __name__=="__main__":
    # Sample dataset structure (should be DataLoader instances)
    Tokenizer = SmilesTokenizer(load_vocab=True)
    train_data = open(f"{DATA_PATH}/moses/train.csv").read().split("\n")
    train_data = TensorDataset(*Tokenizer.encode(smiles_list=train_data, max_length=100, padding=True, truncation=True, return_tensors="pt"))
    train_data = DataLoader(dataset=train_data, batch_size=128, shuffle=True, num_workers=19, persistent_workers=True)

    model = TransforMole(
        dim_model=256,
        vocab_size=29
    )

    # 设置日志记录器
    logger = pl.loggers.TensorBoardLogger(
            save_dir=f"{OUTPUT_PATH}/logs",
            name="pretrain",
            version = 'Version 1',
        )

    checkpoint = ModelCheckpoint(
            dirpath=f"{OUTPUT_PATH}/checkpoints",
            filename='checkpoint-{epoch:02d}',
            save_top_k=-1,
            verbose=True
        )

    molecule_generation_callback = train.MoleculeGenerationCallback(train_csv_path=f"{DATA_PATH}/moses/train.csv")

    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        precision="16-mixed",
        max_epochs=5,
        enable_progress_bar=True,
        logger=logger,
        callbacks=[checkpoint, molecule_generation_callback]
    )
    trainer.fit(model, train_data)