import pytorch_lightning as pl
from transformole.config import OUTPUT_PATH
from pytorch_lightning.callbacks import ModelCheckpoint
import os
import torch

from .plot import calculate_valid_smiles_ratio, generate_molecule_images, calculate_atom_count_distribution, calculate_similarity

def get_train_logger():
    return pl.loggers.TensorBoardLogger(
        save_dir=f"{OUTPUT_PATH}/logs",
        name="train"
    )

def get_checkpoint_callback(name: str):
    return ModelCheckpoint(
        dirpath=f"{OUTPUT_PATH}/checkpoints/{name}",
        filename='checkpoint-{epoch:02d}',
        save_top_k=-1,
        verbose=True
    )

class MoleculeGenerationCallback(pl.Callback):
    def __init__(self, num_samples: int, train_csv_path: str, interval: int, name: str, version: int):
        super().__init__()
        self.tran_csv_path = train_csv_path
        self.num_samples = num_samples
        self.interval = interval
        self.name = name
        self.version = version

    def on_train_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.interval == 0:
            # Generate 100 molecules
            output_dir = os.path.join(OUTPUT_PATH, f'generated/{self.name}-V{self.version}')
            csv_path = pl_module.generate(num_samples=self.num_samples, max_length=100, output_dir=output_dir, vocab_path=f'{OUTPUT_PATH}vocab/vocab.yaml')

            # Calculate metrics
            avg_atom_count = calculate_atom_count_distribution(csv_path, output_path= f'{OUTPUT_PATH}atom_count/{self.name}-V{self.version}')
            valid_ratio = calculate_valid_smiles_ratio(csv_path)
            avg_max_similarity = calculate_similarity(csv_path, self.tran_csv_path, output_path= f'{OUTPUT_PATH}similarity/{self.name}-V{self.version}')

            # Log metrics
            trainer.logger.experiment.add_scalar('valid_smiles_ratio', valid_ratio, trainer.current_epoch)
            trainer.logger.experiment.add_scalar('average_atom_count', avg_atom_count, trainer.current_epoch)
            trainer.logger.experiment.add_scalar('average_max_similarity', avg_max_similarity, trainer.current_epoch)

            # Generate molecule images
            generate_molecule_images(csv_path, output_path=f'{OUTPUT_PATH}image/{self.name}-V{self.version}')


def get_trainer(name: str, version: int, train_csv_path: str, max_epoch: int, num_samples=128, interval: int = 1):
    logger = pl.loggers.TensorBoardLogger(
        save_dir=f"{OUTPUT_PATH}/logs",
        name=name,
        version=f'Version {version}',
    )

    checkpoint = ModelCheckpoint(
        dirpath=f"{OUTPUT_PATH}/checkpoints",
        filename=f'checkpoint-epoch:02d-f{name}',
        save_top_k=-1,
        verbose=True
    )

    molecule_generation_callback = MoleculeGenerationCallback(
        num_samples = num_samples,
        train_csv_path = train_csv_path,
        interval = interval,
        name = name,
        version= version
    )

    return pl.Trainer(
        accelerator="auto",
        devices=-1 if torch.cuda.is_available() else 1,
        precision="16-mixed",
        max_epochs=max_epoch,
        enable_progress_bar=True,
        logger=logger,
        callbacks=[checkpoint, molecule_generation_callback]
    )