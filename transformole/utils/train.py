import pytorch_lightning as pl
from transformole.config import OUTPUT_PATH
from pytorch_lightning.callbacks import ModelCheckpoint
import os
from .plot import calculate_valid_smiles_ratio, generate_molecule_images, calculate_atom_count_distribution, calculate_similarity

def get_train_logger():
    return pl.loggers.TensorBoardLogger(
        save_dir=f"{OUTPUT_PATH}/logs",
        name="train"
    )

def get_checkpoint_callback():
    return ModelCheckpoint(
        dirpath=f"{OUTPUT_PATH}/checkpoints",
        filename='checkpoint-{epoch:02d}',
        save_top_k=-1,
        verbose=True
    )

class MoleculeGenerationCallback(pl.Callback):
    def __init__(self, train_csv_path: str):
        super().__init__()
        self.tran_csv_path = train_csv_path

    def on_train_epoch_end(self, trainer, pl_module):
        # Generate 100 molecules
        output_dir = os.path.join(OUTPUT_PATH, 'generated')
        csv_path = pl_module.generate(num_samples=100, max_length=100, output_dir=output_dir, vocab_path=f'{OUTPUT_PATH}vocab/vocab.yaml')

        # Calculate metrics
        avg_atom_count = calculate_atom_count_distribution(csv_path)
        valid_ratio = calculate_valid_smiles_ratio(csv_path)
        avg_max_similarity = calculate_similarity(csv_path, self.tran_csv_path)

        # Log metrics
        trainer.logger.experiment.add_scalar('valid_smiles_ratio', valid_ratio, trainer.current_epoch)
        trainer.logger.experiment.add_scalar('average_atom_count', avg_atom_count, trainer.current_epoch)
        trainer.logger.experiment.add_scalar('average_max_similarity', avg_max_similarity, trainer.current_epoch)

        # Generate molecule images
        generate_molecule_images(csv_path)