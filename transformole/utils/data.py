from typing import Tuple

from torch.utils.data import Dataset, DataLoader, random_split
from moses import get_dataset
from transformers import GPT2Tokenizer
from pytorch_lightning import LightningDataModule
from collections import Counter
import torch
import numpy as np
from rdkit import Chem
from tqdm import tqdm
import os
from ..config import DATA_PATH

class SmilesDataset(Dataset):
    """
    Dataset class for SMILES strings.
    """
    def __init__(self, smiles_list: list, tokenizer, max_length=100):
        self.tokenizer = tokenizer
        self.smiles = smiles_list
        self.max_length = max_length

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smile = self.smiles[idx]
        encoding = self.tokenizer.encode_plus(
            smile,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }


def canonicalize(smiles):
    """SMILES normalization"""
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def randomize_smiles(smiles, num_versions=3):
    """generate random SMILES strings"""
    mol = Chem.MolFromSmiles(smiles)
    return [
        Chem.MolToSmiles(mol, doRandom=True, canonical=False)
        for _ in range(num_versions)
    ]


def collate_fn(batch):
    """自定义批处理函数"""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    return {'input_ids': input_ids, 'attention_mask': attention_mask}


def process_data(smiles_list, augment=False):
    """
    Data preprocessing pipeline
    :param smiles_list: list of SMILES strings
    :param augment: whether to apply data augmentation
    :return: list of processed SMILES strings
    """

    # Data normalization.
    valid_smiles = [canonicalize(s) for s in smiles_list]
    valid_smiles = [s for s in valid_smiles if s is not None]

    # Data augmentation.
    if augment:
        augmented = []
        for s in valid_smiles:
            augmented.append(s)
            augmented.extend(randomize_smiles(s))
        valid_smiles = augmented

    # Add special tokens.
    valid_smiles = [f'<BOS> {s} <EOS>' for s in valid_smiles]

    # Remove duplicates.
    return list(set(valid_smiles))


class SmilesDataModule(LightningDataModule):
    """
    PyTorch Lightning DataModule for Moses datasets.
    """
    @classmethod
    def from_csv(cls, data_path=DATA_PATH):
        """
        Initialize DataModule from CSV files
        :param data_path: Where the CSV files are stored
        :return: SmilesDataModule instance
        """
        train = open(f'{data_path}train.csv').read().split('\n')
        test = open(f'{data_path}test.csv').read().split('\n')
        return cls(raw_data=(train, test))

    @classmethod
    def from_moses(cls):
        """
        Initialize DataModule from Moses datasets
        :return: SmilesDataModule instance
        """
        return cls(raw_data=(get_dataset('train'), get_dataset('test')))

    def __init__(
            self,
            batch_size: int = 32,
            max_seq_len: int = 128,
            num_workers: int = 4,
            augment: bool = True,
            raw_data: Tuple[list, list] = None
    ):
        super().__init__()
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.num_workers = num_workers
        self.augment = augment

        # Initialize tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        special_tokens = {
            'bos_token': '<BOS>',
            'eos_token': '<EOS>',
            'pad_token': '<PAD>',
            'additional_special_tokens': ['<MASK>']
        }
        self.tokenizer.add_special_tokens(special_tokens)
        self.vocab_size = len(self.tokenizer)
        self.raw_data = raw_data
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def setup(self, stage: str = None):
        """
        Load and preprocess data
        :param stage: The stage of the experiment
        :return:
        """
        # Load raw data
        if stage == 'fit' or stage is None:
            self.train_data, self.val_data = random_split(
                self.raw_data[0],
                [int(0.9 * len(self.raw_data[0])), int(0.1 * len(self.raw_data[0]))]
            )
            self.train_data = process_data(self.train_data, augment=self.augment)
            self.val_data = process_data(self.val_data)
        elif stage == 'test' or stage is None:
            self.test_data = process_data(get_dataset('test'))
        else:
            raise ValueError(f'Invalid stage: {stage}')

    def train_dataloader(self) -> DataLoader:
        return self.create_dataloader(self.train_data, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self.create_dataloader(self.val_data)

    def test_dataloader(self) -> DataLoader:
        return self.create_dataloader(self.test_data)

    def create_dataloader(self, data, shuffle=False) -> DataLoader:
        dataset = SmilesDataset(
            data,
            self.tokenizer,
            max_length=self.max_seq_len
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )


def analyze_dataset(smiles_list: list)-> tuple:
    """
    Analyze the dataset
    :param smiles_list: list of SMILES strings
    :return: tuple of average length, std, max, min, and atom types distribution
    """
    lengths = [len(s) for s in smiles_list]
    print(f"Average length: {np.mean(lengths):.2f} ± {np.std(lengths):.2f}")
    print(f"Max length: {max(lengths)}")
    print(f"Min length: {min(lengths)}")

    atoms = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol:
            atoms.extend([a.GetSymbol() for a in mol.GetAtoms()])
    print("\nAtom types distribution:")
    print(Counter(atoms).most_common(10))
    return np.mean(lengths), np.std(lengths), max(lengths), min(lengths), Counter(atoms).most_common(10)

def write_csv(stage: str):
    """
    Generate CSV file for Moses dataset
    :param stage: 'train', 'val', or 'test'
    """
    data = tqdm(get_dataset(stage), desc=f'Processing {stage} data')
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
    with open(f'{DATA_PATH}{stage}.csv', 'w') as f:
        for s in data:
            f.write(f'{s}\n')

    print(f'{stage}.csv saved successfully!')

if __name__ == '__main__':
    pass