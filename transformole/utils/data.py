from typing import Tuple
from torch.utils.data import Dataset, DataLoader, TensorDataset
# from moses import get_dataset
from pytorch_lightning import LightningDataModule
from collections import Counter, defaultdict
import torch
import numpy as np
from rdkit import Chem
from tqdm import tqdm
import os
from ..config import DATA_PATH, OUTPUT_PATH
import re
import yaml
from multiprocessing import Pool, cpu_count
import csv

class SmilesDataset(Dataset):
    """
    Dataset class for SMILES strings.
    """

    def __init__(self, smiles_list: list, tokenizer, max_length=75):
        self.tokenizer = tokenizer
        self.smiles = smiles_list
        self.max_length = max_length

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smile = [self.smiles[idx]]
        encoding = self.tokenizer.encode(
            smile,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
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


class SmilesDataModule(LightningDataModule):
    """
    PyTorch Lightning DataModule for Moses datasets.
    """

    @classmethod
    def from_moses(cls, data_path=DATA_PATH):
        """
        Initialize DataModule from Moses datasets
        :return: SmilesDataModule instance
        """
        train = open(f"{data_path}/moses/train.csv").read().split("\n")[:-5000]
        valid = open(f"{data_path}/moses/train.csv").read().split("\n")[5000:-1]
        test = open(f"{data_path}/moses/test.csv").read().split("\n")
        return cls(raw_data=(train, valid, test))

    @classmethod
    def from_guacamol(cls, data_path=DATA_PATH):
        """
        Initialize DataModule from Guacamol datasets
        :return: SmilesDataModule instance
        """
        train = open(f"{data_path}/guacamol/train.csv").read().split("\n")
        valid = open(f"{data_path}/guacamol/valid.csv").read().split("\n")
        test = open(f"{data_path}/guacamol/test.csv").read().split("\n")
        return cls(raw_data=(train, valid, test))

    @classmethod
    def from_anesthesia(cls, data_path=DATA_PATH):
        """
        Initialize DataModule from Anesthesia datasets
        :return: SmilesDataModule instance
        """
        train = open(f'{data_path}/anesthesia/finetune.csv').read().split('\n')
        valid = open(f"{data_path}/guacamol/valid.csv").read().split("\n")
        test = open(f"{data_path}/guacamol/test.csv").read().split("\n")
        return cls(raw_data=(train, valid, test))

    def __init__(
        self,
        batch_size: int = 256,
        max_seq_len: int = 100,
        num_workers: int = 4,
        augment: bool = True,
        raw_data: Tuple[list, list, list] = None,
        load_vocab = True,
        padding_smiles: bool = True,
        truncation_smiles: bool = False,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.num_workers = num_workers
        self.augment = augment

        # Initialize tokenizer
        self.raw_data = raw_data
        self.tokenizer = SmilesTokenizer(load_vocab=load_vocab)
        self.vocab_size = len(self.tokenizer)
        self.train_data = None
        self.valid_data = None
        self.test_data = None
        self.padding_smiles = padding_smiles
        self.truncation_smiles = truncation_smiles

    def setup(self, stage: str = None):
        """
        Load and preprocess data
        :param stage: The stage of the experiment
        :return: None
        """
        # Load raw data
        if stage == "fit" or stage is None:
            self.train_data = TensorDataset(*self.tokenizer.encode(self.raw_data[0], self.max_seq_len, padding=self.padding_smiles, truncation=self.truncation_smiles, return_tensors="pt"))
            self.valid_data = TensorDataset(*self.tokenizer.encode(self.raw_data[1], self.max_seq_len, padding=self.padding_smiles, truncation=self.truncation_smiles, return_tensors="pt"))
        if stage == "test" or stage is None:
            self.valid_data = TensorDataset(*self.tokenizer.encode(self.raw_data[2], self.max_seq_len, padding=self.padding_smiles, truncation=self.truncation_smiles, return_tensors="pt"))

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def valid_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valid_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )


def analyze_dataset(smiles_list: list) -> tuple:
    """
    Analyze the dataset
    :param smiles_list: list of SMILES strings
    :return: tuple of average length, std, max, min, and atom types distribution
    """
    lengths = [len(s) for s in tqdm(smiles_list, desc="Analyzing lengths")]
    print(f"Average length: {np.mean(lengths):.2f} ± {np.std(lengths):.2f}")
    print(f"Max length: {max(lengths)}")
    print(f"Min length: {min(lengths)}")

    atoms = []
    for s in tqdm(smiles_list, desc="Analyzing atom types"):
        mol = Chem.MolFromSmiles(s)
        if mol:
            atoms.extend([a.GetSymbol() for a in mol.GetAtoms()])
    print("\nAtom types distribution:")
    print(Counter(atoms).most_common(10))
    return (
        np.mean(lengths),
        np.std(lengths),
        max(lengths),
        min(lengths),
        Counter(atoms).most_common(10),
    )


# def write_csv(stage: str):
#     """
#     Generate CSV file for Moses dataset
#     :param stage: 'train', 'valid', or 'test'
#     """
#     data = tqdm(get_dataset(stage), desc=f"Saving {stage} data")
#     if not os.path.exists(f"{DATA_PATH}moses/"):
#         os.makedirs(f"{DATA_PATH}moses/")
#     with open(f"{DATA_PATH}moses/{stage}.csv", "w") as f:
#         for s in data:
#             f.write(f"{s}\n")

#     print(f"{stage}.csv saved successfully!")


def smiles_to_csv(smiles_path: str, csv_path: str):
    """
    Convert SMILES file to CSV file
    :param smiles_path: path to SMILES file
    :param csv_path: path to CSV file
    """
    smiles = tqdm(open(smiles_path).read().split("\n"), desc="Processing SMILES")
    with open(csv_path, "w") as f:
        for s in smiles:
            f.write(f"{s}\n")

    print(f"{csv_path} saved successfully!")


class SmilesTokenizer:
    """
    SMILES tokenizer class.
    """
    def __init__(
        self, special_tokens=None, coverage_threshold=0.95, max_vocab_size=100, load_vocab=False
    ):
        self.special_tokens = special_tokens or ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
        self.coverage_threshold = coverage_threshold
        self.max_vocab_size = max_vocab_size
        self.inverse_vocab = {}
        self._regex = re.compile(
            r"($$.*?$$|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\||$|$|\.|=|#|-|\+|\\|/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9]|.)"
        )
        if load_vocab is True:
            self.load_vocab(path=OUTPUT_PATH)
        elif type(load_vocab) is str:
            self.load_vocab(path=load_vocab)
        else:
            self.vocab = {}

    def __len__(self):
        return len(self.vocab)

    def build_vocab(self, smiles_list):
        """
        Build vocabulary from SMILES list for vocab building.
        :param smiles_list: list of SMILES strings
        :return: None
        """
        token_counts = defaultdict(int)
        total_count = 0

        for smile in smiles_list:
            tokens = self._tokenize(smile)
            for token in tokens:
                token_counts[token] += 1
                total_count += 1

        # Add special tokens to counts to ensure they are included in the vocab.
        for token in self.special_tokens:
            token_counts[token] += 1
            total_count += 1

        # Sort tokens by frequency
        sorted_tokens = sorted(token_counts.items(), key=lambda x: -x[1])

        # Calculate cutoff
        cumulative = 0
        cutoff = len(sorted_tokens)
        for i, (token, count) in enumerate(sorted_tokens):
            cumulative += count
            if cumulative / total_count >= self.coverage_threshold:
                cutoff = i + 1
                break

        cutoff = min(cutoff, self.max_vocab_size)

        # Form final vocab
        final_vocab = set()
        # Add high-frequency tokens
        for token, _ in sorted_tokens[:cutoff]:
            final_vocab.add(token)
        # Add special tokens
        for token in self.special_tokens:
            final_vocab.add(token)

        # Create vocab list
        vocab_list = self.special_tokens + [
            token for token, _ in sorted_tokens if token not in self.special_tokens
        ]
        vocab_list = vocab_list[: self.max_vocab_size]

        self.vocab = {token: idx for idx, token in enumerate(vocab_list)}
        self.inverse_vocab = {idx: token for idx, token in enumerate(vocab_list)}
        # Print summary
        actual_coverage = (
            sum(count for token, count in sorted_tokens[:cutoff]) / total_count
        )
        print(f"Vocab size: {len(self.vocab)}")
        print(f"Actual coverage: {actual_coverage:.2%}")
        print(f"High frequency tokens: {vocab_list[:10]}")
        self.save_vocab()
        return self.vocab

    def save_vocab(self, path=OUTPUT_PATH):
        """
        Save vocabulary to file.
        :param path: Path to save the vocabulary.
        :return: None
        """
        print('Saving vocabulary...')
        if not os.path.exists(f'{path}vocab'):
            os.makedirs(f'{path}vocab')
        with open(f'{path}vocab/vocab.yaml', "w") as f:
            yaml.dump(self.vocab, f)
        print('Vocabulary saved successfully.')

    def load_vocab(self, path):
        """
        Load vocabulary from file.
        :param path: Path to load the vocabulary.
        :return: None
        """
        if not os.path.exists(f'{path}vocab/vocab.yaml'):
            raise FileNotFoundError("Vocabulary file not found.")
        else:
            with open(f'{path}vocab/vocab.yaml', "r") as f:
                self.vocab = yaml.load(f, Loader=yaml.FullLoader)
        self.inverse_vocab = {idx: token for token, idx in self.vocab.items()}
        print('Vocabulary loaded successfully.')

    def _tokenize(self, smile):
        return [token for token in self._regex.findall(smile) if token]

    def _encode_smiles(self, smile):
        tokens = [self.vocab["<BOS>"]] + [self.vocab.get(token, self.vocab["<UNK>"]) for token in smile] + [self.vocab["<EOS>"]]
        return tokens

    def _pad_sequence(self, args):
        seq, max_len = args
        padded_seq = seq + [self.vocab["<PAD>"]] * (max_len - len(seq))
        mask = [1] * len(seq) + [0] * (max_len - len(seq))
        return padded_seq, mask

    def encode(self, smiles_list, max_length, padding, truncation, return_tensors):
        if truncation:
            smiles_list = [smile[:max_length - 2] for smile in smiles_list]
        else:
            smiles_list = [smile for smile in smiles_list if len(smile) <= max_length-2]

        with Pool(cpu_count()) as pool:
            encoded = list(tqdm(pool.imap(
                self._encode_smiles,
                smiles_list
            ), total=len(smiles_list), desc="Encoding SMILES"))

        if padding:
            with Pool(cpu_count()) as pool:
                padded_results = list(tqdm(pool.imap(
                    self._pad_sequence,
                    [(seq, max_length) for seq in encoded]
                ), total=len(encoded), desc="Padding sequences"))
            padded, masks = zip(*padded_results)
        else:
            padded = encoded
            masks = [[1] * len(seq) for seq in encoded]

        if return_tensors == "pt":
            return (
                torch.tensor(padded, dtype=torch.long).squeeze(),
                torch.tensor(masks, dtype=torch.float).squeeze(),
            )
        else:
            return (
                np.array(padded),
                np.array(masks),
            )

def sdfs_to_csv(directory_path: str):
    smiles_list = []

    # Loop through all files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".sdf"):
            sdf_path = os.path.join(directory_path, filename)
            # Parse the SDF file
            suppl = Chem.SDMolSupplier(sdf_path)
            for mol in suppl:
                if mol is not None:
                    smiles = Chem.MolToSmiles(mol)
                    smiles_list.append(smiles)

    # Write SMILES to a CSV file
    output_csv_path = os.path.join(directory_path, "smiles.csv")
    with open(output_csv_path, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["SMILES"])
        for smiles in smiles_list:
            writer.writerow([smiles])