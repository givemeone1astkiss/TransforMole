import os
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw,AllChem, DataStructs
from PIL import Image
from ..config.glob import OUTPUT_PATH
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

def generate_molecule_images(csv_path: str) -> None:
    """
    Generate molecule structure images from SMILES strings in a CSV file and save them in a grid format.
    Skip invalid molecules and print their IDs.
    :param csv_path: The path to the CSV file containing molecule IDs and SMILES strings.
    :return: None
    """
    RDLogger.DisableLog('rdApp.*')
    data = pd.read_csv(csv_path)
    ids = data['ID'].values
    smiles = data['SMILES'].values
    mols = []
    legends = []
    invalid_ids = []

    for id_, s in zip(ids, smiles):
        try:
            mol = Chem.MolFromSmiles(s)
            if mol is not None:
                mols.append(mol)
                legends.append(f'ID: {id_}')
            else:
                invalid_ids.append(id_)
        except Exception as e:
            invalid_ids.append(id_)
            print(f"Error processing molecule ID {id_}: {e}")

    images = [Draw.MolToImage(mol, legend=legend, size=(200, 200)) for mol, legend in tqdm(zip(mols, legends), total=len(mols), desc='Generating images')]

    # Create a directory to save the images if it doesn't exist
    output_dir = os.path.join(OUTPUT_PATH, 'image')
    os.makedirs(output_dir, exist_ok=True)

    # Find the largest existing file number
    existing_files = os.listdir(output_dir)
    file_numbers = [int(f.split('_')[1].split('.')[0]) for f in existing_files if f.startswith("molecules_") and f.endswith(".png")]
    largest_existing_number = max(file_numbers, default=-1)

    # Combine images into a grid and save
    for i in range(0, len(images), 100):
        grid_images = images[i:i + 100]
        img_grid = Image.new('RGB', (2000, 2000))
        for j, img in enumerate(grid_images):
            x = (j % 10) * 200
            y = (j // 10) * 200
            img_grid.paste(img, (x, y))
        img_grid.save(os.path.join(output_dir, f'molecules_{largest_existing_number + 1 + (i // 100)}.png'))

    # Print invalid molecule IDs
    if invalid_ids:
        print("Invalid molecule IDs:", invalid_ids)

    # Print invalid molecule IDs
    if invalid_ids:
        print("Invalid molecule IDs:", invalid_ids)


def compute_max_similarities(source_fps, target_fps):
    max_similarities = []
    for source_fp in tqdm(source_fps, desc='Calculating similarities'):
        if source_fp is not None:
            similarities = [DataStructs.TanimotoSimilarity(source_fp, target_fp) for target_fp in target_fps]
            max_similarities.append(max(similarities))
        else:
            max_similarities.append(np.nan)
    return max_similarities

def calculate_similarity(source_csv_path: str, target_csv_path: str) -> float:
    """
    Calculate the maximum similarity between molecules in the source CSV file and target CSV file.
    If the source or target CSV file contains an ECFP4 column, use the fingerprints from this column.
    Otherwise, calculate the fingerprints and save them to the ECFP4 column in the respective CSV file.

    :param source_csv_path: The path to the source CSV file containing molecule IDs and SMILES strings.
    :param target_csv_path: The path to the target CSV file containing SMILES strings.
    :return: None
    """
    RDLogger.DisableLog('rdApp.*')
    source_data = pd.read_csv(source_csv_path)
    target_data = pd.read_csv(target_csv_path, header=None, names=['SMILES'])

    source_ids = source_data['ID'].values

    # Check if source CSV has ECFP4 column
    if 'ECFP4' in source_data.columns:
        source_fps = [DataStructs.cDataStructs.CreateFromBitString(str(fp)) for fp in source_data['ECFP4'].values]
    else:
        source_smiles = source_data['SMILES'].values
        source_fps = []
        invalid_source_smiles = []
        for s in tqdm(source_smiles, desc='Processing source molecules'):
            mol = Chem.MolFromSmiles(s)
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
                source_fps.append(fp)
                source_data.loc[source_data['SMILES'] == s, 'ECFP4'] = DataStructs.BitVectToText(fp)
            else:
                invalid_source_smiles.append(s)

        # Save the updated source CSV with ECFP4 column
        source_data.to_csv(source_csv_path, index=False)

    # Check if target CSV has ECFP4 column
    if 'ECFP4' in target_data.columns:
        target_fps = [DataStructs.cDataStructs.CreateFromBitString(str(fp)) for fp in target_data['ECFP4'].values]
    else:
        target_smiles = target_data['SMILES'].values
        target_fps = []
        invalid_target_smiles = []
        for s in tqdm(target_smiles, desc='Processing target molecules'):
            mol = Chem.MolFromSmiles(s)
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
                target_fps.append(fp)
                target_data.loc[target_data['SMILES'] == s, 'ECFP4'] = DataStructs.BitVectToText(fp)
            else:
                invalid_target_smiles.append(s)

        # Save the updated target CSV with ECFP4 column
        target_data.to_csv(target_csv_path, index=False)

    max_similarities = compute_max_similarities(source_fps, target_fps)

    output_dir = os.path.join(OUTPUT_PATH, 'similarity')
    os.makedirs(output_dir, exist_ok=True)

    # Find the largest existing file number
    existing_files = os.listdir(output_dir)
    file_numbers = [int(f.split('_')[1].split('.')[0]) for f in existing_files if f.startswith("similarity_") and f.endswith(".csv")]
    largest_existing_number = max(file_numbers, default=-1)

    similarity_df = pd.DataFrame({'ID': source_ids, 'Max_Similarity': max_similarities})
    similarity_df.to_csv(os.path.join(output_dir, f'similarity_{largest_existing_number + 1}.csv'), index=False)

    plt.hist([s for s in max_similarities if not np.isnan(s)], bins=20, edgecolor='black')
    plt.xlabel('Maximum Similarity')
    plt.ylabel('Frequency')
    plt.title('Similarity Distribution')
    plt.savefig(os.path.join(output_dir, f'similarity_histogram_{largest_existing_number + 1}.png'))
    plt.close()

    if invalid_source_smiles:
        print("Invalid source SMILES:", invalid_source_smiles)
    if invalid_target_smiles:
        print("Invalid target SMILES:", invalid_target_smiles)
    return np.nanmean(max_similarities)


def calculate_valid_smiles_ratio(csv_path: str) -> float:
    """
    Calculate the ratio of valid SMILES strings in a CSV file.
    :param csv_path: The path to the CSV file containing SMILES strings.
    :return: The ratio of valid SMILES strings.
    """
    RDLogger.DisableLog('rdApp.*')
    data = pd.read_csv(csv_path)
    smiles = data['SMILES'].values

    valid_count = 0
    total_count = len(smiles)

    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            valid_count += 1

    return valid_count / total_count if total_count > 0 else 0.0

def calculate_atom_count_distribution(csv_path: str) -> float:
    """
    Calculate the atom count distribution of SMILES strings in a CSV file and generate a histogram.
    :param csv_path: The path to the CSV file containing SMILES strings.
    :return: None
    """
    RDLogger.DisableLog('rdApp.*')
    data = pd.read_csv(csv_path)
    smiles = data['SMILES'].values

    atom_counts = []
    invalid_smiles = []

    for s in tqdm(smiles, desc='Calculating atom counts'):
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            atom_counts.append(mol.GetNumAtoms())
        else:
            invalid_smiles.append(s)

    # Create a directory to save the histogram if it doesn't exist
    output_dir = os.path.join(OUTPUT_PATH, 'atom_count')
    os.makedirs(output_dir, exist_ok=True)

    # Find the largest existing file number
    existing_files = os.listdir(output_dir)
    file_numbers = [int(f.split('_')[1].split('.')[0]) for f in existing_files if f.startswith("atom_count_") and f.endswith(".png")]
    largest_existing_number = max(file_numbers, default=-1)

    # Generate and save the histogram
    plt.hist(atom_counts, bins=20, edgecolor='black')
    plt.xlabel('Number of Atoms')
    plt.ylabel('Frequency')
    plt.title('Atom Count Distribution')
    plt.savefig(os.path.join(output_dir, f'atom_count_{largest_existing_number + 1}.png'))
    plt.close()

    if invalid_smiles:
        print("Invalid SMILES strings:", invalid_smiles)

    return np.mean(atom_counts) if atom_counts else 0.0