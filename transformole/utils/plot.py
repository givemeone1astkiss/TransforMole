import os
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw,AllChem, DataStructs
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool, cpu_count
from PIL import Image

def generate_molecule_images(csv_path: str, output_path: str) -> None:
    """
    Generate molecule structure images from SMILES strings in a CSV file and save them in a grid format.
    Skip invalid molecules and print their IDs.
    :param output_path: The path to save the grid images.
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

    images = [Draw.MolToImage(mol, legend=legend, size=(200, 200)) for mol, legend in
              tqdm(zip(mols, legends), total=len(mols), desc='Generating images')]

    # Create a directory to save the images if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Find the largest existing folder number
    existing_folders = [f for f in os.listdir(output_path) if
                        os.path.isdir(os.path.join(output_path, f)) and f.startswith("images_")]
    folder_numbers = [int(f.split('_')[1]) for f in existing_folders]
    largest_existing_number = max(folder_numbers, default=-1)

    # Create a new folder with the next largest number
    new_folder_path = os.path.join(output_path, f'images_{largest_existing_number + 1}')
    os.makedirs(new_folder_path)

    # Create and save grids of 10x10 images
    grid_size = 10
    img_width, img_height = images[0].size
    num_images = len(images)
    num_grids = (num_images + grid_size * grid_size - 1) // (grid_size * grid_size)

    for grid_index in range(num_grids):
        grid_img = Image.new('RGB', (img_width * grid_size, img_height * grid_size))
        for i in range(grid_size * grid_size):
            img_index = grid_index * grid_size * grid_size + i
            if img_index >= num_images:
                break
            img = images[img_index]
            grid_x = (i % grid_size) * img_width
            grid_y = (i // grid_size) * img_height
            grid_img.paste(img, (grid_x, grid_y))
        grid_img.save(os.path.join(new_folder_path, f'molecules_{grid_index}.png'))

    # Print invalid molecule IDs
    if invalid_ids:
        print("Invalid molecule IDs:", invalid_ids)


def generate_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    else:
        return None

def calculate_similarity(source_csv_path: str, target_csv_path: str, output_path: str) -> float:
    RDLogger.DisableLog('rdApp.*')
    source_data = pd.read_csv(source_csv_path)
    target_data = pd.read_csv(target_csv_path, header=None, names=['SMILES'])

    source_ids = source_data['ID'].values

    # Check if source CSV has ECFP4 column
    invalid_source_smiles = []
    if 'ECFP4' in source_data.columns:
        source_fps = [DataStructs.cDataStructs.CreateFromBitString(str(fp)) for fp in source_data['ECFP4'].values]
    else:
        source_smiles = source_data['SMILES'].values
        with Pool(cpu_count()) as pool:
            source_fps = list(tqdm(pool.imap(generate_fingerprint, source_smiles), total=len(source_smiles), desc='Processing source molecules'))

    # Check if target CSV has ECFP4 column
    invalid_target_smiles = []
    if 'ECFP4' in target_data.columns:
        target_fps = [DataStructs.cDataStructs.CreateFromBitString(str(fp)) for fp in target_data['ECFP4'].values]
    else:
        target_smiles = target_data['SMILES'].values
        with Pool(cpu_count()) as pool:
            target_fps = list(tqdm(pool.imap(generate_fingerprint, target_smiles), total=len(target_smiles), desc='Processing target molecules'))

    max_similarities = []
    for source_fp in tqdm(source_fps, desc='Calculating similarities'):
        if source_fp is not None:
            similarities = [DataStructs.TanimotoSimilarity(source_fp, target_fp) for target_fp in target_fps if target_fp is not None]
            max_similarities.append(max(similarities))
        else:
            max_similarities.append(np.nan)

    os.makedirs(f"{output_path}similarity/", exist_ok=True)
    os.makedirs(f"{output_path}similarity_histogram/", exist_ok=True)
    existing_files = os.listdir(f"{output_path}similarity/")
    file_numbers = [int(f.split('_')[1].split('.')[0]) for f in existing_files if
                    f.startswith("similarity_") and f.endswith(".csv")]
    largest_existing_number = max(file_numbers, default=-1)
    similarity_df = pd.DataFrame({'ID': source_ids, 'Max_Similarity': max_similarities})
    similarity_df.to_csv(os.path.join(output_path, f'similarity/similarity_{largest_existing_number+1}.csv'), index=False)
    existing_files = os.listdir(f"{output_path}similarity_histogram/")
    file_numbers = [int(f.split('_')[2].split('.')[0]) for f in existing_files if
                    f.startswith("similarity_histogram_") and f.endswith(".png")]
    largest_existing_number = max(file_numbers, default=-1)
    plt.hist([s for s in max_similarities if not np.isnan(s)], bins=20, edgecolor='black')
    plt.xlabel('Maximum Similarity')
    plt.ylabel('Frequency')
    plt.title('Similarity Distribution')
    plt.savefig(os.path.join(output_path, f'similarity_histogram/similarity_histogram_{largest_existing_number+1}.png'))
    plt.close()

    if invalid_source_smiles:
        print("Invalid source SMILES:", invalid_source_smiles)
    if invalid_target_smiles:
        print("Invalid target SMILES:", invalid_target_smiles)

    print("Average maximum similarity:", np.nanmean(max_similarities))
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
        try:
            mol = Chem.MolFromSmiles(s)
            if mol is not None:
                valid_count += 1
        except Exception as e:
            print(f"Error processing molecule with SMILES{s}: {e}")

    return valid_count / total_count if total_count > 0 else 0.0

def calculate_atom_count_distribution(csv_path: str, output_path: str) -> float:
    """
    Calculate the atom count distribution of SMILES strings in a CSV file and generate a histogram.
    :param output_path: The path to save the files.
    :param csv_path: The path to the CSV file containing SMILES strings.
    :return: None
    """
    RDLogger.DisableLog('rdApp.*')
    data = pd.read_csv(csv_path)
    smiles = data['SMILES'].values

    atom_counts = []
    invalid_smiles = []

    for s in tqdm(smiles, desc='Calculating atom counts'):
        try :
            mol = Chem.MolFromSmiles(s)
            if mol is not None:
                atom_counts.append(mol.GetNumAtoms())
            else:
                invalid_smiles.append(s)
        except Exception as e:
            invalid_smiles.append(s)
            print(f"Error processing molecule with SMILES {s}: {e}")
            
    # Create a directory to save the histogram if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Find the largest existing file number
    existing_files = os.listdir(output_path)
    file_numbers = [int(f.split('_')[2].split('.')[0]) for f in existing_files if f.startswith("atom_count_") and f.endswith(".png")]
    largest_existing_number = max(file_numbers, default=-1)

    # Generate and save the histogram
    plt.hist(atom_counts, bins=20, edgecolor='black')
    plt.xlabel('Number of Atoms')
    plt.ylabel('Frequency')
    plt.title('Atom Count Distribution')
    plt.savefig(os.path.join(output_path, f'atom_count_{largest_existing_number + 1}.png'))
    plt.close()

    if invalid_smiles:
        print("Invalid SMILES strings:", invalid_smiles)

    print("Average atom count:", np.mean(atom_counts) if atom_counts else 0.0)
    return np.mean(atom_counts) if atom_counts else 0.0