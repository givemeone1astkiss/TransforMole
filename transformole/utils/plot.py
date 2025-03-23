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

    # Combine images into a grid and save
    for i in range(0, len(images), 100):
        grid_images = images[i:i + 100]
        img_grid = Image.new('RGB', (2000, 2000))
        for j, img in enumerate(grid_images):
            x = (j % 10) * 200
            y = (j // 10) * 200
            img_grid.paste(img, (x, y))
        img_grid.save(os.path.join(output_dir, f'molecules_{i // 100}.png'))

    # Print invalid molecule IDs
    if invalid_ids:
        print("Invalid molecule IDs:", invalid_ids)


def calculate_similarity(source_csv_path: str, target_csv_path: str) -> None:
    RDLogger.DisableLog('rdApp.*')
    source_data = pd.read_csv(source_csv_path)
    target_data = pd.read_csv(target_csv_path, header=None, names=['SMILES'])

    source_ids = source_data['ID'].values
    source_smiles = source_data['SMILES'].values
    target_smiles = target_data['SMILES'].values

    source_fps = []
    invalid_source_indices = []
    for i, s in tqdm(enumerate(source_smiles), total=len(source_smiles), desc='Processing source molecules'):
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            source_fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048))
        else:
            source_fps.append(None)
            invalid_source_indices.append(i)

    target_fps = []
    invalid_target_smiles = []
    for s in tqdm(target_smiles, desc='Processing target molecules'):
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            target_fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048))
        else:
            invalid_target_smiles.append(s)

    max_similarities = []
    for source_fp in tqdm(source_fps, desc='Calculating similarities'):
        if source_fp is not None:
            similarities = [DataStructs.TanimotoSimilarity(source_fp, target_fp) for target_fp in target_fps]
            max_similarities.append(max(similarities))
        else:
            max_similarities.append(np.nan)

    output_dir = os.path.join(OUTPUT_PATH, 'similarity')
    os.makedirs(output_dir, exist_ok=True)

    similarity_df = pd.DataFrame({'ID': source_ids, 'Max_Similarity': max_similarities})
    similarity_df.to_csv(os.path.join(output_dir, 'similarity.csv'), index=False)

    plt.hist([s for s in max_similarities if not np.isnan(s)], bins=20, edgecolor='black')
    plt.xlabel('Maximum Similarity')
    plt.ylabel('Frequency')
    plt.title('Similarity Distribution')
    plt.savefig(os.path.join(output_dir, 'similarity_histogram.png'))
    plt.close()

    if invalid_source_indices:
        print("Invalid source molecule indices:", invalid_source_indices)
    if invalid_target_smiles:
        print("Invalid target SMILES:", invalid_target_smiles)