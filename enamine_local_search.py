import pandas as pd
import bz2
from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols


# Load Enamine dataset from a compressed bz2 file
def load_enamine_dataset(file_path, smiles_column="SMILES", chunk_size=100):
    with bz2.open(file_path, "rt") as file:
        # Read the file in chunks
        chunk_reader = pd.read_csv(file, sep="\t", chunksize=chunk_size)  # Assuming tab-delimited format

        smiles_list = []  # To store SMILES strings

        for chunk in chunk_reader:
            # Check if the smiles_column exists in the chunk
            if smiles_column not in chunk.columns:
                raise ValueError(f"Column '{smiles_column}' not found in the dataset.")

            # Append the non-null SMILES values to the list
            smiles_list.extend(chunk[smiles_column].dropna().tolist())

    return smiles_list


# Compute Tanimoto similarity
def compute_similarity(target_fp, molecule_fp):
    return DataStructs.FingerprintSimilarity(target_fp, molecule_fp)


# Find similar molecules
def find_similar_molecules(target_smile, dataset_smiles, similarity_threshold=0.7):
    target_mol = Chem.MolFromSmiles(target_smile)
    if not target_mol:
        raise ValueError("Invalid SMILES string for the target molecule.")

    target_fp = FingerprintMols.FingerprintMol(target_mol)
    similar_molecules = []

    for smile in dataset_smiles:
        mol = Chem.MolFromSmiles(smile)
        if mol:
            mol_fp = FingerprintMols.FingerprintMol(mol)
            similarity = compute_similarity(target_fp, mol_fp)
            if similarity >= similarity_threshold:
                similar_molecules.append((smile, similarity))

    # Sort by similarity in descending order
    similar_molecules.sort(key=lambda x: x[1], reverse=True)
    return similar_molecules


# Main function
if __name__ == "__main__":
    DATA_PATH = 'benchmark_datasets/CACHE5/20240430_MCHR1_splitted_RJ.csv'
    # Load the dataset
    data = pd.read_csv(DATA_PATH, index_col=0)
    train_folds = [f"Fold_{i}" for i in [0, 1, 2, 3, 5, 6, 7]]
    train_data = data[data["DataSAIL_10f"].isin(train_folds)]

    smiles = train_data['smiles'].values

    dataset_path = "/nethome/pjajoria/Documents/Enamine_REAL_HAC_24_394M_CXSMILES.cxsmiles.bz2"  # Path to the compressed Enamine dataset
    smiles_column = "smiles"  # Adjust if needed for your dataset
    similarity_threshold = 0.8  # Adjust as needed

    # Load dataset
    print("Loading Enamine dataset...")
    dataset_smiles = load_enamine_dataset(dataset_path, smiles_column)

    print(f"Loaded {len(dataset_smiles)} molecules from the dataset.")

    # Find similar molecules
    print("Finding similar molecules...")
    for smile in smiles:
        similar_molecules = find_similar_molecules(smile, dataset_smiles, similarity_threshold)

        # Output results
        print(f"Found {len(similar_molecules)} similar molecules:")
        for smile, similarity in similar_molecules[:10]:  # Display top 10
            print(f"SMILES: {smile}, Similarity: {similarity:.2f}")
