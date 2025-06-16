import os
import pickle
import re
import numpy as np
import pandas as pd
from rdkit import DataStructs
from data_prep import smiles_to_morgan_fingerprint
import heapq


def save_pickle(heap_dict, output_dir = "outputs/similar_molecules"):
    # Ensure the directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get the current files in the directory
    existing_files = os.listdir(output_dir)

    # Pattern to match "part_x" files and extract the number
    part_pattern = re.compile(r"part_(\d+)\.pickle")
    existing_parts = [
        int(part_pattern.search(file).group(1))
        for file in existing_files
        if part_pattern.search(file)
    ]

    # Determine the next part number
    next_part_number = max(existing_parts, default=0) + 1

    # Construct the new filename
    output_filename = f"cache_similar_molecules_part_{next_part_number}.pickle"
    output_path = os.path.join(output_dir, output_filename)

    with open(output_path, "wb") as handle:
        pickle.dump(heap_dict, handle)


def get_bit_vector_from_fp(fp, dim=2048):
    bit_vect_fp = DataStructs.SparseBitVect(size=dim)
    active_indexes = np.where(fp == 1)[0]
    for bit in active_indexes:
        bit_vect_fp.SetBit(int(bit))
    return bit_vect_fp


def get_parquet_files_list(directory_paths):
    all_parquet_files = []
    for directory_path in directory_paths:
        if not os.path.isdir(directory_path):
            print(f"Skipping invalid directory: {directory_path}")
            continue

        parquet_files = [
            os.path.join(directory_path, file)
            for file in os.listdir(directory_path)
            if file.endswith('.parquet')
        ]
        all_parquet_files.extend(parquet_files)
    print("Found {} .parquet files.".format(len(all_parquet_files)))
    return all_parquet_files


def yield_parquet_dataframes(all_parquet_files):
    """
    Generator that finds all .parquet files in a list of directories and yields their DataFrames.

    Args:
        directory_paths (list of str): List of directory paths to search for .parquet files.

    Yields:
        pd.DataFrame: DataFrame loaded from a .parquet file.
    """
    # Collect all .parquet file paths from the provided directories
    # Yield each DataFrame sequentially
    for file_path in all_parquet_files:
        try:
            print(f"Loading file: {file_path}")
            df = pd.read_parquet(file_path)
            yield df
        except Exception as e:
            print(f"Error loading {file_path}: {e}")


def update_heap(key, comparison_datapoint, distance, heap_size=5):
    """
    Maintains the top `heap_size` closest elements for the given key.

    Args:
        key: The key for the heap in heap_dict.
        comparison_datapoint: The current data point being compared.
        distance: The distance of the comparison_datapoint.
        heap_size: The maximum size of the heap to maintain (default 5).
    """
    if key not in heap_dict:
        heap_dict[key] = []
    heap = heap_dict[key]

    # If the heap has space, simply add the new element
    if len(heap) < heap_size:
        heapq.heappush(heap, (distance, comparison_datapoint))  # Negate distance for max-heap behavior
    elif distance > heap[0][0]:  # Check if the current distance is closer than the farthest in the heap
        heapq.heapreplace(heap, (distance, comparison_datapoint))  # Replace the farthest element


def main_iterative():
    data_path = 'benchmark_datasets/CACHE5/20240430_MCHR1_splitted_RJ.csv'
    enamine_dataset_dirs = ["/data/corpora/enamine/with_fingerprint/Enamine_REAL_HAC_29_38_1_3B_Part_1_CXSMILES",
                            "/data/corpora/enamine/with_fingerprint/Enamine_REAL_HAC_29_38_1_3B_Part_2_CXSMILES"]

    all_parquet_files = get_parquet_files_list(enamine_dataset_dirs)
    all_parquet_files = all_parquet_files[:1]

    data = pd.read_csv(data_path, index_col=0)

    # Convert SMILES to Morgan fingerprints
    data["morgan_fp"] = list(map(smiles_to_morgan_fingerprint, data['smiles'].values))

    # Define the fold splits
    train_folds = [f"Fold_{i}" for i in [0, 1, 2, 3, 5, 6, 7]]

    # Create train, validation, and test sets based on the 'DataSAIL_10f' column
    train_data = data[data["DataSAIL_10f"].isin(train_folds)]

    # Create a dict with an empty list as the default factory
    heap_dict = {}
    start_time = time.time()

    train_dataset = [x for x in zip(train_data["smiles"].values, list(map(get_bit_vector_from_fp, train_data["morgan_fp"].values)))]
    # For each file
    for df in yield_parquet_dataframes(all_parquet_files):
        enamine_file = zip(df['smiles'].values, list(map(get_bit_vector_from_fp, df["fingerprints"].values)))
        # For each datapoint in the file
        for i, (enamine_smile, enamine_bv_fp) in enumerate(enamine_file):
            # For each cache datapoint
            for cache_smile, cache_bv_fp in train_dataset:
                sim_score = DataStructs.TanimotoSimilarity(cache_bv_fp, enamine_bv_fp)
                update_heap(cache_smile, enamine_smile, sim_score, heap_dict, heap_size=10)
                if i >= 64:
                    print(f"Completed batch 64 smiles in {time.time() - start_time} seconds")
                    exit("Manual Stop")

    heap_dict = {"files_list": all_parquet_files, "heap_dict": heap_dict}
    save_pickle(heap_dict)


if __name__ == "__main__":
    data_path = 'benchmark_datasets/CACHE5/20240430_MCHR1_splitted_RJ.csv'
    enamine_dataset_dirs = ["/data/corpora/enamine/with_fingerprint/Enamine_REAL_HAC_29_38_1_3B_Part_1_CXSMILES",
                            "/data/corpora/enamine/with_fingerprint/Enamine_REAL_HAC_29_38_1_3B_Part_2_CXSMILES"]

    all_parquet_files = get_parquet_files_list(enamine_dataset_dirs)
    all_parquet_files = all_parquet_files[:1]

    data = pd.read_csv(data_path, index_col=0)

    # Convert SMILES to Morgan fingerprints
    data["morgan_fp"] = list(map(smiles_to_morgan_fingerprint, data['smiles'].values))

    # Define the fold splits
    train_folds = [f"Fold_{i}" for i in [0, 1, 2, 3, 5, 6, 7]]

    # Create train, validation, and test sets based on the 'DataSAIL_10f' column
    train_data = data[data["DataSAIL_10f"].isin(train_folds)]

    # Create a dict with an empty list as the default factory
    heap_dict = {}

    train_dataset = [x for x in zip(train_data["smiles"].values, list(map(get_bit_vector_from_fp, train_data["morgan_fp"].values)))]
    # For each file
    for df in yield_parquet_dataframes(all_parquet_files):
        enamine_file = zip(df['smiles'].values, list(map(get_bit_vector_from_fp, df["fingerprints"].values)))
        # For each datapoint in the file
        for enamine_smile, enamine_bv_fp in enamine_file:
            # For each cache datapoint
            for cache_smile, cache_bv_fp in train_dataset:
                sim_score = DataStructs.TanimotoSimilarity(cache_bv_fp, enamine_bv_fp)
                update_heap(cache_smile, enamine_smile, sim_score)

    heap_dict = {"files_list": all_parquet_files, "heap_dict": heap_dict}
    save_pickle(heap_dict)
