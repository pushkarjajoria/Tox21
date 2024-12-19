import argparse
import os
import pickle
import random
import re
import string
import time

import numpy as np
import pandas as pd
import torch
from rdkit import DataStructs
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from data_prep import smiles_to_morgan_fingerprint
import heapq


class EnamineFileDataset(Dataset):
    def __init__(self, smiles, fps):
        self.smiles = smiles
        self.fps = fps

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return self.smiles[idx], self.fps[idx]


def seconds_to_human_readable(seconds):
    """Convert seconds to human-readable time format (DD:HH:MM:SS)."""
    days = seconds // 86400  # 86400 seconds in a day
    hours = (seconds % 86400) // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{int(days)}:{int(hours):02}:{int(minutes):02}:{int(seconds):02}"


def save_pickle(heap_dict, output_dir="/nethome/pjajoria/Github/Tox21Noisy/outputs/similar_molecules"):
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


def save_checkpoint(heap_dict, processed_file_count, run_identifier, checkpoint_dir="/nethome/pjajoria/Github/Tox21Noisy/outputs/similar_molecules/checkpoints"):
    """
    Save a checkpoint file with a consistent name to track intermediate progress.

    Args:
        heap_dict (dict): The dictionary containing heap information.
        processed_file_count (int): Number of files processed so far.
        checkpoint_dir (str): Directory to save the checkpoint.
    """
    # Ensure the directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Define the checkpoint filename
    checkpoint_filename = f"checkpoint_latest_{run_identifier}.pickle"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)

    # Include metadata about the progress
    checkpoint_data = {
        "heap_dict": heap_dict,
        "processed_file_count": processed_file_count,
    }

    with open(checkpoint_path, "wb") as handle:
        pickle.dump(checkpoint_data, handle)

    print(f"Checkpoint saved: {checkpoint_path}")


def tensorized_tanimoto_similarity(cache_molecules, enamine_batch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cache_molecules = torch.tensor(np.stack(cache_molecules)).to(device)
    enamine_batch = enamine_batch.to(device)
    batch_a, batch_b = cache_molecules.shape[-2], enamine_batch.shape[-2]

    A_intersection_B = torch.logical_and(cache_molecules.unsqueeze(1), enamine_batch.unsqueeze(0)).sum(dim=2)

    sum_A = torch.sum(cache_molecules, dim=1)
    sum_B = torch.sum(enamine_batch, dim=1)
    broadcast_A = sum_A.unsqueeze(1).repeat(1, batch_b)
    broadcast_B = sum_B.unsqueeze(1).repeat(1, batch_a).T
    tanimoto_similarity = A_intersection_B/(broadcast_A + broadcast_B - A_intersection_B)
    return tanimoto_similarity


def get_bit_vector_from_fp(fp, dim=2048):
    bit_vect_fp = DataStructs.SparseBitVect(size=dim)
    active_indexes = np.where(fp == 1)[0]
    for bit in active_indexes:
        bit_vect_fp.SetBit(int(bit))
    return bit_vect_fp


def get_parquet_files_list(directory_paths, job_type=None):
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
    if job_type:
        filtered_files = []
        for file in all_parquet_files:
            # Extract batch number from the file name
            try:
                batch_number = int(file.split("_batch_")[1].split(".parquet")[0])
                if (job_type == "odd" and batch_number % 2 != 0) or \
                   (job_type == "even" and batch_number % 2 == 0):
                    filtered_files.append(file)
            except (IndexError, ValueError):
                print(f"Skipping file with invalid naming convention: {file}")
        all_parquet_files = filtered_files
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


def update_heap(key, comparison_datapoint, distance, heap_dict, comparison_fingerprint, heap_size=5):
    """
    Maintains the top `heap_size` closest elements for the given key.

    Args:
        key: The key for the heap in heap_dict.
        comparison_datapoint: The current data point being compared.
        distance: The distance of the comparison_datapoint.
        heap_size: The maximum size of the heap to maintain (default 5).
        :param heap_dict: A dictionary containing the heap data for each cache molecule.
    """
    if key not in heap_dict:
        heap_dict[key] = []
    heap = heap_dict[key]

    # If the heap has space, simply add the new element
    if len(heap) < heap_size:
        heapq.heappush(heap, (distance, comparison_datapoint, comparison_fingerprint))  # Negate distance for max-heap behavior
    elif distance > heap[0][0]:  # Check if the current distance is closer than the farthest in the heap
        heapq.heapreplace(heap, (distance, comparison_datapoint, comparison_fingerprint))  # Replace the farthest element


def main_tensorized(enamine_dataset_dirs, batch_size, job_type):
    run_identifier = ''.join(random.choices(string.ascii_letters + string.digits, k=5))
    data_path = '/nethome/pjajoria/Github/Tox21Noisy/benchmark_datasets/CACHE5/20240430_MCHR1_splitted_RJ.csv'

    # Number of similar molecules from the Enamine dataset per molecule in cache train dataset
    k = 10

    all_parquet_files = get_parquet_files_list(enamine_dataset_dirs, job_type)
    # all_parquet_files = all_parquet_files[:1]

    data = pd.read_csv(data_path, index_col=0)

    # Convert SMILES to Morgan fingerprints
    data["morgan_fp"] = list(map(smiles_to_morgan_fingerprint, data['smiles'].values))

    # Define the fold splits
    train_folds = [f"Fold_{i}" for i in [0, 1, 2, 3, 5, 6, 7]]

    # Create train, validation, and test sets based on the 'DataSAIL_10f' column
    train_data = data[data["DataSAIL_10f"].isin(train_folds)]
    cache_smiles, cache_fps = train_data['smiles'].values, train_data['morgan_fp'].values

    # Create a dict with an empty list as the default factory
    save_interval = 1
    heap_dict = {}
    file_counter = 0
    total_time = 0
    for df in yield_parquet_dataframes(all_parquet_files):
        start_time = time.time()
        enamine_file_smiles, enamine_file_fps = df['smiles'].values, df["fingerprints"].values
        enamine_file_dataset = EnamineFileDataset(enamine_file_smiles, enamine_file_fps)
        enamine_dataloader = DataLoader(enamine_file_dataset, batch_size=batch_size, shuffle=True)
        for batch_idx, (batch_smiles, batch_fps) in enumerate(enamine_dataloader):
            batch_similarity = tensorized_tanimoto_similarity(cache_fps, batch_fps)
            values, indices = torch.topk(batch_similarity, k, largest=True, dim=1)
            values, indices = values.detach().cpu().numpy(), indices.detach().cpu().numpy()
            for iter, c_smile in enumerate(cache_smiles):
                for similarity, index in zip(values[iter], indices[iter]):
                    update_heap(key=c_smile, comparison_datapoint=enamine_file_smiles[index], distance=similarity, comparison_fingerprint=enamine_file_fps[index], heap_dict=heap_dict, heap_size=k)
        file_counter += 1
        if file_counter % save_interval == 0:
            save_checkpoint(heap_dict, file_counter, run_identifier)
        time_taken = time.time() - start_time
        total_time += time_taken
        running_av_time = total_time / file_counter
        print(f"Processed file {file_counter} out of total:{len(all_parquet_files)} in {seconds_to_human_readable(time_taken)} seconds")
        print(f"Estimated time remaining = {seconds_to_human_readable(running_av_time * (len(all_parquet_files) - file_counter))}")
    heap_dict = {"processed_files_list": all_parquet_files, "heap_dict": heap_dict}
    save_pickle(heap_dict)


if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Process .parquet files and find similar molecules using tensors."
    )

    # Add arguments
    parser.add_argument(
        "directory_paths",
        type=str,
        nargs="+",
        help="List of directories containing the .parquet files."
    )
    parser.add_argument(
        "batch_size",
        type=int,
        help="Batch size for processing files."
    )
    parser.add_argument(
        "--job_type",
        type=str,
        choices=["odd", "even"],
        default=None,
        help="Type of job ('odd' or 'even') to filter files. Defaults to processing all files."
    )

    # Parse arguments
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main_tensorized(
        enamine_dataset_dirs=args.directory_paths,
        batch_size=args.batch_size,
        job_type=args.job_type
    )
