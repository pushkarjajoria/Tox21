import bz2
import gzip
import time
from asyncio import as_completed

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from concurrent.futures import ThreadPoolExecutor
import os
import sys
from tqdm import tqdm


def seconds_to_human_readable(seconds):
    """Convert seconds to human-readable time format (HH:MM:SS)."""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{int(hours)}:{int(minutes):02}:{int(seconds):02}"


def smile_to_morgan_fingerprint(smile, radius=2, n_bits=2048):
    """Convert SMILES string to Morgan fingerprint."""
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return None  # Return None for invalid SMILES
    fp = mfpgen.GetFingerprint(mol)
    return np.array(fp)


def process_chunk(chunk, thread_count):
    """Process a chunk of SMILES strings using multithreading."""
    smiles = chunk['smiles']
    fingerprints = []

    # Multithreading for fingerprint computation
    with ThreadPoolExecutor(max_workers=thread_count) as executor:
        results = list(executor.map(smile_to_morgan_fingerprint, smiles))

    for smile, fp in zip(smiles, results):
        if fp is not None:
            fingerprints.append({'smiles': smile, 'fingerprint': list(fp)})

    return fingerprints


# Function to read the `.gz2` file in chunks
def read_gz2_in_chunks(file_path, chunk_size):
    """
    Generator to read a .gz2 file in chunks.
    """
    with gzip.open(file_path, 'rt') as f:
        reader = pd.read_csv(f, chunksize=chunk_size)
        for chunk in reader:
            yield chunk


# Main function to orchestrate parallel processing
def process_file_in_chunks(file_path, chunk_size, output_dir, thread_pool_size):
    """
    Process a large .gz2 file in chunks using a thread pool and save the results to disk as Parquet files.
    """
    chunk_index = 0
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
    results = []

    with ThreadPoolExecutor(max_workers=thread_pool_size) as executor:
        futures = {}
        chunk_generator = read_gz2_in_chunks(file_path, chunk_size)

        # Start reading chunks and assigning them to threads
        for chunk in chunk_generator:
            futures[executor.submit(process_chunk, chunk, chunk_index)] = chunk_index
            chunk_index += 1

            # Maintain a limited number of active threads
            if len(futures) >= thread_pool_size:
                for future in as_completed(futures):
                    index = futures[future]
                    result = future.result()
                    results.append((index, result))
                futures.clear()  # Reset for next batch of threads

        # Wait for remaining threads to complete
        for future in as_completed(futures):
            index = futures[future]
            result = future.result()
            results.append((index, result))

    # Save processed chunks to Parquet incrementally
    for index, result in sorted(results, key=lambda x: x[0]):  # Ensure order by index
        output_file = os.path.join(output_dir, f"chunk_{index}.parquet")
        result.to_parquet(output_file, index=False)
        print(f"Saved chunk {index} to {output_file}")

    print(f"All chunks processed and saved to {output_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python create_enamine_fingerprint_dataset.py <input_file> <output_file> <[true/false]test_run>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    try:
        test_run = sys.argv[3] == 'true'
    except IndexError:
        test_run = False
    print(f"Running the python script with command \"python create_enamine_fingerprint_dataset.py {input_file} {output_file} {test_run}\"")
    start_time = time.time()
    # input_file = "/nethome/pjajoria/Documents/Enamine_REAL_HAC_24_394M_CXSMILES.cxsmiles.bz2"
    # output_file = "/nethome/pjajoria/Documents/Enamine_REAL_HAC_24_394M_CXSMILES.cxsmiles.parquet"

    # Modify thread_count as per available resources
    file_row_map = {
        "Enamine_REAL_HAC_22_23_471M_CXSMILES.cxsmiles.bz2": 471_000_000,
        "Enamine_REAL_HAC_28_803M_CXSMILES.cxsmiles.bz2": 803_000_000,
        "Enamine_REAL_HAC_24_394M_CXSMILES.cxsmiles.bz2": 394_000_000,
        "Enamine_REAL_HAC_29_38_1.3B_Part_1_CXSMILES.cxsmiles.bz2": 1_300_000_000,
        "Enamine_REAL_HAC_25_789M_CXSMILES.cxsmiles.bz2": 789_000_000,
        "Enamine_REAL_HAC_29_38_1.3B_Part_2_CXSMILES.cxsmiles.bz2": 1_300_000_000,
        "Enamine_REAL_HAC_26_766M_CXSMILES.cxsmiles.bz2": 766_000_000,
        "Enamine_REAL_HAC_6_21_420M_CXSMILES.cxsmiles.bz2": 420_000_000,
        "Enamine_REAL_HAC_27_872M_CXSMILES.cxsmiles.bz2": 872_000_000
    }
    total_rows_from_filename = file_row_map[input_file.split("/")[-1]]
    process_file(input_file, output_file, estimated_total_rows=total_rows_from_filename, chunk_size=100000, thread_count=8, test_run=test_run)
    print(f"Finished processing {input_file} in {time.time() - start_time} seconds")
