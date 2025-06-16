import bz2
import gc
import time
from multiprocessing.pool import ThreadPool
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import os
import sys
from tqdm import tqdm


def seconds_to_human_readable(seconds):
    """Convert seconds to human-readable time format (HH:MM:SS)."""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{int(hours)}:{int(minutes):02}:{int(seconds):02}"


def read_progress(progress_file):
    """Read the progress file to get the last processed chunk index."""
    if os.path.exists(progress_file):
        with open(progress_file, "r") as file:
            return int(file.read().strip())
    return 0


def write_progress(progress_file, chunks_processed):
    """Write the number of processed chunks to the progress file."""
    with open(progress_file, "w") as file:
        file.write(str(chunks_processed))


def smile_to_morgan_fingerprint(smile, radius=2, n_bits=2048):
    """Convert SMILES string to Morgan fingerprint."""
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return None  # Return None for invalid SMILES
    fp = mfpgen.GetFingerprint(mol)
    return np.array(fp)


def process_chunk(chunk):
    """Process a chunk of SMILES strings."""
    smiles = chunk['smiles']
    results = map(smile_to_morgan_fingerprint, smiles)
    return pd.DataFrame({"smiles": smiles, "fingerprints": results})


def process_file_in_chunks(file_path, chunk_size, output_dir, thread_pool_size, total_rows, resume=False):
    """
    Processes a large file in chunks with resume functionality.
    """
    os.makedirs(output_dir, exist_ok=True)
    progress_file = os.path.join(output_dir, "progress.txt")
    chunks_processed = read_progress(progress_file) if resume else 0

    total_chunks = total_rows // chunk_size
    if total_rows % chunk_size != 0:
        total_chunks += 1  # Include the last chunk

    with bz2.open(file_path, "rt") as file:
        chunk_iterator = pd.read_csv(file, chunksize=chunk_size, delimiter="\t")
        # Skip already processed chunks
        for _ in range(chunks_processed):
            next(chunk_iterator)

        batch_index = chunks_processed // thread_pool_size
        with tqdm(total=total_chunks, initial=chunks_processed, desc="Progress", unit="Chunk") as pbar:
            while True:
                pool = ThreadPool(thread_pool_size)
                results = []
                current_batch_size = 0

                for _ in range(thread_pool_size):
                    try:
                        chunk = next(chunk_iterator)
                        result = pool.apply_async(process_chunk, args=(chunk,))
                        results.append(result)
                        current_batch_size += 1
                    except StopIteration:
                        print("[INFO] Reached the end of the file.")
                        break  # End of file

                if current_batch_size == 0:
                    print("[INFO] No more chunks to process left in the file. Breaking.")
                    break  # No more chunks to process

                pool.close()
                print(f"[INFO] Waiting for threads to complete...", flush=True)
                pool.join()

                # Combine and save results
                combined_results = [result.get() for result in results if result is not None]
                if combined_results:
                    combined_df = pd.concat(combined_results, ignore_index=True)
                    output_file = os.path.join(output_dir, f"output_batch_{batch_index + 1}.parquet")
                    combined_df.to_parquet(output_file, engine="pyarrow", index=False, compression="snappy")
                    del combined_results, combined_df
                    gc.collect()

                # Update progress
                chunks_processed += current_batch_size
                write_progress(progress_file, chunks_processed)
                batch_index += 1
                pbar.update(current_batch_size)

    print(f"Processing completed. Total chunks processed: {chunks_processed}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python create_enamine_fingerprint_dataset.py <input_file> <output_file> <chunk_size> <thread_pool_size>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_dir = sys.argv[2]
    try:
        chunk_size = int(sys.argv[3])
    except (IndexError, ValueError):
        chunk_size = 200_000
    try:
        thread_pool_size = int(sys.argv[4])
    except (IndexError, ValueError):
        thread_pool_size = 8

    # input_file = "/nethome/pjajoria/Documents/Enamine_REAL_HAC_24_394M_CXSMILES.cxsmiles.bz2"
    # output_dir = "/nethome/pjajoria/Documents/Enamine_REAL_HAC_24_394M_CXSMILES"
    # chunk_size = 20_000
    # thread_pool_size = 8
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

    # Execute processing with resume functionality
    start_time = time.time()
    process_file_in_chunks(input_file, chunk_size, output_dir, thread_pool_size, total_rows_from_filename, resume=True)
    print(f"Finished processing {input_file} in {seconds_to_human_readable(time.time() - start_time)}")
