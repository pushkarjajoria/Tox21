import pandas as pd
import time
import os
import requests
from tqdm import tqdm


# Create outputs directory if it doesn't exist
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Constants
DATA_PATH = 'benchmark_datasets/CACHE5/20240430_MCHR1_splitted_RJ.csv'
API_RETRY_LIMIT = 5  # Maximum retries for connection failures
MAX_BACKOFF_TIME = 60  # Maximum backoff time in seconds (1 minute)


def find_similar_pubchem(smiles, threshold=0.8):
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/"
    similarity_url = f"{base_url}compound/similarity/smiles/JSON"
    params = {
        "smiles": smiles,
        "Threshold": int(threshold * 100)  # Threshold must be between 0-100
    }

    retries = 0
    while retries < API_RETRY_LIMIT:
        try:
            response = requests.get(similarity_url, params=params, timeout=10)
            if response.status_code == 202:  # Request accepted but not completed
                list_key = response.json()["Waiting"]["ListKey"]
                while True:
                    poll_url = f"{base_url}compound/listkey/{list_key}/JSON"
                    poll_response = requests.get(poll_url, timeout=10)
                    if poll_response.status_code == 200:
                        return poll_response.json()
                    elif poll_response.status_code == 202:
                        print("No Response yet. Waiting 5 seconds")
                        time.sleep(5)  # Wait before retrying
                    else:
                        return f"Error: {poll_response.status_code} - {poll_response.text}"
            elif response.status_code == 200:
                return response.json()
            else:
                print(f"Error: {response.status_code} - {response.text}")
                break  # Do not retry for non-retryable errors
        except requests.exceptions.RequestException as e:
            retries += 1
            backoff_time = min(2 ** retries, MAX_BACKOFF_TIME)
            print(f"Retry {retries}/{API_RETRY_LIMIT}: {e}. Retrying in {backoff_time} seconds...")
            time.sleep(backoff_time)
    print(f"Failed to fetch data for SMILES: {smiles} after {API_RETRY_LIMIT} retries.")
    return None


if __name__ == "__main__":

    # Constants
    DATA_PATH = 'benchmark_datasets/CACHE5/20240430_MCHR1_splitted_RJ.csv'

    # Load the dataset
    data = pd.read_csv(DATA_PATH, index_col=0)
    train_folds = [f"Fold_{i}" for i in [0, 1, 2, 3, 5, 6, 7]]
    train_data = data[data["DataSAIL_10f"].isin(train_folds)]

    smiles = train_data['smiles'].values

    # Sanity test with a subset of the dataset
    sanity_smiles = smiles[:10]
    similar_smiles_mapping = []
    threshold = 0.5

    for smile_query in tqdm(sanity_smiles, desc="Processing SMILES"):
        result = find_similar_pubchem(smile_query, threshold)
        if result and isinstance(result, dict):
            smiles_list = []
            for compound in result.get("PC_Compounds", []):
                for prop in compound.get("props", []):
                    if prop.get("urn", {}).get("label") == "SMILES":
                        smiles_list.append(prop["value"]["sval"])
            similar_smiles_mapping.append({
                "query_smiles": smile_query,
                "similar_smiles": smiles_list
            })

    # Save similar SMILES and mappings to files
    output_file_smiles = os.path.join(OUTPUT_DIR, "sanity_test_similar_smiles.csv")
    output_file_mapping = os.path.join(OUTPUT_DIR, "sanity_test_smiles_mapping.csv")

    # Save as DataFrame
    pd.DataFrame(similar_smiles_mapping).to_csv(output_file_mapping, index=False)

    # For simplicity, save all similar smiles as a single list (flattened)
    all_similar_smiles = [smile for entry in similar_smiles_mapping for smile in entry["similar_smiles"]]
    pd.DataFrame({"similar_smiles": all_similar_smiles}).to_csv(output_file_smiles, index=False)

    print(f"Sanity test completed. Results saved in '{OUTPUT_DIR}'.")


