import pandas as pd  # for typehinting below
from sklearn.ensemble import RandomForestClassifier
from smallworld_api import SmallWorld, NoMatchError
from IPython.display import display

data_path = 'benchmark_datasets/CACHE5/20240430_MCHR1_splitted_RJ.csv'
data = pd.read_csv(data_path, index_col=0)
train_folds = [f"Fold_{i}" for i in [0, 1, 2, 3, 5, 6, 7]]
train_data = data[data["DataSAIL_10f"].isin(train_folds)]

smiles = train_data['smiles']

smiles = smiles.values
global_results = []  # Global list to store results

for smile in smiles:
    print(f"Molecules similar to {smile}:")
    sw = SmallWorld()
    try:
        results: pd.DataFrame = sw.search(smile, dist=5, db=sw.REAL_dataset)
        global_results.append(results)  # Append results to global list
        display(results)
    except Exception as e:
        print(f"No match found for {smile}. Error: {e}")

if global_results:
    combined_results = pd.concat(global_results, ignore_index=True)
else:
    combined_results = pd.DataFrame()  # Fallback in case global_results is empty

# Check if there are valid data points to label
if not combined_results.empty:
    # Prepare features for prediction
    # Replace 'feature_columns' with actual column names from combined_results
    feature_columns = [col for col in combined_results.columns if col != 'target']  # Assuming 'target' isn't a feature
    features = combined_results[feature_columns]

    # Load the pretrained model
    # Replace this with actual code to load the trained model (e.g., using joblib or pickle)
    # Example:
    # from joblib import load
    # model = load('best_model.joblib')
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)  # Placeholder, use your best model

    # Generate labels
    labels = model.predict(features)

    # Add labels to the DataFrame
    combined_results['predicted_label'] = labels

    # Display or save the labeled data
    print("Labeled Results:")
    print(combined_results.head(5))
    # combined_results.to_csv("labeled_results.csv", index=False)  # Optional: Save to CSV
else:
    print("No data available for labeling.")