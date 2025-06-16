import pickle
import joblib
import numpy as np
import pandas as pd

# Load the pre-trained RandomForest model from the .joblib file
model = joblib.load("models/rf_classification_morgan_fp.joblib")

# Load the DataFrame
df = pd.read_pickle("/nethome/pjajoria/Github/Tox21Noisy/outputs/similar_molecules/old_checkpoints/weighted_tanimoto/similar_molecules_dataframe.pkl")
X_similar = np.array(df["Fingerprint"].tolist())
# predictions = model.predict(X_similar)

# X_similar_np = X_similar.numpy()

# Get prediction probabilities
probabilities = model.predict_proba(X_similar)

# Define a custom threshold (e.g., 0.6)
custom_threshold = 0.4

df["RandomForestLabels"] = (probabilities[:, 1] >= custom_threshold)

positive_samples = df["RandomForestLabels"].sum()
total_samples = len(df)
print(f"Positive Samples: {positive_samples}/{total_samples}")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# baseline_model = Cache5AntagonistPredictor().to(device)
# baseline_model.load_state_dict(torch.load("models/nal_model.pt"))
#
# dl_predictions = predict(baseline_model, X_similar)
#
# print(len(dl_predictions[dl_predictions == 1]))
