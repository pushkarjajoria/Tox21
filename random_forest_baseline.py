import joblib
import torch
from rdkit import DataStructs

# Load the pre-trained RandomForest model from the .joblib file
model = joblib.load("models/rf_classification_morgan_fp.joblib")

# Example input data for classification (make sure this matches the shape of your model's expected input)
# For example, if the model expects 5 features, we should provide a 2D array with 5 features for each sample
input_tensors = torch.randint(0, 2, (10, 2048), dtype=torch.uint8)
# Use the model to predict the class labels for the input data
predictions = model.predict(input_tensors)

# Print the predicted class labels
print("Predicted class labels:", predictions)

# sim = DataStructs.TanimotoSimilarity(fp1,fp2)
