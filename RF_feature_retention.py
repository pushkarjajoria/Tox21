import joblib
from data_prep import smiles_to_morgan_fingerprint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier


def get_feature_importances(model):
    feature_importances = model.feature_importances_
    indices = np.argsort(feature_importances)[::-1]  # Sorted indices (descending)
    sorted_importances = np.sort(feature_importances)[::-1]
    cumulative_importances = np.cumsum(sorted_importances)
    total_features = len(feature_importances)
    return feature_importances, indices, sorted_importances, cumulative_importances, total_features


def compute_num_features(cumulative_importances, total_features, percentages):
    num_features = []
    for p in percentages:
        # Find first index where cumulative importance meets or exceeds the threshold
        idx = np.argmax(cumulative_importances >= p)
        num_features.append(idx + 1)  # +1 to convert index to count
    return num_features


def load_data(csv_path):
    data = pd.read_csv(csv_path, index_col=0)
    data["morgan_fp"] = list(map(smiles_to_morgan_fingerprint, data['smiles'].values))

    train_folds = [f"Fold_{i}" for i in [0, 1, 2, 3, 5, 6, 7]]
    train_data = data[data["DataSAIL_10f"].isin(train_folds)]
    val_data = data[data["DataSAIL_10f"] == "Fold_4"]
    test_data = data[data["DataSAIL_10f"].isin([f"Fold_{i}" for i in [8, 9]])]

    X_train = np.array(train_data["morgan_fp"].to_list())
    y_train = train_data["class"].values
    X_val = np.array(val_data["morgan_fp"].to_list())
    y_val = val_data["class"].values
    X_test = np.array(test_data["morgan_fp"].to_list())
    y_test = test_data["class"].values

    return X_train, y_train, X_val, y_val, X_test, y_test


def compute_baseline_metrics(y):
    # Majority classifier: always predict 0
    baseline_preds = np.zeros_like(y)
    baseline_probs = np.zeros_like(y, dtype=float)
    try:
        auroc = roc_auc_score(y, baseline_probs)
    except ValueError:
        auroc = 0.5  # Neutral value if AUROC is ill-defined
    return {
        "AUROC": auroc,
        "Accuracy": accuracy_score(y, baseline_preds),
        "F1": f1_score(y, baseline_preds, zero_division=0),
        "Precision": precision_score(y, baseline_preds, zero_division=0),
        "Recall": recall_score(y, baseline_preds, zero_division=0)
    }


def train_and_evaluate_model(X_train, y_train, X_data, y_data, mask, seed):
    # Apply feature mask to training and evaluation data
    X_train_masked = X_train * mask
    X_data_masked = X_data * mask

    model = RandomForestClassifier(n_estimators=1000, min_samples_split=4,
                                   max_features=0.3, random_state=seed)
    model.fit(X_train_masked, y_train)
    probs = model.predict_proba(X_data_masked)[:, 1]
    preds = (probs >= 0.5).astype(int)

    return {
        "AUROC": roc_auc_score(y_data, probs),
        "Accuracy": accuracy_score(y_data, preds),
        "F1": f1_score(y_data, preds, zero_division=0),
        "Precision": precision_score(y_data, preds, zero_division=0),
        "Recall": recall_score(y_data, preds, zero_division=0)
    }


def plot_metrics(percents, metrics_mean, metrics_std, baseline, title):
    plt.figure(figsize=(12, 8))

    # For each metric, plot error bars (mean +/- std)
    plt.errorbar(percents, metrics_mean["AUROC"], yerr=metrics_std["AUROC"],
                 marker='o', linestyle='-', label='AUROC')
    plt.errorbar(percents, metrics_mean["Accuracy"], yerr=metrics_std["Accuracy"],
                 marker='s', linestyle='--', label='Accuracy')
    plt.errorbar(percents, metrics_mean["F1"], yerr=metrics_std["F1"],
                 marker='^', linestyle='-.', label='F1 Score')
    plt.errorbar(percents, metrics_mean["Precision"], yerr=metrics_std["Precision"],
                 marker='v', linestyle=':', label='Precision')
    plt.errorbar(percents, metrics_mean["Recall"], yerr=metrics_std["Recall"],
                 marker='D', linestyle='-', label='Recall')

    # Plot baseline metrics as horizontal lines
    plt.axhline(baseline["AUROC"], color='b', linestyle=':', label='Baseline AUROC')
    plt.axhline(baseline["Accuracy"], color='orange', linestyle=':', label='Baseline Accuracy')
    plt.axhline(baseline["F1"], color='g', linestyle=':', label='Baseline F1')
    plt.axhline(baseline["Precision"], color='r', linestyle=':', label='Baseline Precision')
    plt.axhline(baseline["Recall"], color='purple', linestyle=':', label='Baseline Recall')

    plt.xlabel('Percentage of Features Retained (%)')
    plt.ylabel('Score')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.xticks(percents)
    plt.show()


def main():
    percentages = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    random_seeds = [0, 1, 2, 3, 4]  # 5 random seeds for variability

    # Load pre-trained model and get feature importances
    cache_rf = joblib.load("/nethome/pjajoria/Github/Tox21Noisy/models/rf_classification_morgan_fp.joblib")
    feature_importances, indices, sorted_importances, cumulative_importances, total_features = get_feature_importances(
        cache_rf)
    num_features = compute_num_features(cumulative_importances, total_features, percentages)

    print("Features needed for each threshold:")
    for p, n in zip(percentages, num_features):
        print(f"{int(p * 100)}% importance: {n} features ({n / total_features:.1%} of total)")

    # Load dataset
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(
        'benchmark_datasets/CACHE5/20240430_MCHR1_splitted_RJ.csv')

    # Compute baseline metrics for Test and Validation data
    baseline_test = compute_baseline_metrics(y_test)
    baseline_val = compute_baseline_metrics(y_val)

    print("\nBaseline (Majority classifier) metrics on Test data:")
    print(f"AUROC: {baseline_test['AUROC']:.3f}")
    print(f"Accuracy: {baseline_test['Accuracy']:.3f}")
    print(f"F1 Score: {baseline_test['F1']:.3f}")
    print(f"Precision: {baseline_test['Precision']:.3f}")
    print(f"Recall: {baseline_test['Recall']:.3f}")

    print("\nBaseline (Majority classifier) metrics on Validation data:")
    print(f"AUROC: {baseline_val['AUROC']:.3f}")
    print(f"Accuracy: {baseline_val['Accuracy']:.3f}")
    print(f"F1 Score: {baseline_val['F1']:.3f}")
    print(f"Precision: {baseline_val['Precision']:.3f}")
    print(f"Recall: {baseline_val['Recall']:.3f}")

    # Dictionaries to store mean and std of metrics for each threshold and dataset
    metrics_val_mean = {"AUROC": [], "Accuracy": [], "F1": [], "Precision": [], "Recall": []}
    metrics_val_std = {"AUROC": [], "Accuracy": [], "F1": [], "Precision": [], "Recall": []}
    metrics_test_mean = {"AUROC": [], "Accuracy": [], "F1": [], "Precision": [], "Recall": []}
    metrics_test_std = {"AUROC": [], "Accuracy": [], "F1": [], "Precision": [], "Recall": []}

    # Evaluate model performance for each feature retention threshold using multiple seeds
    for p, num_f in zip(percentages, num_features):
        # Create a mask selecting the top 'num_f' features
        mask = np.zeros(total_features, dtype=bool)
        mask[indices[:num_f]] = True

        # Collect metrics over different random seeds
        val_metrics_seed = {"AUROC": [], "Accuracy": [], "F1": [], "Precision": [], "Recall": []}
        test_metrics_seed = {"AUROC": [], "Accuracy": [], "F1": [], "Precision": [], "Recall": []}

        for seed in random_seeds:
            current_val = train_and_evaluate_model(X_train, y_train, X_val, y_val, mask, seed)
            current_test = train_and_evaluate_model(X_train, y_train, X_test, y_test, mask, seed)
            for key in val_metrics_seed:
                val_metrics_seed[key].append(current_val[key])
                test_metrics_seed[key].append(current_test[key])

        # Compute mean and std for each metric across seeds
        for key in metrics_val_mean:
            mean_val = np.mean(val_metrics_seed[key])
            std_val = np.std(val_metrics_seed[key])
            metrics_val_mean[key].append(mean_val)
            metrics_val_std[key].append(std_val)

            mean_test = np.mean(test_metrics_seed[key])
            std_test = np.std(test_metrics_seed[key])
            metrics_test_mean[key].append(mean_test)
            metrics_test_std[key].append(std_test)

        print(
            f"Percentage: {p * 100:.1f}% ({num_f} features), Val AUROC: {np.mean(val_metrics_seed['AUROC']):.3f} ± {np.std(val_metrics_seed['AUROC']):.3f}, "
            f"Test AUROC: {np.mean(test_metrics_seed['AUROC']):.3f} ± {np.std(test_metrics_seed['AUROC']):.3f}")

    # Prepare x-axis ticks in percentages
    percents_plot = np.array(percentages) * 100

    # Plot results for validation and test sets with error bars
    plot_metrics(percents_plot, metrics_val_mean, metrics_val_std, baseline_val,
                 "Model Performance vs Feature Retention on Validation Data")
    plot_metrics(percents_plot, metrics_test_mean, metrics_test_std, baseline_test,
                 "Model Performance vs Feature Retention on Test Data")


if __name__ == "__main__":
    main()
