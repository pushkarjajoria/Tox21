import pickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def plot_comparison_figure(noise_levels, plot_name="baseline_vs_noise_adaptation", **kwargs):
    # Data from the log
    baseline_accuracy = kwargs['baseline_accuracy']
    noise_layer_accuracy = kwargs['noise_layer_accuracy']

    baseline_precision = kwargs['baseline_precision']
    noise_layer_precision = kwargs['noise_layer_precision']

    baseline_recall = kwargs['baseline_recall']
    noise_layer_recall = kwargs['noise_layer_recall']

    baseline_f1 = kwargs['baseline_f1']
    noise_layer_f1 = kwargs['noise_layer_f1']

    # Assuming the data for each metric is provided as an array of arrays (5 seeds)
    baseline_accuracy = np.array(baseline_accuracy)
    noise_layer_accuracy = np.array(noise_layer_accuracy)

    baseline_precision = np.array(baseline_precision)
    noise_layer_precision = np.array(noise_layer_precision)

    baseline_recall = np.array(baseline_recall)
    noise_layer_recall = np.array(noise_layer_recall)

    baseline_f1 = np.array(baseline_f1)
    noise_layer_f1 = np.array(noise_layer_f1)

    # Calculate means and standard deviations
    baseline_accuracy_mean = baseline_accuracy.mean(axis=0)
    noise_layer_accuracy_mean = noise_layer_accuracy.mean(axis=0)
    baseline_accuracy_std = baseline_accuracy.std(axis=0)
    noise_layer_accuracy_std = noise_layer_accuracy.std(axis=0)

    baseline_precision_mean = baseline_precision.mean(axis=0)
    noise_layer_precision_mean = noise_layer_precision.mean(axis=0)
    baseline_precision_std = baseline_precision.std(axis=0)
    noise_layer_precision_std = noise_layer_precision.std(axis=0)

    baseline_recall_mean = baseline_recall.mean(axis=0)
    noise_layer_recall_mean = noise_layer_recall.mean(axis=0)
    baseline_recall_std = baseline_recall.std(axis=0)
    noise_layer_recall_std = noise_layer_recall.std(axis=0)

    baseline_f1_mean = baseline_f1.mean(axis=0)
    noise_layer_f1_mean = noise_layer_f1.mean(axis=0)
    baseline_f1_std = baseline_f1.std(axis=0)
    noise_layer_f1_std = noise_layer_f1.std(axis=0)

    # Model info from kwargs
    model_info = kwargs.get('model_info', {})

    # Format model info for display
    model_info_text = '\n'.join([f'{key}: {value}' for key, value in model_info.items()])

    # Plotting
    plt.figure(figsize=(16, 10))

    # Accuracy
    plt.subplot(2, 2, 1)
    plt.errorbar(noise_levels, baseline_accuracy_mean, yerr=baseline_accuracy_std, marker='o', label='Baseline', capsize=5)
    plt.errorbar(noise_levels, noise_layer_accuracy_mean, yerr=noise_layer_accuracy_std, marker='o', label='Noise Layer', capsize=5)
    plt.title('Accuracy')
    plt.xlabel('Noise Level (%)')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    # Precision
    plt.subplot(2, 2, 2)
    plt.errorbar(noise_levels, baseline_precision_mean, yerr=baseline_precision_std, marker='o', label='Baseline', capsize=5)
    plt.errorbar(noise_levels, noise_layer_precision_mean, yerr=noise_layer_precision_std, marker='o', label='Noise Layer', capsize=5)
    plt.title('Precision')
    plt.xlabel('Noise Level (%)')
    plt.ylabel('Precision (%)')
    plt.legend()

    # Recall
    plt.subplot(2, 2, 3)
    plt.errorbar(noise_levels, baseline_recall_mean, yerr=baseline_recall_std, marker='o', label='Baseline', capsize=5)
    plt.errorbar(noise_levels, noise_layer_recall_mean, yerr=noise_layer_recall_std, marker='o', label='Noise Layer', capsize=5)
    plt.title('Recall')
    plt.xlabel('Noise Level (%)')
    plt.ylabel('Recall (%)')
    plt.legend()

    # F1 Score
    plt.subplot(2, 2, 4)
    plt.errorbar(noise_levels, baseline_f1_mean, yerr=baseline_f1_std, marker='o', label='Baseline', capsize=5)
    plt.errorbar(noise_levels, noise_layer_f1_mean, yerr=noise_layer_f1_std, marker='o', label='Noise Layer', capsize=5)
    plt.title('F1 Score')
    plt.xlabel('Noise Level (%)')
    plt.ylabel('F1 Score (%)')
    plt.legend()

    # Add model and hyperparameter info as a text box
    plt.gcf().text(0.90, 0.5, model_info_text, fontsize=10, verticalalignment='center', bbox=dict(facecolor='white', alpha=0.6))

    plt.tight_layout(rect=(0, 0, 0.9, 1))  # Adjust to make room for the textbox

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    plot_name = f"/nethome/pjajoria/Github/Tox21Noisy/plots/{plot_name}_{current_time}.png"

    # Save the plot with the new name
    plt.savefig(plot_name)
    # plt.show()


def plot_results(results_dict):
    try:
        num_seeds = results_dict["num_seed"]
        noise_levels = results_dict["noise_levels"]
    except KeyError:
        noise_levels = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.50]
        num_seeds = 3
    tasks = ['SR-HSE', 'NR-AR', 'SR-ARE', 'NR-Aromatase', 'NR-ER-LBD', 'NR-AhR',
             'SR-MMP', 'NR-ER', 'NR-PPAR-gamma', 'SR-p53', 'SR-ATAD5', 'NR-AR-LBD']
    num_noise_levels = len(noise_levels)
    # Prepare data for plotting
    baseline_f1 = np.array(
        [[[results_dict['baseline_f1'][seed][noise][task] for task in tasks] for noise in range(num_noise_levels)] for seed in
         range(num_seeds)])
    noise_f1 = np.array(
        [[[results_dict['noise_layer_f1'][seed][noise][task] for task in tasks] for noise in range(num_noise_levels)] for seed in
         range(num_seeds)])

    baseline_roc_auc = np.array(
        [[[results_dict['baseline_roc_auc'][seed][noise][task] for task in tasks] for noise in range(num_noise_levels)] for seed in
         range(num_seeds)])
    noise_roc_auc = np.array(
        [[[results_dict['noise_layer_roc_auc'][seed][noise][task] for task in tasks] for noise in range(num_noise_levels)] for seed in
         range(num_seeds)])

    # Plot 1: Average F1 and ROC-AUC across noise levels (with error bars for seeds)
    plt.figure(figsize=(12, 6))

    # Subplot 1: F1
    plt.subplot(1, 2, 1)
    baseline_f1_mean = baseline_f1.mean(axis=(2, 0))  # Average over tasks and seeds
    baseline_f1_std = baseline_f1.std(axis=(2, 0))  # Std over seeds

    noise_f1_mean = noise_f1.mean(axis=(2, 0))
    noise_f1_std = noise_f1.std(axis=(2, 0))

    plt.errorbar(noise_levels, baseline_f1_mean, yerr=baseline_f1_std, label='Baseline F1', fmt='-o')
    plt.errorbar(noise_levels, noise_f1_mean, yerr=noise_f1_std, label='Noise Layer F1', fmt='-o')
    plt.xlabel('Noise Level')
    plt.ylabel('Average F1')
    plt.title('Average F1 Across Noise Levels')
    plt.legend()

    # Subplot 2: ROC AUC
    plt.subplot(1, 2, 2)
    baseline_roc_auc_mean = baseline_roc_auc.mean(axis=(2, 0))  # Average over tasks and seeds
    baseline_roc_auc_std = baseline_roc_auc.std(axis=(2, 0))  # Std over seeds

    noise_roc_auc_mean = noise_roc_auc.mean(axis=(2, 0))
    noise_roc_auc_std = noise_roc_auc.std(axis=(2, 0))

    plt.errorbar(noise_levels, baseline_roc_auc_mean, yerr=baseline_roc_auc_std, label='Baseline ROC-AUC', fmt='-o')
    plt.errorbar(noise_levels, noise_roc_auc_mean, yerr=noise_roc_auc_std, label='Noise Layer ROC-AUC', fmt='-o')
    plt.xlabel('Noise Level')
    plt.ylabel('Average ROC-AUC')
    plt.title('Average ROC-AUC Across Noise Levels')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Plot 2: Per-task statistics (12 subplots)
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle('Per Task F1 across Noise Levels', fontsize=16)

    for i, task in enumerate(tasks):
        row, col = divmod(i, 4)

        # Plot for each task
        ax = axes[row, col]
        baseline_f1_task_mean = baseline_f1[:, :, i].mean(axis=0)
        baseline_f1_task_std = baseline_f1[:, :, i].std(axis=0)

        noise_f1_task_mean = noise_f1[:, :, i].mean(axis=0)
        noise_f1_task_std = noise_f1[:, :, i].std(axis=0)

        ax.errorbar(noise_levels, baseline_f1_task_mean, yerr=baseline_f1_task_std, label='Baseline F1', fmt='-o')
        ax.errorbar(noise_levels, noise_f1_task_mean, yerr=noise_f1_task_std, label='Noise Layer F1', fmt='-o')

        ax.set_title(f'Task: {task}')
        ax.set_xlabel('Noise Level')
        ax.set_ylabel('F1 Score')

    plt.legend()
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # Plot 3: Per-task ROC-AUC (12 subplots)
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle('Per Task ROC-AUC across Noise Levels', fontsize=16)

    for i, task in enumerate(tasks):
        row, col = divmod(i, 4)

        # Plot for each task
        ax = axes[row, col]
        baseline_roc_auc_task_mean = baseline_roc_auc[:, :, i].mean(axis=0)
        baseline_roc_auc_task_std = baseline_roc_auc[:, :, i].std(axis=0)

        noise_roc_auc_task_mean = noise_roc_auc[:, :, i].mean(axis=0)
        noise_roc_auc_task_std = noise_roc_auc[:, :, i].std(axis=0)

        ax.errorbar(noise_levels, baseline_roc_auc_task_mean, yerr=baseline_roc_auc_task_std, label='Baseline ROC-AUC',
                    fmt='-o')
        ax.errorbar(noise_levels, noise_roc_auc_task_mean, yerr=noise_roc_auc_task_std, label='Noise Layer ROC-AUC',
                    fmt='-o')

        ax.set_title(f'Task: {task}')
        ax.set_xlabel('Noise Level')
        ax.set_ylabel('ROC-AUC')

    plt.legend()
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# Example usage
# plot_results(results_dict)  # Call this function with your results dict


if __name__ == "__main__":
    with open("/nethome/pjajoria/Github/Tox21Noisy/outputs/result_pickles/results_pickle_2024-10-17_16-20.pkl", "rb") as f:
        results = pickle.load(f)
    plot_results(results)
