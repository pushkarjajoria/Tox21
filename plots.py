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


if __name__ == "__main__":
    # Updated noise levels
    NOISE_LEVELS = np.linspace(0.2, 0.5, 6)
    # Define the number of noise levels and seeds
    num_noise_levels = 6
    num_seeds = 5

    # Adjusted dummy values for baseline accuracy with performance decreasing across noise levels for each seed
    multiple_seed_baseline_accuracy = [
        [88, 83, 78, 72, 67, 62],  # Seed 1
        [87, 82, 77, 71, 66, 61],  # Seed 2
        [86, 81, 76, 70, 65, 60],  # Seed 3
        [84, 79, 75, 69, 64, 59],  # Seed 4
        [85, 80, 74, 71, 66, 63]  # Seed 5
    ]

    # Adjusted dummy values for noise layer accuracy
    multiple_seed_noise_layer_accuracy = [
        [85, 80, 75, 70, 65, 60],  # Seed 1
        [84, 79, 74, 69, 64, 59],  # Seed 2
        [83, 78, 73, 68, 63, 58],  # Seed 3
        [82, 76, 72, 67, 62, 57],  # Seed 4
        [83, 77, 74, 69, 66, 61]  # Seed 5
    ]

    # Adjusted dummy values for baseline precision
    multiple_seed_baseline_precision = [
        [0.88, 0.83, 0.78, 0.73, 0.68, 0.63],  # Seed 1
        [0.87, 0.82, 0.77, 0.72, 0.67, 0.62],  # Seed 2
        [0.86, 0.81, 0.76, 0.71, 0.66, 0.61],  # Seed 3
        [0.84, 0.79, 0.75, 0.70, 0.65, 0.60],  # Seed 4
        [0.85, 0.80, 0.74, 0.71, 0.66, 0.63]  # Seed 5
    ]

    # Adjusted dummy values for noise layer precision
    multiple_seed_noise_layer_precision = [
        [0.85, 0.80, 0.75, 0.70, 0.65, 0.60],  # Seed 1
        [0.84, 0.79, 0.74, 0.69, 0.64, 0.59],  # Seed 2
        [0.83, 0.78, 0.73, 0.68, 0.63, 0.58],  # Seed 3
        [0.82, 0.76, 0.72, 0.67, 0.62, 0.57],  # Seed 4
        [0.83, 0.77, 0.74, 0.69, 0.66, 0.61]  # Seed 5
    ]

    # Adjusted dummy values for baseline recall
    multiple_seed_baseline_recall = [
        [0.88, 0.83, 0.78, 0.73, 0.68, 0.63],  # Seed 1
        [0.87, 0.82, 0.77, 0.72, 0.67, 0.62],  # Seed 2
        [0.86, 0.81, 0.76, 0.71, 0.66, 0.61],  # Seed 3
        [0.84, 0.79, 0.75, 0.70, 0.65, 0.60],  # Seed 4
        [0.85, 0.80, 0.74, 0.71, 0.66, 0.63]  # Seed 5
    ]

    # Adjusted dummy values for noise layer recall
    multiple_seed_noise_layer_recall = [
        [0.85, 0.80, 0.75, 0.70, 0.65, 0.60],  # Seed 1
        [0.84, 0.79, 0.74, 0.69, 0.64, 0.59],  # Seed 2
        [0.83, 0.78, 0.73, 0.68, 0.63, 0.58],  # Seed 3
        [0.82, 0.76, 0.72, 0.67, 0.62, 0.57],  # Seed 4
        [0.83, 0.77, 0.74, 0.69, 0.66, 0.61]  # Seed 5
    ]

    # Adjusted dummy values for baseline F1 score
    multiple_seed_baseline_f1 = [
        [0.88, 0.83, 0.78, 0.73, 0.68, 0.63],  # Seed 1
        [0.87, 0.82, 0.77, 0.72, 0.67, 0.62],  # Seed 2
        [0.86, 0.81, 0.76, 0.71, 0.66, 0.61],  # Seed 3
        [0.84, 0.79, 0.75, 0.70, 0.65, 0.60],  # Seed 4
        [0.85, 0.80, 0.74, 0.71, 0.66, 0.63]  # Seed 5
    ]

    # Adjusted dummy values for noise layer F1 score
    multiple_seed_noise_layer_f1 = [
        [0.85, 0.80, 0.75, 0.70, 0.65, 0.60],  # Seed 1
        [0.84, 0.79, 0.74, 0.69, 0.64, 0.59],  # Seed 2
        [0.83, 0.78, 0.73, 0.68, 0.63, 0.58],  # Seed 3
        [0.82, 0.76, 0.72, 0.67, 0.62, 0.57],  # Seed 4
        [0.83, 0.77, 0.74, 0.69, 0.66, 0.61]  # Seed 5
    ]

    plot_comparison_figure(
        noise_levels=NOISE_LEVELS,
        baseline_accuracy=multiple_seed_baseline_accuracy,
        noise_layer_accuracy=multiple_seed_noise_layer_accuracy,
        baseline_precision=multiple_seed_baseline_precision,
        noise_layer_precision=multiple_seed_noise_layer_precision,
        baseline_recall=multiple_seed_baseline_recall,
        noise_layer_recall=multiple_seed_noise_layer_recall,
        baseline_f1=multiple_seed_baseline_f1,
        noise_layer_f1=multiple_seed_noise_layer_f1,
        model_info={"Comments": "Testing with transposed noise-influenced performance"}
    )
