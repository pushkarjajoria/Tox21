import matplotlib.pyplot as plt
from datetime import datetime


def plot_comparison_figure(noise_levels, **kwargs):
    # Data from the log
    baseline_accuracy = kwargs['baseline_accuracy']
    noise_layer_accuracy = kwargs['noise_layer_accuracy']

    baseline_precision = kwargs['baseline_precision']
    noise_layer_precision = kwargs['noise_layer_precision']

    baseline_recall = kwargs['baseline_recall']
    noise_layer_recall = kwargs['noise_layer_recall']

    baseline_f1 = kwargs['baseline_f1']
    noise_layer_f1 = kwargs['noise_layer_f1']

    # Model info from kwargs
    model_info = kwargs.get('model_info', {})

    # Format model info for display
    model_info_text = '\n'.join([f'{key}: {value}' for key, value in model_info.items()])

    # Plotting
    plt.figure(figsize=(14, 10))

    # Accuracy
    plt.subplot(2, 2, 1)
    plt.plot(noise_levels, baseline_accuracy, marker='o', label='Baseline')
    plt.plot(noise_levels, noise_layer_accuracy, marker='o', label='Noise Layer')
    plt.title('Accuracy')
    plt.xlabel('Noise Level (%)')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    # Precision
    plt.subplot(2, 2, 2)
    plt.plot(noise_levels, baseline_precision, marker='o', label='Baseline')
    plt.plot(noise_levels, noise_layer_precision, marker='o', label='Noise Layer')
    plt.title('Precision')
    plt.xlabel('Noise Level (%)')
    plt.ylabel('Precision (%)')
    plt.legend()

    # Recall
    plt.subplot(2, 2, 3)
    plt.plot(noise_levels, baseline_recall, marker='o', label='Baseline')
    plt.plot(noise_levels, noise_layer_recall, marker='o', label='Noise Layer')
    plt.title('Recall')
    plt.xlabel('Noise Level (%)')
    plt.ylabel('Recall (%)')
    plt.legend()

    # F1 Score
    plt.subplot(2, 2, 4)
    plt.plot(noise_levels, baseline_f1, marker='o', label='Baseline')
    plt.plot(noise_levels, noise_layer_f1, marker='o', label='Noise Layer')
    plt.title('F1 Score')
    plt.xlabel('Noise Level (%)')
    plt.ylabel('F1 Score (%)')
    plt.legend()

    # Add model and hyperparameter info as a text box
    plt.gcf().text(0.95, 0.5, model_info_text, fontsize=10, verticalalignment='center', bbox=dict(facecolor='white', alpha=0.6))

    plt.tight_layout(rect=(0, 0, 0.9, 1))  # Adjust to make room for the textbox

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    plot_name = f"plots/baseline_vs_noise_adaptation_{current_time}.png"

    # Save the plot with the new name
    plt.savefig(plot_name)
    plt.show()
