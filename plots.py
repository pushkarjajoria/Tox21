import matplotlib.pyplot as plt

# Data from the log
noise_levels = [0, 10, 20, 30]

baseline_accuracy = [0.9615, 0.9615, 0.9408, 0.9587]
noise_layer_accuracy = [0.9532, 0.9573, 0.9594, 0.9560]

baseline_precision = [0.5769, 0.5800, 0.3721, 0.5400]
noise_layer_precision = [0.4706, 0.5179, 0.5490, 0.5000]

baseline_recall = [0.4688, 0.4531, 0.5000, 0.4219]
noise_layer_recall = [0.5000, 0.4531, 0.4375, 0.3125]

baseline_f1 = [0.5172, 0.5088, 0.4267, 0.4737]
noise_layer_f1 = [0.4848, 0.4833, 0.4870, 0.3846]

# Plotting
plt.figure(figsize=(12, 8))

# Accuracy
plt.subplot(2, 2, 1)
plt.plot(noise_levels, baseline_accuracy, marker='o', label='Baseline')
plt.plot(noise_levels, noise_layer_accuracy, marker='o', label='Noise Layer')
plt.title('Accuracy')
plt.xlabel('Noise Level (%)')
plt.ylabel('Accuracy')
plt.legend()

# Precision
plt.subplot(2, 2, 2)
plt.plot(noise_levels, baseline_precision, marker='o', label='Baseline')
plt.plot(noise_levels, noise_layer_precision, marker='o', label='Noise Layer')
plt.title('Precision')
plt.xlabel('Noise Level (%)')
plt.ylabel('Precision')
plt.legend()

# Recall
plt.subplot(2, 2, 3)
plt.plot(noise_levels, baseline_recall, marker='o', label='Baseline')
plt.plot(noise_levels, noise_layer_recall, marker='o', label='Noise Layer')
plt.title('Recall')
plt.xlabel('Noise Level (%)')
plt.ylabel('Recall')
plt.legend()

# F1 Score
plt.subplot(2, 2, 4)
plt.plot(noise_levels, baseline_f1, marker='o', label='Baseline')
plt.plot(noise_levels, noise_layer_f1, marker='o', label='Noise Layer')
plt.title('F1 Score')
plt.xlabel('Noise Level (%)')
plt.ylabel('F1 Score')
plt.legend()

plt.tight_layout()
plt.show()
