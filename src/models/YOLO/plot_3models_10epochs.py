import pandas as pd
import matplotlib.pyplot as plt
import os

plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "legend.fontsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "lines.linewidth": 2.5,
    "lines.markersize": 8
})

experiment_root = 'experiments'
report_epochs = 10

models = {
    'v8n': f'yolov8n_{report_epochs}epochs',
    'v8s': f'yolov8s_{report_epochs}epochs',
    'v8m': f'yolov8m_{report_epochs}epochs'
}

fig, axs = plt.subplots(1, 2, figsize=(18, 6))

colors = {'v8n': 'blue', 'v8s': 'green', 'v8m': 'red'}

for label, folder in models.items():
    results_path = os.path.join(experiment_root, folder, 'results.csv')
    if not os.path.exists(results_path):
        print(f"Warning: {results_path} not found, skipping {label}")
        continue

    df = pd.read_csv(results_path)
    epochs = df.index + 1

    total_loss = df['train/box_loss'] + df['train/cls_loss']
    axs[0].plot(epochs, total_loss, label=label, color=colors[label])

    axs[1].plot(epochs, df['metrics/mAP50(B)'] * 100, label=label, color=colors[label])  # scaled to percentage

axs[0].set_title('Training Total Loss vs Epoch', weight='bold')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')
axs[0].legend()
axs[0].grid(True, linewidth=0.7, linestyle='--', alpha=0.6)

axs[1].set_title('Val mAP@0.5 vs Epoch', weight='bold')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('mAP %')
axs[1].legend()
axs[1].grid(True, linewidth=0.7, linestyle='--', alpha=0.6)

plt.tight_layout()

output_path = f'comparison_curves_{report_epochs}_epochs.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()
