import pandas as pd
import matplotlib.pyplot as plt
import os

plt.rcParams.update({
    "font.size": 14,
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

experiment_folder = 'experiments/yolov8m_50epochs'
results_path = os.path.join(experiment_folder, 'results.csv')
df = pd.read_csv(results_path)

epochs = df.index + 1

val_map50 = df['metrics/mAP50(B)'] * 100

plt.figure(figsize=(8, 5))
plt.plot(epochs, val_map50, label='Val mAP@0.5', color='orange', marker='o')
plt.xlabel('Epoch')
plt.ylabel('mAP@0.5 %')
plt.title('Validation mAP@0.5 over Epochs', weight='bold')
plt.grid(True)
plt.legend()
plt.tight_layout()

output_path = os.path.join(experiment_folder, 'val_map50_curve.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()
