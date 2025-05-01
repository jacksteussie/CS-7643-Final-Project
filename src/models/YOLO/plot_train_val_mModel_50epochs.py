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

train_box_loss = df['train/box_loss']
train_cls_loss = df['train/cls_loss']
val_box_loss = df['val/box_loss']
val_cls_loss = df['val/cls_loss']

plt.figure(figsize=(8, 5))

plt.plot(epochs, train_box_loss, label='Train Box Loss', color='blue', linestyle='-')
plt.plot(epochs, train_cls_loss, label='Train Cls Loss', color='green', linestyle='-')

plt.plot(epochs, val_box_loss, label='Val Box Loss', color='blue', linestyle='--')
plt.plot(epochs, val_cls_loss, label='Val Cls Loss', color='green', linestyle='--')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train vs Val Losses', weight='bold')
plt.legend()
plt.grid(True, linewidth=0.7, linestyle='--', alpha=0.6)
plt.tight_layout(pad=3.0)

output_path = os.path.join(experiment_folder, 'train_val_losses.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()
