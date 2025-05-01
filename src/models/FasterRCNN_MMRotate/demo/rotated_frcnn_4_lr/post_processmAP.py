import os
import re
from matplotlib import pyplot as plt

# Paths
training_data = "/teamspace/studios/this_studio/mmrotate/demo/rotated_frcnn_4_lr/results_dist_train.log"
test_data = "/teamspace/studios/this_studio/mmrotate/demo/rotated_frcnn_4_lr/results_dist_test.log"

def getMapPerEpoch(myfile):
    epochs, maps = [],[]
    with open(myfile, "r") as f:
        lines=f.readlines()
    for line in lines:
        if "| mAP" in line:
            maps.append(float(line.replace("|","").replace("mAP","").replace("\n","").replace(" ","")) * 100)
        elif ".pth" in line:
            current_epoch = line.split('epoch_')
            current_epoch = int(re.findall(r'\d+', current_epoch[1])[0])
            if current_epoch not in epochs:
                epochs.append(current_epoch)
    return epochs[::-1], maps[::-1]

train_epochs, train_mAP = getMapPerEpoch(training_data)
val_epochs, val_mAP = getMapPerEpoch(test_data)

plt.figure(figsize=(4,3))
plt.tight_layout()
plt.plot(train_epochs, train_mAP,label="train")
plt.plot(val_epochs, val_mAP,label="val")
plt.title("mAP")
plt.xlabel("Epochs")
plt.ylabel("%")
plt.legend()
plt.grid(True)    
plt.savefig("/teamspace/studios/this_studio/mmrotate/demo/rotated_frcnn_4_lr/plots/mAP.png", bbox_inches='tight')
plt.show()
print("✅ Conversion complete.")