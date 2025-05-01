import json
import matplotlib.pyplot as plt
import math

# Customize this path to your log file
LOG_FILE = "/teamspace/studios/this_studio/mmrotate/demo/rotated_frcnn_4_lr/20250427_042006.log.json"

# Load log data
def load_log(filepath):
    with open(filepath, "r") as f:
        data = [json.loads(line) for line in f if line.strip()]
    return data

# Extract training and validation metrics
def extract_metrics(log_data, mode="train"):
    
    epochs={}
    counter = 0
    for entry in log_data:
        if entry.get("mode") == mode:
            if entry.get("epoch",0) not in epochs:
                epochs[entry.get("epoch",0)]={}          
                counter = 1      
                loss_rpn_cls = 0
                loss_rpn_bbox = 0
                acc = 0
                loss_bbox = 0
                loss_cls = 0
                loss = 0
            if mode == "val":
                if "mAP" in entry:
                    epochs[entry.get("epoch",0)]["mAP"] = entry.get("mAP",0)
                else:
                    #"loss_rpn_cls_val": 0.0623, "loss_rpn_bbox_val": 0.02123, "loss_cls_val": 0.11477, "acc_val": 96.33878, "loss_bbox_val": 0.11859, "loss_val": 0.31689, "time": 0.18764}
                    epochs[entry.get("epoch",0)]["loss_rpn_cls"] = entry.get("loss_rpn_cls_val",0)
                    epochs[entry.get("epoch",0)]["loss_rpn_bbox"] = entry.get("loss_rpn_bbox_val",0)
                    epochs[entry.get("epoch",0)]["acc"] = entry.get("acc_val",0)
                    epochs[entry.get("epoch",0)]["loss_bbox"] = entry.get("loss_bbox_val",0)                    
                    epochs[entry.get("epoch",0)]["loss_cls"] = entry.get("loss_cls_val",0)
                    epochs[entry.get("epoch",0)]["loss"] = entry.get("loss_val",0)
            else:
                loss_rpn_cls+=entry.get("loss_rpn_cls",0)
                loss_rpn_bbox+=entry.get("loss_rpn_bbox",0)
                acc+=entry.get("acc",0)
                loss_bbox+=entry.get("loss_bbox",0)
                loss+=entry.get("loss",0)
                epochs[entry.get("epoch",0)]["loss_rpn_cls"] = loss_rpn_cls/counter
                epochs[entry.get("epoch",0)]["loss_rpn_bbox"] = loss_rpn_bbox/counter
                epochs[entry.get("epoch",0)]["acc"] = acc/counter
                epochs[entry.get("epoch",0)]["loss_cls"] = loss_cls/counter
                epochs[entry.get("epoch",0)]["loss_bbox"] = loss_bbox/counter
                epochs[entry.get("epoch",0)]["loss"] = loss/counter
                counter += 1
    return epochs

# Plot metrics
def plot_metrics(x, y, a, b, title, xlabel, ylabel):
    plt.figure(figsize=(4,3))
    plt.tight_layout()
    plt.plot(x, y,label="train")
    plt.plot(a,b,label="val")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)    
    plt.savefig("/teamspace/studios/this_studio/mmrotate/demo/rotated_frcnn_4_lr/plots/"+title+".png", bbox_inches='tight')
    plt.show()

# Main execution
log_data = load_log(LOG_FILE)

# Extract training data
traindata = extract_metrics(log_data, "train")

# Extract validation data
valdata= extract_metrics(log_data, "val")

# Plot Training Loss & Accuracy
#train_losses = [key: val["loss"] for key, val in traindata["loss"].items() if "loss" in val]
plot_metrics(traindata.keys(), [val["loss"] for key,val in traindata.items()], valdata.keys(), [val["loss"] for key,val in valdata.items()], "Loss Over Time", "Epochs", "Loss")
plot_metrics(traindata.keys(), [val["acc"] for key,val in traindata.items()], valdata.keys(), [val["acc"] for key,val in valdata.items()], "Acc Over Time", "Epochs", "Acc")
plot_metrics(traindata.keys(), [val["loss_bbox"] for key,val in traindata.items()], valdata.keys(), [val["loss_bbox"] for key,val in valdata.items()], "LossBBox Over Time", "Epochs", "Loss")
plot_metrics(traindata.keys(), [val["loss_bbox"] for key,val in traindata.items()], valdata.keys(), [val["loss_cls"] for key,val in valdata.items()], "LossCls Over Time", "Epochs", "Loss")
plot_metrics(traindata.keys(), [val["loss_rpn_cls"] for key,val in traindata.items()], valdata.keys(), [val["loss_rpn_cls"] for key,val in valdata.items()], "Loss RPN Cls", "Epochs", "Loss")
plot_metrics(traindata.keys(), [val["loss_rpn_bbox"] for key,val in traindata.items()], valdata.keys(), [val["loss_rpn_bbox"] for key,val in valdata.items()], "Loss RPN BBox", "Epochs", "Loss")


print("Analysis complete! Modify the script to add more metrics if needed.")