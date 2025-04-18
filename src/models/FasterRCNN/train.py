# Source: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

from torch.utils import train_one_epoch, evaluate
import torch.utils
import torch.utils.data
from datasets import DotaDataset
from faster_rcnn_mobilenet_v3 import create_model
import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

num_classes = 16
dataset = DotaDataset()
val_dataset = DotaDataset(folder="val")
test_dataset = DotaDataset(folder="test")

data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=128, # TODO
    shuffle=True,
    # collate_fn=torch.utils.collate_fn # TODO: is this needed?
)

data_loader_test = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=1, # TODO
    shuffle=False,
    # collate_fn=torch.utils.collate_fn # TODO: is this needed?
)

model = create_model(16, True, True)
model.to(device=device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    params=params,
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005
)

lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=3,
    gamma=0.1
)

num_epochs = 10

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, data_loader_test, device=device)
