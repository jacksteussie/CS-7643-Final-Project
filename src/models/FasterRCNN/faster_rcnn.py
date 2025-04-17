# Source: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.faster_rcnn import FasterRCNN_MobileNet_V3_Large_320_FPN_Weights

# TODO: Pretrained model on COCO
model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT)
