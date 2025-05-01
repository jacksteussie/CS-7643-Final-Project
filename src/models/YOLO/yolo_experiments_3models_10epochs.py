from ultralytics import YOLO

epochs=10

common_args = {
    'data': 'dota.yaml',
    'imgsz': 640,
    'epochs': epochs,
    'batch': 16,
    'device': 0,
    'project': 'experiments',
    'val': True,
    'verbose': True,
    'plots': True
}

model_n = YOLO('yolov8n.pt')
model_n.train(
    name=f'yolov8n_{epochs}epochs',
    **common_args
)

model_s = YOLO('yolov8s.pt')
model_s.train(
    name=f'yolov8s_{epochs}epochs',
    **common_args
)

model_m = YOLO('yolov8m.pt')
model_m.train(
    name=f'yolov8m_{epochs}epochs',
    **common_args
)
