from ultralytics import YOLO

epochs=50

common_args = {
    'data': 'dota.yaml',
    'imgsz': 640,
    'epochs': epochs,
    'batch': 16,
    'device': 0,
    'project': 'experiments',
    'name': 'yolov8m_50epochs',
    'val': True,
    'verbose': True,
    'plots': True
}

model_m = YOLO('yolov8m.pt')
model_m.train(**common_args)