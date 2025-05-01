from ultralytics import YOLO

model = YOLO('experiments/yolov8m_50epochs/weights/best.pt')
model.predict(source='data/dota-mod/images/test/P1796__640__3584___3072.jpg', save=True, conf=0.25)