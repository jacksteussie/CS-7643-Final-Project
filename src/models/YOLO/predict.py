from ultralytics import YOLO

model = YOLO('experiments/yolov8m_50epochs/weights/best.pt')
model.predict(source='data/dota-mod/images/test/P1796__1024__3584___2688.jpg', save=True, conf=0.25)