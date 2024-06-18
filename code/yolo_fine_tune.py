import os
from ultralytics import YOLO

model = YOLO('yolov8n.pt')


results = model.train(data="data/images/config.yaml", epochs=50)

results = model.predict(
    source='data/images/test/', 
    device=0, 
    save_txt=True, 
    save_conf=True
)