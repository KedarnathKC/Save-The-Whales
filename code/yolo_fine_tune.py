import os
from ultralytics import YOLO

def train():
    # Change the model to standard yolov8n.pt model while training
    # model = YOLO('runs/detect/train/weights/best.pt')
    model = YOLO('yolov8n.pt')

    # This will save the training outputs in runs/detect/train folder
    results = model.train(data="data/config.yaml", epochs=100)

    # This will save the predicted bounding box co-ordinates as .txt file in runs/detect/predict folder
    # results = model.predict(
    #     source='data/images/test/', 
    #     device=0, 
    #     save_txt=True, 
    #     save_conf=True
    # )

    
if __name__ == '__main__':
    train()