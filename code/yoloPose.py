'''
Link to the model parameters:   
https://github.com/ultralytics/ultralytics/blob/main/docs/en/modes/train.md

k-Obj_loss being zero:
x
'''
from ultralytics import YOLO

def train():

    model = YOLO("yolov8s-pose.pt") 
    # model = YOLO('runs/pose/train11/weights/last.pt')
    
    # Train the model
    results = model.train(data="data/config_pose.yaml", epochs=50, imgsz=640, batch=16, name="AdamW lr0.02 lrf0.001",optimizer='AdamW',lr0=0.02, lrf=0.001)

    # results = model.predict(
    #     source='data/images/test/', 
    #     device=0, 
    #     save_txt=True, 
    #     save_conf=True
    # )

    # path = 'data/images/test/2016.08.12.500.04.JPG'

    # result = model(path)[0]



if __name__ == '__main__':
    train()