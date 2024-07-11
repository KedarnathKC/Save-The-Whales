from ultralytics import YOLO
# from PIL import Image
import cv2
from torchvision import transforms
from matplotlib import pyplot as plt

import matplotlib.patches as patches



def get_gt_labels_in_img_dim(gt_path,img_h,img_w):
    bboxes=list()
    with open(gt_path,'r') as f:
        for line in f:
            components = line.split()
            cx, cy, w, h = [float(x) for x in components[1:5]]
            bbox = (cx*img_w,cy*img_h,w*img_w,h*img_h)
            bboxes.append(bbox)
    return bboxes
        
def get_kpts_in_img_dim(gt_path,img_h,img_w):
    kpts=list()
    with open(gt_path,'r') as f:
        for line in f:
            components=line.split()[5:]
            # print("Components: ",components)
            kpt=list()
            for i in range(0,len(components),3):
                # print(components[i])
                # print(components[i+1])
                kpts.append((float(components[i])*img_w,float(components[i+1])*img_h))
            # kpts.append(kpt)
    return kpts
            
    
def inference(model_path,img_path,gt_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # define transform function
    h,w,_=img.shape
    gt = get_gt_labels_in_img_dim(gt_path,h,w)
    kpts= get_kpts_in_img_dim(gt_path,h,w)
    # print("GT BBox: ",gt)
    # print("GT KPts: ",kpts)
    fig, ax = plt.subplots(1,2,figsize=(24, 12))
    ax[0].imshow(img)
    ax[1].imshow(img)
    
    for box in gt:
        cx,cy,w,h = box
        x = cx - w/2
        y = cy - h/2
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='b', facecolor='none')
        ax[0].add_patch(rect)
    for kpt in kpts:
        x,y = kpt
        ax[0].plot(x, y, 'bo', markersize=5)  
        
    model = YOLO(model_path)
    results= model(img_path)[0]
    pred_boxes = results.boxes.xywh.cpu().tolist()
    pred_kpts = results.keypoints.xy.cpu().flatten().tolist()
    # print("Pred Kpts: ",pred_kpts)
    for box in pred_boxes:
        # print("Box: ",box)
        x,y,w,h = box
        x = x -w/2
        y = y -h/2
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
        ax[1].add_patch(rect)
    for i in range(0,len(pred_kpts),2):
        x,y = pred_kpts[i],pred_kpts[i+1]
        # print(x,y)
        ax[1].plot(x, y, 'ro', markersize=5)  
    ax[0].set_title("Ground Truth")
    ax[1].set_title("Prediction")
    plt.show()
        
if __name__=='__main__':
    model_path = '../runs/pose/train11/weights/best.pt'
    img_name = '2016.06.28.69.04'
    img_path = '../data/images/test/'+img_name+'.jpg'
    gt_path = '../data/labels/test/'+img_name+'.txt'
    inference(model_path,img_path,gt_path)
    
    
    