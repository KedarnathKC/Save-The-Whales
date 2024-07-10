# instantiate the pretrained SAM model
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import numpy as np
import torch
import pandas as pd
import pickle
from torchvision import transforms

def get_YOLO_boxes(box_dir, image_height, image_width, image_name):

    detections_path = box_dir + image_name + ".txt"

    bboxes = []
    conf_scores = []

    with open(detections_path, 'r') as file:
      for line in file:
        components = line.split()
        # class_id = int(components[0])
        confidence = float(components[5])
        cx, cy, w, h = [float(x) for x in components[1:5]]

        # Convert from normalized [0, 1] to image scale
        cx *= image_width
        cy *= image_height
        w *= image_width
        h *= image_height

        # Convert the center x, y, width, and height to xmin, ymin, xmax, ymax
        xmin = cx - w / 2
        ymin = cy - h / 2
        xmax = cx + w / 2
        ymax = cy + h / 2

        # class_ids.append(class_id)
        bboxes.append((xmin, ymin, xmax, ymax))
        conf_scores.append(confidence)

    return bboxes, conf_scores

def generate_SAM_masks(bboxes):

  masks = []

  for bbox in zip(bboxes):
    # print(f'BBox coordinates: {bbox}')
    input_box = np.array(bbox)

    # Generate the mask for the current bounding box
    mask, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box,
        multimask_output=False,
    )

    mask_val = np.where(mask[0] == True, 1.0, 0.0)
    # print("mask_val - ", mask_val.shape)
    masks.append(torch.Tensor(mask_val))
    # print(len(masks))
  # print("length of masks - ", len(masks))
  return masks

# match
# for a given image, select all the boxes bboxes
# find the number of entries in the sheet for those boxes
# for each entry in the sheet, find the closest box wrt midx and midy (should be lower than a threshold) remove all other boxes
# for each entry in sheet, generate masks for the boxes and create x and y
# store x = masks, y = value from sheet into tensors as bundle
# randomly diide into train, val and test

def find_entries(labels, image_name):
    entries = labels[labels['filename'] == image_name + '.JPG']
    return entries

def match_entries(entries, bboxes):
    transform = transforms.ToTensor()
    y = []
    boxes = []
    for index, row in entries.iterrows():
        mid_x = row['len_mid_x']
        mid_y = row['len_mid_y']
        val = math.sqrt(mid_x**2 + mid_y**2)
        minD = 9999
        minBox = None
        for bbox in bboxes:
          # xmin, ymin, xmax, ymax
          box_mid_x = (bbox[2]-bbox[0])/2
          box_mid_y = (bbox[3]-bbox[1])/2
          box_val = math.sqrt(box_mid_x**2 + box_mid_y**2)
          if(mid_x >= bbox[0] and mid_x <= bbox[2] and mid_y >= bbox[1] and mid_y <= bbox[3] and box_val < minD):
              minD = box_val
              minBox = bbox
        if(minBox != None):
          # print(torch.Tensor(list(row['rostrum_x':'fluke_y'].values)).shape)
          y.append(torch.Tensor(list(row['rostrum_x':'fluke_y'].values)))
          # print(transform(row['rostrum_x':'fluke_y'].values).shape)
          boxes.append(minBox)
          bboxes.remove(minBox)

    return y, boxes

# Load the data
train_label_path = "labels/labels_2016_train.csv"
val_label_path = "labels/labels_2016_val.csv"
test_label_path = "labels/labels_2016_test.csv"
all_labels_path = "labels/labels_2016.csv"
train_labels = pd.read_csv(train_label_path)
val_labels = pd.read_csv(val_label_path)
test_labels = pd.read_csv(test_label_path)
all_labels = pd.read_csv(all_labels_path)

sam_checkpoint = "./models/sam/sam_vit_h_4b8939.pth"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
predictor = SamPredictor(sam)


# Save the data using Pickle
transform = transforms.ToTensor()

sam_image_tensors = []
image_names_list = []
y_tensors = []
all_boxes = []

labels = train_labels
box_dir = train_box_dir
save_dir = "images/images_2016/sam/data/train/data7.pkl"
image_dir = raw_train_image_dir

count = 0

for index, row in labels.iterrows():
    image_name = row['filename'][:-4]
    image_path = image_dir + image_name + '.JPG'
    image = cv2.imread(image_path)
    # print("image_name - ", image_name)
    # print("image shape - ", image.shape)
    #   continue


    # Read YOLO box values
    bboxes, conf_scores = get_YOLO_boxes(box_dir, image.shape[0], image.shape[1], image_name)

    # Keep the most confident box and remove others
    # conf_mask_index = conf_scores.index(max(conf_scores))
    # bboxes = [bboxes[conf_mask_index]]
    # conf_scores = [conf_scores[conf_mask_index]]

    entries = find_entries(all_labels, image_name)
    # print("Number of entries - ", entries.shape[0])

    y_list, box_list = match_entries(entries, bboxes)
    # print("number of y - ", len(y_list))
    # print("number of box - ", len(box_list))
    if(len(y_list) == 0):
      print("No matches for " + image_name + '.JPG')
      continue

    # Generate SAM masks
    predictor.set_image(image)
    masks = generate_SAM_masks(box_list)

    # Concatenate the SAM mask as the 4th channel
    image_tensor = torch.Tensor(image).permute(2, 0, 1)

    if(len(y_list) == len(box_list) and len(y_list) == len(masks)):
      for i in range(len(masks)):
          current_image_tensor = torch.cat((image_tensor, masks[i].unsqueeze(0)), dim=0)
          # print("current_image_temnsor - ", current_image_tensor.shape)
          # current_image_tensor[3,:,:] = masks[i]
          sam_image_tensors.append(current_image_tensor)
          y_tensors.append(y_list[i])
          all_boxes.append(box_list[i])
          image_names_list.append(image_name + '.JPG')
    else:
      print("masks, boxes and y lengths not equal")

    print("Done - " + str(count) + "/397")
    count+=1
print(len(sam_image_tensors))
print(len(image_names_list))
print(len(y_tensors))
print(len(all_boxes))

data = {'x': sam_image_tensors, 'image_names': image_names_list, 'y':y_tensors, 'bboxes': all_boxes}

with open(save_dir, 'wb') as f:
    pickle.dump(data, f)

# x = torch.stack(image_tensors)
# # x = F.interpolate(x, size=(x.shape[2]//4, x.shape[3]//4), mode='bilinear', align_corners=False)
# print("Shape of the batch tensor: ", x.shape)

# save tensors (with SAM masks) in a file
# torch.save(tensor, 'images/images_2016/yolov9/pretrained_yolov9e/SAM_single_whale.pt')