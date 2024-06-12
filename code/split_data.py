'''
This file is to split the data into train, test and val set.
RUN THIS FILE ONCE AT THE BEGINING
'''

import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import shutil

def find_file(root_dir, file_name):
    for root, dirs, files in os.walk(root_dir):
        if file_name in files:
            return os.path.join(root, file_name)
    return None

device = torch.device("cpu")

if torch.cuda.is_available():
   print("Training on GPU")
   device = torch.device("cuda:0")




label_path = "data/labels/labels_2016.csv"
labels = pd.read_csv(label_path)
labels = labels.dropna()
print("shape - ", labels.shape)
# duplicate_values = labels.duplicated(subset=['filename'], keep=False)
# # Remove rows with duplicate values in 'filename'
# labels_one_whale = labels[duplicate_values]
# labels_no_one_whale = labels_one_whale.drop_duplicates(subset=['filename'])
# print("shape - ", labels_no_one_whale.shape)
labels = labels.drop_duplicates(subset=['filename'])
print("shape without duplicates - ", labels.shape)

# create train and test set
train_labels, test_labels = train_test_split(labels, test_size=0.15, random_state=42)
train_labels, val_labels = train_test_split(train_labels, test_size=0.10, random_state=42)

print("Training set size:", train_labels.shape)
print("Validation set size:", val_labels.shape)
print("Test set size:", test_labels.shape)

train_labels.to_csv('data/labels/labels_2016_train.csv', index=False)
val_labels.to_csv('data/labels/labels_2016_val.csv', index=False)
test_labels.to_csv('data/labels/labels_2016_test.csv', index=False)

# move the annotated images to a different location

source_path = 'data/images/'
train_destination_path = 'data/images/train/'
test_destination_path =  'data/images/test/'
val_destination_path =  'data/images/val/'

# Move the train images
count=0
print("Moving train images....")
for index, row in train_labels.iterrows():
    image_name = row['filename']
    file_path = find_file(source_path, image_name)
    if file_path == None:
      print(image_name + " not found")
      continue
    try:
      shutil.copy(file_path, train_destination_path)
      count+=1
    except PermissionError:
       print("Permission Denied")
    except Exception as e:
      print(e)
      print(image_name + " not copied")
      break
print("Moved ", count, " images")

count=0
print("Moving validation images....")
for index, row in val_labels.iterrows():
    image_name = row['filename']
    file_path = find_file(source_path, image_name)
    
    if file_path == None:
      print(image_name + " not found")
      continue
    try:
      shutil.copy(file_path, val_destination_path)
      count+=1
    except:
      print(image_name + " not copied")
print("Moved ", count, " images")

# # Move the test images
print("Moving test images....")
count = 0
for index, row in test_labels.iterrows():
    image_name = row['filename']
    file_path = find_file(source_path, image_name)
    if file_path == None:
      print(image_name + " not found")
      continue
    try:
      shutil.copy(file_path, test_destination_path)
      count += 1
    except:
      print(image_name + " not copied")
print("Moved ", count, " images")