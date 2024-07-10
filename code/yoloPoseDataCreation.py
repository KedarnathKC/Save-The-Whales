import pandas as pd

pose_label_path = 'data/pose'
bbox_label_path = 'data/labels'
keypoint_labels_path = 'data/key_point_labels/labels_2016.csv'

keypoints_df = pd.read_csv(keypoint_labels_path)

print(keypoints_df)
