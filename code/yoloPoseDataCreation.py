import pandas as pd
import shutil

def move_files(df,src_root_dir,dest_dir):
    for index,row in df.iterrows():
        filename = row.filename[:-4]
        file_path = src_root_dir+filename+'.txt'
        try:
          shutil.copy(file_path, dest_dir)
        except PermissionError:
           print("Permission Denied")
        except Exception as e:
          print(e)
          print(filename + " not copied")
          print(file_path)
          break

def match_whale_bbox(bboxes,mid_x,mid_y,img_h,img_w):
    min_index = 0
    min_dist=999999
    for i in range(len(bboxes)):
        cx,cy,_,_ = bboxes[i]
        dist =( (cx*img_w-mid_x)**2 + (cy*img_h-mid_y)**2 )**(1/2)
        if(dist<min_dist):
            min_dist = dist
            min_index = i
    return min_index
    
    
def normalize(keypoints,h,w):
    n = len(keypoints)
    visibility=n//2
    norm = [1]*(n+visibility)
    for i in range(0,n,2):
        norm[i+i//2]=keypoints[i]/w
    for i in range(1,n,2):
        norm[i+i//2]=keypoints[i]/h
    return norm
        

def main():
    pose_label_path = 'data/labels/all_images/'
    bbox_label_path = 'data/bbox_labels/all_images/'
    keypoint_labels_path = 'data/key_point_labels/labels_2016.csv'

    keypoints_df = pd.read_csv(keypoint_labels_path)

    # Creating a dict with key as the filename and bbox co-ordinates as the list of tuples as value.
    bbox_dict = dict()
    for index,row in keypoints_df.iterrows():
        filename = row.filename[:-4]
        value = bbox_dict.get(filename,list())
        with open(bbox_label_path+filename+'.txt', 'r') as file:
            for line in file:
                components = line.split()
                cx, cy, w, h = [float(x) for x in components[1:5]]
                bbox = (cx,cy,w,h)
                value.append(bbox)
        bbox_dict[filename]=value


    cls=0
    for index,row in keypoints_df.iterrows():
        filename = row.filename[:-4]
        # print("File Name: ",filename)
        # print("bbox dict: ",  bbox_dict[filename])
        h,w = row['Image.Length'], row['Image.Width']
        index = 0
        if len(bbox_dict[filename])>1:
            index = match_whale_bbox(bbox_dict[filename],row.len_mid_x,row.len_mid_y,h,w)
        # print("Matched index: ",index)
        keypoints = normalize(list(row['rostrum_x':'fluke_y'].values),h,w)
        bbox = bbox_dict[filename][index]
        bbox_dict[filename].pop(index)
        pose = [cls]
        pose.extend(bbox)
        pose.extend(keypoints)
        # print("Len of pose: ",len(pose))
        with open(pose_label_path+filename+'.txt', 'a') as file: 
            file.write(' '.join(map(str, pose)) + '\n')
    print("Succesfully created the keypoint labels")
    
    train_label_path = 'data/key_point_labels/labels_2016_train.csv'
    val_label_path = 'data/key_point_labels/labels_2016_val.csv'
    test_label_path = 'data/key_point_labels/labels_2016_test.csv'

    train_labels = pd.read_csv(train_label_path)
    val_labels = pd.read_csv(val_label_path)
    test_labels = pd.read_csv(test_label_path)

    train_destination_path = 'data/labels/train/'
    val_destination_path = 'data/labels/val/'
    test_destination_path = 'data/labels/test/'

    # Moving Train
    move_files(train_labels,pose_label_path,train_destination_path)
    print("Successfully Moved Train Labels")

    # Moving Val
    move_files(val_labels,pose_label_path,val_destination_path)
    print("Successfully Moved Val Labels")

    # Moving Test
    move_files(test_labels,pose_label_path,test_destination_path)
    print("Successfully Moved Test Labels")

if __name__ == '__main__':
    main()
        
        
        
        
            
            

        