import numpy as np
import pandas as pd
import os
import random
import cv2
import re
import json
import argparse

def circle_mask(size=(480, 480), border=0, fix=(0,0)):
    mask = np.zeros((size[0], size[1]), dtype=np.uint8) 
    mask = cv2.circle(mask, (int(size[0]/2 -1), int(size[1]/2 -1)), int(size[0]/2 -13), 1, thickness=-1)
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    return mask

def inv_foreground(ref_frame, diff, offset=0.0):
        
        ref_frame = np.float32(ref_frame)
        diff = np.float32(diff)
        diff = (diff*2 - 255) 
        frame = ref_frame + diff
        mask = circle_mask()
        frame = (frame).astype(np.uint8)
        frame = frame*mask
        return frame
    
def foreground(frame, ref_frame, offset=0.0):
        
        frame = np.float32(frame)
        ref_frame = np.float32(ref_frame)
        diff_frame = frame - ref_frame
        diff_frame = diff_frame  + 255
        mask = circle_mask()
        diff_frame = (diff_frame/2).astype(np.uint8)
        diff_frame = diff_frame*mask
        return diff_frame

def main(args):
    random.seed(42)
    
    json_path = f'./datasets/data_Allsight/json_data/{args.data_type}_{args.data_set}_{args.data_num}_transformed.json'
    copy_to_path = f'./datasets/data_Allsight/diff_images/{args.data_type}_{args.data_num}/{args.data_set}/'
    
    # Create the directory if it doesn't exist
    if not os.path.exists(copy_to_path):
        os.makedirs(copy_to_path)


    df_data = pd.read_json(json_path).transpose()
    df_data['diff_frame'] = df_data['frame']

    if args.save:
        for idx in range(len(df_data)):
            frame = (cv2.imread(df_data['frame'][idx])).astype(np.uint8)
            ref_frame = (cv2.imread(df_data['ref_frame'][idx])).astype(np.uint8)
            
            diff_image = foreground(frame, ref_frame, offset=0.2)
            save_path = os.path.join(copy_to_path, f'diff_{args.data_type}{idx}.jpg')
            
            cv2.imwrite(save_path, diff_image)
            df_data['diff_frame'][idx] = save_path
            
        # Save df to json
        to_dict = {}
        for index, row in list(df_data.iterrows()):
            to_dict[index] = dict(row)
        with open(f'{json_path}', 'w') as json_file:
            json.dump(to_dict, json_file, indent=3)
            print("[INFO] json saved")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process images and related JSON data.')
    parser.add_argument('--data_type', type=str, default='real', help='real, sim')
    parser.add_argument('--data_set', type=str, default='train', help='train, test')
    parser.add_argument('--data_num', type=int, default= 8, help='num JSON path')
    parser.add_argument('--save', default=False)
    args = parser.parse_args()

    main(args)