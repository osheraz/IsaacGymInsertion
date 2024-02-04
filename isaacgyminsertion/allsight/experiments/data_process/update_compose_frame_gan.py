import numpy as np
import pandas as pd
import os
import random
import cv2
import re
import json
import argparse
import torch
import sys

PATH = os.path.join(os.path.dirname(__file__), "../")
sys.path.insert(0, PATH)

from util.util import tensor2im
from models import networks, pre_process

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

def circle_mask(size=(224, 224), border=10, fix=(0,0)):
    mask = np.zeros((size[0], size[1]), dtype=np.uint8) 
    m_center = (size[0] // 2, size[1] // 2)
    m_radius = min(size[0], size[1]) // 2 - border
    mask = cv2.circle(mask, m_center, m_radius, 1, thickness=-1)
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

def is_image_file(filename):
    # Check if the file has an image extension
    image_extensions = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff")
    return filename.lower().endswith(image_extensions)

def get_image_paths_from_folder(folder_path):
    # Get all image paths from a given folder
    image_files = [f for f in os.listdir(folder_path) if is_image_file(f)]
    image_paths = [os.path.join(folder_path, image_file) for image_file in image_files]
    return image_paths

def get_image_number(image_path):
    # Extract image number from image file name
    filename = os.path.basename(image_path)
    image_name = os.path.splitext(filename)[0]
    match = re.search(r'\d+', image_name)
    return int(match.group()) if match else float('inf')

def main(args):
    random.seed(42)
    
    json_gan_p = f'./datasets/data_Allsight/json_data/{args.gan_type}_test_{args.gan_num}_{args.sim_data_num}_{args.gan_epoch}_transformed.json'
    json_gan_p_new = f'./datasets/data_Allsight/json_data/{args.gan_type}_test_{args.gan_num}_{args.sim_data_num}_{args.gan_epoch}_transformed_ref.json'
    copy_to_path = f'./datasets/data_Allsight/{args.gan_type}_data/test_{args.gan_num}_{args.gan_epoch}/compose_frames/'
    
    # Create the directory if it doesn't exist
    if not os.path.exists(copy_to_path):
        os.makedirs(copy_to_path)
           
    df_data = pd.read_json(json_gan_p).transpose()

    if args.save:
        # Save real image df
        for idx in range(len(df_data)):
            # = cv2.resize(cv2.cvtColor(cv2.imread(self.real_df['ref_frame'][self.real_A_num]), cv2.COLOR_BGR2RGB),(224,224))
            ref_image = cv2.resize(cv2.imread(df_data['ref_frame'][idx]),(224,224))
            diff_image = cv2.imread(df_data['diff_frame'][idx])
            comp_image = inv_foreground(ref_image, diff_image, offset=0.0)
            # sim_ref_image = cv2.cvtColor(sim_ref_image, cv2.COLOR_BGR2RGB).astype(np.uint8)
            

            # gan_ref_image = cv2.cvtColor(gan_ref_image, cv2.COLOR_RGB2BGR)
            
            save_path = os.path.join(copy_to_path, f'{args.gan_type}_comp_{idx}.jpg')
            cv2.imwrite(save_path, comp_image)
            df_data['frame'][idx] = save_path

        # Save real df to json
        to_dict = {}
        for index, row in list(df_data.iterrows()):
            to_dict[index] = dict(row)
        with open(f'{json_gan_p_new}', 'w') as json_file:
            json.dump(to_dict, json_file, indent=3)
            
            print("[INFO] json updated")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process images and related JSON data.')
    parser.add_argument('--sim_data_num', type=int, default= 8, help='sim JSON path')
    parser.add_argument('--data_kind', type=str, default='transformed', help='transformed, aligned')
    parser.add_argument('--gan_num', type=str, default= 60)
    parser.add_argument('--gan_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
    parser.add_argument('--gan_type', type=str, default='cgan', help='cgan, distil_cgan, mask_cgan')
    parser.add_argument('--save', default=False)
    args = parser.parse_args()

    main(args)