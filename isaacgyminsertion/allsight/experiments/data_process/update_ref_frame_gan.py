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
    
    model_G_path = f'./checkpoints/allsight_{args.gan_num}/{args.gan_epoch}_net_G_B.pth'
    json_gan_p = f'./datasets/data_Allsight/json_data/{args.gan_type}_test_{args.gan_num}_{args.sim_data_num}_{args.gan_epoch}_transformed.json'
    json_gan_p_new = f'./datasets/data_Allsight/json_data/{args.gan_type}_test_{args.gan_num}_{args.sim_data_num}_{args.gan_epoch}_transformed_ref.json'
    copy_to_path = f'./datasets/data_Allsight/{args.gan_type}_data/test_{args.gan_num}_{args.gan_epoch}/ref_frames/'
    
    # Create the directory if it doesn't exist
    if not os.path.exists(copy_to_path):
        os.makedirs(copy_to_path)
    
    opt = {
        "preprocess": "resize_and_crop",
        "crop_size": 224,
        "load_size": 224,
        "no_flip": True,
    }  
    transform = pre_process.get_transform(opt=opt)
    model_G = networks.define_G(input_nc=3,
                                    output_nc=3,
                                    ngf=64,
                                    netG="resnet_9blocks",
                                    norm="instance",
                                    )
    model_G.load_state_dict(torch.load(model_G_path))
    model_G = model_G.to(device)
    model_G.eval()
    
        
    df_data = pd.read_json(json_gan_p).transpose()

    if args.save:
        # Save real image df
        for idx, img_path in enumerate(df_data['ref_frame']):
            sim_ref_image = (cv2.imread(img_path))
            sim_ref_image = cv2.cvtColor(sim_ref_image, cv2.COLOR_BGR2RGB).astype(np.uint8)
            
            gan_im_tensor = transform(sim_ref_image).unsqueeze(0)
            gan_im_tensor = gan_im_tensor.to(device)
            gan_ref_image = tensor2im(model_G(gan_im_tensor))
            gan_ref_image = cv2.cvtColor(gan_ref_image, cv2.COLOR_RGB2BGR)
            
            save_path = os.path.join(copy_to_path, f'{args.gan_type}{idx}.jpg')
            cv2.imwrite(save_path, gan_ref_image)
            df_data['ref_frame'][idx] = save_path

        # Save real df to json
        to_dict = {}
        for index, row in list(df_data.iterrows()):
            to_dict[index] = dict(row)
        with open(f'{json_gan_p_new}', 'w') as json_file:
            json.dump(to_dict, json_file, indent=3)
            
            print("[INFO] json updated")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process images and related JSON data.')
    parser.add_argument('--sim_data_num', type=int, default= 7, help='sim JSON path')
    parser.add_argument('--data_kind', type=str, default='transformed', help='transformed, aligned')
    parser.add_argument('--gan_num', type=str, default= 28)
    parser.add_argument('--gan_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
    parser.add_argument('--gan_type', type=str, default='distil_cgan', help='cgan, distil_cgan, mask_cgan')
    parser.add_argument('--save', default=True)
    args = parser.parse_args()

    main(args)