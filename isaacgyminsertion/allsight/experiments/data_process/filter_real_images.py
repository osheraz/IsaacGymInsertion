###########################
# Import
###########################
import numpy as np
import pandas as pd
import os
from glob import glob
import matplotlib.pyplot as plt
import random
import argparse
import json
import cv2
import matplotlib.pyplot as plt
import time

random.seed(42)


def display_images(df):
    # List of image paths and contact_px
    image_paths = df['frame'].tolist()
    contact_px = df['contact_px'].tolist()

    # Display 16 images at a time
    for i in range(0, len(image_paths), 16):
        fig = plt.figure(figsize=(10, 10))

        for j in range(i, min(i + 16, len(image_paths))):
            # Open the image
            img = cv2.imread(image_paths[j])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert color from BGR to RGB

            # Draw a circle around the specified pixel
            center_coordinates = (int(contact_px[j][0]), int(contact_px[j][1]))  # (x,y)
            radius = int(contact_px[j][2])
            color = (255, 0, 0)  # red color in RGB
            thickness = 2  # line thickness
            img = cv2.circle(img, center_coordinates, radius, color, thickness)

            # Plot the image
            fig.add_subplot(4, 4, j % 16 + 1)
            # cv2.imshow('Image', img)
            plt.imshow(img)
            plt.axis('off')

        plt.show()
        time.sleep(1)

    return


def filter_id_time(df, num):
    df_sorted = df.sort_values(by=['sensor_id', 'time'], ascending=[True, False])
    # Create an empty DataFrame to store the filtered results
    filtered_df = pd.DataFrame()
    # Iterate through unique IDs
    for unique_id in df_sorted['sensor_id'].unique():
        # Get the top num samples with the largest time for each ID
        top_samples = df_sorted[df_sorted['sensor_id'] == unique_id].head(num)
        filtered_df = filtered_df.append(top_samples)

    # Reset the index of the filtered DataFrame
    filtered_df = filtered_df.reset_index(drop=True)
    return filtered_df


def get_third_element(lst):
    return lst[2]


def get_buffer_paths(leds, gel, indenter, train_sensor_id, test_sensor_id,
                     dataset_path='/datasets/all_data/allsight_dataset'):
    # sensor id = train sensors id
    trained_sensor_id_final = []
    test_sensor_id_final = []

    buffer_paths_to_train = []
    buffer_paths_to_test = []

    if leds == 'combined':
        leds_list = ['rrrgggbbb', 'rgbrgbrgb', 'white']
    else:
        leds_list = [leds]

    for l in leds_list:
        paths = [f".{dataset_path}/{gel}/{l}/data/{ind}" for ind in indenter]
        buffer_paths = []
        summ_paths = []

        for p in paths:
            buffer_paths += [y for x in os.walk(p) for y in glob(os.path.join(x[0], '*_transformed_annotated.json'))]
            summ_paths += [y for x in os.walk(p) for y in glob(os.path.join(x[0], 'summary.json'))]

        for bp, s in zip(buffer_paths, summ_paths):
            with open(s, 'rb') as handle:
                summ = json.load(handle)
                summ['sensor_id'] = summ['sensor_id'] if isinstance(summ['sensor_id'], list) else [summ['sensor_id']]

            for sm in summ['sensor_id']:
                if sm in train_sensor_id:
                    buffer_paths_to_train.append(bp)
                    trained_sensor_id_final.append(sm)
                    print(f'Sensor {sm} is in the train set')
                elif sm in test_sensor_id:
                    buffer_paths_to_test.append(bp)
                    test_sensor_id_final.append(sm)
                    print(f'Sensor {sm} is in the test set')
                else:
                    print(f'Sensor {sm} is not in the train set and not in the test set')

    return buffer_paths_to_train, buffer_paths_to_test, list(trained_sensor_id_final), list(test_sensor_id_final)


def main(args):
    print("----------------------")
    print("[INFO] start filter_real_images ")

    ###########################
    # Define paths and data
    ###########################
    pc_name = os.getlogin()
    leds = args.leds
    gel = 'clear'  # clear / markers
    indenter = ['ellipse', 'square', 'sphere3', 'sphere4', 'sphere5', 'hexagon']  # id 3 only 20 (3mm radius)
    data_name_1 = f'real_train_{args.real_data_num}'
    data_name_2 = f'real_test_{args.real_data_num}'
    # real_paths = [f"./datasets/data_Allsight/all_data/allsight_dataset/{gel}/{leds}/data/{ind}" for ind in indenter]
    JSON_FILE_1 = f"./datasets/data_Allsight/json_data/{data_name_1}.json"
    JSON_FILE_2 = f"./datasets/data_Allsight/json_data/{data_name_2}.json"

    n_sam_train = 4000
    n_sam_test = 1500

    # ,13,14,16,17,18,19
    buffer_train_paths, buffer_test_paths, sensors_1, sensors_2 = get_buffer_paths(leds, gel, indenter,
                                                                                   train_sensor_id=[12],
                                                                                   test_sensor_id=[15])

    # ind = ['ellipse', 'ellipse','square' ,'square']
    ###########################
    # Define data frames
    ###########################

    for idx, p in enumerate(buffer_train_paths):
        if idx == 0:
            df_data_real_train = pd.read_json(p).transpose()
            df_data_real_train['sensor_id'] = sensors_1[idx]
        else:
            df = pd.read_json(p).transpose()
            df['sensor_id'] = sensors_1[idx]
            df_data_real_train = pd.concat([df_data_real_train, df], axis=0)

    for idx, p in enumerate(buffer_test_paths):
        if idx == 0:
            df_data_real_test = pd.read_json(p).transpose()
            df_data_real_test['sensor_id'] = sensors_2[idx]
            # df_data_real_test['indenter'] = ind[idx]
        else:
            df = pd.read_json(p).transpose()
            df['sensor_id'] = sensors_2[idx]
            # df['indenter'] = ind[idx]
            df_data_real_test = pd.concat([df_data_real_test, df], axis=0)

    # If we do filter by force         
    # df_data_real_train['ft_z'] = df_data_real_train['ft_ee_transformed'].apply(get_third_element)
    # df_data_real_test['ft_z'] = df_data_real_test['ft_ee_transformed'].apply(get_third_element)    

    ###########################
    # Filter and sample
    ###########################

    old_path = "/home/osher/catkin_ws/src/allsight/dataset/"
    new_path = f"./datasets/data_Allsight/all_data/allsight_dataset/"

    df_data_real_test['frame'] = df_data_real_test['frame'].str.replace(old_path, new_path)
    df_data_real_test['ref_frame'] = df_data_real_test['ref_frame'].str.replace(old_path, new_path)
    df_data_real_train['frame'] = df_data_real_train['frame'].str.replace(old_path, new_path)
    df_data_real_train['ref_frame'] = df_data_real_train['ref_frame'].str.replace(old_path, new_path)

    df_data_real_train = filter_id_time(df_data_real_train, 730)

    # df_data_real_train = df_data_real_train[df_data_real_train.time > 3.5]
    df_data_real_test = df_data_real_test[df_data_real_test.time > 4.3]
    ###
    df_train_up = df_data_real_train[df_data_real_train.num > 8]
    df_train_down = df_data_real_train[df_data_real_train.num <= 8]
    df_train_up = df_train_up.drop(df_train_up.index[1::3])
    df_train_up = df_train_up.reset_index(drop=True)
    df_train = pd.concat([df_train_down, df_train_up], ignore_index=True)

    df_test_up = df_data_real_test[df_data_real_test.num > 8]
    df_test_down = df_data_real_test[df_data_real_test.num <= 8]
    df_test_up = df_test_up.drop(df_test_up.index[1::2])
    df_test_up = df_test_up.reset_index(drop=True)
    df_test = pd.concat([df_test_down, df_test_up], ignore_index=True)
    ###
    # df_train = df_data_real_train
    # df_test = df_data_real_test
    ###
    # df_train = df_train.sample(n=n_sam_train, random_state=42)
    df_test = df_test.sample(n=n_sam_test, random_state=42)

    print(f'train: {df_train.shape}')
    for i in range(9):
        print(df_train[df_train.num == i + 2].shape)
    print(f'test: {df_test.shape}')
    for i in range(9):
        print(df_test[df_test.num == i + 2].shape)
    # df_data_real_train = df_data_real_train[df_data_real_train.num < 7.0]
    # df_data_real_train = df_data_real_train[df_data_real_train.time < 4.0]
    # display_images(df_data_real_train)  
    # df_train_real = df_train_real.sample(n=1000, random_state=42)
    df_data_real_train = df_train.reset_index(drop=True)
    df_data_real_test = df_test.reset_index(drop=True)
    ###########################
    # Save real df to json
    ########################### 
    if args.save:

        to_dict = {}
        for index, row in list(df_data_real_train.iterrows()):
            to_dict[index] = dict(row)
        with open(r'{}_transformed.json'.format(JSON_FILE_1[:-5]), 'w') as json_file:
            json.dump(to_dict, json_file, indent=3)

            print("[INFO] real train saved")

        to_dict2 = {}
        for index, row in list(df_data_real_test.iterrows()):
            to_dict2[index] = dict(row)
        with open(r'{}_transformed.json'.format(JSON_FILE_2[:-5]), 'w') as json_file:
            json.dump(to_dict2, json_file, indent=3)

            print("[INFO] real test saved")

    print("[INFO] finish filter_real_images")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process images and related JSON data.')
    parser.add_argument('--real_data_num', type=int, default=9, help='real JSON path')
    parser.add_argument('--save', default=False)
    parser.add_argument('--leds', type=str, default='white', help='rrrgggbbb | white')
    args = parser.parse_args()

    main(args)
