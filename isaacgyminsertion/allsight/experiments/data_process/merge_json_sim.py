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

random.seed(42)


def filter_id_sample(df, num):
    
    df_sorted = df
    # Create an empty DataFrame to store the filtered results
    filtered_df = pd.DataFrame()
    # Iterate through unique IDs
    for unique_id in df_sorted['sensor_id'].unique():
        # Get the top num samples with the largest time for each ID
        samples_id = df_sorted[df_sorted['sensor_id'] == unique_id].sample(n=num, random_state=42)
        filtered_df = filtered_df.append(samples_id)

    # Reset the index of the filtered DataFrame
    filtered_df = filtered_df.reset_index(drop=True)
    return filtered_df

def main(args):
    print("----------------------")
    print("[INFO] start merge_sim")

    ###########################
    # Define paths and data
    ###########################

    leds = args.leds
    gel = 'clear' #clear / markers
    indenter = ['sphere3']#, 'cube']
    data_name_1 = f'sim_train_{args.sim_data_num}'
    real_paths = [f"./datasets/data_Allsight/all_data/allsight_sim_dataset/{gel}/{leds}/data/{ind}" for ind in indenter]
    JSON_FILE_1 = f"./datasets/data_Allsight/json_data/{data_name_1}.json"

    buffer_sim_paths = []
    for p in real_paths:
        buffer_sim_paths += [y for x in os.walk(p) for y in glob(os.path.join(x[0], '*.json'))]

    for idx, p in enumerate(buffer_sim_paths):
        if idx == 0:
            df_data_sim = pd.read_json(p).transpose()
            df_data_sim['sensor_id'] = idx + 12
        else:
            df = pd.read_json(p).transpose()
            df['sensor_id'] = idx + 12
            df_data_sim = pd.concat([df_data_sim, df], axis=0)
    #train
    df_data_sim = filter_id_sample(df_data_sim, 510)
    #test
    # df_data_sim = df_data_sim[df_data_sim['sensor_id'] == 15].sample(n=3100, random_state=42)
    #
    df_data_sim = df_data_sim.sample(n=args.samples, random_state=42)
    ###########################
    # Filter and sample
    ###########################        
    old_path = "allsight_sim_dataset/"
    new_path = f"./datasets/data_Allsight/all_data/allsight_sim_dataset/"

    df_data_sim['frame'] = df_data_sim['frame'].str.replace(old_path, new_path)
    df_data_sim['frame'] = df_data_sim['frame'].str.replace(":", "_")
    
    df_data_sim['ref_frame'] = df_data_sim['ref_frame'].str.replace(old_path, new_path)
    df_data_sim['ref_frame'] = df_data_sim['ref_frame'].str.replace(":", "_")
    
    df_data_sim = df_data_sim.reset_index(drop=True)
    print(df_data_sim.shape)

    ###########################
    # Save real df to json
    ########################### 
    if args.save:
        import json

        to_dict = {}
        for index, row in list(df_data_sim.iterrows()):
            to_dict[index] = dict(row)
        with open(r'{}_transformed.json'.format(JSON_FILE_1[:-5]), 'w') as json_file:
            json.dump(to_dict, json_file, indent=3)
        
            print("[INFO] Json saved")


    print("[INFO] finish merge_sim")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process images and related JSON data.')
    parser.add_argument('--sim_data_num', type=int, default= 8, help='sim JSON path')
    parser.add_argument('--samples', type=int, default= 4000, help='sim JSON path')
    parser.add_argument('--leds', type=str, default='white', help='rrrgggbbb | white')
    parser.add_argument('--save', default=False)
    args = parser.parse_args()

    main(args)