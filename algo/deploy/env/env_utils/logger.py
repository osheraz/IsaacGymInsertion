import cv2
import os
from datetime import datetime
import json


class DataLogger():

    def __init__(self, conf):

        self.data_dict = {}
        self.img_press_dict = {}
        self.save = conf['save']
        # Init of the dataset dir paths with the current day and time
        self.date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S")
        dir_path = '/home/osher/catkin_ws/src/allsight'
        self.dataset_path_images = dir_path + "/dataset/{}/{}/images/{}/img_{}".format(conf['gel'], conf['leds'], conf['indenter'], self.date)
        self.dataset_path_data = dir_path + "/dataset/{}/{}/data/{}/data_{}".format(conf['gel'], conf['leds'], conf['indenter'], self.date)

        if self.save:
            if not os.path.exists(self.dataset_path_images): os.mkdir(self.dataset_path_images)
            if not os.path.exists(self.dataset_path_data): os.mkdir(self.dataset_path_data)

            # create a summary
            file_path = os.path.join(self.dataset_path_data, 'summary.json')
            with open(file_path, 'w') as fp:
                json.dump(conf, fp, indent=3)

    def append(self, i, q, frame, trans, rot, trans_ee, rot_ee, ft, save_time=0.0):

        ref_id = 'image{}_{:.2f}_0.jpg'.format(i, q)
        ref_path = os.path.join(self.dataset_path_images, ref_id)

        if save_time:
            img_id = 'image{}_{:.2f}_{:.2f}.jpg'.format(i, q, save_time)
            img_path = os.path.join(self.dataset_path_images, img_id)
        else:
            img_id = ref_id
            img_path = ref_path

        self.img_press_dict[img_path] = frame

        self.data_dict[img_id] = {'frame': img_path,
                                  'ref_frame': ref_path,
                                  'pose': (trans, rot),
                                  'pose_ee': (trans_ee, rot_ee),
                                  'theta': q,
                                  'ft': ft.tolist(),
                                  'time': save_time,
                                  'num': i}

    def save_batch_images(self):
        # Save images
        for key in self.img_press_dict.keys():
            if not cv2.imwrite(key, self.img_press_dict[key]):
                raise Exception("Could not write image")

        # Clear the dict
        self.img_press_dict.clear()

    def save_data_dict(self):

        path = os.path.join(self.dataset_path_data, 'data_{}.json'.format(self.date))
        if self.save:
            with open(path, 'w') as json_file:
                json.dump(self.data_dict, json_file, indent=3)
