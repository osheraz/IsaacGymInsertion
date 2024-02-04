import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import time
import pandas as pd
import cv2
import numpy as np


# Load json file
p = './datasets/data_Allsight/json_data/real_train_8_transformed.json'
df_data = pd.read_json(p).transpose()

# List of image paths and contact_px
image_paths = df_data['frame'].tolist()
contact_px = df_data['contact_px'].tolist()

m = 49
# Display 16 images at a time
for i in range(0, len(image_paths), m):
    fig = plt.figure(figsize=(10, 10))

    for j in range(i, min(i+m, len(image_paths))):
        # Open the image
        img = cv2.imread(image_paths[j])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert color from BGR to RGB

        # Draw a circle around the specified pixel
        center_coordinates = (int(contact_px[j][0]), int(contact_px[j][1]))  # (x,y)
        radius = int(contact_px[j][2])
        color = (255, 0, 0)  # red color in RGB
        thickness = 1  # line thickness
        img = cv2.circle(img, center_coordinates, radius, color, thickness)

        # Plot the image
        fig.add_subplot(int(np.sqrt(m)), int(np.sqrt(m)), j % m + 1)
        # cv2.imshow('Image', img)
        plt.imshow(img)
        plt.axis('off')
        

    plt.show()
    time.sleep(1)
    # wait = 1000 if i == 0 else 1
    # cv2.waitKey(wait) & 0xff