# %%

import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

from glob import glob
import random

# %%
import yaml
with open('best_params_task.yaml', 'r') as file:
    file = yaml.load(file, Loader=yaml.Loader)

all_paths = glob('/home/osher/Desktop/isaacgym/python/IsaacGymInsertion/isaacgyminsertion/outputs/test/data/datastore_42_test/*/*.npz')
print(len(all_paths))
path = random.sample(all_paths, 1)[0]

data = np.load(path)
# path = random.sample(all_paths, 1)[0]
print(path)

done_idx = data['done'].nonzero()[-1][0]


print(done_idx)
ax = plt.figure(figsize=(10, 20)).add_subplot(projection='3d')

if False:
    # ax.scatter(data['socket_pos'][:done_idx, 0],
    #            data['socket_pos'][:done_idx, 1],
    #            zs=data['socket_pos'][:done_idx, 2],  color='r')

    # ax.plot(data['noisy_socket_pos'][:done_idx, 0],
    #         data['noisy_socket_pos'][:done_idx, 1],
    #         zs=data['noisy_socket_pos'][:done_idx, 2])

    # ax.plot(data['plug_pos'][:done_idx, 0], data['plug_pos'][:done_idx, 1], zs=data['plug_pos'][:done_idx, 2])
    ax.plot(data['eef_pos'][:done_idx, 0], data['eef_pos'][:done_idx, 1], zs=data['eef_pos'][:done_idx, 2])
    ax.set_xlabel('$X$', fontsize=20, rotation=150)
    ax.set_ylabel('$Y$',fontsize=20, rotation=150)
    ax.set_zlabel('$Z$', fontsize=30, rotation=60)

    delta = 0.06
    # ax.axes.set_xlim3d(left=0.5 - delta, right=0.5 + delta)
    # ax.axes.set_ylim3d(bottom=0 - delta, top=0 + delta)
    # ax.axes.set_zlim3d(bottom=0, top=0.2)

    plt.show()
    # %%
if False:
    plt.scatter(data['noisy_socket_pos'][1:done_idx, 0], data['noisy_socket_pos'][1:done_idx, 1], color='b')
    plt.plot(data['plug_pos'][1:done_idx, 0], data['plug_pos'][1:done_idx, 1])
    plt.plot(data['eef_pos'][1:done_idx, 0], data['eef_pos'][1:done_idx, 1])
    plt.scatter(data['socket_pos'][1:done_idx, 0], data['socket_pos'][1:done_idx, 1], color='r', s=35)
    plt.show()

# %%
if False:
    latent = data['latent']
    plt.plot(latent[:done_idx, :])
    plt.show()

# %%
if False:
    plt.scatter(data['noisy_socket_pos'][1:done_idx, 0], data['noisy_socket_pos'][1:done_idx, 1], color='b')
    plt.plot(data['plug_pos'][1:done_idx, 0], data['plug_pos'][1:done_idx, 1])
    plt.plot(data['eef_pos'][1:done_idx, 0], data['eef_pos'][1:done_idx, 1])
    plt.scatter(data['socket_pos'][1:done_idx, 0], data['socket_pos'][1:done_idx, 1], color='r', s=35)
    plt.show()



if False:
    import cv2
    import numpy as np
    from tqdm import tqdm

    # choose codec according to format needed
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter('video.avi', fourcc, 20, (112, 672), isColor=False)

    for j in tqdm(range(0, done_idx)):

        tactile_img = data['tactile']
        img1 = tactile_img[j][0]
        img2 = tactile_img[j][1]
        img3 = tactile_img[j][2]

        img = np.concatenate([img1, img2, img3], axis=1)
        # video.write((img * 255).astype(np.uint8))
        cv2.imshow('img', img)
        cv2.waitKey(1)

    cv2.destroyAllWindows()
    video.release()

