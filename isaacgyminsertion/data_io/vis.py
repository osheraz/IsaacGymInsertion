# %%

import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import os
from glob import glob
import random

# %%
import yaml

all_paths = glob('/home/roblab20/tactile_insertion/datastore_42_contact2/*/*.npz')
print(len(all_paths))

test=False

if False:
    for i in range(len(all_paths)):
        # while not opened:
        try:
            # path = random.sample(all_paths, 1)[0]
            path = all_paths[i]
            data = np.load(path)
            # print(path)
            done_idx = data['done'].nonzero()[-1][0]
            if done_idx > 400 or done_idx < 2:
                continue

            opened = True
        except Exception as e:
            print(f"Error loading file {path}: {e}")
            # Remove the bad file path from the list
            all_paths.remove(path)
            try:
                # Remove the file from the computer
                os.remove(path)
                print(f"File {path} removed from the computer.")
            except Exception as e:
                print(f"Error removing file {path}: {e}")

        # path = random.sample(all_paths, 1)[0]
        print(path)
        # all_paths.remove(path)

        opened = True

        print(done_idx)

        if True:
            import cv2
            import numpy as np
            from tqdm import tqdm


            # choose codec according to format needed
            # fourcc = cv2.VideoWriter_fourcc(*'XVID')
            # video = cv2.VideoWriter('video.avi', fourcc, 20, (112, 672), isColor=False)

            def reverse_normalize(image):
                mean = np.array([0.5, 0.5, 0.5])
                std = np.array([0.5, 0.5, 0.5])
                reversed_image = (image * std) + mean
                return reversed_image


            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            import numpy as np

            print(data.files)

            # tactile_img = data['tactile'][:done_idx, ...]
            # latent = data['latent'][:done_idx, ...]
            plug_pos = data['eef_pos'][:done_idx, ...]
            socket_pos = data['eef_pos'][:done_idx, ...]

            plug_pos_x = plug_pos[:, 0]
            plug_pos_y = plug_pos[:, 1]
            plug_pos_z = plug_pos[:, 2]
            # pose_orientations = plug_pos[:, 3:].reshape(plug_pos.shape[0], 3,
            #                                             3)  # Extracting the orientation part of the pose matrix
            #
            # from scipy.spatial.transform import Rotation as R
            #
            # euler_angles = np.array([R.from_matrix(pose_orientations[i]).as_euler('xyz', degrees=True)
            #                          for i in range(len(pose_orientations))])
            # Converting quaternion to Euler angles (roll, pitch, yaw)

            socket_pos_x = socket_pos[:, 0]
            socket_pos_y = socket_pos[:, 1]
            socket_pos_z = socket_pos[:, 2]

            if not test:
                # Create the initial figures and axes
                fig = plt.figure(figsize=(18, 10))
                # ax1 = fig.add_subplot(311)
                # ax4 = fig.add_subplot(312)
                ax2 = fig.add_subplot(121)
                ax3 = fig.add_subplot(122, projection='3d')
                test = True
                # Initialize the line objects for dynamic updating
            line_latent, = ax2.plot([], [], '-')
            line_plug, = ax3.plot([], [], [])
            line_socket, = ax3.plot([], [], [], '-')

            # ax1.set_title('Tactile Image')
            # ax1.axis('off')
            # ax4.set_title('Diff Tactile Image')
            # ax4.axis('off')
            # ax2.set_title('Latent Representation')
            # ax2.axis('on')
            # ax2.set_xlim([-0.1, 8.1])
            # ax2.set_ylim([-1.1, 1.1])

            ax3.set_title('Plug and Socket Positions')
            ax3.set_xlabel('X')
            ax3.set_ylabel('Y')
            ax3.set_zlabel('Z')
            # ax3.legend()

            # ax3.set_xlim([min(min(plug_pos_x), min(socket_pos_x)), max(max(plug_pos_x), max(socket_pos_x))])
            ax3.set_xlim([0.47, 0.52])
            ax3.set_ylim([min(min(plug_pos_y), min(socket_pos_y)), max(max(plug_pos_y), max(socket_pos_y))])
            ax3.set_zlim([min(min(plug_pos_z), min(socket_pos_z)), max(max(plug_pos_z), max(socket_pos_z))])

            quiver_orientations = []
            # plt.ion()  # Turn on interactive mode

            for j in tqdm(range(0, done_idx)):
                # img1 = tactile_img[j][0]
                # img2 = tactile_img[j][1]
                # img3 = tactile_img[j][2]
                #
                # img1 = reverse_normalize(img1)
                # img2 = reverse_normalize(img2)
                # img3 = reverse_normalize(img3)
                # img = np.concatenate((img1, img2, img3), axis=1)
                #
                # if j != 0:
                #     ax4.imshow((img - last) * 20)
                #     ax4.set_title('Diff Tactile Image')
                #     ax4.axis('off')
                #
                # last = img

                # Update and redraw the tactile image
                # ax1.imshow(img)
                # ax1.set_title('Tactile Image')
                # ax1.axis('off')

                # Update and redraw the latent representation
                # line_latent.set_xdata(range(len(latent[j])))
                # line_latent.set_ydata(latent[j])
                # ax2.set_title('Latent Representation')
                # ax2.axis('on')

                # Update and redraw the plug and socket positions in 3D
                line_plug.set_xdata(plug_pos_x[:j + 1])
                line_plug.set_ydata(plug_pos_y[:j + 1])
                line_plug.set_3d_properties(plug_pos_z[:j + 1])

                if False:
                    # Update and redraw the orientation using quiver plot
                    if j == 0:
                        for quiver in quiver_orientations:
                            quiver.remove()
                        quiver_orientations = []

                    for i in range(j + 1):
                        quiver_orientations.append(ax3.quiver(plug_pos_x[i], plug_pos_y[i], plug_pos_z[i],
                                                              pose_orientations[i, 0, :], pose_orientations[i, 1, :],
                                                              pose_orientations[i, 2, :],
                                                              color='green', length=0.001, arrow_length_ratio=0.3))

                line_socket.set_xdata(socket_pos_x[:j + 1])
                line_socket.set_ydata(socket_pos_y[:j + 1])
                line_socket.set_3d_properties(socket_pos_z[:j + 1])

                ax3.set_title('Plug and Socket Positions')
                ax3.set_xlabel('X')
                ax3.set_ylabel('Y')
                ax3.set_zlabel('Z')
                # ax3.legend()

                # plt.pause(0.00001)
                # fig.canvas.flush_events()

            # plt.ioff()  # Turn off interactive mode at the end
    plt.show()

if False:
    import cv2
    import numpy as np
    from tqdm import tqdm

    path = random.sample(all_paths, 1)[0]
    # path = all_paths[i]
    data = np.load(path)
    done_idx = data['done'].nonzero()[-1][0]


    # choose codec according to format needed
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # video = cv2.VideoWriter('video.avi', fourcc, 20, (112, 672), isColor=False)

    def reverse_normalize(image):
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.5, 0.5, 0.5])
        reversed_image = (image * std) + mean
        return reversed_image


    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np

    print(data.files)

    tactile_img = data['tactile'][:done_idx, ...]
    latent = data['latent'][:done_idx, ...]
    plug_pos = data['plug_pos'].reshape(-1, 4, 4)[:done_idx, ...]
    socket_pos = data['socket_pos'].reshape(-1, 4, 4)[:done_idx, ...]

    plug_pos_x = plug_pos[:, 0, 3]
    plug_pos_y = plug_pos[:, 1, 3]
    plug_pos_z = plug_pos[:, 2, 3]
    pose_orientations = plug_pos[:, :3, :3]  # Extracting the orientation part of the pose matrix

    from scipy.spatial.transform import Rotation as R

    euler_angles = np.array([R.from_matrix(pose_orientations[i]).as_euler('xyz', degrees=True)
                             for i in range(len(pose_orientations))])
    # Converting quaternion to Euler angles (roll, pitch, yaw)

    socket_pos_x = socket_pos[:, 0, 3]
    socket_pos_y = socket_pos[:, 1, 3]
    socket_pos_z = socket_pos[:, 2, 3]

    # Create the initial figures and axes
    fig = plt.figure(figsize=(18, 10))
    ax1 = fig.add_subplot(311)
    ax4 = fig.add_subplot(312)
    ax2 = fig.add_subplot(325)
    ax3 = fig.add_subplot(326, projection='3d')

    # Initialize the line objects for dynamic updating
    line_latent, = ax2.plot([], [], 'o')
    line_plug, = ax3.plot([], [], [], label='Plug', color='blue')
    line_socket, = ax3.plot([], [], [], label='Socket', color='red')

    ax1.set_title('Tactile Image')
    ax1.axis('off')
    ax4.set_title('Diff Tactile Image')
    ax4.axis('off')
    ax2.set_title('Latent Representation')
    ax2.axis('on')
    ax2.set_xlim([-0.1, 8.1])
    ax2.set_ylim([-1.1, 1.1])

    ax3.set_title('Plug and Socket Positions')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.legend()

    # ax3.set_xlim([min(min(plug_pos_x), min(socket_pos_x)), max(max(plug_pos_x), max(socket_pos_x))])
    ax3.set_xlim([0.47, 0.52])
    ax3.set_ylim([min(min(plug_pos_y), min(socket_pos_y)), max(max(plug_pos_y), max(socket_pos_y))])
    ax3.set_zlim([min(min(plug_pos_z), min(socket_pos_z)), max(max(plug_pos_z), max(socket_pos_z))])

    quiver_orientations = []
    # plt.ion()  # Turn on interactive mode

    for j in tqdm(range(0, done_idx)):
        img1 = tactile_img[j][0]
        img2 = tactile_img[j][1]
        img3 = tactile_img[j][2]

        img1 = reverse_normalize(img1)
        img2 = reverse_normalize(img2)
        img3 = reverse_normalize(img3)
        img = np.concatenate((img1, img2, img3), axis=1)
        #
        # if j != 0:
        #     ax4.imshow((img - last) * 20)
        #     ax4.set_title('Diff Tactile Image')
        #     ax4.axis('off')
        #
        # last = img

        # Update and redraw the tactile image
        ax1.imshow(img)
        ax1.set_title('Tactile Image')
        ax1.axis('off')

        # Update and redraw the latent representation
        line_latent.set_xdata(range(len(latent[j])))
        line_latent.set_ydata(latent[j])
        ax2.set_title('Latent Representation')
        ax2.axis('on')

        # Update and redraw the plug and socket positions in 3D
        line_plug.set_xdata(plug_pos_x[:j + 1])
        line_plug.set_ydata(plug_pos_y[:j + 1])
        line_plug.set_3d_properties(plug_pos_z[:j + 1])

        if False:
            # Update and redraw the orientation using quiver plot
            if j == 0:
                for quiver in quiver_orientations:
                    quiver.remove()
                quiver_orientations = []

            for i in range(j + 1):
                quiver_orientations.append(ax3.quiver(plug_pos_x[i], plug_pos_y[i], plug_pos_z[i],
                                                      pose_orientations[i, 0, :], pose_orientations[i, 1, :],
                                                      pose_orientations[i, 2, :],
                                                      color='green', length=0.001, arrow_length_ratio=0.3))

        line_socket.set_xdata(socket_pos_x[:j + 1])
        line_socket.set_ydata(socket_pos_y[:j + 1])
        line_socket.set_3d_properties(socket_pos_z[:j + 1])

        ax3.set_title('Plug and Socket Positions')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        ax3.legend()

        plt.pause(0.00001)
        fig.canvas.flush_events()

    plt.ioff()  # Turn off interactive mode at the end
    plt.show()

if False:
    # ax.scatter(data['socket_pos'][:done_idx, 0],
    #            data['socket_pos'][:done_idx, 1],
    #            zs=data['socket_pos'][:done_idx, 2],  color='r')

    # ax.plot(data['noisy_socket_pos'][:done_idx, 0],
    #         data['noisy_socket_pos'][:done_idx, 1],
    #         zs=data['noisy_socket_pos'][:done_idx, 2])
    path = random.sample(all_paths, 1)[0]
    # path = all_paths[i]
    data = np.load(path)
    done_idx = data['done'].nonzero()[-1][0]

    fig = plt.figure(figsize=(18, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(data['priv_obs'][:done_idx, 0], data['priv_obs'][:done_idx, 1], zs=data['priv_obs'][:done_idx, 2],
            marker='o')
    # ax.plot(data['plug_pos'][:done_idx, 0], data['plug_pos'][:done_idx, 1], zs=data['plug_pos'][:done_idx, 2])
    # ax.plot(data['eef_pos'][:done_idx, 0], data['eef_pos'][:done_idx, 1], zs=data['eef_pos'][:done_idx, 2])
    ax.set_xlabel('$X$', fontsize=20, rotation=150)
    ax.set_ylabel('$Y$', fontsize=20, rotation=150)
    ax.set_zlabel('$Z$', fontsize=30, rotation=60)

    delta = 0.06
    # ax.axes.set_xlim3d(left=0.5 - delta, right=0.5 + delta)
    # ax.axes.set_ylim3d(bottom=0 - delta, top=0 + delta)
    # ax.axes.set_zlim3d(bottom=0, top=0.2)

    plt.show()
    # %%

if False:
    path = random.sample(all_paths, 1)[0]
    # path = all_paths[i]
    data = np.load(path)
    done_idx = data['done'].nonzero()[-1][0]

    fig = plt.figure(figsize=(18, 10))
    ax = fig.add_subplot(111)
    plt.scatter(data['noisy_socket_pos'][1:done_idx, 0], data['noisy_socket_pos'][1:done_idx, 1], color='b')
    plt.plot(data['plug_pos'][1:done_idx, 0], data['plug_pos'][1:done_idx, 1], color='black')
    plt.plot(data['eef_pos'][1:done_idx, 0], data['eef_pos'][1:done_idx, 1])
    plt.scatter(data['socket_pos'][1:done_idx, 0], data['socket_pos'][1:done_idx, 1], color='r', s=35)
    plt.show()
from scipy.spatial.transform import Rotation

if False:
    fig = plt.figure(figsize=(18, 10))
    ax = fig.add_subplot(111)

    for i in range(100):
        path = random.sample(all_paths, 1)[0]
        print(path)
        # path = all_paths[i]
        data = np.load(path)
        done_idx = data['done'].nonzero()[-1][0]

        euler_angles = Rotation.from_quat(data["plug_hand_quat"][1:done_idx, :]).as_euler('xyz')
        euler_angles[:,0] = np.where(euler_angles[:,0] > 0, euler_angles[:,0] - np.pi, euler_angles[:,0] + np.pi)

        # Extract and reshape rotation matrices for the plug and the EEF
        # rot_mats_plug = data["plug_pos"][:, 3:].reshape(-1, 3, 3)[1:done_idx, :]
        # rot_mats_eef = data["eef_pos"][:, 3:].reshape(-1, 3, 3)[1:done_idx, :]

        # # Step 1: Invert (transpose) the EEF's rotation matrices
        # rot_mats_eef_inv = np.transpose(rot_mats_eef, (0, 2, 1))
        #
        # # Step 2: Compute the relative rotation matrix
        # rel_rot_mats = np.matmul(rot_mats_eef_inv, rot_mats_plug)
        #
        # # Convert the relative rotation matrices to Rotation objects
        # rel_rot_objs = Rotation.from_matrix(rel_rot_mats)
        #
        # # Step 3: Convert to Euler angles ('xyz' sequence)
        # rel_euler_angles = rel_rot_objs.as_euler('xyz')

        # euler_angles = rel_euler_angles

        sin_cos_representation = np.hstack((np.sin(euler_angles[:, 0:1]), np.cos(euler_angles[:, 0:1]),
                                            np.sin(euler_angles[:, 1:2]), np.cos(euler_angles[:, 1:2]),
                                            np.sin(euler_angles[:, 2:3]), np.cos(euler_angles[:, 2:3])))

        to_show_roll = np.arctan2(np.sin(euler_angles[:, 0:1]), np.cos(euler_angles[:, 0:1])) * 180 / np.pi
        to_show_pitch = np.arctan2(np.sin(euler_angles[:, 1:2]), np.cos(euler_angles[:, 1:2])) * 180 / np.pi
        to_show_yaw = np.arctan2(np.sin(euler_angles[:, 2:3]), np.cos(euler_angles[:, 2:3])) * 180 / np.pi
        #

        # to_show_roll_unwrapped = np.unwrap(to_show_roll)
        # to_show_pitch_unwrapped = np.unwrap(to_show_pitch)
        # to_show_yaw_unwrapped = np.unwrap(to_show_yaw)
        plt.plot(to_show_roll[:, 0], 'o')

        # plt.plot(sin_cos_representation[:,2:4], 'ko')
        # plt.plot(sin_cos_representation[:,4:6], 'go')

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

    fig = plt.figure(figsize=(18, 10))
    ax = fig.add_subplot(111)
    path = random.sample(all_paths, 1)[0]
    # path = all_paths[i]
    data = np.load(path)
    done_idx = data['done'].nonzero()[-1][0]


    def reverse_normalize(image):
        mean = np.array([0.5])
        std = np.array([0.5])
        reversed_image = (image * std) + mean
        return reversed_image


    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np

    print(data.files)

    tactile_img = data['tactile'][:done_idx, ...]

    for j in tqdm(range(0, done_idx)):
        img1 = tactile_img[j][0]
        img2 = tactile_img[j][1]
        img3 = tactile_img[j][2]

        # img1 = np.transpose(img1, (1, 2, 0))
        # img2 = np.transpose(img2, (1, 2, 0))
        # img3 = np.transpose(img3, (1, 2, 0))

        img1 = reverse_normalize(img1)
        img2 = reverse_normalize(img2)
        img3 = reverse_normalize(img3)
        img = np.concatenate((img1, img2, img3), axis=1)

        # Update and redraw the tactile image
        cv2.imshow('test', (img*255).astype(np.uint8))
        cv2.waitKey(5)

if True:
    fig = plt.figure(figsize=(18, 10))
    ax = fig.add_subplot(111)
    a = []
    for i in range(1000):
        path = random.sample(all_paths, 1)[0]
        # print(path)
        # path = all_paths[i]
        data = np.load(path)
        done_idx = data['done'].nonzero()[-1][0]
        label = Rotation.from_quat(data["plug_hand_quat"][:done_idx, :]).as_euler('xyz', degrees=True)  # data["latent"] #
        to_plot = label - label[0, :]

        if i == 0:
            a = to_plot
        else:
            a = np.vstack((a, to_plot))

        plt.plot(to_plot[:,2], 'o')

        # plt.plot(sin_cos_representation[:,2:4], 'ko')
        # plt.plot(sin_cos_representation[:,4:6], 'go')

    plt.show()