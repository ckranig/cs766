from PIL import Image
import numpy as np
from typing import Union, Tuple, List
import os
from scipy.ndimage import convolve, gaussian_filter
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

def mod_lap(img: np.ndarray, w_size:int) -> np.ndarray:
    #normalize image by mean light
    #img = img / np.mean(img)
    avg_kernel = np.ones((5, 5))/25
    #avg_img = convolve(img, avg_kernel)
    avg_img = convolve2d(img, avg_kernel, mode='same', boundary='symm')
    #x_kernel = np.array([[0, 0, 0], [1, -2, 1], [0, 0, 0]])
    #y_kernel = np.array([[0, 1, 0], [0, -2, 0], [0, 1, 0]])
    #ml_x = np.abs(convolve(avg_img, x_kernel, mode='constant'))
    #ml_y = np.abs(convolve(avg_img, y_kernel, mode='constant'))
    #mf_initial = ml_x + ml_y
    x5_laplacian = np.array([[0, 0,1,0,0 ],[ 0,1,2,1,0 ],[ 1,2, -16, 2,1 ],[ 0,1,2,1,0 ],[ 0,0,1,0,0 ]])
    x3_laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    x3_laplacian_stable = np.array([[0.25, 0.5, 0.25], [0.5, -3, 0.5],[0.25, 0.5, 0.25]])
    #mf_initial = np.abs(convolve(avg_img,x3_laplacian))
    mf_initial = np.abs(convolve2d(avg_img, x3_laplacian, mode='same', boundary='symm'))
    ml_kernel = np.ones((2 * w_size + 1, 2 * w_size + 1))
    #mf_final = convolve(mf_initial, ml_kernel, mode='constant')
    mf_final = convolve2d(mf_initial, ml_kernel, mode='same', boundary='fill')
    #mf_avg = mf_final / np.mean(mf_final)
    return mf_final

def generateIndexMap(gray_list: List[np.ndarray], w_size: int) -> np.ndarray:
    # Generate an index map for the refocusing application
    # Input:
    #   gray_list - List of K gray-scale images
    #   w_size - half window size used for focus measure computation
    # Output:
    #   index_map - mxn index map
    #               index_map(i, j) is the index of the image that is in focus
    #               at pixel (i, j)
    index_map = np.zeros(gray_list[0].shape)
    # Compute focus measure for each pixel in each image
    for idx, img in enumerate(gray_list):
        #blurred_img = gaussian_filter(img, sigma=1)
        focus_measure = mod_lap(img, w_size)
        #focus_measure = modified_laplacian(img, w_size)
        # Update index map
        mask = focus_measure > index_map
        index_map[mask] = idx
    avg_kernel = np.ones((3, 3))/9
    index_map_avg = convolve2d(index_map, avg_kernel, mode='same', boundary='symm')
    return index_map_avg.astype(np.uint8)


def loadFocalStack(focal_stack_dir: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    # Load the focal stack
    # Input:
    #   focal_stack_dir - directory of the focal stack
    # Output:
    #   rgb_list - List of RGB images for varying focal lengths
    #   gray_list - List of gray-scale images for varying focal lengths
    rgb_list = []
    gray_list = []
    files = os.listdir(focal_stack_dir)
    sorted_files = sorted(files, key=lambda x: int(''.join(filter(str.isdigit, x))))
    for filename in sorted_files:
        filepath = os.path.join(focal_stack_dir, filename)
        if os.path.isfile(filepath) and any(filename.endswith(extension) for extension in ['.jpg', '.jpeg', '.png']):
            rgb_img = Image.open(filepath)
            grayscale_img = rgb_img.convert('L')
            rgb_image_arr = np.array(rgb_img)
            gray_image_arr = np.array(grayscale_img)
            rgb_list.append(rgb_image_arr)
            gray_list.append(gray_image_arr)

    return rgb_list, gray_list


def refocusApp(rgb_list: List[np.ndarray], depth_map: np.ndarray) -> None:
    # Refocusing application
    # Input:
    #   rgb_list - List of RGB images for varying focal lengths
    #   depth_map - mxn index map
    #               depth_map(i, j) is the index of the image that is in focus
    #               at pixel (i, j)
    img_idx = 0
    while True:
        # Display an image from the focal stack
        img = rgb_list[img_idx]
        plt.imshow(img)
        plt.title("Image {}".format(img_idx))
        plt.xlabel("Choose a scene point (click on image)")
        plt.show(block = False)

        # Ask the user to choose a scene point
        scene_point = plt.ginput()
        if not scene_point:
            print("Terminating program...")
            break
        scene_point = np.array(scene_point[0], dtype=int)
        if scene_point[0] < 0 or scene_point[1] < 0 or scene_point[0] >= img.shape[1] or scene_point[1] >= img.shape[0]:
            print("Point is outside of the image frame. Terminating program...")
            break

        # Refocus to the image such that the scene point is focused
        img_idx = depth_map[scene_point[1], scene_point[0]]