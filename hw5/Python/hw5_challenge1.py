from PIL import Image
import numpy as np
from typing import Union, Tuple, List
import skimage

def findSphere(img: np.ndarray) -> Tuple[np.ndarray, float]:
    # Find the center and radius of the sphere
    # Input:
    #   img - the image of the sphere
    # Output:
    #   center - 2x1 vector of the center of the sphere
    #   radius - radius of the sphere
    thresh = skimage.filters.threshold_otsu(img)
    binary = img >= thresh
    #labeled_img = skimage.measure.label(img)
    labeled_img = skimage.measure.label(binary)
    #region1 = skimage.measure.regionprops(labeled_img)
    region = skimage.measure.regionprops(labeled_img)
    props = region[0]
    circle_radius = np.sqrt(props.area / np.pi)
    #circ_rad2 = 0.5*props.axis_major_length
    return props.centroid,  circle_radius

def computeLightDirections(center: np.ndarray, radius: float, images: List[np.ndarray]) -> np.ndarray:
    # Compute the light source directions
    # Input:
    #   center - 2x1 vector of the center of the sphere
    #   radius - radius of the sphere
    #   images - list of N images
    # Output:
    #   light_dirs_5x3 - 5x3 matrix of light source directions
    light_dirs = []
    for image in images:
        brightest_pixel = np.unravel_index(np.argmax(image), image.shape)
        x_max = brightest_pixel[1]
        y_max = brightest_pixel[0]

        # Translate world coordinates by the negative of the sphere's center
        x_prime = x_max - center[0]
        y_prime = y_max - center[1]
        z_prime = np.sqrt(radius**2 - (x_prime)**2 - (y_prime)**2)

        # Normalize the resulting vector to obtain a unit vector
        magnitude = np.sqrt(x_prime**2 + y_prime**2 + z_prime**2) #Used for testing
        N = np.array([x_prime, y_prime, z_prime]) / magnitude * image[brightest_pixel] #Scale to Brightness

        light_dirs.append(N)

    return np.array(light_dirs)

def computeMask(images: List[np.ndarray]) -> np.ndarray:
    # Compute the mask of the object
    # Input:
    #   images - list of N images
    # Output:
    #   mask - HxW binary mask
    #mask = images[0] > 0
    #for image in images[1:]:
    #    mask = np.logical_or(mask, image > 0)
    mask = np.any(images, axis=0)
    return mask

def computeNormals(light_dirs: np.ndarray, images: List[np.ndarray], mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Compute the surface normals and albedo of the object
    # Input:
    #   light_dirs - Nx3 matrix of light directions
    #   images - list of N images
    #   mask - binary mask
    # Output:
    #   normals - HxWx3 matrix of surface normals
    #   albedo_img - HxW matrix of albedo values
    H = images[0].shape[0]
    W = images[0].shape[1]
    normals = np.zeros((H,W,3))
    albedo_img = np.zeros((H,W))
    images_arr = np.array(images)
    #S_pseudo_inverse = np.linalg.inv((light_dirs.T @ light_dirs)) @ light_dirs.T
    S_p_inv = np.linalg.pinv(light_dirs)
    for x in range(W):
        for y in range(H):
            if mask[y,x]:
                I = images_arr[:,y,x]
                N = S_p_inv @ I
                N_norm = np.linalg.norm(N)
                n = N/N_norm
                albedo = N_norm*np.pi
                normals[y,x] = n
                albedo_img[y,x] = albedo
    
    #Normalize
    (albedo_img-np.min(albedo_img))/(np.max(albedo_img)-np.min(albedo_img))
    return normals, albedo_img

