from PIL import Image
import numpy as np
from typing import Union, Tuple, List

def corr(arr1, arr2):
    return np.sum(np.abs(arr1 - arr2))

def computeFlow(img1: np.ndarray, img2: np.ndarray, win_radius: int, template_radius: int, grid_MN: List[int]) -> np.ndarray:
    # Compute optical flow using template matching
    # Input:
    #   img1 - HxW matrix of the first image
    #   img2 - HxW matrix of the second image
    #   win_radius - half size of the search window
    #   template_radius - half size of the template window
    #   grid_MN - 1x2 vector for the number of rows and cols in the grid
    # Output:
    #   result - HxWx2 matrix of optical flow vectors
    #     result[:,:,0] is the y component of the flow vectors
    #     result[]:,:,1] is the x component of the flow vectors
    # Compute optical flow using template matching
    
    # Get image dimensions
    H, W = img1.shape
    
    # Initialize result matrix for optical flow vectors
    result = np.zeros((H, W, 2))
    
    # Iterate through grid cells
    rows = np.linspace(template_radius, H, num=grid_MN[0] + 1, dtype=np.int64)
    cols = np.linspace(template_radius, W, num=grid_MN[1] + 1, dtype=np.int64)
    for row in rows[:-1]:
        if row - template_radius < 0 or row + template_radius > H:
            continue
        for col in cols[:-1]:
            if col - template_radius < 0 or col + template_radius > W:
                continue
            # Define the template window around the current pixel
            t1 = img1[row - template_radius:row + template_radius + 1, col - template_radius:col + template_radius + 1]
            
            min_row = row
            min_col = col
            min_val = float("inf")
            min_r = max(0 + template_radius, row - (win_radius-template_radius))
            max_r = min(H - template_radius - 1, row + (win_radius-template_radius))
            min_c = max(0 + template_radius, col - (win_radius-template_radius))
            max_c = min(W - template_radius - 1, col + (win_radius-template_radius))
            for r2 in range(min_r,max_r+1):
                for c2 in range(min_c,max_c+1):
                    t2 = img2[r2 - template_radius:r2 + template_radius + 1, c2 - template_radius:c2 + template_radius + 1]
                    
                    cur_val = np.sum(np.abs(t1 - t2))
                    if cur_val < min_val:
                        min_val = cur_val
                        min_row = r2
                        min_col = c2
            
            # Store the optical flow vector in the result matrix
            result[row, col, 0] = min_row - row
            result[row, col, 1] = min_col - col
    
    
    return result
