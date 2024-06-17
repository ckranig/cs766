from PIL import Image, ImageDraw
import numpy as np
from typing import Union, Tuple, List
import os

def draw_rectangle(image, rect):
    draw = ImageDraw.Draw(image)
    draw.rectangle(rect, outline='red', width=2)
    del draw

def corr(arr1, arr2):
    return np.sum(np.abs(arr1 - arr2))

def calc_color_map(arr, num_bins, max_val=255.0):
    bins_per_color = int(num_bins/3)
    bin_width = float(max_val)/bins_per_color
    color_map = np.zeros((bins_per_color + 1,bins_per_color + 1,bins_per_color + 1), dtype = np.int32)
    h, w, _ = arr.shape
    for row in range(h):
        for col in range(w):
            r,g,b = arr[row,col]
            r_bin = int(r // bin_width)
            g_bin = int(g // bin_width)
            b_bin = int(b // bin_width)
            color_map[r_bin, g_bin, b_bin] += 1
    return color_map

def calc_color_map_2d(arr, num_bins, max_val=255.0):
    bins_per_color = int(num_bins/3)
    bin_width = float(max_val)/bins_per_color
    color_map = np.zeros((bins_per_color + 1,bins_per_color + 1,bins_per_color + 1), dtype = np.int32)
    h, _ = arr.shape
    for row in range(h):
        r,g,b = arr[row]
        r_bin = int(r // bin_width)
        g_bin = int(g // bin_width)
        b_bin = int(b // bin_width)
        color_map[r_bin, g_bin, b_bin] += 1
    return color_map

def calc_rect(search_arr, color_map, box_h, box_w, num_bins, max_val=255.0):
    h, w, _ = search_arr.shape
    min_row = 0
    min_col = 0
    base_row_map = calc_color_map(search_arr[0:box_h, 0:box_w], num_bins, max_val)
    min_cor = float('inf')
    for row in range(h-box_h):
        new_row_map = calc_color_map_2d(search_arr[row+box_h, 0:box_w], num_bins, max_val)
        base_row_map += new_row_map
        base_col_map = np.copy(base_row_map)
        for col in range(w-box_w):
            new_col_map = calc_color_map_2d(search_arr[row:row+box_h+1, col+box_w], num_bins, max_val)
            base_col_map += new_col_map
            cur_corr = corr(base_col_map, color_map)
            if cur_corr < min_cor:
                min_cor = cur_corr
                min_row = row
                min_col = col
            prev_col_map = calc_color_map_2d(search_arr[row:row+box_h+1, col], num_bins, max_val)
            base_col_map -= prev_col_map
        prev_row_map = calc_color_map_2d(search_arr[row, 0:box_w], num_bins, max_val)
        base_row_map -= prev_row_map
    return min_col, min_row

def trackingTester(data_params, tracking_params):
    # Tracking Tester
    # Input:
    #   data_params - data parameters
    #   tracking_params - tracking parameters
    # Create output directory if it doesn't exist
    os.makedirs(data_params.out_dir, exist_ok=True)
    
    cur_x, cur_y, w, h = tracking_params.rect
    frame_path = data_params.gen_data_fname(0)
    roi_image = Image.open(frame_path).convert('RGB')
    roi_image_arr = np.array(roi_image)
    roi = roi_image_arr[cur_y:cur_y+h, cur_x:cur_x+w]
    draw_rectangle(roi_image, (cur_x,cur_y,cur_x+w,cur_y+h))

    # Generate color map
    color_map = calc_color_map(roi, tracking_params.bin_n)

    for frame_id in data_params.frame_ids:
        # Load frame image
        frame_path = data_params.gen_data_fname(frame_id)
        frame_image = Image.open(frame_path).convert('RGB')
        frame_arr = np.array(frame_image)

        min_row = max(0, cur_y - tracking_params.search_half_window_size)
        max_row = min(frame_arr.shape[0]-1, cur_y + h + tracking_params.search_half_window_size)
        min_col = max(0, cur_x - tracking_params.search_half_window_size)
        max_col = min(frame_arr.shape[1]-1, cur_x + w + tracking_params.search_half_window_size)
        d_x, d_y = calc_rect(frame_arr[min_row:max_row+1, min_col:max_col+1], color_map, h, w, tracking_params.bin_n)
        cur_x += (d_x - tracking_params.search_half_window_size)
        cur_y += (d_y - tracking_params.search_half_window_size)
        # Draw rectangle around the target
        draw_rectangle(frame_image, (cur_x,cur_y,cur_x+w,cur_y+h))
        
        # Save annotated frame
        output_path = data_params.gen_out_fname(frame_id)
        frame_image.save(output_path)
        print(f"Annotated frame {frame_id + 1} saved at {output_path}")