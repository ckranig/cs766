from PIL import Image, ImageDraw
import numpy as np
import math

min_theta = -np.pi/2
max_theta = np.pi/2
def generateHoughAccumulator(edge_image: np.ndarray, theta_num_bins: int, rho_num_bins: int) -> np.ndarray:
    '''
    Generate the Hough accumulator array.
    Arguments:
        edge_image: the edge image.
        theta_num_bins: the number of bins in the theta dimension.
        rho_num_bins: the number of bins in the rho dimension.
    Returns:
        hough_accumulator: the Hough accumulator array.
    '''
    #x sin(theta) - y cos(theta) + p = 0
    diag_len = int(np.ceil(math.hypot(edge_image.shape[0], edge_image.shape[1])))  # max_dist
    rhos = np.linspace(-diag_len, diag_len, rho_num_bins)
    thetas = np.linspace(min_theta, max_theta, theta_num_bins)

    #Precalulate_thetas
    cos_thetas = np.cos(thetas)
    sin_thetas = np.sin(thetas)
    # Initialize the Hough accumulator array
    hough_accumulator = np.zeros((rho_num_bins, theta_num_bins), dtype=int)

    # Iterate through only edge pixels
    y_idxs, x_idxs = np.nonzero(edge_image)  # (row, col) indexes to edges
    dist = 0
    for edge_idx in range(len(y_idxs)):
        x = x_idxs[edge_idx]
        y = y_idxs[edge_idx]
        for theta_idx in range(len(cos_thetas)):
            rho = x*cos_thetas[theta_idx] + y*sin_thetas[theta_idx]
            rho_idx = np.argmin(np.abs(rhos - rho))
            hough_accumulator[max(0,rho_idx-dist):min(rho_num_bins,rho_idx+dist+1), max(0,theta_idx-dist):min(theta_num_bins,theta_idx+dist+1)] += 1

    # Scale Results Back
    hough_accumulator = 255 * (hough_accumulator / np.max(hough_accumulator))
    return hough_accumulator


def lineFinder(orig_img: np.ndarray, hough_img: np.ndarray, hough_threshold: float):
    '''
    Find the lines in the image.
    Arguments:
        orig_img: the original image.
        hough_img: the Hough image.
        hough_threshold: the threshold for the Hough accumulator array.
    Returns: 
        line_img: PIL image with lines drawn.

    '''

    #Only look at peaks
    hough_peaks = []
    hough_img_copy = np.copy(hough_img)
    sorted_lines = np.column_stack(np.unravel_index(np.argsort(hough_img, axis=None)[::-1], hough_img.shape, order='C'))
    dist = 5
    #Combine Lines
    for rho_idx, theta_idx in sorted_lines:
        if hough_img[rho_idx,theta_idx] < hough_threshold:
            break
        elif hough_img_copy[rho_idx,theta_idx] < hough_threshold:
            continue
        hough_img_copy[max(0,rho_idx-dist):min(hough_img.shape[0],rho_idx+dist+1), max(0,theta_idx-dist):min(hough_img.shape[1],theta_idx+dist+1)] = 0
        hough_peaks.append((rho_idx,theta_idx))
    #hough_peaks = np.column_stack(np.where(hough_img >= hough_threshold))

    
    #Get bin values
    diag_len = int(np.ceil(math.hypot(orig_img.shape[0], orig_img.shape[1])))  # max_dist
    rhos = np.linspace(-diag_len, diag_len, hough_img.shape[0])
    thetas = np.linspace(min_theta, max_theta, hough_img.shape[1])


    line_img = Image.fromarray(orig_img.astype(np.uint8)).convert('RGB')
    draw = ImageDraw.Draw(line_img)
    for rho_idx, theta_idx in hough_peaks:
        #choose x = 0 and max width and then solve for y
        rho = rhos[rho_idx]
        theta = thetas[theta_idx]
        xp0 = 0
        xp1 = orig_img.shape[1]
        yp0 = int(((-np.cos(theta) * xp0) / np.sin(theta)) + (rho / np.sin(theta)))
        yp1 = int(((-np.cos(theta) * xp1) / np.sin(theta)) + (rho / np.sin(theta)))
        draw.line((xp0, yp0, xp1, yp1), fill=128)
    line_img.show()
    return  line_img

def lineSegmentFinder(orig_img: np.ndarray, edge_img: np.ndarray, hough_img: np.ndarray, hough_threshold: float):
    '''
    Find the line segments in the image.
    Arguments:
        orig_img: the original image.
        edge_img: the edge image.
        hough_img: the Hough image.
        hough_threshold: the threshold for the Hough accumulator array.
    Returns:
        line_segement_img: PIL image with line segments drawn.
    '''
    #Only look at peaks
    hough_peaks = np.column_stack(np.where(hough_img >= hough_threshold))
    #Get bin values
    diag_len = int(np.ceil(math.hypot(orig_img.shape[0], orig_img.shape[1])))  # max_dist
    rhos = np.linspace(-diag_len, diag_len, hough_img.shape[0])
    thetas = np.linspace(min_theta, max_theta, hough_img.shape[1])

    #line_img = np.copy(np.asarray(Image.fromarray(orig_img.astype(np.uint8)).convert('RGB'), dtype=np.uint8))
    line_img = Image.fromarray(orig_img.astype(np.uint8)).convert('RGB')
    draw = ImageDraw.Draw(line_img)
    for rho_idx, theta_idx in hough_peaks:
        #choose x = 0 and max width and then solve for y
        rho = rhos[rho_idx]
        theta = thetas[theta_idx]
        sin_t = np.sin(theta)
        cos_t = np.cos(theta)
        
        prev_edge = 0
        threshold = 10
        start_x = None
        start_y = None
        end_x = None
        end_y = None
        for x in range(orig_img.shape[1]):
            y = min(max(int(((-cos_t * x) / sin_t) + (rho / sin_t)), 0), orig_img.shape[0]-1)
            #min_y = max(0, y-1)
            #max_y = min(edge_img.shape[0]-1, y+1)
            #min_x = max(0, x-1)
            #max_x = min(edge_img.shape[1]-1,x+1)
            #if np.any(edge_img[min_y:max_y,min_x:max_x] > 0):
            if edge_img[y,x] > 0:
                if start_x is None:
                    start_x = x - 1
                    start_y = min(max(int(((-cos_t * start_x) / sin_t) + (rho / sin_t)), 0), orig_img.shape[0]-1)
                end_x = x
                end_y = y
                prev_edge = 0
            else:
                prev_edge += 1
                if prev_edge > threshold and start_x is not None and end_x is not None:
                    draw.line((start_x, start_y, end_x, end_y), fill=128)
                    start_x = None
                    start_y = None
                    end_x = None
                    end_y = None

        if start_x is not None and end_x is not None:
            draw.line((start_x, start_y, end_x, end_y), fill=128)

    #line_img = Image.fromarray(line_img).convert('RGB')
    line_img.show()
    return  line_img
