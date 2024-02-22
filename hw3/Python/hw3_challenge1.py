from PIL import Image, ImageDraw
import numpy as np

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
    raise NotImplementedError


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

    # hough_peaks = ???

    # line_img = Image.fromarray(orig_img.astype(np.uint8)).convert('RGB')
    # draw = ImageDraw.Draw(line_img) 

    # for i, j in hough_peaks:
    #     xp0, yp0, xp1, yp1 = ???
    #     draw.line((xp0, yp0, xp1, yp1), fill=128)
    
    # line_img.show()
    # return  line_img
    raise NotImplementedError

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
    raise NotImplementedError
