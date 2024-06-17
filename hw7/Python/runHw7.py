import argparse
from runTests import run_tests
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import os

def runHw7():
    # runHw7 is the "main" interface that lets you execute all the 
    # walkthroughs and challenges in this homework. It lists a set of 
    # functions corresponding to the problems that need to be solved.
    #
    # Note that this file also serves as the specifications for the functions 
    # you are asked to implement. In some cases, your submissions will be 
    # auto-graded.  Thus, it is critical that you adhere to all the specified 
    # function signatures.
    #
    # Before your submission, make sure you can run runHw7('all') 
    # without any error.
    #
    # Usage:
    # python runHw7.py                  : list all the registered functions
    # python runHw7.py 'function_name'  : execute a specific test
    # python runHw7.py 'all'            : execute all the registered functions
    parser = argparse.ArgumentParser(
        description='Execute a specific test or all tests.')
    parser.add_argument(
        'function_name', type=str, nargs='?', default='all',
        help='Name of the function to test or "all" to execute all the registered functions')
    args = parser.parse_args()

    # Call test harness
    fun_handles = {
        'honesty': honesty, 
        'debug1a': debug1a, 
        'challenge1a': challenge1a, 
        'challenge2a': challenge2a, 
        'challenge2b': challenge2b, 
        'challenge2c': challenge2c, 
    }
    run_tests(args.function_name, fun_handles)

###########################################################################
# Academic Honesty Policy
###########################################################################
def honesty():
    from signAcademicHonestyPolicy import sign_academic_honesty_policy
    # Type your full name and uni (both in string) to state your agreement 
    # to the Code of Academic Integrity.
    sign_academic_honesty_policy('Calvin Kranig', '9083889825')

###########################################################################
# Tests for Challenge 1: Optical flow using template matching
###########################################################################

def flow_to_img(img_arr, flow):
    rgb_array = np.stack((img_arr * 255,)*3, axis=-1)
    line_img = Image.fromarray(rgb_array.astype(np.uint8), "RGB")
    draw = ImageDraw.Draw(line_img)
    H, W, _ = flow.shape
    for row in range(H):
        for col in range(W):
            d_row, d_col = flow[row][col]
            if d_row != 0 or d_col != 0:
                draw.line((col, row, col + d_col, row + d_row), fill=128)
    return line_img

def debug1a():
    from hw7_challenge1 import computeFlow
    img1 = np.array(Image.open('data/simple1.png')) / 255.
    img2 = np.roll(img1, shift=(4, 4), axis=(0, 1))

    search_half_window_size = 3   # Half size of the search window
    template_half_window_size = 2 # Half size of the template window
    grid_MN = [int(img1.shape[0] /10), int(img1.shape[1] / 10)]              # Number of rows and cols in the grid

    result = computeFlow(img1, img2, search_half_window_size, template_half_window_size, grid_MN)

    out_img = flow_to_img(img1, result)
    out_img.save('outputs/simple_result.png')

def challenge1a():
    from hw7_challenge1 import computeFlow
    img_list = [f'data/flow/flow{i+1}.png' for i in range(6)]
    img_stack = [np.array(Image.open(img)) / 255. for img in img_list]
    
    search_half_window_size = 10   # Half size of the search window
    template_half_window_size = 3 # Half size of the template window
    grid_MN = [int(img_stack[0].shape[0] /10), int(img_stack[0].shape[1] / 10)]              # Number of rows and cols in the grid

    os.makedirs('outputs/flow/', exist_ok=True)
    for i in range(1, len(img_stack)):
        result = computeFlow(img_stack[i-1], img_stack[i], search_half_window_size, template_half_window_size, grid_MN)

        out_img = flow_to_img(img_stack[i-1], result)
        out_img.save(f'outputs/flow/result{i}.png')



###########################################################################
# Tests for Challenge 2: Tracking with color histogram template
###########################################################################
def challenge2a():
    from helpers import chooseTarget, generateVideo, generateVideo2
    from hw7_challenge2 import trackingTester
    #-------------------
    # Parameters
    #-------------------
    class data_params:
        out_dir = 'outputs/walking_person/result'
        data_dir = 'data/walking_person'
        frame_ids = list(range(250))
        gen_data_fname = lambda x: f'{data_params.data_dir}/frame{x+1}.png'
        gen_out_fname = lambda x: f'{data_params.out_dir}/frame{x+1}.png'

    # ****** IMPORTANT ******
    # In your submission, replace the call to "chooseTarget" with actual 
    # parameters to specify the target of interest.
    class tracking_params:
        #chooseTarget(data_params)
        rect = rect = np.array([204, 85, 19, 35])
        search_half_window_size = 10 # Half size of the search window
        bin_n = (256/8)*3                    # Number of bins in the color histogram

    # Pass the parameters to trackingTester
    #trackingTester(data_params, tracking_params)

    # Take all the output frames and generate a video
    generateVideo(data_params)

def challenge2b():
    from helpers import chooseTarget, generateVideo, generateVideo2
    from hw7_challenge2 import trackingTester
    #-------------------
    # Parameters
    #-------------------
    class data_params:
        data_dir = 'data/rolling_ball'
        out_dir = 'outputs/rolling_ball_result'
        frame_ids = list(range(250))
        gen_data_fname = lambda x: f'{data_params.data_dir}/frame{x+1}.png'
        gen_out_fname = lambda x: f'{data_params.out_dir}/frame{x+1}.png'

    # ****** IMPORTANT ******
    # In your submission, replace the call to "chooseTarget" with actual parameters
    # to specify the target of interest
    class tracking_params:
        #rect = chooseTarget(data_params)
        rect = np.array([154, 131, 45, 49])
        search_half_window_size = 10  # Half size of the search window
        bin_n = (256/8)*3                   # Number of bins in the color histogram

    # Pass the parameters to trackingTester
    trackingTester(data_params, tracking_params)

    # Take all the output frames and generate a video
    generateVideo(data_params)


def challenge2c():#
    from helpers import chooseTarget, generateVideo, generateVideo2
    from hw7_challenge2 import trackingTester
    #-------------------
    # Parameters
    #-------------------
    class data_params:
        data_dir = 'data/basketball'
        out_dir = 'outputs/basketball_result'
        frame_ids = list(range(250))
        gen_data_fname = lambda x: f'{data_params.data_dir}/frame{x+1}.png'
        gen_out_fname = lambda x: f'{data_params.out_dir}/frame{x+1}.png'

    # ****** IMPORTANT ******
    # In your submission, replace the call to "chooseTarget" with actual parameters
    # to specify the target of interest
    class tracking_params:
        #rect = chooseTarget(data_params)
        rect = np.array([417, 303, 27, 83])
        search_half_window_size = 10  # Half size of the search window
        bin_n = (256/8)*3                   # Number of bins in the color histogram
        
    # Pass the parameters to trackingTester
    trackingTester(data_params, tracking_params)

    # Take all the output frames and generate a video
    generateVideo(data_params)

if __name__ == '__main__':
    runHw7()