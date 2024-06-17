import argparse
from runTests import run_tests
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def runHw6():
    # runHw6 is the "main" interface that lets you execute all the 
    # walkthroughs and challenges in this homework. It lists a set of 
    # functions corresponding to the problems that need to be solved.
    #
    # Note that this file also serves as the specifications for the functions 
    # you are asked to implement. In some cases, your submissions will be 
    # auto-graded.  Thus, it is critical that you adhere to all the specified 
    # function signatures.
    #
    # Before your submission, make sure you can run runHw6('all') 
    # without any error.
    #
    # Usage:
    # python runHw6.py                  : list all the registered functions
    # python runHw6.py 'function_name'  : execute a specific test
    # python runHw6.py 'all'            : execute all the registered functions
    parser = argparse.ArgumentParser(
        description='Execute a specific test or all tests.')
    parser.add_argument(
        'function_name', type=str, nargs='?', default='all',
        help='Name of the function to test or "all" to execute all the registered functions')
    args = parser.parse_args()

    # Call test harness
    fun_handles = {
        'honesty': honesty, 
        'challenge1a': challenge1a, 
        'challenge1b': challenge1b, 
    }
    run_tests(args.function_name, fun_handles)

# Academic Honesty Policy
def honesty():
    from signAcademicHonestyPolicy import sign_academic_honesty_policy
    # Type your full name and uni (both in string) to state your agreement 
    # to the Code of Academic Integrity.
    sign_academic_honesty_policy('Calvin Kranig', '9083889825')

###########################################################################
# Tests for Challenge 1: Refocusing Application
###########################################################################

def challenge1a():
    from hw6_challenge1 import generateIndexMap, loadFocalStack
    focal_stack_dir = 'data/stack'
    rgb_stack, gray_stack = loadFocalStack(focal_stack_dir)
    # rgb_stack is an mxnx3k matrix, where m and n are the height and width of
    # the image, respectively, and 3k is the number of images in a focal stack
    # multiplied by 3 (each image contains RGB channels). 
    #
    # rgb_stack will only be used for the refocusing app viewer (it is not used
    # here).
    #
    # gray_stack is an mxnxk matrix.

    # Specify the (half) window size used for focus measure computation
    half_window_size = 1

    # Generate an index map, here we will only use the gray-scale images
    index_map = generateIndexMap(gray_stack, half_window_size)
    Image.fromarray((index_map * 255).astype(np.uint8)).save('outputs/index_map_org.png')
    Image.fromarray(((index_map / (len(gray_stack) - 1)) * 255).astype(np.uint8)).save('outputs/index_map.png')

def challenge1b():
    from hw6_challenge1 import loadFocalStack, refocusApp
    focal_stack_dir = 'data/stack'
    rgb_stack, gray_stack = loadFocalStack(focal_stack_dir)

    index_map = ((np.array(Image.open('outputs/index_map.png')) / 255) * (len(gray_stack) - 1)).astype(np.uint8)
    refocusApp(rgb_stack, index_map)

if __name__ == '__main__':
    #challenge1a()
    #challenge1b()
    runHw6()