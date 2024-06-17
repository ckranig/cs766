import argparse
from runTests import run_tests
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def runHw5():
    # runHw5 is the "main" interface that lets you execute all the 
    # walkthroughs and challenges in this homework. It lists a set of 
    # functions corresponding to the problems that need to be solved.
    #
    # Note that this file also serves as the specifications for the functions 
    # you are asked to implement. In some cases, your submissions will be 
    # auto-graded.  Thus, it is critical that you adhere to all the specified 
    # function signatures.
    #
    # Before your submission, make sure you can run runHw5('all') 
    # without any error.
    #
    # Usage:
    # python runHw5.py                  : list all the registered functions
    # python runHw5.py 'function_name'  : execute a specific test
    # python runHw5.py 'all'            : execute all the registered functions
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
        'challenge1c': challenge1c, 
        'challenge1d': challenge1d, 
        'demoSurfaceReconstruction': demoSurfaceReconstruction, 
    }
    run_tests(args.function_name, fun_handles)

# Academic Honesty Policy
def honesty():
    from signAcademicHonestyPolicy import sign_academic_honesty_policy
    # Type your full name and uni (both in string) to state your agreement 
    # to the Code of Academic Integrity.
    sign_academic_honesty_policy('Calvin Kranig', '9083889825')

###########################################################################
# Tests for Challenge 1: Photometric Stereo
###########################################################################

# Compute the properties of the sphere
def challenge1a():
    from hw5_challenge1 import findSphere

    img = Image.open('data/sphere0.png')
    img = np.array(img) / 255.0

    center, radius = findSphere(img)
    np.savez('outputs/sphere_properties.npz', center=center, radius=radius)
    print(f"Sphere center: {center}\nSphere radius: {radius}")


# Compute the directions of light sources
def challenge1b(): 
    from hw5_challenge1 import computeLightDirections
    
    img_list = [
        np.array(Image.open(f'data/sphere{i}.png')) / 255.0
        for i in range(1, 6)
    ]

    data = np.load('outputs/sphere_properties.npz')
    center = data['center']
    radius = data['radius']

    light_dirs_5x3 = computeLightDirections(center, radius, img_list)

    np.save('outputs/light_dirs.npy', light_dirs_5x3)
    print(f"Light directions:\n{light_dirs_5x3}")

# Compute the mask of the object
def challenge1c():
    from hw5_challenge1 import computeMask
    vase_img_list = [
        np.array(Image.open(f'data/vase{i}.png')) / 255.0
        for i in range(1, 6)
    ]

    mask = computeMask(vase_img_list)
    mask = Image.fromarray((mask * 255).astype(np.uint8))
    mask.save('outputs/vase_mask.png')

# Compute surface normals and albedos of the object
def challenge1d():
    from hw5_challenge1 import computeNormals
    
    # Load the mask image and cast it back to logical
    mask = np.array(Image.open('outputs/vase_mask.png')).astype(bool)

    # Load the light directions
    light_dirs_5x3 = np.load('outputs/light_dirs.npy')

    # Load the images of the vase
    vase_img_list = [
        np.array(Image.open(f'data/vase{i}.png')) / 255.0
        for i in range(1, 6)
    ]

    # Compute the surface normals and albedo
    normals, albedo_img = computeNormals(light_dirs_5x3, vase_img_list, mask)

    # normals is a mxnx3 matrix which contains 
    # the x-, y-, z- components of the normal of each pixel.

    # Visualize the surface normals as a normal map image. 
    # Normal maps are images that store normals directly 
    # in the RGB values of an image. The mapping is as follows:
    # X (-1.0 to +1.0) maps to Red (0-255)
    # Y (-1.0 to +1.0) maps to Green (0-255)
    # Z (-1.0 to +1.0) maps to Blue (0-255)
    # A normal map thumbnail sphere_normal_map.png 
    # for a sphere is included for your reference.
    normal_map_img = ((normals + 1) / 2 * 255).astype(np.uint8)
    albedo_img = (albedo_img * 255).astype(np.uint8)

    # Save the images and normals
    Image.fromarray(normal_map_img).save('outputs/vase_normal_map.png')
    Image.fromarray(albedo_img).save('outputs/vase_albedo.png')
    np.save('outputs/normals.npy', normals)

    fig, axs = plt.subplots(2, 1)
    axs[0].imshow(normal_map_img); axs[0].set_title('Normal map')
    axs[1].imshow(albedo_img); axs[1].set_title('Albedo')
    plt.show()
    print(f"Surface normals:\n{normals}")
    

# Demo (no submission required)
def demoSurfaceReconstruction():
    from helpers import reconstructSurf
    # Load the normals
    normals = np.load('outputs/normals.npy')

    # Load the mask image and cast it back to logical
    mask = np.array(Image.open('outputs/vase_mask.png')).astype(bool)

    # reconstructSurf demonstrates surface reconstruction 
    # using the Frankot-Chellappa algorithm
    surf_img = reconstructSurf(normals, mask)
    surf_img_img = Image.fromarray((surf_img * 255).astype(np.uint8))
    surf_img_img.save('outputs/vase_surface.png')

    # Use the surf tool to visualize the 3D reconstruction
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.arange(surf_img.shape[0]), np.arange(surf_img.shape[1])
    
    X, Y = np.meshgrid(X, Y, indexing='ij')
    ax.plot_surface(X, Y, surf_img)
    plt.show()

if __name__ == '__main__':
    runHw5()
    demoSurfaceReconstruction()