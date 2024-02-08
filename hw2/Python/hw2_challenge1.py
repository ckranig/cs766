import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

from PIL import Image
import numpy as np

import skimage


def generateLabeledImage(gray_img: np.ndarray, threshold: float) -> np.ndarray:
    '''
    Generates a labeled image from a grayscale image by assigning unique labels to each connected component.
    Arguments:
        gray_img: grayscale image.
        threshold: threshold for the grayscale image.
    Returns:
        labeled_img: the labeled image.
    '''
    bw_img = gray_img > threshold
    labeled_img, num_objs = skimage.measure.label(bw_img, return_num = True)
    return labeled_img

def tmp(labeled_img: np.ndarray):
    from scipy.ndimage import measurements
    num_objs = np.unique(labeled_img)
    for obj in range(1, len(num_objs)):
        bw_img = labeled_img == obj
        cx, cy = measurements.center_of_mass(bw_img)
        print(cx, cy)

def compute2DProperties(orig_img: np.ndarray, labeled_img: np.ndarray) ->  np.ndarray:
    '''
    Compute the 2D properties of each object in labeled image.
    Arguments:
        orig_img: the original image.
        labeled_img: the labeled image.
    Returns:
        obj_db: the object database, where each row contains the properties
            of one object.
    '''
    num_objs = np.unique(labeled_img)
    """
    Columns:
    1.	Object	label,
    2.	Row	position	of	the	center,
    3.	Column	position	of	the	center,
    4.	The	minimum	moment	of	inertia,
    5.	The	orientation	(angle	in	degrees	between	the	axis	of	minimum	inertia	and	
    the	horizontal	axis,	positive	=	clockwise	from	the	horizontal	axis),	
    6.	The	roundness.
    """
    obj_db = np.zeros((len(num_objs)-1,6))
    row_sums = np.zeros(len(num_objs)-1)
    col_sums = np.zeros(len(num_objs)-1)
    A_sum = np.zeros(len(num_objs)-1)
    #Calculate Initial Sums
    for row in range(labeled_img.shape[0]):
        for col in range(labeled_img.shape[1]):
            obj = labeled_img[row][col] - 1
            if obj >= 0:
                row_sums[obj] = row_sums[obj] + row
                col_sums[obj] = col_sums[obj] + col
                A_sum[obj] = A_sum[obj] + 1
    #Calculate Averages
    for obj in range(len(num_objs)-1):
        obj_db[obj][0] = obj + 1
        obj_db[obj][1] = round(row_sums[obj] / A_sum[obj])
        obj_db[obj][2] = round(col_sums[obj] / A_sum[obj])
    #Calculate Shifted a,b,c
    a_s = np.zeros(len(num_objs)-1)
    b_s = np.zeros(len(num_objs)-1)
    c_s = np.zeros(len(num_objs)-1)
    for row in range(labeled_img.shape[0]):
        for col in range(labeled_img.shape[1]):
            obj = labeled_img[row][col] - 1
            if obj >= 0:
                mean_x = obj_db[obj][2]
                mean_y = obj_db[obj][1]
                x_p = col - mean_x
                y_p = row - mean_y
                a_s[obj] = a_s[obj] + x_p**2
                b_s[obj] = b_s[obj] + x_p * y_p
                c_s[obj] = c_s[obj] + y_p**2
    #Calculate thetas, orientation, min moment and roundness
    for obj in range(len(num_objs)-1):
        theta1 = 0.5 * np.arctan2(2 * b_s[obj], a_s[obj] - c_s[obj])
        theta2 = theta1 + np.pi/2
        #E = asin^2(theta) - bsin(theta)cos(theta) + ccos^2(theta)
        E1 = a_s[obj]*(np.sin(theta1)**2) - 2*b_s[obj]*np.sin(theta1)*np.cos(theta1) + c_s[obj]*(np.cos(theta1)**2)
        E2 = a_s[obj]*(np.sin(theta2)**2) - 2*b_s[obj]*np.sin(theta2)*np.cos(theta2) + c_s[obj]*(np.cos(theta2)**2)
        if E1 < E2:
            obj_db[obj][3] = E1
            obj_db[obj][4] = theta1
            obj_db[obj][5] = E1/E2
        else:
            obj_db[obj][3] = E2
            obj_db[obj][4] = theta2
            obj_db[obj][5] = E2/E1
    return obj_db

def recognizeObjects(orig_img: np.ndarray, labeled_img: np.ndarray, obj_db: np.ndarray, output_fn: str):
    '''
    Recognize the objects in the labeled image and save recognized objects to output_fn
    Arguments:
        orig_img: the original image.
        labeled_img: the labeled image.
        obj_db: the object database, where each row contains the properties 
            of one object.
        output_fn: filename for saving output image with the objects recognized.
    '''
    measured_properties = [5]
    thresholds = [0.05]
    new_obj_db = compute2DProperties(orig_img, labeled_img)
    fig, ax = plt.subplots()
    plt.axis(False)
    ax.imshow(orig_img, cmap='gray')
    for i in range(new_obj_db.shape[0]):
        match = False
        for j in range(obj_db.shape[0]):
            out = abs(new_obj_db[i][measured_properties] - obj_db[j][measured_properties]) < thresholds
            if np.all(out):
                match = True
                break
        if match:
            length = 50
            x1 = new_obj_db[i][2]
            y1 = new_obj_db[i][1]
            x2 = x1 + length * np.cos(new_obj_db[i][4])
            y2 = y1 + length * np.sin(new_obj_db[i][4])
            x3 = x1 + length * np.cos(new_obj_db[i][4] + np.pi)
            y3 = y1 + length * np.sin(new_obj_db[i][4] + np.pi)
            # plot the position
            ax.plot(x1,y1, 'r*',markersize=10)
            # plot the orientation
            ax.plot([x2, x3], [y2, y3], color='r', linestyle='-', linewidth=2)
    plt.savefig(output_fn)

def hw2_challenge1a():
    import matplotlib.cm as cm
    from skimage.color import label2rgb
    from hw2_challenge1 import generateLabeledImage
    img_list = ['two_objects.png', 'many_objects_1.png', 'many_objects_2.png']
    threshold_list = [0.5,0.5,0.5]#[0.3, 0.315, 0.3]   # You need to find the right thresholds

    for i in range(len(img_list)):
        orig_img = Image.open(f"data/{img_list[i]}")
        orig_img = np.array(orig_img.convert('L')) / 255.
        labeled_img = generateLabeledImage(orig_img, threshold_list[i])
        Image.fromarray(labeled_img.astype(np.uint8)).save(
            f'outputs/labeled_{img_list[i]}')
        
        cmap = np.array(cm.get_cmap('Set1').colors)
        rgb_img = label2rgb(labeled_img, colors=cmap, bg_label=0)
        Image.fromarray((rgb_img * 255).astype(np.uint8)).save(
            f'outputs/rgb_labeled_{img_list[i]}')

def hw2_challenge1b():
    labeled_two_obj = Image.open('outputs/labeled_two_objects.png')
    labeled_two_obj = np.array(labeled_two_obj)
    orig_img = Image.open('data/two_objects.png')
    orig_img = np.array(orig_img.convert('L')) / 255.
    obj_db  = compute2DProperties(orig_img, labeled_two_obj)
    np.save('outputs/obj_db.npy', obj_db)
    print(obj_db)
    
    # TODO: Plot the position and orientation of the objects
    # Use a dot or star to annotate the position and a short line segment originating from the dot for orientation
    # Refer to demoTricksFun.py for examples to draw dots and lines. 
    fig, ax = plt.subplots()
    plt.axis(False)
    ax.imshow(orig_img, cmap='gray')
    for i in range(obj_db.shape[0]):
        length = 50
        x1 = obj_db[i][2]
        y1 = obj_db[i][1]
        x2 = x1 + length * np.cos(obj_db[i][4])
        y2 = y1 + length * np.sin(obj_db[i][4])
        x3 = x1 + length * np.cos(obj_db[i][4] + np.pi)
        y3 = y1 + length * np.sin(obj_db[i][4] + np.pi)
        # plot the position
        ax.plot(x1,y1, 'r*',markersize=10)
        # plot the orientation
        ax.plot([x2, x3], [y2, y3], color='r', linestyle='-', linewidth=2)
    plt.savefig('outputs/two_objects_properties.png')
    plt.show()

def hw2_challenge1c():
    obj_db = np.load('outputs/obj_db.npy')
    img_list = ['many_objects_1.png', 'many_objects_2.png']

    for i in range(len(img_list)):
        labeled_img = Image.open(f'outputs/labeled_{img_list[i]}')
        labeled_img = np.array(labeled_img)
        orig_img = Image.open(f"data/{img_list[i]}")
        orig_img = np.array(orig_img.convert('L')) / 255.

        recognizeObjects(orig_img, labeled_img, obj_db,
                         f'outputs/testing1c_{img_list[i]}')

if __name__ == "__main__":
    #hw2_challenge1a()
    #hw2_challenge1b()
    hw2_challenge1c()