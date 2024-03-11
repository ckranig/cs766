from PIL import Image, ImageDraw
import numpy as np
from typing import Union, Tuple, List
from helpers import genSIFTMatches


def computeHomography(src_pts_nx2: np.ndarray, dest_pts_nx2: np.ndarray) -> np.ndarray:
    '''
    Compute the homography matrix.
    Arguments:
        src_pts_nx2: the coordinates of the source points (nx2 numpy array).
        dest_pts_nx2: the coordinates of the destination points (nx2 numpy array).
    Returns:
        H_3x3: the homography matrix (3x3 numpy array).
    '''
    src_pts_nx2 = np.array(src_pts_nx2)
    dest_pts_nx2 = np.array(dest_pts_nx2)
    A = np.zeros((2 * src_pts_nx2.shape[0], 9))
    for i in range(src_pts_nx2.shape[0]):
        x, y = src_pts_nx2[i]
        u, v = dest_pts_nx2[i]
        A[2 * i] = [-x, -y, -1, 0, 0, 0, x * u, y * u, u]
        A[2 * i + 1] = [0, 0, 0, -x, -y, -1, x * v, y * v, v]
    _, eigenvectors = np.linalg.eig(A.T @ A)
    hom_mat = eigenvectors[:, np.argmin(np.abs(_))].reshape(3,3)

    # Normalize the homography matrix
    hom_mat /= hom_mat[2, 2]

    return hom_mat


def applyHomography(H_3x3: np.ndarray, src_pts_nx2: np.ndarray) ->  np.ndarray:
    '''
    Apply the homography matrix to the source points.
    Arguments:
        H_3x3: the homography matrix (3x3 numpy array).
        src_pts_nx2: the coordinates of the source points (nx2 numpy array).
    Returns:
        dest_pts_nx2: the coordinates of the destination points (nx2 numpy array).
    '''
    src_pts_nx2 = np.array(src_pts_nx2)
    H_3x3 = np.array(H_3x3)
    homogeneous_src_pts = np.column_stack((src_pts_nx2, np.ones(src_pts_nx2.shape[0])))
    homogeneous_dest_pts = np.dot(H_3x3, homogeneous_src_pts.T).T
    #Normalize
    dest_pts_nx2 = homogeneous_dest_pts[:, :2] / homogeneous_dest_pts[:, 2:]

    return dest_pts_nx2


def showCorrespondence(img1: Image.Image, img2: Image.Image, pts1_nx2: np.ndarray, pts2_nx2: np.ndarray) -> Image.Image:
    '''
    Show the correspondences between the two images.
    Arguments:
        img1: the first image.
        img2: the second image.
        pts1_nx2: the coordinates of the points in the first image (nx2 numpy array).
        pts2_nx2: the coordinates of the points in the second image (nx2 numpy array).
    Returns:
        result: image depicting the correspondences.
    '''
    if type(img1) != Image.Image:
        img1 = Image.fromarray(img1)
    if type(img2) != Image.Image:
        img2 = Image.fromarray(img2)
    max_height = max(img1.height, img2.height)
    combined_width = img1.width + img2.width
    combined_image = Image.new('RGB', (combined_width, max_height))
    combined_image.paste(img1, (0, 0))
    combined_image.paste(img2, (img1.width, 0))
    draw = ImageDraw.Draw(combined_image)

    for i in range(pts1_nx2.shape[0]):
        draw.line((pts1_nx2[i][0], pts1_nx2[i][1], pts2_nx2[i][0]+img1.width, pts2_nx2[i][1]), fill="red", width=4)

    return combined_image

# function [mask, result_img] = backwardWarpImg(src_img, resultToSrc_H, dest_canvas_width_height)

def backwardWarpImg(src_img: Image.Image, destToSrc_H: np.ndarray, canvas_shape: Union[Tuple, List]) -> Tuple[Image.Image, Image.Image]:
    '''
    Backward warp the source image to the destination canvas based on the
    homography given by destToSrc_H. 
    Arguments:
        src_img: the source image.
        destToSrc_H: the homography that maps points from the destination
            canvas to the source image.
        canvas_shape: shape of the destination canvas (height, width).
    Returns:
        dest_img: the warped source image.
        dest_mask: a mask indicating sourced pixels. pixels within the
            source image are 1, pixels outside are 0.
    '''
    # Get the height and width of the destination canvas
    dest_height, dest_width = canvas_shape

    # Generate a meshgrid for the destination canvas
    dest_x, dest_y = np.meshgrid(np.arange(dest_width), np.arange(dest_height))
    dest_pts_nx2 = np.column_stack((dest_x.flatten(), dest_y.flatten()))

    # Apply the inverse homography to map destination points to source image
    src_pts_nx2 = applyHomography(destToSrc_H, dest_pts_nx2)

    # Round and convert the source points to integers
    src_pts_nx2 = np.round(src_pts_nx2).astype(int)

    dest_img_array = np.zeros((dest_height, dest_width, 3))
    dest_mask_array = np.zeros((dest_height, dest_width), dtype=np.uint8)
    src_array = np.array(src_img)
    src_width = src_array.shape[1]
    src_height = src_array.shape[0]

    inside_source = np.logical_and.reduce((src_pts_nx2[:, 0] >= 0, src_pts_nx2[:, 0] < src_width,
                                           src_pts_nx2[:, 1] >= 0, src_pts_nx2[:, 1] < src_height))

    dest_img_array[dest_y.flatten()[inside_source], dest_x.flatten()[inside_source], :] = src_array[src_pts_nx2[inside_source, 1], src_pts_nx2[inside_source, 0], :]
    dest_mask_array[dest_y.flatten()[inside_source], dest_x.flatten()[inside_source]] = 1

    # Convert NumPy arrays back to PIL Images
    dest_img = Image.fromarray((dest_img_array * 255).astype(np.uint8), mode="RGB")
    dest_mask = Image.fromarray(dest_mask_array, mode="L")

    return dest_img, dest_mask


def blendImagePair(img1: Image.Image, mask1: Image.Image, img2: Image.Image, mask2: Image.Image, mode: str) -> Image.Image:
    '''
    Blend the warped images based on the masks.
    Arguments:
        img1: source image.
        mask1: source mask.
        img2: destination image.
        mask2: destination mask.
        mode: either 'overlay' or 'blend'
    Returns:
        out_img: blended image.
    '''
    mask1_normalized = np.array(mask1).astype(np.float32)
    mask2_normalized = np.array(mask2).astype(np.float32)

    mask1_stacked = np.stack([mask1_normalized, mask1_normalized, mask1_normalized], axis=2)
    mask2_stacked = np.stack([mask2_normalized, mask2_normalized, mask2_normalized], axis=2)
    img1 = np.array(img1)
    img2 = np.array(img2)

    if mode == "overlay":
        ret = np.array(img1)
        ret = ret*np.logical_not(mask2_stacked) + (img2 * mask2_stacked)
    else: #Mode is blend
        ret = (img1 * mask1_stacked + img2 * mask2_stacked)
        combinded_max = mask1_stacked + mask2_stacked
        combinded_max[combinded_max == 0] = 1
        ret /= combinded_max

    return Image.fromarray((ret * 255).astype(np.uint8), mode="RGB")

def runRANSAC(src_pt: np.ndarray, dest_pt: np.ndarray, ransac_n: int, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Run the RANSAC algorithm to find the inliers between the source and
    destination points.
    Arguments:
        src_pt: the coordinates of the source points (nx2 numpy array).
        dest_pt: the coordinates of the destination points (nx2 numpy array).
        ransac_n: the number of iterations to run RANSAC.
        eps: the threshold for considering a point to be an inlier.
    Returns:
        inliers_id: the indices of the inliers (kx1 numpy array).
        H: the homography matrix (3x3 numpy array).
    '''
    num_pixels = 4
    best_inliers_id = np.array([])
    best_H = np.zeros((3, 3))
    for i in range(ransac_n):
        random_indices = np.random.choice(src_pt.shape[0], num_pixels, replace=False)
        H_3x3 = computeHomography(src_pt[random_indices, :], dest_pt[random_indices, :])
        pred_pts = applyHomography(H_3x3, src_pt)
        distances = np.linalg.norm(pred_pts - dest_pt, axis=1)
        inliers_id = np.where(distances < eps)[0]
        if len(inliers_id) > len(best_inliers_id):
            best_inliers_id = inliers_id
            best_H = H_3x3

    return best_inliers_id, best_H

def add_translation_to_homography(H, tx, ty):
    translation_matrix = np.array([[1, 0, tx],
                                   [0, 1, ty],
                                   [0, 0, 1]])
    H_new = np.dot(translation_matrix, H)

    return H_new

def stitchImg(*args: Image.Image) -> Image.Image:
    '''
    Stitch a list of images.
    Arguments:
        args: a variable number of input images.
    Returns:
        stitched_img: the stitched image.
    '''
    imgs = [np.array(img) for img in args]
    max_width = 0
    max_height = 0
    for img in imgs:
        max_width += img.shape[1]
        max_height += img.shape[0]
    canvas = np.zeros((max_height,max_width,3))
    #Assume first image is in top left
    min_x = int(max_width/2) - int(imgs[0].shape[1]/2)
    min_y = int(max_height/2) - int(imgs[0].shape[0]/2)
    max_x = min_x + imgs[0].shape[1]
    max_y = min_y + imgs[0].shape[0]
    canvas[min_y:max_y, min_x:max_x, :] = imgs[0]
    #tmp = Image.fromarray((canvas * 255).astype(np.uint8), mode="RGB")
    #tmp.save('outputs/tmp_stitch_{}.png'.format(0))
    for i in range(1, len(imgs)):
        curImg = imgs[i]
        canvas_mask = np.any(canvas > 0, axis=-1)
        positive_indices = np.where(canvas > 0)
        upper_left = (min(positive_indices[0]), min(positive_indices[1]))
        bottom_right = (max(positive_indices[0]), max(positive_indices[1]))
        curCanv = canvas[upper_left[0]:bottom_right[0],upper_left[1]:bottom_right[1]]
        # Compute homography between consecutive images
        t_xs, t_xd = genSIFTMatches(curImg, curCanv)
        xs = np.zeros(t_xs.shape)
        xd = np.zeros(t_xd.shape)
        xs[:,0] = t_xs[:,1]
        xs[:,1] = t_xs[:,0]
        xd[:,0] = t_xd[:,1]
        xd[:,1] = t_xd[:,0]
        inliers_id, H_3x3 = runRANSAC(xs, xd, 100, 2)
        after_img = showCorrespondence((curImg* 255).astype(np.uint8), (curCanv* 255).astype(np.uint8), xs[inliers_id, :], xd[inliers_id, :])
        #after_img.save('outputs/tmp_corr_{}.png'.format(i))

        # Warp the second image onto the canvas
        H_3x3 = add_translation_to_homography(H_3x3, upper_left[1], upper_left[0])
        dest_img, dest_mask = backwardWarpImg(curImg, np.linalg.inv(H_3x3), (max_height, max_width))
        #dest_img.save('outputs/tmp_dstImg_{}.png'.format(i))
        # Blend the warped image onto the canvas
        newCanvas = blendImagePair(canvas, canvas_mask, np.array(dest_img)/255.0, np.array(dest_mask)/255.0, "blend")
        #newCanvas.save('outputs/tmp_stitch_{}.png'.format(i))
        canvas = np.array(newCanvas)/ 255.0

    positive_indices = np.where(canvas > 0)
    upper_left = (min(positive_indices[0]), min(positive_indices[1]))
    bottom_right = (max(positive_indices[0]), max(positive_indices[1]))
    curCanv = canvas[upper_left[0]:bottom_right[0],upper_left[1]:bottom_right[1]]
    #curCanv = canvas[min_y:max_y,upper_left[1]:bottom_right[1],:]
    return Image.fromarray((curCanv* 255).astype(np.uint8), mode="RGB")