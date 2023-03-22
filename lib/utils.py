
import numpy as np
from numpy import linalg

def get_corners(img_shape):
    # check img corner points for later use
    # corners: 3x4
    height, width, channel = img_shape
    lt = np.array([0,0,1]).reshape(3, 1)
    lb = np.array([0,height-1,1]).reshape(3, 1)
    rt = np.array([width-1,0,1]).reshape(3, 1)
    rb = np.array([width-1,height-1,1]).reshape(3, 1)
    corner = np.concatenate([lt, lb, rt, rb], axis=1)

    return corner

def compute_canvas_size(img_shape, Hs, n_imgs):
    idx_ref = 0
    corners_pair = get_corners(img_shape) # corners: 3x4
    minxy = np.min(corners_pair, axis=1, keepdims=True)
    maxxy = np.max(corners_pair, axis=1, keepdims=True)
    for idx_pair in range(1, n_imgs):
        
        # IMPLEMENT HERE -------------------------------
        # convert pair image corner points coordinates to the reference canvas coordinates.
        corner_ref = None
        # ----------------------------------------------

        minxy = np.floor(np.min(np.minimum(corner_ref, minxy), axis=1, keepdims=True))
        maxxy = np.ceil(np.max(np.maximum(corner_ref, maxxy), axis=1, keepdims=True))

    return minxy, maxxy