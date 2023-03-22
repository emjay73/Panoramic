#########################################
# Hartley p206~207
#########################################

import os 
import cv2
import numpy as np
from lib.ransac import find_robust_H_from_ransac
from lib.feature_matching import extract_feature, match_features
from lib.utils import compute_canvas_size

# image set 1 --------------------
folder_db = 'images1'
folder_out = 'outputs'
folder_dump = 'dump'
# --------------------------------- 

# image set 2 ----------------------
# folder_db = 'images2'
# folder_out = 'outputs2'
# folder_dump = 'dump2'
# --------------------------------- 

os.makedirs(folder_out, exist_ok=True)
os.makedirs(folder_dump, exist_ok=True)

# read images -------------------------------------------------------
list_img =[]
for file_img in sorted(os.listdir(folder_db)):
    print(f'load image {file_img}')
    list_img.append(cv2.imread(os.path.join(folder_db, file_img)))
n_imgs = len(list_img)

# prepare ref image ---------------------------------------------------
idx_ref = 0;
img_ref  = list_img[idx_ref]
kp_ref, des_ref = extract_feature(img_ref, dump_path = os.path.join(folder_dump, f'sift_keypoints_{idx_ref}.jpg'))

# estimate Hs ------------------------------------------------------------
Hs = {}
idx_pre = idx_ref
img_pre = img_ref
kp_pre  = kp_ref
des_pre = des_ref
for idx_now in range(idx_ref+1, n_imgs):

    print(f'using image pairs {idx_pre} and {idx_now}')
    img_now = list_img[idx_now]
    
    # extract features    
    kp_now, des_now = extract_feature(img_now, dump_path = os.path.join(folder_dump, f'sift_keypoints_{idx_now}.jpg'))

    # match features
    matches = match_features(kp_pre, kp_now, des_pre, des_now, dump_path=os.path.join(folder_dump, f'sift_matching_{idx_pre}_{idx_now}.jpg'), img1 = img_pre, img2 = img_now)

    # find H (ref -> pair)
    H = find_robust_H_from_ransac(matches, kp_pre, kp_now, ransac_iter=20000)
    Hs[(idx_pre, idx_now)] = H
    
    # IMPLEMENT HERE -------------------------------
    # find H(idx_ref->idx_now)    
    Hs[(idx_ref, idx_now)] = None # 3x3 matrix H that converts the homogeneous coordinates of the idx_ref image to that of the idx_now image
    # ----------------------------------------------

    # save previous information
    idx_pre = idx_now
    kp_pre  = kp_now
    des_pre = des_now
    img_pre = img_now


# original image size range
lt = np.array([0,0,1]).reshape(3, 1)
rb = np.array([img_ref.shape[1]-1,img_ref.shape[0]-1,1]).reshape(3, 1)

# estimate canvas size
minxy, maxxy = compute_canvas_size(img_ref.shape, Hs, n_imgs)

# get canvas coordinates
xs, ys = np.meshgrid(np.arange(minxy[0], maxxy[0]), np.arange(minxy[1], maxxy[1]))
canvas_h, canvas_w = xs.shape
xy1_canvas = np.stack([xs.flatten(), ys.flatten(), np.ones_like(xs.flatten())], axis=0)

# add reference image to the canvas
# https://docs.opencv.org/3.4/d1/da0/tutorial_remap.html
# https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#ga5bb5a1fea74ea38e1a5445ca803ff121
panoramic = cv2.remap(img_ref, xs.astype(np.float32), ys.astype(np.float32), interpolation=cv2.INTER_LANCZOS4)
cv2.imwrite(os.path.join(folder_out, f'widecanvas_{idx_ref}.jpg'), panoramic)

for idx_pair in range(1, n_imgs):
    print(f'stitching image {idx_pair}')
    img_pair = list_img[idx_pair]

    # IMPLEMENT HERE -------------------------------
    # canvas coordinates (xy1_canvas) (shape: (3,(canvas_h*canvas_w))) -> pair img coordinates (xy1_pair) (shape: (3, (canvas_h*canvas_w)))
    xy1_pair = None 
    # ----------------------------------------------

    # backward warping (draw pair image in the canvas)
    xy1_pair = xy1_pair.astype(np.float32)
    xs_pair = xy1_pair[0, :].reshape(xs.shape)
    ys_pair = xy1_pair[1, :].reshape(ys.shape)
    canvas_pair = cv2.remap(img_pair, xs_pair, ys_pair, interpolation=cv2.INTER_LANCZOS4)
    
    # IMPLEMENT HERE -------------------------------
    # estimate canvas mask named 'mask_pair'
    # true for the pixels that can be mapped to pair image pixels
    # shape: (canvas_h, canvas_w, 3)
    mask_pair = None
    # ----------------------------------------------

    # fuse pair image with canvas
    panoramic = np.where(mask_pair, canvas_pair, panoramic)    
    cv2.imwrite(os.path.join(folder_out, f'widecanvas_{idx_pair}.jpg'), panoramic)    




