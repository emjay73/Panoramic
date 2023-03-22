
import random 
import numpy as np
from lib.NDLT import NDLT, DLT
from scipy.optimize import least_squares

def find_robust_H_from_ransac(matches, kp_ref, kp_pair, ransac_iter=500, ransac_d=6):
    
    # keypoints
    kpm_ref     = keypoints2homogeneousmatrix(kp_ref)
    kpm_pair    = keypoints2homogeneousmatrix(kp_pair)
    
    # paired keypoints (keep matched keypoints only)
    n_matches   = len(matches)
    ids_ref     = [matches[i][0].queryIdx for i in range(n_matches)]
    ids_pair    = [matches[i][0].trainIdx for i in range(n_matches)]
    kpm_ref     = kpm_ref[:, ids_ref]
    kpm_pair    = kpm_pair[:, ids_pair]
    
    n_max_inlier = -1    
    tf_best = None
    for _ in range(ransac_iter):
        
        ids_random = random.sample(range(n_matches), 4)
                
        p4_ref    = kpm_ref[:, ids_random]
        p4_pair   = kpm_pair[:, ids_random]
        
        H = NDLT(p4_ref, p4_pair)
        
        # IMPLEMENT HERE ---------------------------------------------------
        # transfer error Hartley p94                
        err_est = None # squared dist
        # ------------------------------------------------------------------

        tf_inlier = err_est < ransac_d
        n_inlier = tf_inlier.sum()
        if n_inlier > n_max_inlier:
            n_max_inlier = n_inlier            
            tf_best = tf_inlier

    print(f'max num inlier: {n_max_inlier}')

    # IMPLEMENT HERE ---------------------------------------------------
    # re-estimate H using the inliers only 
    # Hartley p188    
    H           = None
    # ------------------------------------------------------------------

    # find over-determined solution H
    # Hartley p90    
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
    H = H.flatten()[:-1]
    H_opt = least_squares(residual, H, method='lm', args=(kin_ref, kin_pair))
    H = np.concatenate((H_opt.x, [1]), axis=0).reshape(3,3)

    return H

def keypoints2homogeneousmatrix(kp):
    # make 3xn matrix    
    kp_matrix = np.ones((3, len(kp)))
    for i, p in enumerate(kp):
        kp_matrix[0, i] = p.pt[0]
        kp_matrix[1, i] = p.pt[1]
    return kp_matrix

def residual(H, kin_ref, kin_pair):
    H = np.concatenate((H, [1]), axis=0).reshape(3,3)
    est = H@kin_ref
    kin_pair_est = est/est[2:3, :]
    err = ((kin_pair - kin_pair_est)**2).sum(axis=0)
    return err
    
