
# MVG ch4, p91, p109
import numpy as np
from numpy import linalg

def NDLT(kp1, kp2):
    T1, pnorm1 = Normalization(kp1)
    T2, pnorm2 = Normalization(kp2)
    H = DLT(pnorm1, pnorm2)

    # IMPLEMENT HERE -------------------------
    # denormalize homography    
    # Hartley p109    
    H = None
    # ----------------------------------------
    
    return H

def Normalization(pts):
    # pts: ndarray, 3xn
    # T: ndarray, 3x3
    # pts_norm: ndarray, 3xn
    
    centroid    = pts.mean(axis=1, keepdims=True)
    pts         = (pts - centroid)
    d_mean      = np.sqrt( (pts**2).sum(axis=0) ).mean()
    scale       = np.sqrt(2)/d_mean
    pts_norm    = pts * scale
    pts_norm[-1, :] = 1

    # similar transformatin under the assumption that no rotation
    centroid = centroid.squeeze()
    T = np.array([  [scale,      0, -scale*centroid[0]],
                    [0,      scale, -scale*centroid[1]],
                    [0,          0,          1] ])
    
    return T, pts_norm


def DLT(p1, p2):
    # Direct Linear Transformation
    # Hartley p89

    p1 = p1.transpose()
    p2 = p2.transpose()
    
    # IMPLEMENT HERE -------------------------
    # estimate H using svd  
    # ref: https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html    
    H = None
    # ----------------------------------------    
    
    return H


