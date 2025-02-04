import numpy as np

def get_depth(dispM, K1, K2, R1, R2, t1, t2):
    """
    creates a depth map from a disparity map (dispM).
    """
    # baseline
    b = np.linalg.norm(t1 - t2)

    # focal length f from the camera matrix K1 
    f = K1[0, 0]

    depthM = np.zeros_like(dispM, dtype=float)

    # avoid division by zero
    non_zero_disp = dispM > 0

    depthM[non_zero_disp] = (b * f) / dispM[non_zero_disp]

    return depthM
