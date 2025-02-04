import numpy as np

def rectify_pair(K1, K2, R1, R2, t1, t2):
    """
    takes left and right camera paramters (K, R, T) and returns left
    and right rectification matrices (M1, M2) and updated camera parameters. You
    can test your function using the provided script testRectify.py
    """
    # YOUR CODE HERE
    # optical center c1 and c2 of each camera
    c1 = -np.linalg.inv(R1) @ t1.reshape(-1, 1)
    c2 = -np.linalg.inv(R2) @ t2.reshape(-1, 1)

    # normalized vector from c1 to c2
    r1 = (c1 - c2).flatten() / np.linalg.norm(c1 - c2)

    # Ensure r2 points downwards in image plane 
    r2 = np.cross(r1, R1[2, :])
    r3 = np.cross(r2, r1)

    # new rotation matrix R_new
    R_new = np.column_stack((r1, r2, r3))

    # normalize R_new to be orthonormal
    U, _, Vt = np.linalg.svd(R_new)
    R_new = U @ Vt

    ########
    if np.linalg.det(R_new) < 0:
        R_new *= -1

    # new intrinsic matrix for both cameras
    K_new = K2

    # new translations
    t1n = -R_new @ c1
    t2n = -R_new @ c2

    # rectification matrices M1 and M2
    M1 = K_new @ R_new @ np.linalg.inv(K1)
    M2 = K_new @ R_new @ np.linalg.inv(K2)

    # camera parameters after rectification
    K1n = K_new
    K2n = K_new
    R1n = R_new
    R2n = R_new

    return M1, M2, K1n, K2n, R1n, R2n, t1n, t2n
