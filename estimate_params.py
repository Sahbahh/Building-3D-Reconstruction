import numpy as np
from scipy.linalg import svd, rq

def estimate_params(P):
    """
    Computes the intrinsic K, rotation R, and translation t from
    given camera matrix P.

    Args:
        P: Camera matrix
    """
    # compute the camera center 
    U, S, Vt = svd(P)
    c = Vt[-1]
    # normalize the camera center
    c = c[:3] / c[3]  

    # intrinsic K and rotation R using rq decomposition
    M = P[:, :3]  # left 3x3 matrix 
    K, R = rq(M)

    # ensuring diagonals are positive
    T = np.diag(np.sign(np.diag(K)))
    K = K @ T
    R = T @ R

    # normalize K to make the last entry 1
    K /= K[2, 2]

    # correct sign of R 
    if np.linalg.det(R) < 0:
        R = -R

    t = -R @ c

    return K, R, t
