import numpy as np
from refineF import refineF

def eightpoint(pts1, pts2, M):

    """
    eightpoint:
        pts1 - Nx2 matrix of (x,y) coordinates
        pts2 - Nx2 matrix of (x,y) coordinates
        M    - max(imwidth, imheight)
    """
    
    # Normalize points
    pts1 = pts1 / M
    pts2 = pts2 / M

    # Construct matrix A
    A = np.vstack([pts2[:, 0] * pts1[:, 0], pts2[:, 0] * pts1[:, 1], pts2[:, 0],
                   pts2[:, 1] * pts1[:, 0], pts2[:, 1] * pts1[:, 1], pts2[:, 1],
                   pts1[:, 0], pts1[:, 1], np.ones(len(pts1))]).T

    # Compute fundamental matrix 
    _, _, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)

    # rank 2 constraint
    U, S, Vt = np.linalg.svd(F)
    S[2] = 0
    F = U @ np.diag(S) @ Vt

    # Flatten F before passing to refineF
    F_flattened = F.flatten()
    F_refined_flattened = refineF(F_flattened, pts1, pts2)

    # Reshape F back to 3x3  after refinement
    F_refined = F_refined_flattened.reshape(3, 3)

    # Unnormalize F
    T = np.array([[1/M, 0, 0], [0, 1/M, 0], [0, 0, 1]])
    F = T.T @ F_refined @ T

    return F
