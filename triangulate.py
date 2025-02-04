import numpy as np

def triangulate(P1, pts1, P2, pts2):
    """
    Estimate the 3D positions of points from 2d correspondence
    Args:
        P1:     projection matrix with shape 3 x 4 for image 1
        pts1:   coordinates of points with shape N x 2 on image 1
        P2:     projection matrix with shape 3 x 4 for image 2
        pts2:   coordinates of points with shape N x 2 on image 2

    Returns:
        Pts3d:  coordinates of 3D points with shape N x 3
    """
    num_points = pts1.shape[0]
    pts3d = np.zeros((num_points, 4))  # homogeneous coordinates

    for i in range(num_points):
        # Construct matrix A for each point pair
        A = np.zeros((4, 4))
        A[0] = pts1[i, 1] * P1[2] - P1[1]
        A[1] = pts1[i, 0] * P1[2] - P1[0]
        A[2] = pts2[i, 1] * P2[2] - P2[1]
        A[3] = pts2[i, 0] * P2[2] - P2[0]

        # Solve using SVD
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        # normalize
        pts3d[i] = X / X[3]  

    #non-homogeneous coordinates
    return pts3d[:, :3]  
