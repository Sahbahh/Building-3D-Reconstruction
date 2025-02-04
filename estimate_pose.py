# import numpy as np
# from scipy.linalg import svd
# from scipy.optimize import least_squares

# ## normalizes 2D points so that the centroid is at the origin 
# ## and the average distance to the origin is sqrt(2).
# ## source: https://stackoverflow.com/questions/47544099/trying-to-understand-what-is-happening-in-this-python-function
# def normalize_points(points):
#     centroid = np.mean(points, axis=1)
#     scale = np.sqrt(2) / np.mean(np.sqrt(np.sum((points - centroid[:, np.newaxis]) ** 2, axis=0)))
#     T = np.array([[scale, 0, -scale * centroid[0]],
#                   [0, scale, -scale * centroid[1]],
#                   [0, 0, 1]])
#     normalized_points = T @ np.vstack([points, np.ones((1, points.shape[1]))])
#     return normalized_points, T


# ## reprojection error for camera matrix estimation.
# def reprojection_error(P_vec, x, X):
#     P = P_vec.reshape(3, 4)
#     x_projected = P @ X
#     x_projected /= x_projected[2]
#     error = x_projected[:2] - x
#     return error.ravel()



# def estimate_pose(x, X):
#     """
#     Estimates the camera matrix P given 2D and 3D points.
#     """
#     # normalize 
#     x_normalized, T_x = normalize_points(x)

#     num_points = x.shape[1]
#     A = np.zeros((num_points * 2, 12))
#     for i in range(num_points):
#         X_i = X[:, i]
#         x_i = x_normalized[:, i]
#         A[2*i] = np.array([-X_i[0], -X_i[1], -X_i[2], -1, 0, 0, 0, 0, x_i[1]*X_i[0], x_i[1]*X_i[1], x_i[1]*X_i[2], x_i[1]])
#         A[2*i+1] = np.array([0, 0, 0, 0, X_i[0], X_i[1], X_i[2], 1, -x_i[0]*X_i[0], -x_i[0]*X_i[1], -x_i[0]*X_i[2], -x_i[0]])

#     # solve for P using SVD
#     U, S, Vt = svd(A)
#     P = Vt[-1].reshape(3, 4)

#     # refine P using non linear least squares
#     X_homog = np.vstack((X, np.ones((1, num_points))))
#     P_vec = least_squares(reprojection_error, P.ravel(), args=(x_normalized[:2], X_homog)).x
#     P_refined = P_vec.reshape(3, 4)

#     # unnormalize 
#     P_refined = np.linalg.inv(T_x) @ P_refined

#     return P_refined



import numpy as np
from scipy.linalg import svd

def estimate_pose(x, X):
    """
    Estimates the camera matrix P given 2D and 3D points.

    Args:
        x: 2D points with shape [2, N]
        X: 3D points with shape [3, N]
    """
    # normalize points
    def normalize_points(points):
        centroid = np.mean(points, axis=1)
        scale = np.sqrt(2) / np.mean(np.sqrt(np.sum((points - centroid[:, np.newaxis])**2, axis=0)))
        T = np.array([[scale, 0, -scale * centroid[0]],
                      [0, scale, -scale * centroid[1]],
                      [0, 0, 1]])
        normalized_points = T @ np.vstack([points, np.ones((1, points.shape[1]))])
        return normalized_points, T

    # Normalize 2D points
    x_norm, T_x = normalize_points(x)

    # Construct matrix A for DLT
    num_points = X.shape[1]
    A = np.zeros((num_points * 2, 12))
    for i in range(num_points):
        X_i = np.hstack([X[:, i], 1])
        x_i = x_norm[:, i]
        A[2*i:2*i+2] = [[0, 0, 0, 0, -X_i[0], -X_i[1], -X_i[2], -1, x_i[1]*X_i[0], x_i[1]*X_i[1], x_i[1]*X_i[2], x_i[1]],
                        [X_i[0], X_i[1], X_i[2], 1, 0, 0, 0, 0, -x_i[0]*X_i[0], -x_i[0]*X_i[1], -x_i[0]*X_i[2], -x_i[0]]]

    # Solve for P using SVD
    _, _, Vt = svd(A)
    P = Vt[-1].reshape(3, 4)

    # unnormalize camera matrix P
    P = np.linalg.inv(T_x) @ P

    return P


# ######################################################################
# import numpy as np
# from scipy.linalg import svd

# def estimate_pose(x, X):
#     """
#     computes the pose matrix (camera matrix) P given 2D and 3D points.

#     Args:
#         x: 2D points with shape [2, N]
#         X: 3D points with shape [3, N]
#     """
#     # Convert points to homogeneous coordinates
#     ones = np.ones((1, X.shape[1]))
#     X_homog = np.vstack((X, ones))
#     x_homog = np.vstack((x, ones))

#     # Number of correspondences
#     N = X.shape[1]

#     # Construct matrix A
#     A = np.zeros((2*N, 12))
#     for i in range(N):
#         X_i = X_homog[:, i]
#         x_i = x_homog[:, i]
#         A[2*i] = np.array([
#             0, 0, 0, 0,
#             -X_i[0], -X_i[1], -X_i[2], -1,
#             x_i[1]*X_i[0], x_i[1]*X_i[1], x_i[1]*X_i[2], x_i[1]
#         ])
#         A[2*i + 1] = np.array([
#             X_i[0], X_i[1], X_i[2], 1,
#             0, 0, 0, 0,
#             -x_i[0]*X_i[0], -x_i[0]*X_i[1], -x_i[0]*X_i[2], -x_i[0]
#         ])

#     # Solve for P using SVD
#     U, S, Vt = svd(A)
#     P = Vt[-1].reshape(3, 4)  # The last row of Vt reshaped into 3x4 matrix

#     return P
