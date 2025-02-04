import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.io as sio
from PIL import Image
from eightpoint import eightpoint
from epipolarCorrespondence import epipolarCorrespondence
from essentialMatrix import essentialMatrix
from camera2 import camera2
from triangulate import triangulate
from displayEpipolarF import displayEpipolarF
from epipolarMatchGUI import epipolarMatchGUI

# Load images and points
img1 = cv2.imread('data/im1.png')
img2 = cv2.imread('data/im2.png')
pts = np.load('data/someCorresp.npy', allow_pickle=True).tolist()
pts1 = pts['pts1']
pts2 = pts['pts2']
M = max(img1.shape[0], img1.shape[1])
# M = pts['M']   ##


# ------------------
# 3.1.1
# find the fundamental matrix 
F = eightpoint(pts1, pts2, M)

np.save('results/fundamental_matrix.npy', F)

print("\nRecovered Fundamental Matrix F:")
print(F)

displayEpipolarF(img1, img2, F)


# ------------------
# 3.1.2
coordsIM1, coordsIM2 = epipolarMatchGUI(img1, img2, F)


# ------------------
# 3.1.3
intrinsics = np.load('data/intrinsics.npy', allow_pickle=True).item()
K1 = intrinsics['K1']  # Intrinsic camera matrix for the image 1
K2 = intrinsics['K2']  # image 2

E = essentialMatrix(F, K1, K2)
np.save('results/essential_matrix.npy', E)

print("\nEstimated Essential Matrix E:")
print(E)
print("\n")


# ------------------
# 3.1.4
# first camera matrix
P1 = np.hstack((np.eye(3), np.zeros((3, 1))))  
P1 = K1 @ P1  # Combine with intrinsic matrix K1

# compute the four candidates for P2
P2s = camera2(E)

# best results
best_P2 = None
best_pts3d = None
min_error = float('inf')
max_points_in_front = 0

# each candidate P2 matrix
for i in range(4):
    P2 = P2s[:, :, i]
    P2_full = K2 @ np.hstack((P2[:, :3], P2[:, 3:4]))

    coordsIM2 = epipolarCorrespondence(img1, img2, F, pts1)
    pts3d = triangulate(P1, pts1, P2_full, coordsIM2)

    # Check points in front of both cameras
    in_front_of_both_cameras = (pts3d[:, 2] > 0) & \
                               ((P2_full @ np.hstack((pts3d, np.ones((pts3d.shape[0], 1)))).T)[2, :] > 0)
    num_points_in_front = np.sum(in_front_of_both_cameras)

    # re-projection error
    reproj_pts1 = P1 @ np.hstack((pts3d, np.ones((pts3d.shape[0], 1)))).T
    reproj_pts2 = P2_full @ np.hstack((pts3d, np.ones((pts3d.shape[0], 1)))).T
    reproj_pts1 /= reproj_pts1[2]
    reproj_pts2 /= reproj_pts2[2]
    error = np.mean(np.linalg.norm(reproj_pts1[:2].T - pts1, axis=1)) + \
            np.mean(np.linalg.norm(reproj_pts2[:2].T - coordsIM2, axis=1))

    # best = most points in front of both cameras and lowest error
    if num_points_in_front > max_points_in_front or (num_points_in_front == max_points_in_front and error < min_error):
        min_error = error
        max_points_in_front = num_points_in_front
        best_P2 = P2_full
        best_pts3d = pts3d

print("Re-projection error:", min_error)
print("\n")



# ------------------
# 3.1.5
temple_coords = np.load('data/templeCoords.npy', allow_pickle=True).item()
temple_coords = temple_coords['pts1']

coordsIM2_temple = epipolarCorrespondence(img1, img2, F, temple_coords)

# like last part
for i in range(4):
    P2 = P2s[:, :, i]
    P2_full = K2 @ np.hstack((P2[:, :3], P2[:, 3:4]))

    pts3d_temple = triangulate(P1, temple_coords, P2_full, coordsIM2_temple)

    in_front_of_both_cameras = (pts3d_temple[:, 2] > 0) & \
                               ((P2_full @ np.hstack((pts3d_temple, np.ones((pts3d_temple.shape[0], 1)))).T)[2, :] > 0)

    num_points_in_front = np.sum(in_front_of_both_cameras)

    if num_points_in_front > min_error:
        best_P2 = P2_full
        best_pts3d = pts3d_temple[in_front_of_both_cameras]



R1 = np.eye(3)
t1 = np.zeros((3, 1))
R2 = best_P2[:, :3]
t2 = best_P2[:, 3]

np.save('results/extrinsics.npy', {'R1': R1, 't1': t1, 'R2': R2, 't2': t2})
# np.save('results/extrinsics', {'R1': R1, 't1': t1, 'R2': R2, 't2': t2})

# Plot the 3D points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(best_pts3d[:, 0], best_pts3d[:, 1], best_pts3d[:, 2])
plt.show()

