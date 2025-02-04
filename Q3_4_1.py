import numpy as np
import cv2
import os

# Function to read camera parameters from the file
# source: https://stackoverflow.com/questions/6213336/reading-file-string-into-an-array-in-a-pythonic-way
def read_camera_parameters(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    camera_parameters = []
    for line in lines:
        parts = line.split()
        if parts[0].endswith('.png'):
            K = np.array(parts[1:10], dtype=float).reshape((3, 3))
            R = np.array(parts[10:19], dtype=float).reshape((3, 3))
            t = np.array(parts[19:22], dtype=float).reshape((3, 1))
            P = K @ np.hstack((R, t))
            camera_parameters.append((P, parts[0]))
    return camera_parameters

# project 3D points into 2D using the camera matrix
def project_points(P, X):
    # Ensure X is a 2D array with shape (3, N)
    X = np.atleast_2d(X)
    if X.shape[0] != 3:
        raise ValueError(f"X should have 3 rows, but has {X.shape[0]}.")
    X_homog = np.vstack([X, np.ones((1, X.shape[1]))])
    x_homog = P @ X_homog
    x = x_homog[:2] / x_homog[2]
    return x.astype(int).squeeze()



## ------------------
## 3.4.2 functions:
# Function to compute a 3D coordinate that projects to pixel q in image I0 and has a depth d
def Get3dCoord(q, P, d):
    x, y = q
    P1_12 = P[:3, :3]
    P1_14 = P[:, 3]
    right_side = np.array([[x], [y], [d]]) - P1_14[:, np.newaxis]
    X = np.linalg.inv(P1_12).dot(right_side)
    return X.flatten()


# Function to compute Normalized Cross Correlation
def NormalizedCrossCorrelation(C0, C1):
    C0_mean = np.mean(C0, axis=1)
    C0_normalized = C0 - C0_mean[:, np.newaxis]
    C0_normalized /= np.linalg.norm(C0_normalized, axis=0)
    
    C1_mean = np.mean(C1, axis=1)
    C1_normalized = C1 - C1_mean[:, np.newaxis]
    C1_normalized /= np.linalg.norm(C1_normalized, axis=0)
    
    return np.sum(C0_normalized * C1_normalized)

# Function to compute consistency between two images
def ComputeConsistency(I0, I1, P0, P1, X):
    x0 = project_points(P0, X.reshape(3, 1))
    x1 = project_points(P1, X.reshape(3, 1))
    
    # Ensure the projected points are within the image bounds
    if (0 <= x0[0] < I0.shape[1]) and (0 <= x0[1] < I0.shape[0]) and \
       (0 <= x1[0] < I1.shape[1]) and (0 <= x1[1] < I1.shape[0]):
        C0 = I0[int(x0[1]), int(x0[0])]
        C1 = I1[int(x1[1]), int(x1[0])]
        return NormalizedCrossCorrelation(C0, C1)
    else:
        return 0  # Return a low score if the projection is out of bounds



def is_background(pixel, threshold=10):
    # Assuming a grayscale image, background is close to black
    return np.all(pixel < threshold)

def compute_depth_map(I0, camera_matrices, min_depth, max_depth, depth_step, window_size):
    # Initialize the depth map with the maximum depth (or a large value that indicates no depth)
    depthM = np.full(I0.shape[:2], max_depth, dtype=np.float32)
    # Iterate through each pixel in I0
    for i in range(window_size // 2, I0.shape[0] - window_size // 2):
        for j in range(window_size // 2, I0.shape[1] - window_size // 2):
            if is_background(I0[i, j]):
                depthM[i, j] = 0  # Assign zero to background pixels
                continue

            best_score = -np.inf
            best_depth = max_depth  # Initialize with the maximum depth
            # Test depth hypotheses
            for d in np.arange(min_depth, max_depth, depth_step):
                X = Get3dCoord((j, i), camera_matrices[0], d)
                scores = [ComputeConsistency(I0, images[k], camera_matrices[0], camera_matrices[k], X)
                          for k in range(1, len(camera_matrices))]
                average_score = np.mean(scores)
                if average_score > best_score:
                    best_score = average_score
                    best_depth = d
            # Set the depth for the current pixel
            depthM[i, j] = best_depth

    # Invert the depth values: further points will have lower values, closer points higher values
    depthM = max_depth - depthM
    # Set background pixels to the maximum value (255 after normalization)
    depthM[depthM == max_depth] = 0

    # Normalize the depth map
    depthM = cv2.normalize(depthM, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return depthM




# Function to visualize the depth map
def visualize_depth_map(depth_map):
    normalized_depth_map = cv2.normalize(depth_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    cv2.imshow('Depth Map', normalized_depth_map.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()



#----------------------
#  bounding box corners
min_xyz = np.array([-0.023121, -0.038009, -0.091940])
max_xyz = np.array([0.078626, 0.121636, -0.017395])
corners = np.array([[min_xyz[0], min_xyz[1], min_xyz[2]],
                    [max_xyz[0], min_xyz[1], min_xyz[2]],
                    [min_xyz[0], max_xyz[1], min_xyz[2]],
                    [min_xyz[0], min_xyz[1], max_xyz[2]],
                    [max_xyz[0], max_xyz[1], min_xyz[2]],
                    [min_xyz[0], max_xyz[1], max_xyz[2]],
                    [max_xyz[0], min_xyz[1], max_xyz[2]],
                    [max_xyz[0], max_xyz[1], max_xyz[2]]]).T



script_directory = os.path.dirname(os.path.abspath(__file__))  # directory where the script is located
data_directory = os.path.join(script_directory, '..', 'data')
results_directory = os.path.join(script_directory, '..', 'results')

# load camera parameters
camera_params_file = os.path.join(data_directory, 'templeR_par.txt')
camera_parameters = read_camera_parameters(camera_params_file)

image_names = [
    'templeR0013.png', 
    'templeR0014.png', 
    'templeR0016.png', 
    'templeR0043.png', 
    'templeR0045.png'
]


# project corners and visualize
for P, image_name in camera_parameters:
    if image_name in image_names:

        # read image
        image_path = os.path.join(data_directory, image_name)
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image not found at path: {image_path}")

        # project the corners onto the image
        projected_corners = project_points(P, corners)
        
        # draw the corners on the image
        for x, y in projected_corners.T:
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

        # save n show
        output_image_path = os.path.join(results_directory, f'projected_{image_name}')
        cv2.imwrite(output_image_path, image)

        cv2.imshow(f'Projected Corners - {image_name}', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


## 3.4.2

images = [cv2.imread(os.path.join(data_directory, name), cv2.IMREAD_GRAYSCALE) for name in image_names]
I0 = images[0]


# Hyperparameters for depth estimation
min_depth = -0.02  
max_depth = 0.07   
depth_range = max_depth - min_depth
depth_step = depth_range / 150  # calculate step size for 150 steps
S = 5  # window size for computing consistency

# Compute the depth map for the reference image
depthM = compute_depth_map(I0, [param[0] for param in camera_parameters], min_depth, max_depth, depth_step, S)
               

# visualize n save the depth map
visualize_depth_map(depthM)
depth_map_path = os.path.join(results_directory, 'depth_map.png')
cv2.imwrite(depth_map_path, depthM.astype(np.uint8))