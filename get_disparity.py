import numpy as np
import cv2

def get_disparity(im1, im2, maxDisp, windowSize):
    """
    creates a disparity map from a pair of rectified images im1 and im2,
    given the maximum disparity maxDisp and the window size windowSize.
    """
    # grayscale
    if len(im1.shape) > 2:
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    if len(im2.shape) > 2:
        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    
    # disparity map
    dispM = np.zeros_like(im1, dtype=np.float32)
    
    # get image dimensions
    height, width = im1.shape

    # window size should be odd
    if windowSize % 2 == 0:
        windowSize += 1
    
    # for pixel access
    half_window = windowSize // 2

    # left image
    for y in range(half_window, height - half_window):
        for x in range(half_window, width - half_window):
            best_ssd = float('inf')
            best_d = 0

            for d in range(maxDisp):
                # stay within the image boundaries
                if x - d < half_window:
                    continue
                
                # window in the left and right images
                w1 = im1[y-half_window:y+half_window+1, x-half_window:x+half_window+1]
                w2 = im2[y-half_window:y+half_window+1, x-d-half_window:x-d+half_window+1]
                
                # SSD
                ssd = np.sum((w1.astype(np.float32) - w2.astype(np.float32))**2)
                
                # update the best SSD if the current SSD is lower
                if ssd < best_ssd:
                    best_ssd = ssd
                    best_d = d
            
            dispM[y, x] = best_d
    
    return dispM
