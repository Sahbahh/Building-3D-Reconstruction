import numpy as np
import cv2

def epipolarCorrespondence(im1, im2, F, pts1):
    """
    Args:
        im1:    Image 1
        im2:    Image 2
        F:      Fundamental Matrix from im1 to im2
        pts1:   coordinates of points in image 1
    Returns:
        pts2:   coordinates of points in image 2
    """
    w = 4
    pts2 = np.zeros_like(pts1)
    search_range = 32  # limit the search range along the epipolar line

    for i, (x1, y1) in enumerate(pts1):
        x1, y1 = int(round(x1)), int(round(y1))
        p1 = np.array([x1, y1, 1])
        eline = F @ p1
        eline /= np.linalg.norm(eline[:2])

        x2_start = max(int(x1 - search_range), w)
        x2_end = min(int(x1 + search_range), im2.shape[1] - w)

        best_match_error = np.inf
        best_x2, best_y2 = -1, -1

        window_im1 = cv2.getRectSubPix(im1, (2*w+1, 2*w+1), (x1, y1))

        for x2i in range(x2_start, x2_end):
            y2i = int(-(eline[0] * x2i + eline[2]) / eline[1])
            if not (w <= y2i < im2.shape[0] - w):
                continue

            window_im2 = cv2.getRectSubPix(im2, (2*w+1, 2*w+1), (x2i, y2i))
            
            # Euclidean distance
            score = np.linalg.norm(window_im1.astype(np.float64) - window_im2.astype(np.float64))

            if score < best_match_error:
                best_match_error = score
                best_x2, best_y2 = x2i, y2i

        pts2[i] = [best_x2, best_y2]

    return pts2