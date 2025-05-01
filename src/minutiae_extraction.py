import cv2
import numpy as np

def extract_minutiae_CN(thinned):
    """Extract ridge endings based on Crossing Number (CN) method."""
    minutiae = []
    rows, cols = thinned.shape
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            if thinned[i, j] == 255:
                neighbors = thinned[i-1:i+2, j-1:j+2]
                count = np.sum(neighbors == 255) - 1
                if count == 1:
                    minutiae.append((i, j))
    return np.array(minutiae)

def extract_minutiae_Harris(img, threshold=0.01):
    """Extract points of interest using Harris corner detection."""
    harris_corners = cv2.cornerHarris(np.float32(img), 2, 3, 0.04)
    coords = np.argwhere(harris_corners > threshold * harris_corners.max())
    coords = [(int(y), int(x)) for y, x in coords]  # (row, col) format
    return np.array(coords)
