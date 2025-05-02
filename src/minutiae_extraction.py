import cv2
import numpy as np
from scipy.ndimage import maximum_filter

def extract_minutiae_CN(thinned):
    """Extract ridge endings based on the correct Crossing Number (CN) method."""
    minutiae = []
    rows, cols = thinned.shape

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if thinned[i, j] == 255:
                # 8-connected neighbors in circular order (P1 to P8)
                P = [
                    thinned[i - 1, j],     # P1
                    thinned[i - 1, j + 1], # P2
                    thinned[i, j + 1],     # P3
                    thinned[i + 1, j + 1], # P4
                    thinned[i + 1, j],     # P5
                    thinned[i + 1, j - 1], # P6
                    thinned[i, j - 1],     # P7
                    thinned[i - 1, j - 1]  # P8
                ]

                # Convert to binary (0/1)
                P_bin = [1 if x == 255 else 0 for x in P]

                # Compute Crossing Number
                CN = 0
                for k in range(8):
                    CN += abs(P_bin[k] - P_bin[(k + 1) % 8])
                CN = CN // 2

                if CN == 1:
                    minutiae.append((i, j))

    return np.array(minutiae)


def non_max_suppression(R, window_size=3):
    """Apply non-maximum suppression to the Harris response."""
    max_filtered = maximum_filter(R, size=window_size)
    return (R == max_filtered) & (R > 0)


def extract_minutiae_grayscale(image):
    """Extract ridge endings using Harris corner detection and image gradients."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Sobel gradients
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)

    # Harris corner response
    k = 0.04
    Ix2 = sobel_x**2
    Iy2 = sobel_y**2
    Ixy = sobel_x * sobel_y

    S_Ix2 = cv2.GaussianBlur(Ix2, (5, 5), 0)
    S_Iy2 = cv2.GaussianBlur(Iy2, (5, 5), 0)
    S_Ixy = cv2.GaussianBlur(Ixy, (5, 5), 0)

    det_M = S_Ix2 * S_Iy2 - S_Ixy**2
    trace_M = S_Ix2 + S_Iy2
    R = det_M - k * (trace_M**2)

    # Threshold and non-maximum suppression
    R_thresh = R > (0.01 * np.max(R))
    nms_mask = non_max_suppression(R, window_size=3)
    final_mask = R_thresh & nms_mask

    corners = np.argwhere(final_mask)

    # Filter: only keep corners on ridges (high gradient magnitude)
    grad_threshold = np.percentile(grad_mag, 75)  # top 25% of gradient
    minutiae = []
    for (y, x) in corners:
        if grad_mag[y, x] >= grad_threshold:
            minutiae.append((y, x))

    return np.array(minutiae)
