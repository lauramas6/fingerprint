import cv2
import numpy as np

def extract_minutiae_CN(thinned):
    """Extract ridge endings based on Crossing Number (CN) method."""
    minutiae = []
    rows, cols = thinned.shape

    # Iterate through each pixel (excluding the borders)
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            # Check if the pixel is a part of the ridge (foreground)
            if thinned[i, j] == 255:
                # Define the 3x3 neighborhood centered at (i, j)
                neighbors = thinned[i-1:i+2, j-1:j+2]

                # Count the number of transitions from black (0) to white (255) in the neighborhood
                crossing_number = 0
                for k in range(0, 3):
                    for l in range(0, 3):
                        # Transition occurs between consecutive pixels in the circular neighborhood
                        if neighbors[k, l] == 255:
                            # Compare to the next pixel in the circular sequence
                            next_k = (k + 1) % 3
                            next_l = (l + 1) % 3
                            if neighbors[next_k, next_l] == 0:
                                crossing_number += 1

                # If the crossing number is 1, it's a ridge ending (minutiae point)
                if crossing_number == 1:
                    minutiae.append((i, j))

    return np.array(minutiae)

def extract_minutiae_grayscale(image):
    """Extract ridge endings using Harris corner detection and image gradients."""
    # Convert the image to grayscale if it's not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply Gaussian smoothing to reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Compute image gradients using Sobel filters
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute gradient magnitude and direction
    grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    grad_dir = np.arctan2(sobel_y, sobel_x) * 180 / np.pi
    
    # Harris corner detection
    k = 0.04  # Harris corner constant
    Ix2 = sobel_x**2
    Iy2 = sobel_y**2
    Ixy = sobel_x * sobel_y
    
    # Compute the sums of products of gradients in a window
    S_Ix2 = cv2.GaussianBlur(Ix2, (5, 5), 0)
    S_Iy2 = cv2.GaussianBlur(Iy2, (5, 5), 0)
    S_Ixy = cv2.GaussianBlur(Ixy, (5, 5), 0)
    
    # Compute the Harris response
    det_M = S_Ix2 * S_Iy2 - S_Ixy**2
    trace_M = S_Ix2 + S_Iy2
    R = det_M - k * (trace_M**2)
    
    # Threshold the Harris response to detect corners (ridge endings)
    threshold = np.max(R) * 0.01  # You can adjust the threshold value
    corners = np.argwhere(R > threshold)
    
    # Filter out corner points that are not ridge endings
    minutiae = []
    for (y, x) in corners:
        if grad_mag[y, x] > 0:  # Ridge ending criteria: non-zero gradient magnitude
            minutiae.append((y, x))
    
    return np.array(minutiae)