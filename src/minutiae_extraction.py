import cv2
import numpy as np

def extract_minutiae_CN(thinned):
    """Extract minutiae (ridge endings and bifurcations) using Crossing Number method."""
    minutiae = []
    rows, cols = thinned.shape
    
    # Define 8-neighborhood offsets in circular order
    neighbors = [(-1,-1), (-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1), (-1,-1)]
    
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            if thinned[i, j] == 255:  # Ridge pixel
                # Get pixel values in 8-neighborhood
                pixel_values = []
                for di, dj in neighbors[:8]:
                    pixel_values.append(thinned[i+di, j+dj])
                
                # Calculate crossing number
                cn = 0
                for k in range(8):
                    # Transition from white to black
                    if pixel_values[k] == 255 and pixel_values[k+1] == 0:
                        cn += 1
                
                # Ridge ending (CN=1) or bifurcation (CN=3)
                if cn == 1 or cn == 3:
                    minutiae.append((i, j, cn))  # Store coordinates and type
    
    return np.array(minutiae)

def extract_minutiae_grayscale(image):
    """Extract minutiae using image gradients and ridge characteristics."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Enhance ridges
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Compute gradients
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute gradient magnitude and orientation
    grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    grad_dir = np.arctan2(sobel_y, sobel_x) * 180 / np.pi
    
    # Normalize magnitude
    grad_mag = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX)
    
    # Threshold to find potential minutiae
    _, binary = cv2.threshold(grad_mag, 0.7 * np.max(grad_mag), 255, cv2.THRESH_BINARY)
    binary = binary.astype(np.uint8)
    
    # Find contours (potential minutiae)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    minutiae = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 10:  # Filter small areas
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
                # Determine if it's a ridge ending or bifurcation based on local characteristics
                # (This would need more sophisticated analysis in a real implementation)
                minutiae.append((cY, cX, 1))  # Default to ridge ending
    
    return np.array(minutiae)