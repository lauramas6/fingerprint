import cv2

def binarize_image(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def thin_image(binary):
    """Apply morphological thinning to a binary image."""
    skeleton = cv2.ximgproc.thinning(binary)
    return skeleton

def gradient_magnitude(img):
    """Compute gradient magnitude using Sobel operators."""
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = (grad_x**2 + grad_y**2)**0.5
    return magnitude
