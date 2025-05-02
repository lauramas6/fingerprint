from src.dataset import load_image
from src.preprocessing import binarize_image, thin_image
from src.minutiae_extraction import extract_minutiae_CN, extract_minutiae_grayscale
from src.display import plot_minutiae
import os

# Choose one fingerprint image from SOCOFing
img_path = "data/SOCOFing/Real/101__M_Left_index_finger.BMP"  # adjust if needed
print(os.path.exists(img_path))  # Should print True
img = load_image(img_path)

# --- CN Method ---
binary = binarize_image(img)
thinned = thin_image(binary)
minutiae_cn = extract_minutiae_CN(thinned)

# --- Grayscale Method ---
minutiae_grayscale = extract_minutiae_grayscale(img)

# --- Visualize Both ---
print(f"CN Minutiae Count: {len(minutiae_cn)}")
print(f"Grayscale Minutiae Count: {len(minutiae_grayscale)}")

plot_minutiae(img, minutiae_cn, minutiae_grayscale)

