import os
import cv2

print(f"Current working directory: {os.getcwd()}")

def load_image_paths(dataset_path, extension='.BMP'):
    """Load all image paths with specified extension."""
    image_paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith(extension)]
    print(f"Found image paths: {image_paths}")  # Add this line to see the paths
    return image_paths

def load_image(image_path):
    """Load an image from file and handle any read errors."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading image at {image_path}")
    else:
        print(f"Loaded image: {image_path}")
    return img

def extract_label_from_filename(filename):
    """Extract subject ID from filename, assuming the format is 'ID__gender_finger'."""
    base = os.path.basename(filename)
    parts = base.split('__')
    subject_id = int(parts[0])  # The first part before '__' is the subject ID
    return subject_id


