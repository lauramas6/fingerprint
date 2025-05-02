import os
import cv2

print(f"Current working directory: {os.getcwd()}")

def load_image_paths(dataset_path, extension='.BMP'):
    """Load all image paths with specified extension."""
    image_paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith(extension)]
    return image_paths

def load_image(image_path):
    """Load an image from file and handle any read errors."""
    img = cv2.imread(image_path)
    return img

def extract_label_from_filename(filename):
    """Extract subject ID from filename, assuming the format is 'ID__gender_finger'."""
    base = os.path.basename(filename)
    parts = base.split('__')
    subject_id = int(parts[0])  # The first part before '__' is the subject ID
    return subject_id


