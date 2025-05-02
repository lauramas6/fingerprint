import os
import cv2

def load_image_paths(dataset_path, extension='.BMP'):
    """Load all image paths with specified extension."""
    return [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith(extension)]

def load_image(filepath):
    """Load a fingerprint image as grayscale."""
    return cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

def extract_label_from_filename(filename):
    """Extract subject ID from filename, assuming the format is 'ID__gender_finger'."""
    base = os.path.basename(filename)
    parts = base.split('__')
    subject_id = int(parts[0])  # The first part before '__' is the subject ID
    return subject_id

