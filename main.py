import os
import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from src.dataset import load_image_paths, load_image, extract_label_from_filename
from src.preprocessing import binarize_image, thin_image
from src.minutiae_extraction import extract_minutiae_CN, extract_minutiae_grayscale
from src.features import extract_feature_vector


def authenticate(template_db, query_vector, subject_id, threshold=0.85):
    """
    Compare query fingerprint vector with enrolled template of claimed subject ID.
    Returns True (accept) or False (reject).
    """
    if subject_id not in template_db:
        return False

    enrolled_vector = template_db[subject_id]
    sim = cosine_similarity([enrolled_vector], [query_vector])[0][0]
    return sim >= threshold


def build_templates(image_paths, method='CN', grid_size=8):
    """
    Build enrollment templates using one fingerprint per subject.
    """
    enrolled = {}
    for path in image_paths:
        subject_id = extract_label_from_filename(path)
        if subject_id in enrolled:
            continue  # only use one image per subject for enrollment

        img = load_image(path)
        if method == 'CN':
            binary = binarize_image(img)
            thinned = thin_image(binary)
            minutiae = extract_minutiae_CN(thinned)
        else:
            minutiae = extract_minutiae_grayscale(img)

        if len(minutiae) == 0:
            continue

        fv = extract_feature_vector(minutiae, img.shape, grid_size)
        enrolled[subject_id] = fv

    return enrolled


def run_authentication(image_paths, template_db, method='CN', grid_size=8, threshold=0.85):
    """
    Attempt to authenticate all fingerprints against their claimed ID.
    """
    total = 0
    correct = 0

    for path in image_paths:
        subject_id = extract_label_from_filename(path)

        # Skip images used in enrollment
        if subject_id in template_db and path.endswith('__1.BMP'):
            continue

        img = load_image(path)
        if method == 'CN':
            binary = binarize_image(img)
            thinned = thin_image(binary)
            minutiae = extract_minutiae_CN(thinned)
        else:
            minutiae = extract_minutiae_grayscale(img)

        if len(minutiae) == 0:
            continue

        fv = extract_feature_vector(minutiae, img.shape, grid_size)
        is_auth = authenticate(template_db, fv, subject_id, threshold)

        total += 1
        if is_auth:
            correct += 1

    accuracy = correct / total if total > 0 else 0.0
    return accuracy


def main():
    dataset_path = 'data/SOCOFing/Real'
    image_paths = load_image_paths(dataset_path)

    # Enroll one image per subject (usually ending in __1.BMP)
    template_CN = build_templates(image_paths, method='CN')
    template_Gray = build_templates(image_paths, method='grayscale')

    # Authenticate all other images
    acc_CN = run_authentication(image_paths, template_CN, method='CN')
    acc_Gray = run_authentication(image_paths, template_Gray, method='grayscale')

    print(f"\n[Authentication Accuracy]")
    print(f"Crossing Number (CN):     {acc_CN * 100:.2f}%")
    print(f"Grayscale-Based Method:   {acc_Gray * 100:.2f}%")


if __name__ == '__main__':
    main()
