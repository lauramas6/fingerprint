from src.dataset import load_image_paths, load_image, extract_label_from_filename
from src.preprocessing import binarize_image, thin_image
from src.minutiae_extraction import extract_minutiae_CN
from src.features import extract_feature_vector
from src.train import train_classifier, evaluate_classifier
from src.display import plot_confusion_matrix

import numpy as np

def main():
    dataset_path = 'data/SOCOFing/Real'  # <-- update this if needed
    image_paths = load_image_paths(dataset_path)

    X_CN, X_Grayscale, Y = [], [], []

    for path in image_paths:
        img = load_image(path)

        # CN method
        binary = binarize_image(img)
        thinned = thin_image(binary)
        minutiae_CN = extract_minutiae_CN(thinned)
        if len(minutiae_CN) == 0:
            continue  # skip images with no detected minutiae
        fv_CN = extract_feature_vector(minutiae_CN, img.shape, grid_size=8)

        # Grayscale method
        minutiae_grayscale = extract_minutiae_grayscale(img)
        if len(minutiae_grayscale) == 0:
            continue
        fv_grayscale = extract_feature_vector(minutiae_grayscale, img.shape, grid_size=8)

        # Append both
        X_CN.append(fv_CN)
        X_Grayscale.append(fv_grayscale)
        Y.append(extract_label_from_filename(path))

    # Convert to arrays
    X_CN = np.array(X_CN)
    X_Grayscale = np.array(X_Grayscale)
    Y = np.array(Y)

    # Split once, use same split for both classifiers
    XCN_train, XCN_test, Y_train, Y_test = train_test_split(X_CN, Y, test_size=0.2, random_state=42)
    XH_train, XH_test, _, _ = train_test_split(X_Grayscale, Y, test_size=0.2, random_state=42)

    # Train + Evaluate CN
    clf_CN = train_classifier(XCN_train, Y_train)
    acc_CN, cm_CN = evaluate_classifier(clf_CN, XCN_test, Y_test)

    # Train + Evaluate Grayscale
    clf_H = train_classifier(XH_train, Y_train)
    acc_H, cm_H = evaluate_classifier(clf_H, XH_test, Y_test)

    print("\n=== Results: Crossing Number (CN) ===")
    print(f"Accuracy: {acc_CN * 100:.2f}%")
    plot_confusion_matrix(cm_CN)

    print("\n=== Results: Grayscale  ===")
    print(f"Accuracy: {acc_H * 100:.2f}%")
    plot_confusion_matrix(cm_H)

if __name__ == '__main__':
    main()