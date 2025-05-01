from src.dataset import load_image_paths, load_image, extract_label_from_filename
from src.preprocessing import binarize_image, thin_image
from src.minutiae import extract_minutiae_CN
from src.features import extract_feature_vector
from src.train import train_classifier, evaluate_classifier
from src.utils import plot_confusion_matrix

import numpy as np

def main():
    dataset_path = 'data/SOCOFing'  # <-- update this if needed
    image_paths = load_image_paths(dataset_path)

    X = []
    Y = []

    for path in image_paths:
        img = load_image(path)

        binary = binarize_image(img)
        thinned = thin_image(binary)
        minutiae_points = extract_minutiae_CN(thinned)

        if len(minutiae_points) == 0:
            continue  # skip images with no detected minutiae

        feature_vector = extract_feature_vector(minutiae_points, img.shape, grid_size=8)
        X.append(feature_vector)
        Y.append(extract_label_from_filename(path))

    X = np.array(X)
    Y = np.array(Y)

    # Split
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Train
    clf = train_classifier(X_train, Y_train)

    # Evaluate
    accuracy, cm = evaluate_classifier(clf, X_test, Y_test)
    print(f"Accuracy: {accuracy*100:.2f}%")

    # Plot confusion matrix
    plot_confusion_matrix(cm)

if __name__ == '__main__':
    main()
