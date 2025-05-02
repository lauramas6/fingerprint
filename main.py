from src.dataset import load_image_paths, load_image
from src.preprocessing import binarize_image, thin_image
from src.minutiae_extraction import extract_minutiae_CN, extract_minutiae_grayscale
from src.features import extract_feature_vector
from src.authentication import authenticate
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np

def plot_roc_curve(fpr, tpr):
    """Plot the ROC curve."""
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.show()

def main():
    dataset_path = 'data/SOCOFing/Real'

    # Step 1: Load gallery images
    gallery_paths = load_image_paths(dataset_path)[:100]  # e.g., 100 for enrollment
    gallery_db = []

    for path in gallery_paths:
        img = load_image(path)
        thinned = thin_image(binarize_image(img))
        minutiae = extract_minutiae_CN(thinned)
        if len(minutiae) == 0:
            continue
        fv = extract_feature_vector(minutiae, img.shape)
        subject_id = path.split('/')[-1].split('__')[0]
        gallery_db.append((subject_id, fv))

    # Step 2: Authenticate a query fingerprint
    query_path = 'data/SOCOFing/Real/001__M_Left_index.BMP'
    query_img = load_image(query_path)
    thinned_query = thin_image(binarize_image(query_img))
    minutiae_query = extract_minutiae_CN(thinned_query)
    query_fv = extract_feature_vector(minutiae_query, query_img.shape)

    claimed_id = '001'  # Claiming to be subject 001
    method = 'CN'  # Choose 'CN' or 'Grayscale' method for authentication

    # Perform authentication
    accuracy, TPR, FPR, FNR, TNR, TP, TN, FP, FN, best_match_id, best_similarity = authenticate(
        claimed_id, query_fv, gallery_db, method=method
    )

    # Print metrics
    print(f"Authentication result: {'MATCH' if best_match_id == claimed_id else 'NO MATCH'} (Score: {best_similarity:.2f})")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"TPR: {TPR:.2f} | FPR: {FPR:.2f} | FNR: {FNR:.2f} | TNR: {TNR:.2f}")

    # Plot ROC curve
    fpr_values = [FPR]
    tpr_values = [TPR]
    plot_roc_curve(fpr_values, tpr_values)

if __name__ == '__main__':
    main()
