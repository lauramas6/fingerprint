import numpy as np
from scipy.spatial.distance import cosine

def compare_minutiae(fv_query, fv_gallery):
    """Compare two feature vectors using cosine similarity."""
    similarity = 1 - cosine(fv_query, fv_gallery)  # Cosine similarity score (higher is better)
    return similarity

def compute_metrics(true_labels, predicted_labels):
    """Compute common performance metrics."""
    TP = np.sum((true_labels == 1) & (predicted_labels == 1))  # True positives
    TN = np.sum((true_labels == 0) & (predicted_labels == 0))  # True negatives
    FP = np.sum((true_labels == 0) & (predicted_labels == 1))  # False positives
    FN = np.sum((true_labels == 1) & (predicted_labels == 0))  # False negatives

    # Compute metrics with safe division
    accuracy = (TP + TN) / len(true_labels) if len(true_labels) > 0 else 0
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0  # True Positive Rate
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0  # False Positive Rate
    FNR = FN / (FN + TP) if (FN + TP) > 0 else 0  # False Negative Rate
    TNR = TN / (TN + FP) if (TN + FP) > 0 else 0  # True Negative Rate

    return accuracy, TPR, FPR, FNR, TNR, TP, TN, FP, FN

def authenticate(claimed_id, query_fv, gallery_db, method, threshold=0.8):
    """Authenticate a fingerprint by comparing with gallery using CN or Grayscale method."""
    """Authenticate a fingerprint by comparing with gallery."""
    best_match_id = None
    best_similarity = 0

    for subject_id, fv_gallery in gallery_db:
        similarity = compare_minutiae(query_fv, fv_gallery)
        if similarity > best_similarity:
            best_similarity = similarity
            best_match_id = subject_id

    # Determine if authentication succeeded
    is_match = (best_match_id == claimed_id and best_similarity >= threshold)

    return is_match, best_match_id, best_similarity