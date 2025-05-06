import numpy as np
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity


def compare_minutiae(fv_query, fv_gallery):
    """Compare two feature vectors using cosine similarity."""
    if np.all(fv_query == 0) or np.all(fv_gallery == 0):
        print("Warning: one of the vectors is all zeros! Assigning similarity = 0.")
        return 0.0
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

def authenticate(query_fv, gallery_db, threshold, query_finger):
    best_similarity = -1
    best_match_id = None

    for entry_id, entry_finger, gallery_fv in gallery_db:
        # Cosine similarity
        similarity = np.dot(query_fv, gallery_fv) / (np.linalg.norm(query_fv) * np.linalg.norm(gallery_fv))
        
        if similarity > best_similarity:
            best_similarity = similarity
            best_match_id = entry_id

    is_match = best_similarity >= threshold
    return is_match, best_match_id, best_similarity
