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

def authenticate(claimed_id, query_fv, gallery_db, method='CN', threshold=0.8):
    """Authenticate a fingerprint by comparing with gallery using CN or Grayscale method."""
    best_match_id = None
    best_similarity = 0

    # Store the true labels and predicted labels for metric calculation
    true_labels = []
    predicted_labels = []

    # Compare against gallery database
    for subject_id, fv_gallery in gallery_db:
        if method in ['CN', 'Grayscale']:
            similarity = compare_minutiae(query_fv, fv_gallery)
        else:
            raise ValueError("Invalid method. Choose 'CN' or 'Grayscale'.")

        # Store the true and predicted labels (1 for match, 0 for non-match)
        true_labels.append(1 if subject_id == claimed_id else 0)
        predicted_labels.append(1 if similarity >= threshold else 0)

        # Track the best match
        if similarity > best_similarity:
            best_similarity = similarity
            best_match_id = subject_id

    # Compute metrics based on all gallery comparisons
    accuracy, TPR, FPR, FNR, TNR, TP, TN, FP, FN = compute_metrics(np.array(true_labels), np.array(predicted_labels))

    return accuracy, TPR, FPR, FNR, TNR, TP, TN, FP, FN, best_match_id, best_similarity
