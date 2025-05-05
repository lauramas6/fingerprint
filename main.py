import os
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
    if len(fpr) >= 2 and len(tpr) >= 2:
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
    else:
        print("Not enough data to plot ROC curve.")

def main():
    # Prompt the user to select the minutiae extraction method (CN or Grayscale)
    print("Select the minutiae extraction method:")
    print("1 - CN (Crossing Number)")
    print("2 - Grayscale-based")
    choice = input("Your choice: ").strip()

    while choice not in ['1', '2']:
        print("Invalid choice. Please enter 1 for CN or 2 for Grayscale.")
        choice = input("Enter your choice (1 or 2): ").strip()

    method = 'CN' if choice == '1' else 'Grayscale'

    # Use os.path.join to make paths OS-independent
    dataset_path = os.path.join('data', 'SOCOFing', 'Real')

    # Load gallery images and ensure proper path handling
    '''gallery_paths = [
        p for p in load_image_paths(dataset_path)
        if os.path.basename(p).startswith("101__")
    ]'''
    
    # Load all images from the dataset (600 individuals, 10 images each)
    image_paths = load_image_paths(dataset_path)
    gallery_db = []

    '''
    for path in gallery_paths:
        print(f"Processing gallery image: {path}")  # 🟡 Debug print
        img = load_image(path)
        thinned = thin_image(binarize_image(img))

        # Choose the appropriate minutiae extraction method
        if method == 'CN':
            minutiae = extract_minutiae_CN(thinned)
        else:
            minutiae = extract_minutiae_grayscale(thinned)

        if len(minutiae) == 0:
            print(f"No minutiae extracted from gallery image: {path}")  # 🟡 Debug warning
            continue
        fv = extract_feature_vector(minutiae, img.shape)
        
        # Get subject ID from filename (handling OS-specific path separators)
        subject_id = os.path.basename(path).split('__')[0]
        gallery_db.append((subject_id, fv))'''

    for i in range(1, 601):  # 600 individuals, IDs from 1 to 600
        subject_images = [p for p in image_paths if os.path.basename(p).startswith(f"{i}__")]
        if not subject_images:
            print(f"No images found for subject ID {i}")
            continue

        # Use the first image of the subject for gallery (can modify based on your choice)
        path = subject_images[0]
        print(f"Processing gallery image for subject {i}: {path}")
        img = load_image(path)
        thinned = thin_image(binarize_image(img))

        # Choose the appropriate minutiae extraction method
        if method == 'CN':
            minutiae = extract_minutiae_CN(thinned)
        else:
            minutiae = extract_minutiae_grayscale(thinned)

        if len(minutiae) == 0:
            print(f"No minutiae extracted from gallery image: {path}")
            continue

        fv = extract_feature_vector(minutiae, img.shape)
        gallery_db.append((str(i), fv))  # Store the feature vector with subject ID


    # Define queries with paths
    '''queries = [
        {
            "path": os.path.join('data', 'SOCOFing', 'Real', '101__M_Left_index_finger.BMP'),
            "claimed_id": '101'
        },
        {
            "path": os.path.join('data', 'SOCOFing', 'Real', '64__M_Left_index_finger.BMP'),
            "claimed_id": '64'
        }
    ]'''

    # Define queries, assume you have 10 queries for testing
    queries = []
    # Genuine attempts: real image, correct claimed ID
    for i in range(1, 11):  # 10 subjects
        matches = [p for p in image_paths if os.path.basename(p).startswith(f"{i}__")]
        if matches:
            queries.append({
                "path": matches[0],
                "claimed_id": str(i)  # correct claim
            })

    # Impostor attempts: real image, incorrect claimed ID
    for i in range(1, 6):  # pick 5 impostor attempts
        real_id = i
        impostor_id = real_id + 1 if real_id < 600 else real_id - 1
        matches = [p for p in image_paths if os.path.basename(p).startswith(f"{real_id}__")]
        if matches:
            queries.append({
                "path": matches[0],
                "claimed_id": str(impostor_id)  # fake claim
            })

    TP = TN = FP = FN = 0
    threshold = 0.8
    fpr_values = []
    tpr_values = []

    for i, query in enumerate(queries):
        print(f"\nProcessing query {i+1}: {query['path']}")
        query_img = load_image(query["path"])
        thinned_query = thin_image(binarize_image(query_img))

        if method == 'CN':
            minutiae_query = extract_minutiae_CN(thinned_query)
            print(f"Minutiae extracted using CN method: {len(minutiae_query)} minutiae found.")
        else:
            minutiae_query = extract_minutiae_grayscale(thinned_query)
            print(f"Minutiae extracted using Grayscale method: {len(minutiae_query)} minutiae found.")

        if len(minutiae_query) == 0:
            print(f"Query {i+1}: No minutiae extracted, skipping.")
            continue

        query_fv = extract_feature_vector(minutiae_query, query_img.shape)

        is_match, best_match_id, best_similarity = authenticate(
            query["claimed_id"], query_fv, gallery_db, method=method, threshold=threshold
        )
        actual_id = os.path.basename(query["path"]).split("__")[0]
        claimed_id = query["claimed_id"]
        
        is_genuine = (actual_id == claimed_id)
        passed_threshold = best_similarity >= threshold

        # Update confusion matrix
        if is_genuine and passed_threshold:
            TP += 1
        elif not is_genuine and passed_threshold:
            FP += 1
        elif is_genuine and not passed_threshold:
            FN += 1
        elif not is_genuine and not passed_threshold:
            TN += 1

        print(f"Query {i+1}: {query['path']}")
        print(f"Authentication result: {'MATCH' if is_match else 'NO MATCH'} (Score: {best_similarity:.2f})")

        # Save ROC data
        fpr_values.append(FP / (FP + TN) if (FP + TN) > 0 else 0)
        tpr_values.append(TP / (TP + FN) if (TP + FN) > 0 else 0)

    # Final performance metrics
    total = TP + TN + FP + FN
    accuracy = (TP + TN) / total if total > 0 else 0
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
    FNR = FN / (FN + TP) if (FN + TP) > 0 else 0
    TNR = TN / (TN + FP) if (TN + FP) > 0 else 0

    print("\n--- Overall Metrics ---")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"TPR: {TPR:.2f} | FPR: {FPR:.2f} | FNR: {FNR:.2f} | TNR: {TNR:.2f}")

    # Plot ROC
    plot_roc_curve(fpr_values, tpr_values)

if __name__ == '__main__':
    main()
