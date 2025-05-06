import os
from src.dataset import load_image_paths, load_image
from src.preprocessing import binarize_image, thin_image
from src.minutiae_extraction import extract_minutiae_CN, extract_minutiae_grayscale
from src.features import extract_feature_vector
from src.authentication import authenticate
import matplotlib.pyplot as plt
from sklearn.metrics import auc, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import random

def plot_roc_curve(fpr, tpr):
    if len(fpr) >= 2 and len(tpr) >= 2:
        sorted_pairs = sorted(zip(fpr, tpr))
        fpr_sorted, tpr_sorted = zip(*sorted_pairs)
        roc_auc = auc(fpr_sorted, tpr_sorted)

        plt.figure()
        plt.plot(fpr_sorted, tpr_sorted, color='darkorange', lw=2,
                 label='ROC curve (area = %0.2f)' % roc_auc)
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

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def main():
    print("Select the minutiae extraction method:")
    print("1 - CN (Crossing Number)")
    print("2 - Grayscale-based")
    choice = input("Your choice: ").strip()
    while choice not in ['1', '2']:
        choice = input("Enter 1 or 2: ").strip()
    method = 'CN' if choice == '1' else 'Grayscale'

    dataset_path = os.path.join('data', 'SOCOFing', 'Real')
    gallery_db = []
    queries = []

    # === GALLERY: One user with Right_thumb_finger ===
    gallery_id = "101"
    finger_type = "Right_thumb_finger"
    gallery_path = os.path.join(dataset_path, f"{gallery_id}__M_{finger_type}.BMP")
    if os.path.exists(gallery_path):
        img = load_image(gallery_path)
        thinned = thin_image(binarize_image(img))
        minutiae = extract_minutiae_CN(thinned) if method == 'CN' else extract_minutiae_grayscale(thinned)
        if len(minutiae) > 0:
            fv = extract_feature_vector(minutiae, img.shape)
            gallery_db.append((gallery_id, finger_type, fv))
            print(f"✅ Added gallery: {gallery_id}, {finger_type}")
        else:
            print(f"⚠️ No minutiae in gallery image: {gallery_path}")
    else:
        print(f"❌ Missing gallery image: {gallery_path}")
        return

    # === QUERIES: Mix of 100 queries, 10 true matches ===
    all_paths = load_image_paths(dataset_path)
    random.shuffle(all_paths)
    total_queries = 100
    added = 0
    true_match_paths = [
        os.path.join(dataset_path, f"{gallery_id}__M_{finger_type}.BMP")
        for _ in range(10)
    ]

    for path in true_match_paths:
        if os.path.exists(path):
            queries.append({"path": path})
            added += 1

    i = 0
    while added < total_queries and i < len(all_paths):
        path = all_paths[i]
        i += 1
        filename = os.path.basename(path)
        id_part = filename.split("__")[0]
        finger_part = filename.split("__")[1].split('.')[0].split("_", 1)[1]
        if id_part == gallery_id and finger_part == finger_type:
            continue  # skip duplicates of true match
        queries.append({"path": path})
        added += 1

    # === AUTHENTICATION LOOP ===
    TP = TN = FP = FN = 0
    threshold = 0.7
    fpr_values = []
    tpr_values = []
    all_scores = []

    for i, query in enumerate(queries):
        print(f"\n🔍 Query {i+1}: {query['path']}")
        filename = os.path.basename(query["path"])
        try:
            query_finger = filename.split('__')[1].split('.')[0].split('_', 1)[1]
        except IndexError:
            continue

        print(f"🧬 Comparing query finger {query_finger} to gallery finger {finger_type}")

        query_img = load_image(query["path"])
        thinned_query = thin_image(binarize_image(query_img))
        minutiae_query = extract_minutiae_CN(thinned_query) if method == 'CN' else extract_minutiae_grayscale(thinned_query)
        print(f"🧬 Minutiae count: {len(minutiae_query)}")
        if len(minutiae_query) == 0:
            continue
        query_fv = extract_feature_vector(minutiae_query, query_img.shape)

        # === 1-to-N match (no claimed ID) ===
        is_match, best_match_id, best_similarity = authenticate(
            query_fv, gallery_db, threshold, query_finger
        )

        actual_id = filename.split('__')[0]
        is_genuine = (actual_id == gallery_id)
        passed = best_similarity >= threshold
        all_scores.append((is_genuine, best_similarity))

        print(f"🆔 Actual: {actual_id}, Match ID: {best_match_id}")
        print(f"📈 Score: {best_similarity:.4f} → {'MATCH' if is_match else 'NO MATCH'}")

        if is_genuine and passed:
            TP += 1
        elif not is_genuine and passed:
            FP += 1
        elif is_genuine and not passed:
            FN += 1
        elif not is_genuine and not passed:
            TN += 1

        fpr_values.append(FP / (FP + TN) if (FP + TN) > 0 else 0)
        tpr_values.append(TP / (TP + FN) if (TP + FN) > 0 else 0)

    # === METRICS ===
    total = TP + TN + FP + FN
    accuracy = (TP + TN) / total if total > 0 else 0
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
    TNR = TN / (TN + FP) if (TN + FP) > 0 else 0
    FNR = FN / (FN + TP) if (FN + TP) > 0 else 0

    print("\n📊 Metrics")
    print(f"✅ Accuracy: {accuracy:.2f}")
    print(f"TPR: {TPR:.2f} | FPR: {FPR:.2f} | TNR: {TNR:.2f} | FNR: {FNR:.2f}")
    print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")

    # === PLOTS ===
    plot_roc_curve(fpr_values, tpr_values)

    from sklearn.metrics import ConfusionMatrixDisplay
    cm = np.array([[TN, FP], [FN, TP]])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Impostor", "Genuine"])
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    main()
