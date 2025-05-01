import matplotlib.pyplot as plt
import seaborn as sns

def plot_minutiae(img, minutiae_points):
    """Overlay minutiae points on fingerprint image."""
    plt.imshow(img, cmap='gray')
    y, x = zip(*minutiae_points)
    plt.scatter(x, y, c='red', s=10)
    plt.title('Minutiae Points')
    plt.axis('off')
    plt.show()

def plot_confusion_matrix(cm):
    """Plot confusion matrix."""
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=False, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def plot_accuracy_comparison(acc_cn, acc_harris):
    methods = ['Crossing Number (CN)', 'Harris Corner']
    accuracies = [acc_cn * 100, acc_harris * 100]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(methods, accuracies, color=['skyblue', 'salmon'])
    plt.ylabel('Accuracy (%)')
    plt.title('Minutiae Extraction Method Comparison')
    plt.ylim(0, 100)

    for bar, acc in zip(bars, accuracies):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{acc:.2f}%', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()
