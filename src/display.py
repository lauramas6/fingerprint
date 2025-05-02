import matplotlib.pyplot as plt
import seaborn as sns

def plot_minutiae(img, minutiae_cn, minutiae_grayscale, title_cn="Minutiae (Crossing Number)", title_grayscale="Minutiae (Grayscale Method)"):
     """Overlay minutiae points on fingerprint image and display side by side."""
    plt.figure(figsize=(12, 6))

    # Plot Crossing Number method
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    y_cn, x_cn = zip(*minutiae_cn)
    plt.scatter(x_cn, y_cn, c='red', s=10)
    plt.title(title_cn)
    plt.axis('off')

    # Plot Grayscale-based method (Harris corner and gradients)
    plt.subplot(1, 2, 2)
    plt.imshow(img, cmap='gray')
    y_grayscale, x_grayscale = zip(*minutiae_grayscale)
    plt.scatter(x_grayscale, y_grayscale, c='blue', s=10)
    plt.title(title_grayscale)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(cm):
    """Plot confusion matrix."""
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=False, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def plot_accuracy_comparison(acc_cn, acc_grayscale):
    methods = ['Crossing Number (CN)', 'Grayscale']
    accuracies = [acc_cn * 100, acc_grayscale * 100]

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
