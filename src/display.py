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
