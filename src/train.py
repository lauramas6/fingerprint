from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

def train_classifier(X_train, Y_train, kernel='linear'):
    """Train an SVM classifier."""
    clf = SVC(kernel=kernel)
    clf.fit(X_train, Y_train)
    return clf

def evaluate_classifier(clf, X_test, Y_test):
    """Evaluate classifier and return accuracy and confusion matrix."""
    Y_pred = clf.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    cm = confusion_matrix(Y_test, Y_pred)
    return accuracy, cm
