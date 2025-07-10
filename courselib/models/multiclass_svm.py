import numpy as np
from .svm import BinaryKernelSVM

class KernelMulticlassOvR:
    """
    One-vs-Rest (OvR) multiclass classifier using binary SVMs with kernel support.

    For a classification problem with K classes, trains K binary classifiers.
    Each classifier distinguishes one class (positive) from all others (negative).

    At prediction time, all K classifiers produce a decision score.
    The class whose classifier yields the highest score is chosen.
    This approach is also known as "One-vs-All" (OvA).
    """

    def __init__(self, kernel_type='rbf', C=1.0, **kwargs):
        self.kernel_type = kernel_type
        self.C = C
        kwargs.pop('kernel', None)
        self.kwargs = kwargs
        self.models = []
        self.labels = None

    def fit(self, X, Y):
        """
        Train one binary classifier for each class vs the rest.
        """
        class_names = np.unique(Y)
        for cls in class_names:
            Y_encoded = np.where(Y == cls, 1, -1)
            svm = BinaryKernelSVM(
                C=self.C,
                kernel=self.kernel_type,
                **self.kwargs
            )
            svm.fit(X, Y_encoded)
            self.models.append(svm)
        self.labels = class_names

    def decision_function(self, X):
        """
        Compute decision scores for each class using corresponding binary classifiers.
        Returns a matrix of shape (n_samples, n_classes).
        """
        scores = [model.decision_function(X) for model in self.models]
        return np.vstack(scores).T
    
    def __call__(self, X):
        """
        Predict the class with the highest score.
        """
        scores = self.decision_function(X)
        indices = np.argmax(scores, axis=1)
        return self.labels[indices]
    
    def evaluate_models(self, X, Y):
        """
        Evaluate and print accuracy of each individual binary classifier.
        """
        from courselib.utils.metrics import binary_accuracy
        print("📊 Accuracy of each binary model (One-vs-Rest):")
        for cls, model in zip(self.labels, self.models):
            Y_encoded = np.where(Y == cls, 1, -1)
            preds = np.sign(model(X))
            acc = binary_accuracy(preds, Y_encoded)
            print(f"  - Class '{cls}': {acc:.4f}")

    def evaluate_accuracy(self, X, Y):
        """
        Evaluate overall accuracy of the OvR multiclass classifier.
        """
        Y_pred = self(X)
        acc = np.mean(Y_pred == Y) * 100
        print(f"🎯 Overall accuracy (OvR): {acc:.4f} %")

class KernelMulticlassOvO:
    """
    One-vs-One (OvO) multiclass classifier using binary SVMs with kernel support.

    For each unique pair of classes, a binary SVM is trained to distinguish between them.
    During inference, all binary classifiers vote, and the class with the most votes wins.
    """
    def __init__(self, kernel_type='rbf', C=1.0, **kwargs):
        self.kernel_type = kernel_type
        self.C = C
        kwargs.pop('kernel', None)
        self.kwargs = kwargs
        self.models = []  # (class_a, class_b, model)
        self.label_pairs = []  # (class_a, class_b)
        self.labels = None

    def fit(self, X, Y):
        """
        Train binary classifiers for each pair of classes.
        """
        self.labels = np.unique(Y)
        for i, class_a in enumerate(self.labels):
            for class_b in self.labels[i+1:]:
                idx = np.where((Y == class_a) | (Y == class_b))
                X_pair = X[idx]
                Y_pair = Y[idx]
                Y_binary = np.where(Y_pair == class_a, 1, -1)

                svm = BinaryKernelSVM(
                    C=self.C,
                    kernel=self.kernel_type,
                    **self.kwargs
                )
                svm.fit(X_pair, Y_binary)
                self.models.append((class_a, class_b, svm))
                self.label_pairs.append((class_a, class_b))

    def __call__(self, X):
        """
        Predict the class using majority voting from all pairwise classifiers.
        """
        votes = {label: np.zeros(len(X)) for label in self.labels}

        for class_a, class_b, model in self.models:
            preds = np.sign(model(X))
            votes[class_a] += (preds == 1)
            votes[class_b] += (preds == -1)

        all_votes = np.stack([votes[label] for label in self.labels], axis=1)
        predicted_indices = np.argmax(all_votes, axis=1)
        return self.labels[predicted_indices]

    def evaluate_models(self, X, Y):
        """
        Evaluate each OvO binary classifier and print its accuracy.
        """
        from courselib.utils.metrics import binary_accuracy
        print("📊 Accuracy of each OvO binary classifier:")
        for (class_a, class_b, model) in self.models:
            idx = np.where((Y == class_a) | (Y == class_b))
            X_pair = X[idx]
            Y_pair = Y[idx]
            Y_bin = np.where(Y_pair == class_a, 1, -1)
            preds = np.sign(model(X_pair))
            acc = binary_accuracy(preds, Y_bin)
            print(f"  - Classifier '{class_a}' vs '{class_b}': {acc:.4f}")

    def evaluate_accuracy(self, X, Y):
        """
        Evaluate overall accuracy of the OvO multiclass classifier.
        """
        Y_pred = self(X)
        acc = np.mean(Y_pred == Y) * 100
        print(f"🎯 Overall accuracy (OvO): {acc:.4f} %")