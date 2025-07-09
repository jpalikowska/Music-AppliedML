import numpy as np
from .svm import BinaryKernelSVM

class KernelMulticlassOvR:
    def __init__(self, kernel_type='rbf', C=1.0, **kwargs):
        self.kernel_type = kernel_type
        self.C = C
        kwargs.pop('kernel', None)
        self.kwargs = kwargs
        self.models = []
        self.labels = None

    def fit(self, X, Y):
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
        scores = [model.decision_function(X) for model in self.models]
        return np.vstack(scores).T
    
    def __call__(self, X):
        scores = self.decision_function(X)
        indices = np.argmax(scores, axis=1)
        return self.labels[indices]
    
    def evaluate_models(self, X, Y):
        from courselib.utils.metrics import binary_accuracy
        print("📊 Accuracy of each binary model (One-vs-Rest):")
        for cls, model in zip(self.labels, self.models):
            Y_encoded = np.where(Y == cls, 1, -1)
            preds = np.sign(model(X))
            acc = binary_accuracy(preds, Y_encoded)
            print(f"  - Class '{cls}': {acc:.4f} (Support vectors: {len(model.sv_y)})")

    def evaluate_accuracy(self, X, Y):
        Y_pred = self(X)
        acc = np.mean(Y_pred == Y)
        print(f"🎯 Overall accuracy (OvR): {acc:.4f}")

class KernelMulticlassOvO:
    def __init__(self, kernel_type='rbf', C=1.0, **kwargs):
        self.kernel_type = kernel_type
        self.C = C
        kwargs.pop('kernel', None)
        self.kwargs = kwargs
        self.models = []  # (class_a, class_b, model)
        self.label_pairs = []  # (class_a, class_b)
        self.labels = None

    def fit(self, X, Y):
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
        votes = {label: np.zeros(len(X)) for label in self.labels}

        for class_a, class_b, model in self.models:
            preds = np.sign(model(X))
            votes[class_a] += (preds == 1)
            votes[class_b] += (preds == -1)

        all_votes = np.stack([votes[label] for label in self.labels], axis=1)
        predicted_indices = np.argmax(all_votes, axis=1)
        return self.labels[predicted_indices]

    def evaluate_models(self, X, Y):
        from courselib.utils.metrics import binary_accuracy
        print("📊 Accuracy of each OvO binary classifier:")
        for (class_a, class_b, model) in self.models:
            idx = np.where((Y == class_a) | (Y == class_b))
            X_pair = X[idx]
            Y_pair = Y[idx]
            Y_bin = np.where(Y_pair == class_a, 1, -1)
            preds = np.sign(model(X_pair))
            acc = binary_accuracy(preds, Y_bin)
            print(f"  - Classifier '{class_a}' vs '{class_b}': {acc:.4f} (Support vectors: {len(model.sv_y)})")

    def evaluate_accuracy(self, X, Y):
        Y_pred = self(X)
        acc = np.mean(Y_pred == Y)
        print(f"🎯 Overall accuracy (OvO): {acc:.4f}")