import numpy  as np

def binary_accuracy(y_pred,y_true, class_labels=[1,-1]):
    """Accuracy function for binary classification models."""
    pred_labels = np.where(y_pred >= (max(class_labels)-min(class_labels))/2., 
                           max(class_labels), min(class_labels))
    return np.mean(pred_labels == y_true)*100

def mean_squared_error(y_pred,y_true):
    return 0.5*np.mean((y_pred - y_true)**2, axis=0)

