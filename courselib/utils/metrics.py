import numpy  as np

def binary_accuracy(y_pred,y_true, class_labels=[1,-1]):
    """Accuracy function for binary classification models."""
    threshold = min(class_labels) + (max(class_labels) - min(class_labels)) / 2.
    pred_labels = np.where(y_pred >= threshold, max(class_labels), min(class_labels))
    return np.mean(pred_labels == y_true)*100

def accuracy(y_pred, y_true):    
    return np.mean(np.argmax(y_pred,axis=-1) == np.argmax(y_true,axis=-1), axis=0) * 100

def mean_squared_error(y_pred,y_true):
    return 0.5*np.mean((y_pred - y_true)**2)

def mean_absolute_error(y_pred,y_true):
    return np.mean(np.abs(y_pred - y_true))





