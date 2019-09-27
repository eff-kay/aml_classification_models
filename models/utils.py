import numpy as np

def evaluate_acc(y_true, y_pred):
    return (np.sum(y_true == y_pred, axis=0)/len(y_true))