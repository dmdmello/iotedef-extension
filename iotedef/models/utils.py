import sklearn as sk 
import sys
import copy
import logging
import numpy as np
from algorithm import Algorithm
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import sklearn.metrics as metrics 
import tensorflow as tf
import sklearn as sk
from sklearn.metrics import confusion_matrix


def Xrecall_thrsh(predictions, labels, X):
    """
    Returns the threshold necessary to obtain 99% recall with the provided predictions.
    """
    # Compute recall scores for a range of thresholds
    thresholds = np.arange(0, 1, 0.01)
    recalls = [metrics.recall_score(labels, predictions >= t) for t in thresholds]

    # Find the threshold that gives X recall
    idx = np.argmin(np.abs(np.array(recalls) - X))

    return thresholds[idx]


def Xprecision_thrsh(predictions, labels, X):
    """
    Returns the threshold necessary to obtain 99% recall with the provided predictions.
    """
    # Compute recall scores for a range of thresholds
    thresholds = np.arange(0, 1, 0.01)
    recalls = [metrics.precision_score(labels, predictions >= t) for t in thresholds]

    # Find the threshold that gives X precision
    idx = np.argmin(np.abs(np.array(recalls) - X))

    return thresholds[idx]


def get_fraction_of_array(array, fraction):
    return array[:int(len(array)*fraction)]


def recall_at_Xprecision(predictions, labels, X):
    """
    Returns the threshold necessary to obtain 99% recall with the provided predictions.
    """
    th = Xprecision_thrsh(predictions, labels, X)
    recall = metrics.recall_score(labels, predictions >= th) 
    return recall


def precision_at_Xrecall(predictions, labels, X):
    """ 
    Returns the threshold necessary to obtain 99% recall with the provided predictions.
    """
    th = Xrecall_thrsh(predictions, labels, X)
    precision = metrics.precision_score(labels, predictions >= th) 
    return precision