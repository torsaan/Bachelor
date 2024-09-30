import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score
import time
import os
from sklearn.preprocessing import label_binarize


def calculate_f1_score(all_labels, all_preds, average='weighted'):
    """
    Calculates the F1 score.

    Parameters:
    - all_labels: array-like of shape (n_samples,), True labels.
    - all_preds: array-like of shape (n_samples,), Predicted labels.
    - average: string, [None, 'binary' (default), 'micro', 'macro', 'samples', 'weighted'], 
      This parameter is required for multiclass/multilabel targets.
      If None, the scores for each class are returned.

    Returns:
    - f1: float, F1 score of the positive class in binary classification or weighted average of the F1 scores of each class for the multiclass task.
    """
    f1 = f1_score(all_labels, all_preds, average=average)
    return f1



def calculate_auc_score(all_labels, all_preds, num_classes):
    """
    Calculates the AUC score for binary or multiclass classification.

    Parameters:
    - all_labels: array-like of shape (n_samples,), True labels.
    - all_preds: array-like of shape (n_samples,), Predicted labels or scores.
    - num_classes: int, Number of classes in the dataset.

    Returns:
    - auc: float, AUC score.
    """
    one_hot_labels = label_binarize(all_labels, classes=range(num_classes))
    
    if num_classes == 2:
        # For binary classification, roc_auc_score expects shape (n_samples,) for scores
        auc = roc_auc_score(one_hot_labels, all_preds[:, 1], multi_class="raise")
    else:
        # For multiclass, convert predictions to one-hot format for one_hot_labels shape
        one_hot_preds = label_binarize(all_preds, classes=range(num_classes))
        auc = roc_auc_score(one_hot_labels, one_hot_preds, average="micro", multi_class="ovr")
    
    return auc
