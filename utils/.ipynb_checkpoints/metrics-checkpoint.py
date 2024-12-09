import numpy as np
from sklearn.metrics import jaccard_score, precision_recall_curve, precision_score, recall_score, f1_score, average_precision_score, matthews_corrcoef, confusion_matrix

def calculate_iou(mask1, mask2):
    """
    Calcula el Índice de Superposición (IoU) entre dos máscaras binarias.

    Parámetros:
    mask1 (numpy.ndarray): Primera máscara binaria.
    mask2 (numpy.ndarray): Segunda máscara binaria.

    Retorna:
    float: Valor del IoU.
    """
    if np.sum(mask1) == 0 and np.sum(mask2) == 0:
        return 1.0  # Ambas máscaras son cero, IoU es 1
    elif np.sum(mask1) == 0 or np.sum(mask2) == 0:
        return 0.0  # Una de las máscaras es cero, IoU es 0
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union)
    return iou
def calculate_recall(ground_truth_mask, predicted_mask):
    """
    Calcula el Recall entre dos máscaras binarias.

    Parámetros:
    ground_truth_mask (numpy.ndarray): Máscara de referencia (ground truth).
    predicted_mask (numpy.ndarray): Máscara predicha.

    Retorna:
    float: Valor del Recall.
    """
    return recall_score(ground_truth_mask.flatten(), predicted_mask.flatten())

def calculate_precision(ground_truth_mask, predicted_mask):
    """
    Calcula la Precision entre dos máscaras binarias.

    Parámetros:
    ground_truth_mask (numpy.ndarray): Máscara de referencia (ground truth).
    predicted_mask (numpy.ndarray): Máscara predicha.

    Retorna:
    float: Valor del Precision.
    """
    return precision_score(ground_truth_mask.flatten(), predicted_mask.flatten())

def calculate_dice(ground_truth_mask, predicted_mask):
    """
    Calcula el coeficiente DICE entre dos máscaras binarias.

    Parámetros:
    ground_truth_mask (numpy.ndarray): Máscara de referencia (ground truth).
    predicted_mask (numpy.ndarray): Máscara predicha.

    Retorna:
    float: Valor del coeficiente DICE.
    """
    if np.sum(ground_truth_mask) == 0 and np.sum(predicted_mask) == 0:
        return 1.0  # Ambas máscaras son cero, IoU es 1

    intersection = np.logical_and(ground_truth_mask, predicted_mask)
    dice = 2 * np.sum(intersection) / (np.sum(ground_truth_mask) + np.sum(predicted_mask))
    return dice

def calculate_f1(ground_truth_mask, predicted_mask):
    """
    Calcula el F1-Score entre dos máscaras binarias.

    Parámetros:
    ground_truth_mask (numpy.ndarray): Máscara de referencia (ground truth).
    predicted_mask (numpy.ndarray): Máscara predicha.

    Retorna:
    float: Valor del F1-Score.
    """
    return f1_score(ground_truth_mask.flatten(), predicted_mask.flatten())

def calculate_map50(ground_truth_mask, predicted_mask):
    precision, recall, _ = precision_recall_curve(ground_truth_mask.flatten(), predicted_mask.flatten())
    ap = average_precision_score(ground_truth_mask.flatten(), predicted_mask.flatten())
    return ap

def calculate_mcc(ground_truth_mask, predicted_mask):
    """
    Calcula el Matthews Correlation Coefficient (MCC) entre dos máscaras binarias.

    Parámetros:
    ground_truth_mask (numpy.ndarray): Máscara de referencia (ground truth).
    predicted_mask (numpy.ndarray): Máscara predicha.

    Retorna:
    float: Valor del MCC.
    """
    return matthews_corrcoef(ground_truth_mask.flatten(), predicted_mask.flatten())

def calculate_tnr(ground_truth_mask, predicted_mask):
    """
    Calcula el True Negative Rate (TNR) entre dos máscaras binarias.

    Parámetros:
    ground_truth_mask (numpy.ndarray): Máscara de referencia (ground truth).
    predicted_mask (numpy.ndarray): Máscara predicha.

    Retorna:
    float: Valor del TNR.
    """
    tn, fp, fn, tp = confusion_matrix(ground_truth_mask.flatten(), predicted_mask.flatten()).ravel()
    tnr = tn / (tn + fp)
    return tnr
