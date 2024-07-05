import numpy as np
# True Positives (TP): Количество правильных положительных предсказаний.
# False Positives (FP): Количество неправильных положительных предсказаний.
# False Negatives (FN): Количество неправильных отрицательных предсказаний.
# True Negatives (TN): Количество правильных отрицательных предсказаний.
# Точность (Precision): Доля правильных положительных предсказаний среди всех положительных предсказаний.
# Полнота (Recall): Доля правильных положительных предсказаний среди всех положительных образцов.
# F1-мера (F1 Score): Гармоническое среднее точности и полноты.
# Точность (Accuracy): Доля правильных предсказаний среди всех предсказаний.
def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    # True positives, false positives, false negatives, true negatives
    tp = np.sum((prediction == True) & (ground_truth == True))
    fp = np.sum((prediction == True) & (ground_truth == False))
    fn = np.sum((prediction == False) & (ground_truth == True))
    tn = np.sum((prediction == False) & (ground_truth == False))

    # Precision: tp / (tp + fp)
    if tp + fp > 0:
        precision = tp / (tp + fp)
    else:
        precision = 0

    # Recall: tp / (tp + fn)
    if tp + fn > 0:
        recall = tp / (tp + fn)
    else:
        recall = 0

    # F1 score: 2 * (precision * recall) / (precision + recall)
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0

    # Accuracy: (tp + tn) / (tp + fp + fn + tn)
    accuracy = (tp + tn) / (tp + fp + fn + tn)

    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    correct_predictions = np.sum(prediction == ground_truth) # Cчитает количество правильных предсказаний (где предсказание совпадает с истинной меткой).
    total_samples = len(ground_truth) # Возвращает общее количество примеров.
    accuracy = correct_predictions / total_samples

    return accuracy
