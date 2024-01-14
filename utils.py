import numpy as np

def calculate_metrics(predicted, desired):

    predicted = predicted.cpu().detach().numpy()
    desired = desired.cpu().detach().numpy()
    
    predicted = np.where(predicted > 0.5, 1, 0)
    desired = np.where(desired > 0.5, 1, 0)

    accuracy = np.mean(predicted == desired)
    intersection = np.logical_and(predicted, desired)
    union = np.logical_or(predicted, desired)
    iou = np.sum(intersection) / np.sum(union)
    dice = 2 * np.sum(intersection) / (np.sum(predicted) + np.sum(desired))

    return accuracy, iou, dice