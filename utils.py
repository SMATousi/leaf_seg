import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision import models
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import glob
import wandb
import random
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


def save_comparison_figures(model, dataloader, epoch, device, save_dir='comparison_figures', num_samples=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    sample_count = 0
    fig, axs = plt.subplots(num_samples, 3, figsize=(10, num_samples * 5))  # 5 is an arbitrary height multiplier for visibility

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            if sample_count >= num_samples:
                break  # Break if we have already reached the desired number of samples

            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            probs = outputs.sigmoid()
            preds = probs > 0.5  # Apply threshold to get binary mask

            # Access the i-th sample in the batch for both ground truth and prediction
            input_image = inputs[sample_count].squeeze().cpu().numpy().transpose(1,2,0)
            gt_mask = targets[sample_count].squeeze().cpu().numpy()  # Convert to NumPy array for plotting
            pred_mask = preds[sample_count].squeeze().cpu().numpy()

            axs[sample_count, 0].imshow(input_image)
            axs[sample_count, 0].set_title(f'Sample {sample_count + 1} Input')
            axs[sample_count, 0].axis('off')

            axs[sample_count, 1].imshow(gt_mask, cmap='gray')
            axs[sample_count, 1].set_title(f'Sample {sample_count + 1} Ground Truth')
            axs[sample_count, 1].axis('off')

            axs[sample_count, 2].imshow(pred_mask, cmap='gray')
            axs[sample_count, 2].set_title(f'Sample {sample_count + 1} Prediction')
            axs[sample_count, 2].axis('off')

            

            sample_count += 1  # Increment the sample counter

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.suptitle(f'Comparison for Epoch {epoch}', fontsize=16)
    plt.subplots_adjust(top=0.9)  # Adjust the top spacing to accommodate the suptitle

    figure_path = os.path.join(save_dir, f'epoch_{epoch}_comparison.png')
    os.makedirs(save_dir, exist_ok=True)  # Ensure the save directory exists
    plt.savefig(figure_path)
    wandb.log({f'Images/epoch_{epoch}': wandb.Image(f'{save_dir}/epoch_{epoch}_comparison.png')})

    plt.close()