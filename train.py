import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from sklearn.model_selection import train_test_split
import random
import numpy as np
from tqdm import tqdm
import argparse
import wandb
from dataloader import *
from model import *
from Uformer_model import *

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


parser = argparse.ArgumentParser(description="A script with argparse options")

# Add an argument for an integer option
parser.add_argument("--runname", type=str, required=True)
parser.add_argument("--projectname", type=str, required=True)
parser.add_argument("--modelname", type=str, required=True)
parser.add_argument("--batchsize", type=int, default=4)
parser.add_argument("--savingstep", type=int, default=10)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--nottest", help="Enable verbose mode", action="store_true")

args = parser.parse_args()

arg_batch_size = args.batchsize
arg_epochs = args.epochs
arg_runname = args.runname
arg_projectname = args.projectname
arg_modelname = args.modelname
arg_savingstep = args.savingstep

args = parser.parse_args()

wandb.init(
        # set the wandb project where this run will be logged
    project=arg_projectname, name=arg_runname
        
        # track hyperparameters and run metadata
        # config={
        # "learning_rate": 0.02,
        # "architecture": "CNN",
        # "dataset": "CIFAR-100",
        # "epochs": 20,
        # }
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)


batch_size = arg_batch_size
learning_rate = 0.0001
epochs = arg_epochs
number_of_workers = 1
image_size = 256


transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
])


# image_folder = "/home/macula/SMATousi/Gullies/ground_truth/leaf_vein_seg/sorted_data/images_448/"
# label_folder = "/home/macula/SMATousi/Gullies/ground_truth/leaf_vein_seg/sorted_data/labels_448/"

image_folder = "../images_448/"
label_folder = "../labels_448/"


full_dataset = leaf_segmentation_dataset(image_folder, label_folder, transform, threshold=0)


# Calculate the size of the train, validation, and test sets
train_ratio = 0.8  # 80% for training
val_ratio = 0.1    # 10% for validation
test_ratio = 0.1   # 10% for testing

train_size = int(train_ratio * len(full_dataset))
val_size = int(val_ratio * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_dataset, temp_dataset = random_split(full_dataset, [train_size, test_size+val_size])
val_dataset, test_dataset = random_split(temp_dataset, [val_size, test_size])


# Create data loaders and move data to GPU
train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=number_of_workers, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=number_of_workers, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=number_of_workers, pin_memory=True)

print("Data is loaded")


# Instantiate the model
if arg_modelname == 'Unet_1':
    model = UNet_1(n_channels=3, n_classes=1, dropout_rate=0.5).to(device)  # Change n_classes based on your output
if arg_modelname == 'Uformer':
    model = Uformer(img_size=image_size,embed_dim=32,win_size=8,in_chans=1,dd_in=3,token_projection='linear',token_mlp='leff',modulator=False).to(device)
if arg_modelname == 'DepthNet':
    model = DepthNet().to(device)
if arg_modelname == 'Bothnet':
    model = BothNet(in_channels=3, out_channels=1).to(device)  # Replace with your custom model definition


criterion = nn.BCEWithLogitsLoss()  # Replace with your loss function
optimizer = optim.Adam(model.parameters(), lr=learning_rate)



# Training loop
for epoch in range(epochs):
    model.train()  # Set the model to training mode
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)  # Move data to GPU
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(images)  # Forward pass
        loss = criterion(outputs, labels)  # Calculate loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

    # Print training loss or other metrics as needed
    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {loss.item()}")

    # Validation loop
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        val_correct = 0
        val_total = 0
        for images, labels in tqdm(val_loader):
            images, labels = images.to(device), labels.to(device)  # Move data to GPU
            val_outputs = model(images)  # Forward pass
            _, val_predicted = torch.max(val_outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (val_predicted == labels).sum().item()

        val_accuracy = 100 * val_correct / val_total
        print(f"Validation Accuracy: {val_accuracy}%")

# Testing loop
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    test_correct = 0
    test_total = 0
    for images, labels in tqdm(test_loader):
        images, labels = images.to(device), labels.to(device)  # Move data to GPU
        test_outputs = model(images)  # Forward pass
        _, test_predicted = torch.max(test_outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (test_predicted == labels).sum().item()

    test_accuracy = 100 * test_correct / test_total
    print(f"Test Accuracy: {test_accuracy}%")
