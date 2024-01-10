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
from dataloader import *
from model import *

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

print(device)


batch_size = 1
learning_rate = 0.0001
epochs = 2
number_of_workers = 1


transform = transforms.Compose([
#     transforms.Resize((448, 448)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
])


image_folder = "/home/macula/SMATousi/Gullies/ground_truth/leaf_vein_seg/sorted_data/images_448/"
label_folder = "/home/macula/SMATousi/Gullies/ground_truth/leaf_vein_seg/sorted_data/labels_448/"


full_dataset = leaf_segmentation_dataset(image_folder, label_folder, transform, threshold=0)


# Calculate the size of the train, validation, and test sets
train_ratio = 0.8  # 80% for training
val_ratio = 0.1    # 10% for validation
test_ratio = 0.1   # 10% for testing

train_size = int(train_ratio * len(full_dataset))
val_size = int(val_ratio * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

# Split the dataset into train, validation, and test sets
train_dataset, temp_dataset = train_test_split(full_dataset, train_size=train_size + val_size, random_state=0, shuffle=False)
val_dataset, test_dataset = train_test_split(temp_dataset, train_size=val_size, random_state=0, shuffle=False)

# Create data loaders and move data to GPU
train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=number_of_workers, shuffle=False, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=number_of_workers, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=number_of_workers, pin_memory=True)

print("Data is loaded")



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
