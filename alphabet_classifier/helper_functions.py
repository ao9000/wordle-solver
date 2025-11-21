import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.datasets import ImageFolder
import os
from torch.utils.data import DataLoader
import torchvision
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import cv2
import numpy as np
from PIL import Image

# Model definition
class MNISTClassifier(nn.Module):
    """
    v1 Source: https://nextjournal.com/gkoehler/pytorch-mnist
    v2 Source: https://github.com/PyTorch/examples/blob/main/mnist/main.py
    Had to find better model due to not able to make mistake on android app
    """
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 26)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output



def transform_handwritten_alphabet_dataset():
    transform = torchvision.transforms.Compose([
        # Convert to pytorch image tensor
        torchvision.transforms.ToTensor(),
        # Mean and std of mnist digit dataset
        torchvision.transforms.Normalize((0.11070,), (0.2661,)),
    ])
    return transform


def wordle_cell_preprocessing(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    return thresh

def wordle_cells_reduce_noise(alpha_inv):
    # Eliminate surrounding noise
    # Detect contours
    cnts, hierarchy = cv2.findContours(alpha_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Relative area threshold
    # Total area of 28x28 image is 784 pixels
    total_area = alpha_inv.shape[0] * alpha_inv.shape[1]
    # Calculate area threshold based on 1% of total area
    frac = 0.010
    area_thresh = total_area * frac
    # Filter contours over 5 pixel square area
    cnts = [cnt for cnt in cnts if cv2.contourArea(cnt) > area_thresh]

    # Check if any contour is detected
    if cnts:
        # Sort to largest contour (Digit)
        cnt = sorted(cnts, key=lambda x: cv2.contourArea(x), reverse=True)[0]
        # Get coordinates, width, height of contour
        x, y, width, height = cv2.boundingRect(cnt)

        # Create buffer for crop
        crop_buffer = 1
        # Decrement crop buffer if buffer goes out of bounds of image
        while (y-crop_buffer) < 0 or (x-crop_buffer) < 0 or (y+height+crop_buffer) > alpha_inv.shape[0] or (x+width+crop_buffer) > alpha_inv.shape[1]:
            crop_buffer -= 1

            if crop_buffer == 0:
                break

        # Crop area
        alpha_inv = alpha_inv[y-crop_buffer:y + height+crop_buffer, x-crop_buffer:x + width+crop_buffer]
        # Update height & width
        height = height + (crop_buffer*2)
        width = width + (crop_buffer*2)

        # Create a black mat
        new_alpha_inv = np.zeros((28, 28), np.uint8)

        # Standardize all image sizes
        # Maintain aspect ratio, resize via height or width (Whichever is bigger)
        resized_target_height_width = 17

        if height > width:
            # Height is larger
            aspect_ratio = resized_target_height_width / float(height)
            new_dimensions = (int(width * aspect_ratio), resized_target_height_width)
        else:
            # Width is larger
            aspect_ratio = resized_target_height_width / float(width)
            new_dimensions = (resized_target_height_width, int(height * aspect_ratio))

        # Don't allow any dimension to be 0, will result in error
        if new_dimensions[0] <= 3 or new_dimensions[1] <= 3:
            new_dimensions = (resized_target_height_width, resized_target_height_width)

        # Check if original image is larger is smaller
        if height > resized_target_height_width:
            # Shrink
            alpha_inv = cv2.resize(alpha_inv, new_dimensions, interpolation=cv2.INTER_AREA)
        else:
            # Expand
            alpha_inv = cv2.resize(alpha_inv, new_dimensions, interpolation=cv2.INTER_CUBIC)

        # Update width & height
        height, width = alpha_inv.shape

        # Paste detected contour in the middle to center image
        new_alpha_inv[14-height//2:14-height//2+height, 14-width//2:14-width//2+width] = alpha_inv

        return new_alpha_inv
    else:
        # No contour detected
        return None


def handwritten_alphabet_dataset_loader(root_dir, train, transform, batch_size):
    def loader(img_path):
        img = cv2.imread(img_path)
        alpha_thresh = wordle_cell_preprocessing(img)
        denoised_alpha = wordle_cells_reduce_noise(alpha_thresh)
        if denoised_alpha is not None:
            return Image.fromarray(denoised_alpha)
        raise RuntimeError("Bad data")


    split = 'train' if train else 'test'
    data_path = os.path.join(root_dir, split)
    # ImageFolder will assign class indices 0–25 in alphabetical order of folder names (A=0, B=1, … Z=25)
    dataset = ImageFolder(
        data_path,
        loader=loader,
        transform=transform
    )
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=train,
                      pin_memory=True)


def wordle_alphabet_dataset_loader(dataset_path, transform, batch_size):
    def loader(img_path):
        img = cv2.imread(img_path)
        alpha_thresh = wordle_cell_preprocessing(img)
        denoised_alpha = wordle_cells_reduce_noise(alpha_thresh)
        if denoised_alpha is not None:
            return Image.fromarray(denoised_alpha)
        raise RuntimeError("Bad data")

    test_dataset = ImageFolder(
        root=dataset_path,
        loader=loader,
        transform=transform,
    )

    # Return all data
    return DataLoader(test_dataset,
                      batch_size=batch_size,
                      shuffle=True,
                      pin_memory=True,
                      )


def build_model(optimizer_name, **kwargs):
    # Init model network
    model = MNISTClassifier()
    name = optimizer_name.lower()

    if name == "adam":
        # All keyword args (lr, weight_decay, betas, etc.) go into Adam(...)
        optimizer = optim.Adam(model.parameters(), **kwargs)

    elif name == "sgd":
        # For SGD you might want at least lr and optionally momentum, etc.
        optimizer = optim.SGD(model.parameters(), **kwargs)

    elif name == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), **kwargs)

    elif name == "adadelta":
        # Adadelta typically only takes lr and rho; defaults mirror PyTorch example
        optimizer = optim.Adadelta(model.parameters(), **kwargs)

    else:
        raise ValueError(
            f"Unsupported optimizer: {optimizer_name!r}. "
            f"Choose from 'adam', 'sgd', 'rmsprop', 'adadelta'"
        )

    return model, optimizer


def plot_pretrain_accuracy_graph(history):
    epochs = range(1, len(history['train_acc']) + 1)

    fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(MultipleLocator(base=1.0))

    # Plot each curve
    plt.plot(epochs, history['train_acc'], label="Train Accuracy", color='blue')
    plt.plot(epochs, history['test_acc'], label="Test Accuracy", color='orange')
    plt.plot(epochs, history['wordle_test_acc'], label="Wordle Alphabets Test Accuracy", color='green')

    # Labels, title, legend
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper left')

    # Save figure to disk
    plt.savefig("models/pretrain_accuracy.png")
    plt.close(fig)


def plot_pretrain_loss_graph(history):
    epochs = range(1, len(history['train_loss']) + 1)

    fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(MultipleLocator(base=1.0))

    # Plot each curve
    plt.plot(epochs, history['train_loss'], label="Train Loss", color='blue')
    plt.plot(epochs, history['test_loss'], label="Test Loss", color='orange')
    plt.plot(epochs, history['wordle_test_loss'], label="Custom Test Loss", color='green')

    # Labels, title, legend
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper left')

    # Save figure to disk
    plt.savefig("models/pretrain_loss.png")
    plt.close(fig)


def plot_accuracy_graph_ft(history):
    epochs = range(1, len(history['train_acc']) + 1)

    fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(MultipleLocator(base=1.0))

    # Plot each curve
    plt.plot(epochs, history['train_acc'], label="Wordle Train Accuracy", color='blue')
    plt.plot(epochs, history['test_acc'], label="Wordle Test Accuracy", color='orange')

    # Labels, title, legend
    plt.title('Fine-tuning Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper left')

    # Save figure to disk
    plt.savefig("models/ft_accuracy.png")
    plt.close(fig)


def plot_loss_graph_ft(history):
    epochs = range(1, len(history['train_loss']) + 1)

    fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(MultipleLocator(base=1.0))

    # Plot each curve
    plt.plot(epochs, history['train_loss'], label="Wordle Train Loss", color='blue')
    plt.plot(epochs, history['test_loss'], label="Wordle Test Loss", color='orange')

    # Labels, title, legend
    plt.title('Fine-tuning Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper left')

    # Save figure to disk
    plt.savefig("models/ft_loss.png")
    plt.close(fig)