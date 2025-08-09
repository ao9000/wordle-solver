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
        # Convert to grayscale
        torchvision.transforms.Grayscale(num_output_channels=1),  # Convert RGB to single channel
        # Convert to pytorch image tensor
        torchvision.transforms.ToTensor(),
        # Resize to 28x28
        torchvision.transforms.Resize((28, 28)),
        # Mean and std of mnist digit dataset
        torchvision.transforms.Normalize((0.11070,), (0.2661,)),
    ])
    return transform


def handwritten_alphabet_dataset_loader(root_dir, train, transform, batch_size):
    split = 'train' if train else 'test'
    data_path = os.path.join(root_dir, split)
    # ImageFolder will assign class indices 0–25 in alphabetical order of folder names (A=0, B=1, … Z=25)
    dataset = ImageFolder(
        data_path,
        transform=transform
    )
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=train,
                      pin_memory=True)



def wordle_alphabet_dataset_loader(root_dir, transform, batch_size):
    dataset = ImageFolder(root_dir, transform=transform)
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=True,
                      pin_memory=True)


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
    plt.plot(epochs, history['wordle_test_acc'], label="Sudoku Digits Test Accuracy", color='green')

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

    # Labels, title, legend
    plt.title('Fine-tuning Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper left')

    # Save figure to disk
    plt.savefig("models/ft_loss.png")
    plt.close(fig)


def wordle_cell_preprocessing(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    return thresh