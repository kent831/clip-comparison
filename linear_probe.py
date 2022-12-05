import os
import torch
import clip
import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100
from torchvision.datasets import MNIST
from tqdm import tqdm


def linear_probe(dataset):
    #Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('RN50', device)

    #Load dataset
    root = os.path.expanduser("~/.cache")
    if dataset == "CIFAR10":
        train = CIFAR10(root, download=True, train=True, transform=preprocess)
        test = CIFAR10(root, download=True, train=False, transform=preprocess)
    elif dataset == "CIFAR100":
        train = CIFAR100(root, download=True, train=True, transform=preprocess)
        test = CIFAR100(root, download=True, train=False, transform=preprocess)
    elif dataset == "MNIST":
        train = MNIST(root, download=True, train=True, transform=preprocess)
        test = MNIST(root, download=True, train=False, transform=preprocess)

    def get_features(dataset):
        all_features = []
        all_labels = []

        with torch.no_grad():
            for images, labels in DataLoader(dataset, batch_size=100):
                features = model.encode_image(images.to(device))

                all_features.append(features)
                all_labels.append(labels)

        return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()

    # Calculate the image features
    train_features, train_labels = get_features(train)
    test_features, test_labels = get_features(test)

    # Perform logistic regression
    classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=0)
    classifier.fit(train_features, train_labels)

    # Evaluate using the logistic regression classifier
    predictions = classifier.predict(test_features)
    accuracy = np.mean((test_labels == predictions).astype(np.float)) * 100.
    return f"{dataset} Accuracy = {accuracy:.3f}"