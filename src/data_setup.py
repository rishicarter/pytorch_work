"""
Custom DataLoader module for image classification using PyTorch.

This module provides a `CustomDataloader` class that simplifies 
loading image datasets for training and testing.

Example Usage:
    dataloader = CustomDataloader(train_dir="path/to/train",
                                  test_dir="path/to/test",
                                  transform=some_transform)
    
    train_loader, test_loader = dataloader.get_dataloaders()
    class_names = dataloader.classes
"""

import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import random
import torch
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

# Default batch size and number of workers
BATCH_SIZE = 32
NUM_WORKERS = max(1, os.cpu_count() // 2)  # Ensure at least 1 worker

class CustomDataloader:
    """
    Custom DataLoader class for image classification datasets.

    This class loads training and testing datasets from directories,
    applies transformations, and creates PyTorch DataLoaders.

    Attributes
    ----------
    classes : list
        List of class names in the dataset.
    class_to_idx : dict
        Mapping of class names to indices.
    train_dataloader : DataLoader
        DataLoader for the training dataset.
    test_dataloader : DataLoader
        DataLoader for the testing dataset.

    Methods
    -------
    get_dataloaders():
        Returns both train and test DataLoaders as a tuple.
    """

    def __init__(self, train_dir: str, test_dir: str, transform: transforms.Compose = None,
                 batch_size: int = BATCH_SIZE, num_workers: int = NUM_WORKERS, transform_test_data: bool = False):
        """
        Initializes the CustomDataloader with specified parameters.

        Parameters
        ----------
        train_dir : str
            Path to the training dataset directory.
        test_dir : str
            Path to the testing dataset directory.
        transform : torchvision.transforms.Compose, optional
            Transformations to be applied to the training data (default is None).
        batch_size : int, optional
            Number of samples per batch (default is 32).
        num_workers : int, optional
            Number of subprocesses to use for data loading (default is half the CPU count).
        transform_test_data : bool, optional
            Whether to apply the same transform to test data (default is False).

        Raises
        ------
        ValueError
            If train_dir or test_dir does not exist.
            If batch_size or num_workers is not a positive integer.
        """

        if not os.path.exists(train_dir):
            raise ValueError(f"Training directory '{train_dir}' does not exist.")
        if not os.path.exists(test_dir):
            raise ValueError(f"Testing directory '{test_dir}' does not exist.")
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        if num_workers < 0:
            raise ValueError("num_workers must be a non-negative integer.")

        self.train_dir = train_dir
        self.test_dir = test_dir
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = min(num_workers, os.cpu_count())  # Ensure it does not exceed available cores
        self.train_image_paths = list(Path(train_dir).glob("*/*.jpg"))
        self.test_image_paths = list(Path(test_dir).glob("*/*.jpg"))
        self.default_transform = transforms.Compose([transforms.Resize((64, 64)),
                                                     transforms.ToTensor()])
        # Load datasets
        self.train_data = datasets.ImageFolder(self.train_dir, transform=self.transform)
        self.test_data = datasets.ImageFolder(self.test_dir, transform=self.transform if transform_test_data else self.default_transform)

    @property
    def classes(self):
        """Returns a list of class names from the training dataset."""
        return self.train_data.classes

    @property
    def class_to_idx(self):
        """Returns a dictionary mapping class names to indices."""
        return self.train_data.class_to_idx

    @property
    def train_dataloader(self):
        """Returns a DataLoader for the training dataset."""
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    @property
    def test_dataloader(self):
        """Returns a DataLoader for the testing dataset."""
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def get_dataloaders(self):
        """Returns a tuple of (train_dataloader, test_dataloader)."""
        return self.train_dataloader, self.test_dataloader

    def __str__(self):
        """
        Returns a user-friendly string representation of the dataset.
    
        Includes:
        - Number of training & testing samples
        - Number of batches
        - Shape of a batch from train DataLoader
        """
        train_samples = len(self.train_data)
        test_samples = len(self.test_data)
        
        train_batches = len(self.train_dataloader)
        test_batches = len(self.test_dataloader)
        
        # Try fetching a single batch to get its shape
        try:
            sample_batch = next(iter(self.train_dataloader))
            batch_shape = sample_batch[0].shape  # Shape of the image tensor batch
        except Exception as e:
            batch_shape = "Unavailable (Dataset might be empty)"
        
        return (
            f"CustomDataloader Summary:\n"
            f"- Training Samples: {train_samples}\n"
            f"- Testing Samples: {test_samples}\n"
            f"- Train Batches: {train_batches} (Batch Size: {self.batch_size})\n"
            f"- Test Batches: {test_batches} (Batch Size: {self.batch_size})\n"
            f"- Batch Shape: {batch_shape}"
        )
    def __repr__(self):
        """Returns the same output as __str__ for better representation in notebooks."""
        return self.__str__()

    def plot_transformed_images(self, k: int = 10, seed: int = 42):
        """
        Plots randomly selected images from the training dataset with and without transformations.

        Parameters
        ----------
        k : int, optional
            Number of images to display (default is 10, max is 10).
        seed : int, optional
            Random seed for reproducibility (default is 42).

        Raises
        ------
        ValueError
            If the training dataset is empty or k exceeds available images.
        """
        if not self.train_image_paths:
            raise ValueError("No training images found.")

        k = min(k, 10)  # Limit to 10 images
        random.seed(seed)
        random_img_paths = random.sample(self.train_image_paths, k=k)

        for img_path in random_img_paths:
            with Image.open(img_path) as f:
                fig, ax = plt.subplots(1, 2, figsize=(6, 3))
                
                # Original Image
                ax[0].imshow(f)
                ax[0].set_title(f"Original\nSize: {f.size}")
                ax[0].axis("off")

                # Transformed Image
                if self.transform:
                    t_img = self.transform(f)
                    t_img = torch.permute(t_img, (1, 2, 0))
                    ax[1].imshow(t_img)
                    ax[1].set_title(f"Transformed\nSize: {t_img.shape}")
                else:
                    ax[1].imshow(f)
                    ax[1].set_title("No Transform Applied")

                ax[1].axis("off")
                fig.suptitle(f"Class: {img_path.parent.stem}", fontsize=15)

        plt.show()

    def display_random_images(self, n: int = 5, display_shape: bool = True, seed: int = 42):
        """
        Displays `n` random images from the training dataset.
    
        Parameters
        ----------
        n : int, optional
            Number of random images to display (default is 5, max is 10).
        display_shape : bool, optional
            Whether to display image shape in the title (default is True).
        seed : int, optional
            Random seed for reproducibility (default is 42).
    
        Raises
        ------
        ValueError
            If the training dataset is empty.
        """
        # Check if dataset is empty
        if len(self.train_data) == 0:
            raise ValueError("Training dataset is empty. Cannot display images.")
    
        # Limit the number of images for display purposes
        if n > 10:
            n = 10
            display_shape = False  # Disable shape display for better readability
            print(f"Limiting display to 10 images. Disabling shape display for clarity.")
    
        # Set random seed for reproducibility
        random.seed(seed)
    
        # Get random sample indexes
        random_samples_idx = random.sample(range(len(self.train_data)), k=n)
    
        # Set up the figure
        plt.figure(figsize=(16, 8))
    
        # Loop through selected samples and display them
        for i, sample_idx in enumerate(random_samples_idx):
            image, label = self.train_data[sample_idx]
    
            # Adjust tensor shape for plotting: [C, H, W] -> [H, W, C]
            image_adjusted = image.permute(1, 2, 0)
    
            # Create subplot
            plt.subplot(1, n, i + 1)
            plt.imshow(image_adjusted)
            plt.axis("off")
    
            # Set title
            title = f"Class: {self.classes[label]}" if self.classes else f"Label: {label}"
            if display_shape:
                title += f"\nShape: {tuple(image_adjusted.shape)}"
    
            plt.title(title, fontsize=10)
    
        # Display the images
        plt.show()
