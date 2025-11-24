"""
RAF-DB Dataset Utilities

This module provides utilities for loading, visualizing, and preprocessing
the RAF-DB (Real-world Affective Faces Database) dataset for emotion recognition.
"""

import os
import json
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from typing import Tuple, List, Optional
import torch
from torch.utils.data import Dataset


# RAF-DB emotion label mapping (7 basic emotions)
# Label indices: 1=Surprise, 2=Fear, 3=Disgust, 4=Happiness, 5=Sadness, 6=Anger, 7=Neutral
EMOTION_LABELS = {
    1: 'Surprise',
    2: 'Fear',
    3: 'Disgust',
    4: 'Happiness',
    5: 'Sadness',
    6: 'Anger',
    7: 'Neutral'
}

# Reverse mapping: emotion name -> label index
EMOTION_TO_LABEL = {v: k for k, v in EMOTION_LABELS.items()}

# Standard 7-class emotion order for consistency with FER2013
EMOTION_ORDER = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']


def load_rafdb_dataset(root: str, split: str = 'train') -> Tuple[List[str], List[int]]:
    """
    Load RAF-DB dataset images and labels for a given split.
    """
    root_path = Path(root)
    
    # Load label CSV file
    if split == 'train':
        label_file = root_path / 'train_labels.csv'
        dataset_dir = root_path / 'DATASET' / 'train'
    elif split == 'test':
        label_file = root_path / 'test_labels.csv'
        dataset_dir = root_path / 'DATASET' / 'test'
    else:
        raise ValueError(f"Split must be 'train' or 'test', got '{split}'")
    
    if not label_file.exists():
        raise FileNotFoundError(f"Label file not found: {label_file}")
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    
    # Read labels CSV
    df = pd.read_csv(label_file)
    
    images = []
    labels = []
    
    # Find images in the dataset directory structure
    # Images are organized in subdirectories by label (1-7)
    for _, row in df.iterrows():
        image_name = row['image']
        label = int(row['label'])
        
        # Check if image exists in the label subdirectory
        image_path = dataset_dir / str(label) / image_name
        
        if image_path.exists():
            images.append(str(image_path))
            labels.append(label)
        else:
            # Try to find the image in any subdirectory (fallback)
            found = False
            for label_dir in dataset_dir.iterdir():
                if label_dir.is_dir():
                    potential_path = label_dir / image_name
                    if potential_path.exists():
                        images.append(str(potential_path))
                        labels.append(label)
                        found = True
                        break
            if not found:
                print(f"Warning: Image not found: {image_name}")
    
    return images, labels


def visualize_samples(images: List[str], labels: List[int], 
                     num_samples: int = 16, 
                     figsize: Tuple[int, int] = (12, 12),
                     save_path: Optional[str] = None) -> None:
    """
    Visualize a grid of sample images with their emotion labels.
    """
    num_samples = min(num_samples, len(images))
    
    # Randomly sample if we have more images than requested
    if len(images) > num_samples:
        indices = np.random.choice(len(images), num_samples, replace=False)
        sampled_images = [images[i] for i in indices]
        sampled_labels = [labels[i] for i in indices]
    else:
        sampled_images = images
        sampled_labels = labels
    
    # Calculate grid dimensions
    cols = 4
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes
    
    for idx, (img_path, label) in enumerate(zip(sampled_images, sampled_labels)):
        ax = axes[idx]
        
        # Load and display image
        try:
            img = Image.open(img_path).convert('RGB')
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f'{EMOTION_LABELS[label]} ({label})', fontsize=10)
        except Exception as e:
            ax.text(0.5, 0.5, f'Error loading\n{os.path.basename(img_path)}', 
                   ha='center', va='center')
            ax.axis('off')
    
    # Hide unused subplots
    for idx in range(num_samples, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()

class RAFDBDataset(Dataset):
    """
    PyTorch Dataset class for RAF-DB.
    
    This can be used with DataLoader for training.
    """
    
    def __init__(self, root: str, split: str = 'train', transform=None):
        """
        Initialize RAF-DB dataset.
        
        Parameters:
        -----------
        root : str
            Root directory containing RAF-DB data
        split : str
            Dataset split: 'train' or 'test'
        transform : callable, optional
            Optional transform to be applied on a sample
        """
        self.root = root
        self.split = split
        self.transform = transform
        
        # Load images and labels
        self.images, self.labels = load_rafdb_dataset(root, split=split)
        
        # Convert labels to 0-indexed (PyTorch convention)
        self.labels = [l - 1 for l in self.labels]  # 1-7 -> 0-6
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            img = self.transform(img)
        
        return img, label

