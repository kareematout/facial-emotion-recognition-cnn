"""
PyTorch Dataset class and helpers for the FER2013 dataset.

This mirrors the structure of rafdb_dataset.RAFDBDataset so that
both datasets can be used with the same training code.
"""

from pathlib import Path
from typing import List, Tuple

from PIL import Image
from torch.utils.data import Dataset

# --------------------------------------------------------------------
# Emotion label configuration
# --------------------------------------------------------------------
# IMPORTANT: This order should match whatever your partner uses
# for RAF-DB so that label 0,1,...,6 mean the same emotion
# across both datasets.
#
# From your partner's notebook screenshot:
# 1: Surprise, 2: Fear, 3: Disgust, 4: Happiness,
# 5: Sadness, 6: Anger, 7: Neutral

EMOTION_ORDER = [
    "surprise",
    "fear",
    "disgust",
    "happy",
    "sad",
    "anger",
    "neutral",
]

EMOTION_TO_LABEL = {emo: i for i, emo in enumerate(EMOTION_ORDER)}
EMOTION_LABELS = EMOTION_ORDER  # alias, for compatibility


# --------------------------------------------------------------------
# Helper: load FER2013 image paths + labels
# --------------------------------------------------------------------
def load_fer2013_dataset(root: str | Path, split: str = "train") -> Tuple[List[Path], List[int]]:
    """
    Load FER2013 image filepaths and integer labels for a given split.

    Parameters
    ----------
    root : str or Path
        Root directory for processed FER2013 data (e.g. data/fer2013).
        This directory is expected to contain subfolders train/, val/, test/
        and inside each, one subfolder per emotion (angry, disgust, ...).
    split : {"train", "val", "test"}
        Dataset split to use.

    Returns
    -------
    images : list[Path]
        List of image filepaths.
    labels : list[int]
        List of labels as 1–7 (to mirror RAF-DB loader).
        Later, the Dataset class will convert them to 0–6.
    """
    root = Path(root)
    split_dir = root / split

    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    images: List[Path] = []
    labels: List[int] = []

    # We iterate over EMOTION_ORDER so label indices are consistent
    for idx, emotion in enumerate(EMOTION_ORDER, start=1):  # 1–7
        emotion_dir = split_dir / emotion
        if not emotion_dir.exists():
            # You can choose to raise instead if you want strict checking
            print(f"[load_fer2013_dataset] Warning: missing folder {emotion_dir}")
            continue

        for img_path in sorted(emotion_dir.glob("*")):
            if img_path.is_file():
                images.append(img_path)
                labels.append(idx)  # 1–7, same convention as RAF-DB loader

    return images, labels


# --------------------------------------------------------------------
# Dataset class
# --------------------------------------------------------------------
class FER2013Dataset(Dataset):
    """
    PyTorch Dataset class for FER2013.

    This mirrors RAFDBDataset so it can be used with the same
    DataLoader and training code.
    """

    def __init__(self, root: str | Path, split: str = "train", transform=None):
        """
        Initialize FER2013 dataset.

        Parameters
        ----------
        root : str or Path
            Root directory containing processed FER2013 data (data/fer2013).
        split : {"train", "val", "test"}
            Dataset split to use.
        transform : callable, optional
            Optional transform to be applied on a sample image.
        """
        self.root = Path(root)
        self.split = split
        self.transform = transform

        # Load image paths and labels (1–7)
        self.images, self.labels = load_fer2013_dataset(self.root, split=self.split)

        # Convert labels to 0–indexed (PyTorch convention)
        # 1–7 -> 0–6
        self.labels = [l - 1 for l in self.labels]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img_path = self.images[idx]
        label = self.labels[idx]

        # Load image as RGB (even though FER2013 is grayscale)
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, label
