"""Utility functions for loading image data."""

from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image
from skimage import io
from tqdm import tqdm


def load_split(  # noqa: C901
    split_dir: str,
    library: Literal["cv2", "pil", "skimage"],
) -> tuple[list[NDArray[np.uint8]], NDArray[np.uint8]]:
    """Load images and labels from a dataset split directory."""
    # Choose loader once (saves repeated branching)
    if library == "cv2":

        def loader(p: Path) -> NDArray[np.uint8]:
            return cv2.imread(str(p))[..., ::-1]

    elif library == "pil":

        def loader(p: Path) -> NDArray[np.uint8]:
            return np.array(Image.open(p).convert("RGB"))

    elif library == "skimage":

        def loader(p: Path) -> NDArray[np.uint8]:
            return io.imread(p)

    else:
        msg = f"Unsupported library: {library}"
        raise ValueError(msg)

    images = []
    labels = []

    for label_name, label_value in [("REAL", 0), ("FAKE", 1)]:
        class_dir = Path(split_dir) / label_name
        if not class_dir.is_dir():
            msg = f"Directory not found: {class_dir}"
            raise RuntimeError(msg)

        for img_path in tqdm(list(class_dir.glob("*.jpg")), desc=f"Loading {label_name} images"):
            img = loader(img_path)

            if img is None:
                # cv2.imread failure
                continue

            images.append(img)
            labels.append(label_value)

    return images, np.array(labels, dtype=np.uint8)
