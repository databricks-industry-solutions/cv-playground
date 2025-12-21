"""Dataset utilities for CLIP training."""

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from typing import Callable, Optional, List, Tuple
import io


class SyntheticCLIPDataset(Dataset):
    """Synthetic dataset for local testing without real data."""

    def __init__(
        self,
        num_samples: int = 100,
        preprocess: Optional[Callable] = None,
        tokenizer: Optional[Callable] = None,
    ):
        self.num_samples = num_samples
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.captions = [
            "A photo of a dog running in the park",
            "A beautiful sunset over the ocean",
            "A cat sleeping on a couch",
            "People walking in a busy city street",
            "A mountain landscape with snow",
        ]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Generate random RGB image
        img_array = torch.randint(0, 255, (224, 224, 3), dtype=torch.uint8).numpy()
        image = Image.fromarray(img_array, mode="RGB")

        if self.preprocess:
            image = self.preprocess(image)
        else:
            image = torch.randn(3, 224, 224)

        caption = self.captions[idx % len(self.captions)]
        if self.tokenizer:
            text = self.tokenizer([caption])[0]
        else:
            text = torch.zeros(77, dtype=torch.long)

        return image, text


def create_mds_collate_fn(preprocess: Callable, tokenizer: Callable) -> Callable:
    """Create collate function for MDS dataset samples."""

    def collate_fn(samples: List[dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        images = []
        texts = []

        for sample in samples:
            try:
                img = sample["image"].convert("RGB")
                img_tensor = preprocess(img)
                images.append(img_tensor)
            except Exception:
                images.append(torch.zeros(3, 224, 224))

            caption = sample["caption"]
            text_tensor = tokenizer([caption])[0]
            texts.append(text_tensor)

        return torch.stack(images), torch.stack(texts)

    return collate_fn


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int = 4,
    collate_fn: Optional[Callable] = None,
    shuffle: bool = True,
) -> DataLoader:
    """Create DataLoader with optimal settings for training."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        collate_fn=collate_fn,
        shuffle=shuffle,
    )

