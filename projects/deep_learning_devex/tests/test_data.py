"""Tests for data module."""

import torch
import pytest
from dl_dep_mgr.data import SyntheticCLIPDataset, create_dataloader


class TestSyntheticCLIPDataset:
    def test_dataset_length(self):
        dataset = SyntheticCLIPDataset(num_samples=50)
        assert len(dataset) == 50

    def test_dataset_returns_tuple(self):
        dataset = SyntheticCLIPDataset(num_samples=10)
        image, text = dataset[0]
        assert isinstance(image, torch.Tensor)
        assert isinstance(text, torch.Tensor)

    def test_dataset_image_shape_without_preprocess(self):
        dataset = SyntheticCLIPDataset(num_samples=10)
        image, _ = dataset[0]
        assert image.shape == (3, 224, 224)

    def test_dataset_text_shape_without_tokenizer(self):
        dataset = SyntheticCLIPDataset(num_samples=10)
        _, text = dataset[0]
        assert text.shape == (77,)
        assert text.dtype == torch.long

    def test_dataset_with_preprocess(self):
        def mock_preprocess(img):
            return torch.randn(3, 224, 224)

        dataset = SyntheticCLIPDataset(num_samples=10, preprocess=mock_preprocess)
        image, _ = dataset[0]
        assert image.shape == (3, 224, 224)

    def test_dataset_cycles_captions(self):
        dataset = SyntheticCLIPDataset(num_samples=10)
        # Access indices beyond caption list length to test cycling
        _ = dataset[0]
        _ = dataset[5]
        _ = dataset[9]


class TestDataLoader:
    def test_create_dataloader(self):
        dataset = SyntheticCLIPDataset(num_samples=20)
        dataloader = create_dataloader(dataset, batch_size=4, num_workers=0)

        batch = next(iter(dataloader))
        images, texts = batch
        assert images.shape[0] == 4
        assert texts.shape[0] == 4

    def test_dataloader_batches(self):
        dataset = SyntheticCLIPDataset(num_samples=10)
        dataloader = create_dataloader(dataset, batch_size=3, num_workers=0)

        batches = list(dataloader)
        # 10 samples / 3 batch_size = 4 batches (3, 3, 3, 1)
        assert len(batches) == 4

