"""Tests for model module."""

import torch
import pytest
from unittest.mock import MagicMock, patch


class TestCreateClipModel:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_create_model_cuda(self):
        from dl_dep_mgr.model import create_clip_model

        model, preprocess, tokenizer = create_clip_model("ViT-B-32", "openai")
        assert model is not None
        assert preprocess is not None
        assert tokenizer is not None

    def test_create_model_cpu(self):
        """Test model creation works on CPU."""
        pytest.importorskip("open_clip")
        from dl_dep_mgr.model import create_clip_model

        model, preprocess, tokenizer = create_clip_model("ViT-B-32", "openai")
        assert model is not None
        assert callable(preprocess)
        assert callable(tokenizer)


class TestCLIPModelWrapper:
    def test_wrapper_init(self):
        from dl_dep_mgr.model import CLIPModelWrapper

        wrapper = CLIPModelWrapper()
        assert wrapper is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_wrapper_load_context(self):
        from dl_dep_mgr.model import CLIPModelWrapper

        wrapper = CLIPModelWrapper()

        # Mock context
        context = MagicMock()
        context.model_config = {"model_name": "ViT-B-32", "pretrained": "openai"}
        context.artifacts = {}

        wrapper.load_context(context)

        assert wrapper.model is not None
        assert wrapper.preprocess is not None
        assert wrapper.tokenizer is not None


class TestModelForward:
    @pytest.fixture
    def model_and_transforms(self):
        pytest.importorskip("open_clip")
        from dl_dep_mgr.model import create_clip_model

        return create_clip_model("ViT-B-32", "openai")

    def test_encode_image(self, model_and_transforms):
        model, preprocess, _ = model_and_transforms
        model.eval()

        # Create dummy image tensor
        dummy_image = torch.randn(1, 3, 224, 224)

        with torch.no_grad():
            features = model.encode_image(dummy_image)

        assert features.shape == (1, 512)  # ViT-B-32 embedding dim

    def test_encode_text(self, model_and_transforms):
        model, _, tokenizer = model_and_transforms
        model.eval()

        # Tokenize text
        text_tokens = tokenizer(["a photo of a cat"])

        with torch.no_grad():
            features = model.encode_text(text_tokens)

        assert features.shape == (1, 512)  # ViT-B-32 embedding dim

    def test_embeddings_normalized(self, model_and_transforms):
        model, _, tokenizer = model_and_transforms
        model.eval()

        text_tokens = tokenizer(["test text"])

        with torch.no_grad():
            features = model.encode_text(text_tokens)
            normalized = features / features.norm(dim=-1, keepdim=True)

        # Check that normalization produces unit vectors
        norms = normalized.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

