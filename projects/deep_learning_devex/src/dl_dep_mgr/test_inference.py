"""Inference test module for validating deployed CLIP model."""

import argparse
import mlflow


def main():
    """Test inference on registered CLIP model."""
    parser = argparse.ArgumentParser(description="Test CLIP model inference")
    parser.add_argument("--catalog", required=True, help="Unity Catalog name")
    parser.add_argument("--schema", required=True, help="Schema name")
    parser.add_argument("--model-name", required=True, help="Model name")
    parser.add_argument("--alias", default="champion", help="Model alias to test")
    args = parser.parse_args()

    model_uri = f"models:/{args.catalog}.{args.schema}.{args.model_name}@{args.alias}"
    print(f"Loading model from: {model_uri}")

    model = mlflow.pyfunc.load_model(model_uri)
    print("Model loaded successfully")

    # Test text embedding
    print("Testing text embedding...")
    test_input = [{"texts": ["a photo of a cat", "a photo of a dog"]}]
    result = model.predict(test_input)

    # Validate output structure
    if isinstance(result, list):
        result = result[0]

    assert "text_embeddings" in result, "Missing text_embeddings in output"
    embeddings = result["text_embeddings"]
    assert len(embeddings) == 2, f"Expected 2 embeddings, got {len(embeddings)}"
    assert len(embeddings[0]) == 512, f"Expected 512-dim embeddings, got {len(embeddings[0])}"

    print(f"Text embeddings shape: {len(embeddings)} x {len(embeddings[0])}")
    print("Inference test PASSED!")


if __name__ == "__main__":
    main()

