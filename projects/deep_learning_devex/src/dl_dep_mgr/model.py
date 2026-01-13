"""CLIP model wrapper for MLflow logging and serving."""

import torch
import mlflow
from mlflow.models.signature import ModelSignature
from mlflow.types import ColSpec, Schema
from typing import Any, Optional, Tuple
import open_clip


class CLIPModelWrapper(mlflow.pyfunc.PythonModel):
    """MLflow wrapper for CLIP model serving."""

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """Load model artifacts."""
        import open_clip
        import torch

        # Load model config
        config = context.model_config
        model_name = config.get("model_name", "ViT-B-32-quickgelu")
        pretrained = config.get("pretrained", "openai")

        # Create model and load weights
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)

        # Load fine-tuned weights if available
        weights_path = context.artifacts.get("model_weights")
        if weights_path:
            state_dict = torch.load(weights_path, map_location="cpu")
            self.model.load_state_dict(state_dict)

        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

        # Log pip freeze for serving environment audit (appears in model serving logs)
        import subprocess

        result = subprocess.run(["pip", "freeze"], capture_output=True, text=True)
        print("=== MODEL SERVING PIP FREEZE ===")
        print(result.stdout)
        print("=== END PIP FREEZE ===")

    def predict(self, context, model_input):
        """Generate embeddings for images and/or text."""
        import torch
        from PIL import Image
        import base64
        import io
        import pandas as pd

        # Handle DataFrame input (MLflow converts list of dicts to DataFrame)
        if isinstance(model_input, pd.DataFrame):
            single_input = model_input.iloc[0].to_dict()
        elif isinstance(model_input, list):
            single_input = model_input[0]
        else:
            single_input = model_input

        results = {}

        def get_list(value):
            """Convert value to list, handling pandas Series and None."""
            if value is None:
                return []
            if isinstance(value, pd.Series):
                return value.tolist()
            if isinstance(value, list):
                return value
            return [value]

        with torch.no_grad():
            # Process images if provided
            images = get_list(single_input.get("images"))
            if images:
                image_tensors = []
                for img_data in images:
                    if isinstance(img_data, str):
                        img_bytes = base64.b64decode(img_data)
                        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    else:
                        img = img_data
                    image_tensors.append(self.preprocess(img))

                image_batch = torch.stack(image_tensors).to(self.device)
                image_features = self.model.encode_image(image_batch)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                results["image_embeddings"] = image_features.cpu().numpy().tolist()

            # Process text if provided
            texts = get_list(single_input.get("texts"))
            if texts:
                text_tokens = self.tokenizer(texts).to(self.device)
                text_features = self.model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                results["text_embeddings"] = text_features.cpu().numpy().tolist()

            # Compute similarities if both provided
            if "image_embeddings" in results and "text_embeddings" in results:
                img_emb = torch.tensor(results["image_embeddings"])
                txt_emb = torch.tensor(results["text_embeddings"])
                similarities = (img_emb @ txt_emb.T).numpy().tolist()
                results["similarities"] = similarities

        return [results]


def create_clip_model(
    model_name: str = "ViT-B-32-quickgelu", pretrained: str = "openai"
) -> Tuple[Any, Any, Any]:
    """Create CLIP model with transforms and tokenizer."""
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, preprocess, tokenizer


def log_clip_model(
    model_name: str,
    pretrained: str,
    save_path: Optional[str] = None,
    wheel_path: Optional[str] = None,
    requirements_path: Optional[str] = None,
    registered_model_name: Optional[str] = None,
    experiment_name: Optional[str] = None,
) -> Tuple[str, str]:
    """Log CLIP model to MLflow, add wheel library, and register to UC.

    Args:
        save_path: Path to fine-tuned weights. If None, uses pretrained weights.
        wheel_path: Path to wheel file to include with model.
        requirements_path: Absolute path to requirements.txt for pip dependencies.
        experiment_name: MLflow experiment name. Required on Databricks.

    Returns:
        Tuple of (run_id, model_version)
    """
    import os

    # Set experiment (required on Databricks)
    if experiment_name:
        mlflow.set_experiment(experiment_name)

    model_config = {"model_name": model_name, "pretrained": pretrained}

    # Only include artifacts if save_path exists
    artifacts = {}
    if save_path and os.path.exists(save_path):
        artifacts = {"model_weights": save_path}

    # Require pinned dependencies from requirements.txt - no fallback
    if not requirements_path:
        raise ValueError(
            "requirements_path is required - all dependencies must be pinned. "
            "Pass --requirements-path with absolute path to requirements.txt"
        )
    print(f"Using pinned requirements from: {requirements_path}")
    pip_requirements = [f"-r {requirements_path}"]

    # Include wheel directly in pip_requirements
    if wheel_path:
        print(f"Including wheel in model requirements: {wheel_path}")
        pip_requirements.append(wheel_path)

    # Define signature for CLIP model (required for Unity Catalog)
    # Mark images as optional since model can process text-only or image-only
    input_schema = Schema(
        [
            ColSpec("string", "images", required=False),  # Base64-encoded images (optional)
            ColSpec("string", "texts", required=False),  # Text strings (optional)
        ]
    )
    output_schema = Schema(
        [
            ColSpec("double", "image_embeddings", required=False),
            ColSpec("double", "text_embeddings", required=False),
            ColSpec("double", "similarities", required=False),
        ]
    )
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)

    with mlflow.start_run() as run:
        mlflow.log_params(model_config)

        # Input example for signature validation (text-only to avoid torch.stack on empty list)
        input_example = [{"texts": ["example text"]}]

        model_info = mlflow.pyfunc.log_model(
            name="model",
            python_model=CLIPModelWrapper(),
            artifacts=artifacts if artifacts else None,
            model_config=model_config,
            pip_requirements=pip_requirements,
            signature=signature,
            input_example=input_example,
        )

        model_uri = model_info.model_uri
        run_id = run.info.run_id

        model_version = None
        if registered_model_name:
            print(f"Registering model to: {registered_model_name}")
            registered_model = mlflow.register_model(
                model_uri=model_uri,
                name=registered_model_name,
            )
            initial_version = registered_model.version

            # Pre-package all dependencies as wheel files for serving
            # This creates a NEW version with wheels pre-packaged
            registered_model_uri = f"models:/{registered_model_name}/{initial_version}"
            print(f"Packaging dependencies with add_libraries_to_model: {registered_model_uri}")
            wheeled_model_info = mlflow.models.utils.add_libraries_to_model(registered_model_uri)
            # Use the wheeled version (the one with dependencies packaged)
            # add_libraries_to_model returns ModelInfo, version is in registered_model_version
            model_version = wheeled_model_info.registered_model_version
            print(f"Created wheeled model version: {model_version}")

        return run_id, model_version
