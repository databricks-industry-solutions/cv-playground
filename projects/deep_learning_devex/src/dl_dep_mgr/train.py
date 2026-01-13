"""CLIP training entry point with local and distributed modes."""

# Suppress TensorFlow/XLA CUDA warnings (must be before other imports)
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TF warnings
os.environ["CUDA_MODULE_LOADING"] = "LAZY"  # Defer CUDA init

import argparse
import sys
import torch
import torch.nn as nn
import mlflow

import subprocess
from datetime import datetime

from dl_dep_mgr.config import TrainingConfig
from dl_dep_mgr.data import SyntheticCLIPDataset, create_dataloader, create_mds_collate_fn
from dl_dep_mgr.model import create_clip_model, log_clip_model


def save_pip_freeze(output_dir: str, env_name: str = "training") -> str:
    """Save pip freeze output to a file in the specified directory.

    Args:
        output_dir: Directory to save the pip freeze file (e.g. /Volumes/...)
        env_name: Name to identify the environment (e.g. 'training', 'serving')

    Returns:
        Path to the saved file
    """
    result = subprocess.run(["pip", "freeze"], capture_output=True, text=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    freeze_path = f"{output_dir}/pip_freeze_{env_name}_{timestamp}.txt"

    os.makedirs(output_dir, exist_ok=True)
    with open(freeze_path, "w") as f:
        f.write(result.stdout)

    print(f"Saved pip freeze to: {freeze_path}")
    return freeze_path


def train_local(config: TrainingConfig) -> float:
    """Run local training with synthetic data (CPU mode for testing)."""
    print("=== Running Local Training Mode ===")
    print(f"Batch size: {config.local_batch_size}, Epochs: {config.local_num_epochs}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Create model
    model, preprocess, tokenizer = create_clip_model(config.clip_model, config.pretrained)
    model = model.to(device)

    # Create synthetic dataset
    dataset = SyntheticCLIPDataset(num_samples=50, preprocess=preprocess, tokenizer=tokenizer)
    dataloader = create_dataloader(
        dataset, batch_size=config.local_batch_size, num_workers=0, shuffle=True
    )

    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    total_loss = 0.0
    num_batches = 0

    for epoch in range(config.local_num_epochs):
        epoch_loss = 0.0
        for batch_idx, (images, texts) in enumerate(dataloader):
            images = images.to(device)
            texts = texts.to(device)

            optimizer.zero_grad()

            image_features = model.encode_image(images)
            text_features = model.encode_text(texts)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            logit_scale = model.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logits_per_image.t()

            labels = torch.arange(len(images), device=device)
            loss = (loss_fn(logits_per_image, labels) + loss_fn(logits_per_text, labels)) / 2

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")

        avg_epoch_loss = epoch_loss / max(len(dataloader), 1)
        print(f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")
        total_loss = avg_epoch_loss

    # Save model locally for testing
    os.makedirs("./outputs", exist_ok=True)
    save_path = "./outputs/clip_local.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    return total_loss


def train_distributed(config: TrainingConfig) -> float:
    """Run distributed training on Databricks with MDS dataset."""
    # Use patched TorchDistributor for proper multi-node, multi-GPU coordination
    from dl_dep_mgr.distributor import TorchDistributor

    # Validate MDS path exists before starting distributed training
    if not os.path.exists(config.mds_path):
        raise FileNotFoundError(
            f"MDS dataset not found at {config.mds_path}. "
            f"Please create the MDS dataset first or use --local mode for testing."
        )

    ### Move this elsewhere and wrap in a function appropriately as part of an earlier DDL step.
    # Validate model save volume exists before training
    model_volume_path = os.path.dirname(config.model_save_path)
    if not os.path.exists(model_volume_path):
        raise FileNotFoundError(
            f"Model output volume not found at {model_volume_path}. "
            f"Please create the UC volume first:\n"
            f"  CREATE VOLUME {config.catalog}.{config.schema}.{config.model_volume};"
        )

    def train_worker(
        mds_path: str,
        model_name: str,
        pretrained: str,
        batch_size: int,
        num_epochs: int,
        lr: float,
        save_path: str,
        gpus_per_node: int,
    ):
        import os
        import torch
        import torch.nn as nn
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP
        from torch.utils.data import DataLoader
        import open_clip
        from streaming import StreamingDataset

        dist.init_process_group(backend="nccl")

        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

        print(f"[Rank {global_rank}] local_rank={local_rank}, world_size={world_size}")

        # Load model
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        tokenizer = open_clip.get_tokenizer(model_name)
        model = model.to(device)
        model = DDP(model, device_ids=[local_rank])

        # Collate function
        def collate_fn(samples):
            images, texts = [], []
            for sample in samples:
                try:
                    img = sample["image"].convert("RGB")
                    images.append(preprocess(img))
                except Exception:
                    images.append(torch.zeros(3, 224, 224))
                texts.append(tokenizer([sample["caption"]])[0])
            return torch.stack(images), torch.stack(texts)

        # Create streaming dataset
        dataset = StreamingDataset(local=mds_path, shuffle=True, batch_size=batch_size)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        if global_rank == 0:
            print(f"Dataset: {len(dataset)} samples, {len(dataloader)} batches")

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        model.train()
        avg_loss = 0.0

        for epoch in range(num_epochs):
            total_loss, num_batches = 0.0, 0

            for batch_idx, (images, texts) in enumerate(dataloader):
                images = images.to(device, non_blocking=True)
                texts = texts.to(device, non_blocking=True)

                optimizer.zero_grad()

                image_features = model.module.encode_image(images)
                text_features = model.module.encode_text(texts)

                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                logit_scale = model.module.logit_scale.exp()
                logits_per_image = logit_scale * image_features @ text_features.t()
                logits_per_text = logits_per_image.t()

                labels = torch.arange(len(images), device=device)
                loss = (loss_fn(logits_per_image, labels) + loss_fn(logits_per_text, labels)) / 2

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                if global_rank == 0 and (batch_idx + 1) % 50 == 0:
                    print(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")

            avg_loss = total_loss / max(num_batches, 1)
            if global_rank == 0:
                print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")

        # Save from rank 0
        if global_rank == 0:
            torch.save(model.module.state_dict(), save_path)
            print(f"Model saved to {save_path}")

        dist.barrier()
        dist.destroy_process_group()

        return avg_loss if global_rank == 0 else None

    # Run distributed training
    distributor = TorchDistributor(
        num_processes=config.num_processes, local_mode=False, use_gpu=True
    )

    result = distributor.run(
        train_worker,
        config.mds_path,
        config.clip_model,
        config.pretrained,
        config.batch_size,
        config.num_epochs,
        config.learning_rate,
        config.model_save_path,
        config.gpus_per_node,
    )

    return result


def register_model(
    config: TrainingConfig,
    wheel_path: str = None,
    requirements_path: str = None,
    experiment_name: str = None,
) -> str:
    """Register trained model to Unity Catalog with wheel library attached."""
    print("=== Registering model to Unity Catalog ===")
    mlflow.set_registry_uri("databricks-uc")

    # Default experiment name if not provided
    if not experiment_name:
        experiment_name = f"/Users/{os.environ.get('USER', 'default')}/{config.model_name}_training"

    print(f"Using MLflow experiment: {experiment_name}")

    run_id, model_version = log_clip_model(
        model_name=config.clip_model,
        pretrained=config.pretrained,
        save_path=config.model_save_path,
        wheel_path=wheel_path,
        requirements_path=requirements_path,
        registered_model_name=config.uc_model_name,
        experiment_name=experiment_name,
    )

    print(f"Model logged with run_id: {run_id}")

    # Set alias if model was registered
    if model_version:
        client = mlflow.MlflowClient()
        client.set_registered_model_alias(config.uc_model_name, "champion", model_version)
        print(f"Set alias 'champion' to version {model_version}")

    return run_id


def main():
    parser = argparse.ArgumentParser(description="CLIP Training Pipeline")
    parser.add_argument(
        "--local", action="store_true", help="Run in local mode with synthetic data"
    )

    # Output catalog/schema (where models go) - required for distributed mode, validated by config
    parser.add_argument("--catalog", type=str, help="UC catalog for outputs")
    parser.add_argument("--schema", type=str, help="UC schema for outputs")
    parser.add_argument("--model-name", type=str, help="Model name")
    parser.add_argument("--model-volume", type=str, help="Volume name for model outputs")

    # Source data catalog/schema/volume (where MDS dataset lives) - REQUIRED for distributed
    parser.add_argument("--source-catalog", type=str, help="UC catalog for source data")
    parser.add_argument("--source-schema", type=str, help="UC schema for source data")
    parser.add_argument("--source-volume", type=str, help="Volume name for MDS dataset")

    # Training hyperparameters - have sensible defaults
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")

    # Other options
    parser.add_argument(
        "--wheel-path",
        type=str,
        default=None,
        help="Path to wheel file to include with model for serving",
    )
    parser.add_argument(
        "--requirements-path",
        type=str,
        default=None,
        help="Absolute path to requirements.txt for model serving",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="MLflow experiment path (e.g. /Users/user@email.com/my_experiment)",
    )
    parser.add_argument(
        "--skip-register", action="store_true", help="Skip model registration (training only)"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training (register pretrained model directly)",
    )
    parser.add_argument(
        "--logs-volume",
        type=str,
        default=None,
        help="Volume name for logs (used for pip freeze export)",
    )

    args = parser.parse_args()

    config = TrainingConfig(
        catalog=args.catalog,
        schema=args.schema,
        model_name=args.model_name,
        model_volume=args.model_volume,
        source_catalog=args.source_catalog,
        source_schema=args.source_schema,
        source_volume=args.source_volume,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        local_mode=args.local,
    )

    # Export pip freeze for environment auditing
    if args.logs_volume:
        logs_path = f"/Volumes/{args.catalog}/{args.schema}/{args.logs_volume}"
        save_pip_freeze(logs_path, env_name="training")

    if args.skip_training:
        print("=== Skipping training, registering pretrained model ===")
        if not args.skip_register:
            register_model(
                config,
                wheel_path=args.wheel_path,
                requirements_path=args.requirements_path,
                experiment_name=args.experiment,
            )
    elif args.local:
        loss = train_local(config)
        print(f"Local training completed. Final loss: {loss:.4f}")
    else:
        loss = train_distributed(config)
        if loss is not None:
            print(f"Distributed training completed. Final loss: {loss:.4f}")
        else:
            # Loss isn't returned from torchrun subprocesses - it's printed during training
            print("Distributed training completed. (Loss printed during training from rank 0)")

        # Register model to UC (unless skipped)
        if not args.skip_register:
            register_model(
                config,
                wheel_path=args.wheel_path,
                requirements_path=args.requirements_path,
                experiment_name=args.experiment,
            )


if __name__ == "__main__":
    main()
