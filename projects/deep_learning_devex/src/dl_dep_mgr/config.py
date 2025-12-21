"""Configuration management for CLIP training pipeline.

All catalog/schema/path values must be provided via CLI arguments.
Defaults are defined in the Databricks Asset Bundle (databricks.yml), not here.
"""

from dataclasses import dataclass
from typing import Optional


def _require(value: Optional[str], name: str) -> str:
    """Require a value to be provided, raise error if missing."""
    if value is None:
        raise ValueError(
            f"Required parameter '{name}' was not provided. "
            f"Please pass --{name.replace('_', '-')} or set in databricks.yml"
        )
    return value


@dataclass
class TrainingConfig:
    """Configuration for CLIP training.

    All UC path parameters are required and must be provided via CLI.
    Defaults are managed in databricks.yml, not in Python code.
    """

    # Output UC paths (where models and logs go) - REQUIRED
    catalog: Optional[str] = None
    schema: Optional[str] = None
    model_name: Optional[str] = None
    model_volume: Optional[str] = None

    # Source data UC paths (where MDS dataset lives) - REQUIRED for distributed training
    source_catalog: Optional[str] = None
    source_schema: Optional[str] = None
    source_volume: Optional[str] = None

    # Model config - these have sensible defaults
    clip_model: str = "ViT-B-32-quickgelu"
    pretrained: str = "openai"

    # Training hyperparams - these have sensible defaults
    batch_size: int = 32
    num_epochs: int = 3
    learning_rate: float = 1e-5
    num_workers: int = 4

    # Distributed training - these have sensible defaults
    num_processes: int = 8
    gpus_per_node: int = 4

    # Local mode settings
    local_mode: bool = False
    local_batch_size: int = 2
    local_num_epochs: int = 1

    def validate_output_paths(self) -> None:
        """Validate that output path parameters are provided."""
        _require(self.catalog, "catalog")
        _require(self.schema, "schema")
        _require(self.model_name, "model_name")
        _require(self.model_volume, "model_volume")

    def validate_source_paths(self) -> None:
        """Validate that source data path parameters are provided."""
        _require(self.source_catalog, "source_catalog")
        _require(self.source_schema, "source_schema")
        _require(self.source_volume, "source_volume")

    @property
    def mds_path(self) -> str:
        """Path to source MDS dataset."""
        self.validate_source_paths()
        return f"/Volumes/{self.source_catalog}/{self.source_schema}/{self.source_volume}"

    @property
    def model_save_path(self) -> str:
        """Path to save trained model weights."""
        self.validate_output_paths()
        return f"/Volumes/{self.catalog}/{self.schema}/{self.model_volume}/{self.model_name}.pt"

    @property
    def uc_model_name(self) -> str:
        """Full Unity Catalog model name."""
        _require(self.catalog, "catalog")
        _require(self.schema, "schema")
        _require(self.model_name, "model_name")
        return f"{self.catalog}.{self.schema}.{self.model_name}"


@dataclass
class ServingConfig:
    """Configuration for model serving deployment.

    All UC path parameters are required and must be provided via CLI.
    Defaults are managed in databricks.yml, not in Python code.
    """

    # UC paths - REQUIRED
    catalog: Optional[str] = None
    schema: Optional[str] = None
    model_name: Optional[str] = None

    # Serving config - these have sensible defaults
    model_alias: str = "champion"
    endpoint_name: Optional[str] = None
    scale_to_zero: bool = True
    workload_size: str = "Small"

    def validate(self) -> None:
        """Validate that required parameters are provided."""
        _require(self.catalog, "catalog")
        _require(self.schema, "schema")
        _require(self.model_name, "model_name")

    @property
    def uc_model_name(self) -> str:
        self.validate()
        return f"{self.catalog}.{self.schema}.{self.model_name}"

    @property
    def serving_endpoint_name(self) -> str:
        if self.endpoint_name:
            return self.endpoint_name
        self.validate()
        return f"clip_{self.catalog}_{self.schema}_{self.model_name}"
