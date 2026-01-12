"""Tests for configuration module."""

import pytest
from dl_dep_mgr.config import TrainingConfig, ServingConfig


class TestTrainingConfig:
    def test_default_values(self):
        config = TrainingConfig()
        assert config.catalog is None
        assert config.schema is None
        assert config.clip_model == "ViT-B-32-quickgelu"
        assert config.batch_size == 32
        assert config.num_epochs == 3

    def test_mds_path(self):
        config = TrainingConfig(
            source_catalog="test_cat", source_schema="test_schema", source_volume="mds"
        )
        assert config.mds_path == "/Volumes/test_cat/test_schema/mds"

    def test_uc_model_name(self):
        config = TrainingConfig(catalog="cat", schema="sch", model_name="my_model")
        assert config.uc_model_name == "cat.sch.my_model"

    def test_validate_output_paths_missing(self):
        config = TrainingConfig()
        with pytest.raises(ValueError, match="Required parameter 'catalog'"):
            config.validate_output_paths()

    def test_validate_source_paths_missing(self):
        config = TrainingConfig()
        with pytest.raises(ValueError, match="Required parameter 'source_catalog'"):
            config.validate_source_paths()


class TestServingConfig:
    def test_default_values(self):
        config = ServingConfig()
        assert config.catalog is None
        assert config.model_alias == "champion"
        assert config.scale_to_zero is True

    def test_serving_endpoint_name_default(self):
        config = ServingConfig(catalog="cat", schema="sch", model_name="model")
        assert config.serving_endpoint_name == "clip_cat_sch_model"

    def test_serving_endpoint_name_override(self):
        config = ServingConfig(
            catalog="cat", schema="sch", model_name="model", endpoint_name="custom_endpoint"
        )
        assert config.serving_endpoint_name == "custom_endpoint"

    def test_validate_missing(self):
        config = ServingConfig()
        with pytest.raises(ValueError, match="Required parameter 'catalog'"):
            config.validate()
