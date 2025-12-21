"""Tests for configuration module."""

import os
import pytest
from dl_dep_mgr.config import TrainingConfig, ServingConfig


class TestTrainingConfig:
    def test_default_values(self):
        config = TrainingConfig()
        assert config.catalog == "main"
        assert config.schema == "dl_dep_mgr"
        assert config.clip_model == "ViT-B-32"
        assert config.batch_size == 32
        assert config.num_epochs == 3

    def test_mds_path(self):
        config = TrainingConfig(catalog="test_cat", schema="test_schema", mds_volume_name="mds")
        assert config.mds_path == "/Volumes/test_cat/test_schema/mds"

    def test_uc_model_name(self):
        config = TrainingConfig(catalog="cat", schema="sch", model_name="my_model")
        assert config.uc_model_name == "cat.sch.my_model"

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("CATALOG", "env_catalog")
        monkeypatch.setenv("SCHEMA", "env_schema")
        monkeypatch.setenv("BATCH_SIZE", "64")
        monkeypatch.setenv("LOCAL_MODE", "true")

        config = TrainingConfig.from_env()
        assert config.catalog == "env_catalog"
        assert config.schema == "env_schema"
        assert config.batch_size == 64
        assert config.local_mode is True


class TestServingConfig:
    def test_default_values(self):
        config = ServingConfig()
        assert config.catalog == "main"
        assert config.model_alias == "champion"
        assert config.scale_to_zero is True

    def test_serving_endpoint_name_default(self):
        config = ServingConfig(catalog="cat", schema="sch", model_name="model")
        assert config.serving_endpoint_name == "clip_cat_sch_model"

    def test_serving_endpoint_name_override(self):
        config = ServingConfig(endpoint_name="custom_endpoint")
        assert config.serving_endpoint_name == "custom_endpoint"

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("CATALOG", "env_cat")
        monkeypatch.setenv("MODEL_ALIAS", "staging")
        monkeypatch.setenv("SCALE_TO_ZERO", "false")

        config = ServingConfig.from_env()
        assert config.catalog == "env_cat"
        assert config.model_alias == "staging"
        assert config.scale_to_zero is False

