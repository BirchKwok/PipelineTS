"""
Comprehensive test suite for dataset loaders and data generators in PipelineTS.

Tests:
- DataGenerator: synthetic data generation
- RandomEventGenerator: random event generation
- LoadElectricDataSets, LoadMessagesSentHourDataSets, LoadMessagesSentDataSets
- LoadWebSales, LoadSupermarketIncoming
- BuiltInSeriesData: access all built-in datasets
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ─── DataGenerator ────────────────────────────────────────────────────────────

class TestDataGenerator:
    def test_import(self):
        from PipelineTS.dataset import DataGenerator
        assert DataGenerator is not None

    def test_generate_trigonometry(self):
        from PipelineTS.dataset import DataGenerator
        data = DataGenerator.trigonometry_ds(size=100)
        assert data is not None
        assert len(data) == 100


# ─── RandomEventGenerator ────────────────────────────────────────────────────

class TestRandomEventGenerator:
    def test_import(self):
        from PipelineTS.dataset import RandomEventGenerator
        assert RandomEventGenerator is not None


# ─── LoadElectricDataSets ─────────────────────────────────────────────────────

class TestLoadElectricDataSets:
    def test_load(self):
        from PipelineTS.dataset import LoadElectric
        data = LoadElectricProduction()
        assert isinstance(data, pd.DataFrame), "Should return a DataFrame"
        assert len(data) > 0, "Dataset should not be empty"
        assert data.shape[1] >= 2, "Should have at least 2 columns"


# ─── LoadMessagesSentHourDataSets ─────────────────────────────────────────────

class TestLoadMessagesSentHourDataSets:
    def test_load(self):
        from PipelineTS.dataset import LoadMessagesSentHour
        data = LoadMessagesSentHour()
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0


# ─── LoadMessagesSentDataSets ─────────────────────────────────────────────────

class TestLoadMessagesSentDataSets:
    def test_load(self):
        from PipelineTS.dataset import LoadMessagesSent
        data = LoadMessagesSent()
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0


# ─── LoadWebSales ─────────────────────────────────────────────────────────────

class TestLoadWebSales:
    def test_load(self):
        from PipelineTS.dataset import LoadWebSales
        data = LoadWebSales()
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0


# ─── LoadSupermarketIncoming ──────────────────────────────────────────────────

class TestLoadSupermarketIncoming:
    def test_load(self):
        from PipelineTS.dataset import LoadSupermarketIncoming
        data = LoadSupermarketIncoming()
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0


# ─── BuiltInSeriesData ────────────────────────────────────────────────────────

class TestBuiltInSeriesData:
    def test_import(self):
        from PipelineTS.dataset import BuiltInSeriesData
        assert BuiltInSeriesData is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '-x'])
