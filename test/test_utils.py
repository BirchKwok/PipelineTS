"""
Comprehensive test suite for utility functions in PipelineTS.

Tests:
- update_dict_without_conflict
- check_time_col_is_timestamp
- compute_time_interval
- time_diff
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ─── update_dict_without_conflict ─────────────────────────────────────────────

class TestUpdateDictWithoutConflict:
    def test_no_conflict(self):
        from PipelineTS.utils import update_dict_without_conflict
        a = {'x': 1}
        b = {'y': 2}
        result = update_dict_without_conflict(a, b)
        assert result == {'x': 1, 'y': 2}

    def test_with_conflict(self):
        from PipelineTS.utils import update_dict_without_conflict
        a = {'x': 1}
        b = {'x': 99, 'y': 2}
        result = update_dict_without_conflict(a, b)
        assert result['x'] == 1, "Should not overwrite existing keys"
        assert result['y'] == 2

    def test_empty_dicts(self):
        from PipelineTS.utils import update_dict_without_conflict
        result = update_dict_without_conflict({}, {})
        assert result == {}

    def test_empty_a(self):
        from PipelineTS.utils import update_dict_without_conflict
        result = update_dict_without_conflict({}, {'a': 1})
        assert result == {'a': 1}

    def test_empty_b(self):
        from PipelineTS.utils import update_dict_without_conflict
        result = update_dict_without_conflict({'a': 1}, {})
        assert result == {'a': 1}


# ─── check_time_col_is_timestamp ──────────────────────────────────────────────

class TestCheckTimeColIsTimestamp:
    def test_valid_timestamp(self):
        from PipelineTS.utils import check_time_col_is_timestamp
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=5, freq='D'),
            'value': [1, 2, 3, 4, 5]
        })
        check_time_col_is_timestamp(df, 'date')

    def test_invalid_timestamp_raises(self):
        from PipelineTS.utils import check_time_col_is_timestamp
        df = pd.DataFrame({
            'date': ['2020-01-01', '2020-01-02', '2020-01-03'],
            'value': [1, 2, 3]
        })
        with pytest.raises(TypeError):
            check_time_col_is_timestamp(df, 'date')


# ─── compute_time_interval ────────────────────────────────────────────────────

class TestComputeTimeInterval:
    def test_daily_interval(self):
        from PipelineTS.utils import compute_time_interval
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=10, freq='D')
        })
        interval = compute_time_interval(df, 'date')
        assert interval == pd.Timedelta('1D')

    def test_hourly_interval(self):
        from PipelineTS.utils import compute_time_interval
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=10, freq='h')
        })
        interval = compute_time_interval(df, 'date')
        assert interval == pd.Timedelta('1h')


# ─── time_diff ────────────────────────────────────────────────────────────────

class TestTimeDiff:
    def test_time_diff(self):
        from PipelineTS.utils import time_diff
        a = pd.Timestamp('2020-01-05')
        b = pd.Timestamp('2020-01-01')
        result = time_diff(a, b)
        assert result == np.timedelta64(4, 'D')

    def test_time_diff_negative(self):
        from PipelineTS.utils import time_diff
        a = pd.Timestamp('2020-01-01')
        b = pd.Timestamp('2020-01-05')
        result = time_diff(a, b)
        assert result == np.timedelta64(-4, 'D')


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '-x'])
