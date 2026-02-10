"""Data quality report and exploratory analysis for time series data.

Generates a concise summary of data health, statistics, and potential issues.
All computations are vectorized for performance.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union


class TimeSeriesDataQualityReport:
    """Generate a comprehensive data quality report for time series data.

    Parameters
    ----------
    time_col : str
        Name of the datetime column.
    target_col : str or list of str
        Name(s) of the target column(s).

    Examples
    --------
    >>> report = TimeSeriesDataQualityReport(time_col='date', target_col='value')
    >>> summary = report.fit(df)
    >>> report.report(df)
    """

    def __init__(self, time_col: str, target_col: Union[str, list]):
        self.time_col = time_col
        if isinstance(target_col, str):
            self.target_cols = [target_col]
        else:
            self.target_cols = list(target_col)

    def fit(self, data: pd.DataFrame) -> dict:
        """Generate a data quality report.

        Parameters
        ----------
        data : pd.DataFrame
            Input time series data.

        Returns
        -------
        dict
            Report with sections: 'overview', 'time_analysis', 'value_analysis',
            'missing_analysis', 'issues'.
        """
        report = {}
        report['overview'] = self._overview(data)
        report['time_analysis'] = self._time_analysis(data)
        report['value_analysis'] = self._value_analysis(data)
        report['missing_analysis'] = self._missing_analysis(data)
        report['issues'] = self._detect_issues(data, report)
        return report

    def report(self, data: pd.DataFrame) -> None:
        """Print a formatted data quality report.

        Parameters
        ----------
        data : pd.DataFrame
            Input time series data.
        """
        report = self.fit(data)

        print("=" * 60)
        print("  TIME SERIES DATA QUALITY REPORT")
        print("=" * 60)

        # Overview
        ov = report['overview']
        print(f"\n{'─' * 40}")
        print("  OVERVIEW")
        print(f"{'─' * 40}")
        print(f"  Rows:           {ov['n_rows']}")
        print(f"  Columns:        {ov['n_columns']}")
        print(f"  Time column:    {ov['time_col']}")
        print(f"  Target col(s):  {', '.join(ov['target_cols'])}")
        print(f"  Time range:     {ov['time_start']} → {ov['time_end']}")
        print(f"  Duration:       {ov['duration']}")
        print(f"  Memory:         {ov['memory_mb']:.2f} MB")

        # Time Analysis
        ta = report['time_analysis']
        print(f"\n{'─' * 40}")
        print("  TIME ANALYSIS")
        print(f"{'─' * 40}")
        print(f"  Inferred freq:  {ta['inferred_freq']}")
        print(f"  Is regular:     {ta['is_regular']}")
        print(f"  Duplicates:     {ta['n_duplicate_timestamps']}")
        if not ta['is_regular']:
            print(f"  Min gap:        {ta['min_gap']}")
            print(f"  Max gap:        {ta['max_gap']}")
            print(f"  Median gap:     {ta['median_gap']}")

        # Value Analysis
        va = report['value_analysis']
        print(f"\n{'─' * 40}")
        print("  VALUE ANALYSIS")
        print(f"{'─' * 40}")
        for col, stats in va.items():
            print(f"\n  [{col}]")
            print(f"    mean={stats['mean']:.4g}  std={stats['std']:.4g}  "
                  f"min={stats['min']:.4g}  max={stats['max']:.4g}")
            print(f"    median={stats['median']:.4g}  skew={stats['skewness']:.4g}  "
                  f"kurtosis={stats['kurtosis']:.4g}")
            print(f"    zeros={stats['n_zeros']}  negatives={stats['n_negatives']}  "
                  f"infs={stats['n_infs']}")

        # Missing Analysis
        ma = report['missing_analysis']
        print(f"\n{'─' * 40}")
        print("  MISSING ANALYSIS")
        print(f"{'─' * 40}")
        print(f"  Implicit gaps:       {ma['n_implicit_gaps']}")
        print(f"  Completeness ratio:  {ma['completeness_ratio']:.2%}")
        if ma['explicit_nan']:
            for col, n in ma['explicit_nan'].items():
                print(f"  NaN in '{col}':       {n} ({n / len(data):.1%})")
        else:
            print("  Explicit NaN:        None")

        # Issues
        issues = report['issues']
        print(f"\n{'─' * 40}")
        print("  ISSUES DETECTED")
        print(f"{'─' * 40}")
        if issues:
            for i, issue in enumerate(issues, 1):
                print(f"  [{issue['severity']}] {issue['message']}")
        else:
            print("  No issues detected.")

        print(f"\n{'=' * 60}")

    def _overview(self, data: pd.DataFrame) -> dict:
        ts = data[self.time_col]
        return {
            'n_rows': len(data),
            'n_columns': len(data.columns),
            'time_col': self.time_col,
            'target_cols': self.target_cols,
            'time_start': ts.min(),
            'time_end': ts.max(),
            'duration': ts.max() - ts.min(),
            'memory_mb': data.memory_usage(deep=True).sum() / 1024 / 1024,
        }

    def _time_analysis(self, data: pd.DataFrame) -> dict:
        ts = data[self.time_col].sort_values()
        diffs = ts.diff().dropna()

        if len(diffs) == 0:
            return {
                'inferred_freq': None,
                'is_regular': True,
                'n_duplicate_timestamps': 0,
                'min_gap': None,
                'max_gap': None,
                'median_gap': None,
            }

        mode_diff = diffs.mode().iloc[0]
        try:
            freq = pd.tseries.frequencies.to_offset(mode_diff)
        except Exception:
            freq = str(mode_diff)

        return {
            'inferred_freq': str(freq),
            'is_regular': bool((diffs == mode_diff).all()),
            'n_duplicate_timestamps': int(ts.duplicated().sum()),
            'min_gap': diffs.min(),
            'max_gap': diffs.max(),
            'median_gap': diffs.median(),
        }

    def _value_analysis(self, data: pd.DataFrame) -> dict:
        result = {}
        for col in self.target_cols:
            if col not in data.columns:
                continue
            values = data[col].values.astype(np.float64)
            valid = values[~np.isnan(values)]
            if len(valid) == 0:
                result[col] = {k: 0 for k in ['mean', 'std', 'min', 'max', 'median',
                                                'skewness', 'kurtosis', 'n_zeros',
                                                'n_negatives', 'n_infs']}
                continue

            result[col] = {
                'mean': float(np.mean(valid)),
                'std': float(np.std(valid)),
                'min': float(np.min(valid)),
                'max': float(np.max(valid)),
                'median': float(np.median(valid)),
                'skewness': float(pd.Series(valid).skew()),
                'kurtosis': float(pd.Series(valid).kurtosis()),
                'n_zeros': int(np.sum(valid == 0)),
                'n_negatives': int(np.sum(valid < 0)),
                'n_infs': int(np.sum(~np.isfinite(values))),
            }
        return result

    def _missing_analysis(self, data: pd.DataFrame) -> dict:
        ts = data[self.time_col].sort_values()
        diffs = ts.diff().dropna()

        if len(diffs) == 0:
            return {
                'n_implicit_gaps': 0,
                'completeness_ratio': 1.0,
                'explicit_nan': {},
            }

        mode_diff = diffs.mode().iloc[0]
        try:
            freq = pd.tseries.frequencies.to_offset(mode_diff)
            full_range = pd.date_range(start=ts.min(), end=ts.max(), freq=freq)
            n_implicit = len(full_range) - len(data)
            completeness = len(data) / len(full_range) if len(full_range) > 0 else 1.0
        except Exception:
            n_implicit = 0
            completeness = 1.0

        nan_counts = {}
        for col in self.target_cols:
            if col in data.columns:
                n = int(data[col].isna().sum())
                if n > 0:
                    nan_counts[col] = n

        return {
            'n_implicit_gaps': max(0, n_implicit),
            'completeness_ratio': completeness,
            'explicit_nan': nan_counts,
        }

    def _detect_issues(self, data: pd.DataFrame, report: dict) -> list:
        issues = []

        # Duplicate timestamps
        n_dup = report['time_analysis']['n_duplicate_timestamps']
        if n_dup > 0:
            issues.append({
                'severity': 'WARNING',
                'message': f'{n_dup} duplicate timestamps detected.',
            })

        # Irregular frequency
        if not report['time_analysis']['is_regular']:
            issues.append({
                'severity': 'INFO',
                'message': 'Time series has irregular frequency. Consider resampling.',
            })

        # Low completeness
        cr = report['missing_analysis']['completeness_ratio']
        if cr < 0.95:
            issues.append({
                'severity': 'WARNING',
                'message': f'Data completeness is {cr:.1%}. Consider filling gaps.',
            })

        # Explicit NaN
        for col, n in report['missing_analysis']['explicit_nan'].items():
            pct = n / len(data)
            severity = 'ERROR' if pct > 0.1 else 'WARNING'
            issues.append({
                'severity': severity,
                'message': f"Column '{col}' has {n} NaN values ({pct:.1%}).",
            })

        # Extreme skewness
        for col, stats in report['value_analysis'].items():
            if abs(stats['skewness']) > 2:
                issues.append({
                    'severity': 'INFO',
                    'message': f"Column '{col}' is highly skewed ({stats['skewness']:.2f}). "
                               f"Consider log or Box-Cox transform.",
                })
            if stats['n_infs'] > 0:
                issues.append({
                    'severity': 'ERROR',
                    'message': f"Column '{col}' contains {stats['n_infs']} infinite values.",
                })

        # Very short series
        if len(data) < 30:
            issues.append({
                'severity': 'WARNING',
                'message': f'Series has only {len(data)} data points. '
                           f'Many models need more data for reliable results.',
            })

        return issues

    # Backward-compatible aliases
    generate = fit
    print_report = report
