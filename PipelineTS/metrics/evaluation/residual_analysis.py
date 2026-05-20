"""Residual analysis tools for time series model diagnostics.

Provides residual statistics, normality tests, autocorrelation checks,
and optional plotting. All computations are vectorized with numpy.
"""

import numpy as np

from PipelineTS.utils.native_stats import acf as native_acf
from PipelineTS.utils.native_stats import ljung_box


def _skew_kurtosis(values: np.ndarray, mean: float) -> tuple[float, float]:
    """Return pandas-compatible unbiased skewness and excess kurtosis."""
    n = values.size
    skewness = np.nan
    kurtosis = np.nan

    centered = values - mean
    centered_sq = centered * centered
    m2 = float(np.sum(centered_sq) / n)
    if m2 <= np.finfo(np.float64).eps:
        if n >= 3:
            skewness = 0.0
        if n >= 4:
            kurtosis = 0.0
        return skewness, kurtosis

    if n >= 3:
        m3 = float(np.sum(centered_sq * centered) / n)
        skewness = float(np.sqrt(n * (n - 1)) / (n - 2) * m3 / (m2 ** 1.5))

    if n >= 4:
        m4 = float(np.sum(centered_sq * centered_sq) / n)
        g2 = m4 / (m2 * m2) - 3.0
        kurtosis = float((n - 1) / ((n - 2) * (n - 3)) * ((n + 1) * g2 + 6.0))

    return skewness, kurtosis


class ResidualAnalyzer:
    """Analyze forecast residuals for model diagnostics.

    Parameters
    ----------
    y_true : array-like
        True values.
    y_pred : array-like
        Predicted values.

    Examples
    --------
    >>> analyzer = ResidualAnalyzer(y_true, y_pred)
    >>> stats = analyzer.statistics()
    >>> analyzer.report()
    >>> analyzer.plot()  # requires matplotlib
    """

    def __init__(self, y_true, y_pred):
        self.y_true = np.asarray(y_true, dtype=np.float64)
        self.y_pred = np.asarray(y_pred, dtype=np.float64)
        self.residuals = self.y_true - self.y_pred

    def statistics(self) -> dict:
        """Compute residual statistics.

        Returns
        -------
        dict
            Keys: 'mean', 'std', 'min', 'max', 'median', 'skewness',
            'kurtosis', 'mean_abs', 'rmse'.
        """
        r = self.residuals
        mean = float(np.mean(r))
        skewness, kurtosis = _skew_kurtosis(r, mean)
        return {
            'mean': mean,
            'std': float(np.std(r, ddof=1)) if len(r) > 1 else 0.0,
            'min': float(np.min(r)),
            'max': float(np.max(r)),
            'median': float(np.median(r)),
            'skewness': skewness,
            'kurtosis': kurtosis,
            'mean_abs': float(np.mean(np.abs(r))),
            'rmse': float(np.sqrt(np.dot(r, r) / r.size)),
        }

    def normality_test(self) -> dict:
        """Test residuals for normality using Shapiro-Wilk and Jarque-Bera.

        Returns
        -------
        dict
            Keys: 'shapiro' (dict with statistic, p_value, is_normal),
            'jarque_bera' (dict with statistic, p_value, is_normal).
        """
        from scipy import stats as sp_stats

        result = {}
        r = self.residuals

        # Shapiro-Wilk (limit to 5000 for performance)
        sample = r[:5000] if len(r) > 5000 else r
        if len(sample) >= 3:
            stat, p = sp_stats.shapiro(sample)
            result['shapiro'] = {
                'statistic': float(stat),
                'p_value': float(p),
                'is_normal': p >= 0.05,
            }
        else:
            result['shapiro'] = {'statistic': np.nan, 'p_value': np.nan, 'is_normal': None}

        # Jarque-Bera
        if len(r) >= 8:
            stat, p = sp_stats.jarque_bera(r)
            result['jarque_bera'] = {
                'statistic': float(stat),
                'p_value': float(p),
                'is_normal': p >= 0.05,
            }
        else:
            result['jarque_bera'] = {'statistic': np.nan, 'p_value': np.nan, 'is_normal': None}

        return result

    def autocorrelation(self, max_lags: int = 20) -> dict:
        """Compute residual autocorrelation (ACF) values.

        Parameters
        ----------
        max_lags : int, default=20
            Maximum number of lags to compute.

        Returns
        -------
        dict
            Keys: 'acf_values' (np.ndarray), 'significant_lags' (list of int),
            'ljung_box' (dict with statistic, p_value, has_autocorrelation).
        """
        r = self.residuals
        n_lags = min(max_lags, len(r) // 2 - 1)
        if n_lags < 1:
            return {
                'acf_values': np.array([1.0]),
                'significant_lags': [],
                'ljung_box': {'statistic': np.nan, 'p_value': np.nan,
                              'has_autocorrelation': None},
            }

        acf_vals = native_acf(r, nlags=n_lags, fft=True)

        # Significance bound (approximate 95% CI)
        bound = 1.96 / np.sqrt(len(r))
        significant = [i for i in range(1, len(acf_vals)) if abs(acf_vals[i]) > bound]

        # Ljung-Box test
        lb_result = {'statistic': np.nan, 'p_value': np.nan, 'has_autocorrelation': None}
        if len(r) > n_lags + 1:
            lb_result = ljung_box(r, lags=min(10, n_lags))

        return {
            'acf_values': acf_vals,
            'significant_lags': significant,
            'ljung_box': lb_result,
        }

    def bias_analysis(self) -> dict:
        """Analyze systematic bias in residuals.

        Returns
        -------
        dict
            Keys: 'mean_bias', 'bias_direction', 'bias_significant'
            (one-sample t-test on mean=0), 'pct_positive', 'pct_negative'.
        """
        from scipy import stats as sp_stats

        r = self.residuals
        mean_bias = float(np.mean(r))

        if len(r) >= 3:
            t_stat, p_val = sp_stats.ttest_1samp(r, 0)
            significant = p_val < 0.05
        else:
            p_val = np.nan
            significant = None

        pct_pos = float(np.mean(r > 0))
        pct_neg = float(np.mean(r < 0))

        if mean_bias > 0:
            direction = 'under-predicting (positive residuals dominant)'
        elif mean_bias < 0:
            direction = 'over-predicting (negative residuals dominant)'
        else:
            direction = 'unbiased'

        return {
            'mean_bias': mean_bias,
            'bias_direction': direction,
            'bias_significant': significant,
            'bias_p_value': float(p_val) if not np.isnan(p_val) else None,
            'pct_positive': pct_pos,
            'pct_negative': pct_neg,
        }

    def report(self) -> None:
        """Print a formatted residual analysis report."""
        stats = self.statistics()
        norm = self.normality_test()
        acorr = self.autocorrelation()
        bias = self.bias_analysis()

        print("=" * 50)
        print("  RESIDUAL ANALYSIS REPORT")
        print("=" * 50)

        print(f"\n{'─' * 35}")
        print("  BASIC STATISTICS")
        print(f"{'─' * 35}")
        print(f"  Mean:     {stats['mean']:>12.4f}")
        print(f"  Std:      {stats['std']:>12.4f}")
        print(f"  Median:   {stats['median']:>12.4f}")
        print(f"  Min:      {stats['min']:>12.4f}")
        print(f"  Max:      {stats['max']:>12.4f}")
        print(f"  MAE:      {stats['mean_abs']:>12.4f}")
        print(f"  RMSE:     {stats['rmse']:>12.4f}")
        print(f"  Skew:     {stats['skewness']:>12.4f}")
        print(f"  Kurtosis: {stats['kurtosis']:>12.4f}")

        print(f"\n{'─' * 35}")
        print("  NORMALITY TESTS")
        print(f"{'─' * 35}")
        sh = norm['shapiro']
        print(f"  Shapiro-Wilk:  stat={sh['statistic']:.4f}  p={sh['p_value']:.4f}  "
              f"normal={'Yes' if sh['is_normal'] else 'No'}")
        jb = norm['jarque_bera']
        print(f"  Jarque-Bera:   stat={jb['statistic']:.4f}  p={jb['p_value']:.4f}  "
              f"normal={'Yes' if jb['is_normal'] else 'No'}")

        print(f"\n{'─' * 35}")
        print("  AUTOCORRELATION")
        print(f"{'─' * 35}")
        lb = acorr['ljung_box']
        print(f"  Ljung-Box:     stat={lb['statistic']:.4f}  p={lb['p_value']:.4f}")
        print(f"  Has autocorr:  {lb['has_autocorrelation']}")
        if acorr['significant_lags']:
            print(f"  Sig. lags:     {acorr['significant_lags'][:10]}")

        print(f"\n{'─' * 35}")
        print("  BIAS ANALYSIS")
        print(f"{'─' * 35}")
        print(f"  Mean bias:     {bias['mean_bias']:.4f}")
        print(f"  Direction:     {bias['bias_direction']}")
        print(f"  Significant:   {bias['bias_significant']}")
        print(f"  % positive:    {bias['pct_positive']:.1%}")
        print(f"  % negative:    {bias['pct_negative']:.1%}")

        print(f"\n{'=' * 50}")

    def plot(self, figsize: tuple = (14, 10)) -> None:
        """Plot residual diagnostics (4-panel figure).

        Panels: residuals over time, histogram, Q-Q plot, ACF plot.

        Parameters
        ----------
        figsize : tuple, default=(14, 10)
            Figure size.
        """
        import matplotlib.pyplot as plt
        from scipy import stats as sp_stats

        r = self.residuals
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # 1. Residuals over time
        ax = axes[0, 0]
        ax.plot(r, color='steelblue', linewidth=0.8)
        ax.axhline(0, color='red', linestyle='--', linewidth=0.8)
        ax.set_title('Residuals')
        ax.set_xlabel('Index')
        ax.set_ylabel('Residual')
        ax.grid(True, alpha=0.3)

        # 2. Histogram
        ax = axes[0, 1]
        ax.hist(r, bins=min(50, max(10, len(r) // 5)), density=True,
                color='steelblue', alpha=0.7, edgecolor='white')
        x_range = np.linspace(r.min(), r.max(), 100)
        ax.plot(x_range, sp_stats.norm.pdf(x_range, r.mean(), r.std()),
                'r-', linewidth=1.5, label='Normal fit')
        ax.set_title('Residual Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Q-Q plot
        ax = axes[1, 0]
        sp_stats.probplot(r, dist="norm", plot=ax)
        ax.set_title('Q-Q Plot')
        ax.grid(True, alpha=0.3)

        # 4. ACF plot
        ax = axes[1, 1]
        acf_data = self.autocorrelation()
        acf_vals = acf_data['acf_values']
        lags = np.arange(len(acf_vals))
        bound = 1.96 / np.sqrt(len(r))
        ax.bar(lags, acf_vals, color='steelblue', width=0.3)
        ax.axhline(bound, color='red', linestyle='--', linewidth=0.8)
        ax.axhline(-bound, color='red', linestyle='--', linewidth=0.8)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_title('Autocorrelation (ACF)')
        ax.set_xlabel('Lag')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    # Backward-compatible alias
    print_report = report
