import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.signal import welch

class TimeSeriesFeatureExtractor:
    """用于时间序列特征提取的工具类"""
    
    @staticmethod
    def add_time_features(df, time_col):
        """添加时间特征"""
        df = df.copy()
        df['hour'] = df[time_col].dt.hour
        df['day'] = df[time_col].dt.day
        df['month'] = df[time_col].dt.month
        df['year'] = df[time_col].dt.year
        df['dayofweek'] = df[time_col].dt.dayofweek
        df['quarter'] = df[time_col].dt.quarter
        df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
        return df
    
    @staticmethod
    def add_lag_features(df, target_col, lags=None):
        """添加滞后特征"""
        if lags is None:
            lags = [1, 7, 14, 30]
        
        df = df.copy()
        for lag in lags:
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        
        return df
    
    @staticmethod
    def add_rolling_features(df, target_col, windows=None):
        """添加滚动窗口特征"""
        if windows is None:
            windows = [7, 14, 30]
            
        df = df.copy()
        for window in windows:
            df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
            df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window=window).std()
            df[f'{target_col}_rolling_min_{window}'] = df[target_col].rolling(window=window).min()
            df[f'{target_col}_rolling_max_{window}'] = df[target_col].rolling(window=window).max()
            
        return df
    
    @staticmethod
    def add_ewm_features(df, target_col, alphas=None):
        """添加指数加权移动平均特征"""
        if alphas is None:
            alphas = [0.05, 0.1, 0.3]
            
        df = df.copy()
        for alpha in alphas:
            df[f'{target_col}_ewm_{alpha}'] = df[target_col].ewm(alpha=alpha).mean()
            
        return df
    
    @staticmethod
    def add_diff_features(df, target_col, periods=None):
        """添加差分特征"""
        if periods is None:
            periods = [1, 7]
            
        df = df.copy()
        for period in periods:
            df[f'{target_col}_diff_{period}'] = df[target_col].diff(period)
            
        return df
    
    @staticmethod
    def add_seasonal_decomposition(df, target_col, time_col, period=None):
        """添加季节性分解特征"""
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        df = df.copy()
        
        # 如果没有指定周期，尝试自动检测
        if period is None:
            # 默认使用日数据的季节性(7天)
            period = 7
            
            # 检查时间戳的频率
            time_diff = df[time_col].diff().dt.total_seconds().median()
            if time_diff == 3600:  # 小时数据
                period = 24  # 一天的小时数
            elif time_diff == 86400:  # 日数据
                period = 7  # 一周的天数
            elif time_diff == 604800:  # 周数据
                period = 52  # 一年的周数
            elif time_diff == 2592000:  # 月数据(近似30天)
                period = 12  # 一年的月数
        
        # 确保时间序列没有缺失值
        df_filled = df[[target_col]].fillna(method='ffill').fillna(method='bfill')
        
        try:
            result = seasonal_decompose(df_filled[target_col], model='additive', period=period)
            df[f'{target_col}_trend'] = result.trend
            df[f'{target_col}_seasonal'] = result.seasonal
            df[f'{target_col}_residual'] = result.resid
        except:
            # 如果分解失败，就跳过这一步
            pass
            
        return df
    
    @staticmethod
    def add_fourier_features(df, target_col, periods=None, n_harmonics=3):
        """添加傅里叶特征"""
        if periods is None:
            periods = [7, 30]  # 一周和一月的周期
            
        df = df.copy()
        t = np.arange(len(df))
        
        for period in periods:
            for n in range(1, n_harmonics + 1):
                df[f'{target_col}_sin_{period}_{n}'] = np.sin(2 * np.pi * n * t / period)
                df[f'{target_col}_cos_{period}_{n}'] = np.cos(2 * np.pi * n * t / period)
                
        return df
    
    @staticmethod
    def add_spectral_features(df, target_col, fs=1.0, nperseg=None):
        """添加频谱特征"""
        df = df.copy()
        
        # 填充缺失值
        series = df[target_col].fillna(method='ffill').fillna(method='bfill').values
        
        if nperseg is None:
            nperseg = min(256, len(series) // 2)
            
        try:
            # 计算功率谱密度
            frequencies, power = welch(series, fs=fs, nperseg=nperseg)
            
            # 添加一些统计特征
            df[f'{target_col}_spectral_mean'] = np.mean(power)
            df[f'{target_col}_spectral_std'] = np.std(power)
            df[f'{target_col}_spectral_max'] = np.max(power)
            
            # 主频率
            df[f'{target_col}_dominant_freq'] = frequencies[np.argmax(power)]
            
            # 频谱熵
            normalized_power = power / np.sum(power)
            entropy = -np.sum(normalized_power * np.log2(normalized_power + 1e-10))
            df[f'{target_col}_spectral_entropy'] = entropy
        except:
            # 如果计算失败，跳过
            pass
            
        return df
    
    @staticmethod
    def add_statistical_features(df, target_col, windows=None):
        """添加统计特征"""
        if windows is None:
            windows = [30, 60]
            
        df = df.copy()
        
        for window in windows:
            rolling = df[target_col].rolling(window=window)
            
            # 添加各种统计特征
            df[f'{target_col}_skew_{window}'] = rolling.apply(lambda x: stats.skew(x, nan_policy='omit'))
            df[f'{target_col}_kurtosis_{window}'] = rolling.apply(lambda x: stats.kurtosis(x, nan_policy='omit'))
            
            # 分位数特征
            df[f'{target_col}_q25_{window}'] = rolling.quantile(0.25)
            df[f'{target_col}_q75_{window}'] = rolling.quantile(0.75)
            df[f'{target_col}_iqr_{window}'] = df[f'{target_col}_q75_{window}'] - df[f'{target_col}_q25_{window}']
            
        return df


class TimeSeriesAugmenter:
    """时间序列数据增强类"""
    
    @staticmethod
    def jitter(x, sigma=0.03):
        """添加随机噪声"""
        return x + np.random.normal(loc=0., scale=sigma, size=x.shape)
    
    @staticmethod
    def scaling(x, sigma=0.1):
        """随机缩放"""
        factor = np.random.normal(loc=1., scale=sigma, size=(x.shape[0], 1))
        return x * factor
    
    @staticmethod
    def rotation(x):
        """旋转变换"""
        flip = np.random.choice([-1, 1], size=(x.shape[0], 1))
        return flip * x
    
    @staticmethod
    def magnitude_warp(x, sigma=0.2, knot=4):
        """幅度变形"""
        from scipy.interpolate import CubicSpline
        
        orig_steps = np.arange(x.shape[1])
        
        random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot+2))
        warp_steps = (np.ones((x.shape[0], 1)) * (np.linspace(0, x.shape[1]-1., num=knot+2))).astype(int)
        
        ret = np.zeros_like(x)
        for i, pat in enumerate(x):
            warper = CubicSpline(warp_steps[i], random_warps[i])(orig_steps)
            ret[i] = pat * warper
            
        return ret
    
    @staticmethod
    def time_warp(x, sigma=0.2, knot=4):
        """时间变形"""
        from scipy.interpolate import CubicSpline
        
        orig_steps = np.arange(x.shape[1])
        
        random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot+2))
        warp_steps = (np.ones((x.shape[0], 1)) * (np.linspace(0, x.shape[1]-1., num=knot+2))).astype(int)
        
        ret = np.zeros_like(x)
        for i, pat in enumerate(x):
            time_warp = CubicSpline(warp_steps[i], warp_steps[i] * random_warps[i])(orig_steps)
            scale = (x.shape[1] - 1) / time_warp[-1]
            ret[i] = np.interp(orig_steps, np.clip(scale * time_warp, 0, x.shape[1] - 1), pat)
            
        return ret
    
    @staticmethod
    def window_slice(x, reduce_ratio=0.9):
        """窗口切片"""
        target_len = int(reduce_ratio * x.shape[1])
        if target_len >= x.shape[1]:
            return x
        
        starts = np.random.randint(low=0, high=x.shape[1] - target_len, size=(x.shape[0])).astype(int)
        ends = (target_len + starts).astype(int)
        
        ret = np.zeros_like(x)
        for i, pat in enumerate(x):
            for dim in range(x.shape[2]):
                ret[i, :, dim] = np.interp(np.linspace(0, target_len, num=x.shape[1]), 
                                          np.arange(target_len), 
                                          pat[starts[i]:ends[i], dim])
        return ret
    
    @staticmethod
    def window_warp(x, window_ratio=0.1, scales=[0.5, 2.]):
        """窗口变形"""
        warp_scales = np.random.choice(scales, x.shape[0])
        warp_size = np.ceil(window_ratio * x.shape[1]).astype(int)
        window_steps = np.arange(warp_size)
        
        window_starts = np.random.randint(low=1, high=x.shape[1] - warp_size - 1, size=(x.shape[0])).astype(int)
        window_ends = (window_starts + warp_size).astype(int)
        
        ret = np.zeros_like(x)
        for i, pat in enumerate(x):
            for dim in range(x.shape[2]):
                start_seg = pat[:window_starts[i], dim]
                window_seg = np.interp(np.linspace(0, warp_size, num=int(warp_size * warp_scales[i])),
                                      window_steps, pat[window_starts[i]:window_ends[i], dim])
                end_seg = pat[window_ends[i]:, dim]
                warped = np.concatenate((start_seg, window_seg, end_seg))
                ret[i, :, dim] = np.interp(np.arange(x.shape[1]), 
                                          np.linspace(0, x.shape[1], num=warped.size),
                                          warped)
        return ret
    
    @staticmethod
    def subset_augmentation(X_train, y_train, augmenter, subset_ratio=0.3, augment_ratio=2):
        """对数据子集进行增强"""
        n = X_train.shape[0]
        subset_size = int(n * subset_ratio)
        indices = np.random.choice(n, subset_size, replace=False)
        
        X_subset = X_train[indices]
        y_subset = y_train[indices]
        
        aug_size = int(subset_size * augment_ratio)
        
        X_aug = augmenter(X_subset[:aug_size])
        y_aug = y_subset[:aug_size]
        
        X_combined = np.vstack((X_train, X_aug))
        y_combined = np.vstack((y_train, y_aug)) if len(y_train.shape) > 1 else np.concatenate((y_train, y_aug))
        
        return X_combined, y_combined

    @staticmethod
    def augment_batch(x, methods=None, probabilities=None):
        """对批次数据应用多种增强方法"""
        if methods is None:
            methods = [
                TimeSeriesAugmenter.jitter,
                TimeSeriesAugmenter.scaling,
                TimeSeriesAugmenter.rotation,
                TimeSeriesAugmenter.magnitude_warp,
                TimeSeriesAugmenter.time_warp
            ]
            
        if probabilities is None:
            probabilities = [0.3, 0.3, 0.2, 0.1, 0.1]
            
        # 确保概率总和为1
        probabilities = np.array(probabilities) / sum(probabilities)
        
        # 为每个样本随机选择一种增强方法
        n_samples = x.shape[0]
        method_idx = np.random.choice(len(methods), size=n_samples, p=probabilities)
        
        x_aug = np.zeros_like(x)
        for i in range(n_samples):
            x_aug[i] = methods[method_idx[i]](x[i:i+1])
            
        return x_aug


class GAUDataPreprocessor:
    """GAU模型的数据预处理器"""
    
    def __init__(self, scaler=None, feature_extractor=None, augmenter=None):
        self.scaler = scaler if scaler is not None else StandardScaler()
        self.feature_extractor = feature_extractor if feature_extractor is not None else TimeSeriesFeatureExtractor()
        self.augmenter = augmenter if augmenter is not None else TimeSeriesAugmenter()
        self.feature_names = None
        
    def fit_transform(self, data, time_col, target_col, lags=30, add_features=True):
        """
        对数据进行特征工程并转换为模型输入格式
        
        参数:
        - data: DataFrame, 原始数据
        - time_col: str, 时间列名
        - target_col: str, 目标列名
        - lags: int, 使用的历史数据长度
        - add_features: bool, 是否添加额外特征
        
        返回:
        - X: 模型输入特征
        - y: 目标值
        """
        df = data.copy()
        
        # 添加特征
        if add_features:
            df = self._add_all_features(df, time_col, target_col)
        
        # 确保时间排序
        df = df.sort_values(by=time_col)
        
        # 保存特征名称列表，不包括时间列和目标列
        self.feature_names = [col for col in df.columns if col not in [time_col, target_col]]
        
        # 创建特征矩阵
        X, y = self._create_time_series_data(df, target_col, lags)
        
        # 标准化特征
        if X.shape[0] > 0:
            # 将3D数据转为2D以进行缩放
            X_shape = X.shape
            X_reshaped = X.reshape(-1, X.shape[-1])
            
            X_scaled = self.scaler.fit_transform(X_reshaped)
            
            # 转回原来的形状
            X = X_scaled.reshape(X_shape)
        
        return X, y
    
    def transform(self, data, time_col, target_col, lags=30, add_features=True):
        """
        将新数据转换为模型输入格式
        """
        df = data.copy()
        
        # 添加特征
        if add_features:
            df = self._add_all_features(df, time_col, target_col)
        
        # 确保时间排序
        df = df.sort_values(by=time_col)
        
        # 确保所有必要的特征都存在
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = 0  # 使用0填充缺失的特征列
        
        # 只使用已知的特征列
        df = df[[time_col, target_col] + self.feature_names]
        
        # 创建特征矩阵
        X, y = self._create_time_series_data(df, target_col, lags)
        
        # 标准化特征
        if X.shape[0] > 0:
            # 将3D数据转为2D以进行缩放
            X_shape = X.shape
            X_reshaped = X.reshape(-1, X.shape[-1])
            
            X_scaled = self.scaler.transform(X_reshaped)
            
            # 转回原来的形状
            X = X_scaled.reshape(X_shape)
        
        return X, y
    
    def augment(self, X, y, augment_ratio=0.5):
        """
        对训练数据进行增强
        
        参数:
        - X: 输入特征
        - y: 目标值
        - augment_ratio: 增强的数据比例
        
        返回:
        - 增强后的特征和目标值
        """
        n_samples = X.shape[0]
        n_augment = int(n_samples * augment_ratio)
        
        # 随机选择要增强的样本
        indices = np.random.choice(n_samples, n_augment, replace=False)
        X_to_augment = X[indices]
        y_to_augment = y[indices]
        
        # 对选择的样本进行增强
        X_augmented = self.augmenter.augment_batch(X_to_augment)
        
        # 合并原始数据和增强数据
        X_combined = np.vstack((X, X_augmented))
        y_combined = np.concatenate((y, y_to_augment))
        
        return X_combined, y_combined
    
    def _add_all_features(self, df, time_col, target_col):
        """
        添加所有特征工程
        """
        # 确保时间列为datetime类型
        df[time_col] = pd.to_datetime(df[time_col])
        
        # 添加时间特征
        df = self.feature_extractor.add_time_features(df, time_col)
        
        # 添加滞后特征
        df = self.feature_extractor.add_lag_features(df, target_col)
        
        # 添加滚动窗口特征
        df = self.feature_extractor.add_rolling_features(df, target_col)
        
        # 添加指数加权平均特征
        df = self.feature_extractor.add_ewm_features(df, target_col)
        
        # 添加差分特征
        df = self.feature_extractor.add_diff_features(df, target_col)
        
        # 添加季节性分解特征
        df = self.feature_extractor.add_seasonal_decomposition(df, target_col, time_col)
        
        # 添加傅里叶特征
        df = self.feature_extractor.add_fourier_features(df, target_col)
        
        # 去除含有NaN的行
        df = df.dropna()
        
        return df
    
    @staticmethod
    def _create_time_series_data(df, target_col, lags):
        """
        创建时间序列数据
        
        参数:
        - df: 包含特征的DataFrame
        - target_col: 目标列名
        - lags: 使用的历史数据长度
        
        返回:
        - X: 形状为(样本数, 时间步, 特征数)的特征数据
        - y: 目标值
        """
        # 去除目标列和datetime列，剩下的都是特征
        feature_cols = [col for col in df.columns
                        if col != target_col and not pd.api.types.is_datetime64_any_dtype(df[col])]
        
        # 获取特征和目标数据
        data = df[feature_cols].values
        targets = df[target_col].values
        
        X, y = [], []
        
        # 创建时间窗口样本
        for i in range(lags, len(data)):
            X.append(data[i-lags:i])
            y.append(targets[i])
        
        return np.array(X), np.array(y) 