# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 14:18:42 2024

@author: Bart Steemans. Govers Lab.
"""

import pandas as pd
import numpy as np
from scipy import stats
from skimage import filters
from scipy.spatial.distance import pdist, squareform
import logging

bactoscoop_logger = logging.getLogger("logger")
class SignalCorrelation:
    def __init__(
        self,
        df,
        channel1,
        channel2,
        feature,
        method_name,
        prepared_feature=None,
    ):
        self.df = df
        self.channel1 = channel1
        self.channel2 = channel2
        self.feature = feature
        self.method_name = method_name
        self.prepared_feature = prepared_feature
        self.method = self.get_method()
        self.validate_inputs()

    def get_method(self):
        methods = {
            "manders": self.manders_overlap_coefficient,
            "pearson": self.pearson_correlation_coefficient,
            "li_icq": self.li_icq,
            "ratio": self.ratio,
            "spearman": self.spearman_rank_correlation,
            "kendall": self.kendall_tau,
            "distance_corr": self.distance_correlation,
            "covariance": self.covariance,
            "n_cross_corr": self.normalized_cross_correlation,
            "entropy_diff": self.entropy_difference,
            "kurtosis_ratio": self.kurtosis_ratio,
            "skewness_product": self.skewness_product,
            "zero_crossings_diff": self.zero_crossings_difference,
            "fft_peak_ratio": self.fft_peak_ratio,
            "fft_energy_ratio": self.fft_energy_ratio,
            "histogram_intersection": self.histogram_intersection,
            "cosine_similarity": self.cosine_similarity,
        }
        if self.method_name in methods:
            return methods[self.method_name]
        else:
            raise ValueError(f'Method "{self.method_name}" is not recognized')

    def validate_inputs(self):
        feature1 = f"{self.channel1}_{self.feature}"
        feature2 = f"{self.channel2}_{self.feature}"
        if feature1 not in self.df.columns or feature2 not in self.df.columns:
            raise ValueError(
                f"Features {feature1} and {feature2} must be in the DataFrame columns"
            )

        if not callable(self.method):
            raise ValueError(f'The method "{self.method_name}" is not callable')

    def calculate(self):
        prepared_feature = self.prepared_feature
        if prepared_feature is None:
            prepared_feature = self.prepare_feature_pair(
                self.df, self.channel1, self.channel2, self.feature
            )

        signals1 = prepared_feature["signals1"]
        signals2 = prepared_feature["signals2"]
        valid_mask = prepared_feature["valid_mask"]

        results = [np.nan] * len(signals1)
        for index, (signal1, signal2, is_valid) in enumerate(
            zip(signals1, signals2, valid_mask)
        ):
            try:
                if is_valid:
                    results[index] = self.method(signal1, signal2)
            except ValueError as e:
                bactoscoop_logger.debug(f"Encountered a ValueError: {e}")
                continue
            
        method_name = self.method.__name__
        new_feature_name = (
            f"{self.channel1}_{self.channel2}_{self.feature}_{method_name}"
        )
        self.df[new_feature_name] = results
        return self.df

    @classmethod
    def prepare_feature_pair(cls, df, channel1, channel2, feature):
        feature1 = f"{channel1}_{feature}"
        feature2 = f"{channel2}_{feature}"
        signals1 = df[feature1].to_numpy(copy=False)
        signals2 = df[feature2].to_numpy(copy=False)
        valid_mask = np.fromiter(
            (
                cls.is_valid_signal(signal1) and cls.is_valid_signal(signal2)
                for signal1, signal2 in zip(signals1, signals2)
            ),
            dtype=bool,
            count=len(signals1),
        )
        return {"signals1": signals1, "signals2": signals2, "valid_mask": valid_mask}

    @staticmethod
    def is_valid_signal(signal):
        try:
            signal_array = np.asarray(signal, dtype=float)
        except (TypeError, ValueError):
            return False
        return bool(signal_array.size and np.isfinite(signal_array).all())

    @staticmethod
    def ratio(signal1, signal2):
        if (
            np.isscalar(signal1)
            and np.isscalar(signal2)
            and np.isfinite(signal1)
            and np.isfinite(signal2)
            and signal2 != 0
        ):
            return signal1 / signal2
        return np.nan

    @staticmethod
    def manders_overlap_coefficient(signal1, signal2):
        if isinstance(signal1, (list, np.ndarray)) and isinstance(
            signal2, (list, np.ndarray)
        ):
            if len(signal1) > 1 and len(signal2) > 1:
                signal1 = np.array(signal1)
                signal2 = np.array(signal2)
                return np.sum(signal1 * signal2) / np.sqrt(
                    np.sum(signal1**2) * np.sum(signal2**2)
                )
        return np.nan

    @staticmethod
    def pearson_correlation_coefficient(signal1, signal2):
        if isinstance(signal1, (list, np.ndarray)) and isinstance(
            signal2, (list, np.ndarray)
        ):
            if len(signal1) > 1 and len(signal2) > 1:
                return stats.pearsonr(signal1, signal2)[0]
        return np.nan

    @staticmethod
    def li_icq(signal1, signal2):
        if isinstance(signal1, (list, np.ndarray)) and isinstance(
            signal2, (list, np.ndarray)
        ):
            if len(signal1) > 1 and len(signal2) > 1:
                signal1 = np.array(signal1)
                signal2 = np.array(signal2)
                mean1, mean2 = np.mean(signal1), np.mean(signal2)
                product = ((signal1 - mean1) * (signal2 - mean2)) > 0
                return (np.sum(product) / len(signal1) - 0.5) * 2
        return np.nan

    @staticmethod
    def spearman_rank_correlation(signal1, signal2):
        if isinstance(signal1, (list, np.ndarray)) and isinstance(
            signal2, (list, np.ndarray)
        ):
            if len(signal1) > 1 and len(signal2) > 1:
                return stats.spearmanr(signal1, signal2)[0]
        return np.nan

    @staticmethod
    def kendall_tau(signal1, signal2):
        if isinstance(signal1, (list, np.ndarray)) and isinstance(
            signal2, (list, np.ndarray)
        ):
            if len(signal1) > 1 and len(signal2) > 1:
                corr, _ = stats.kendalltau(signal1, signal2)
                return corr
        return np.nan

    @staticmethod
    def distance_correlation(signal1, signal2):
        """Biased sample distance correlation; undefined profiles return NaN."""
        if isinstance(signal1, (list, np.ndarray)) and isinstance(
            signal2, (list, np.ndarray)
        ):
            if len(signal1) > 1 and len(signal2) > 1:
                x = np.asarray(signal1, dtype=float)
                y = np.asarray(signal2, dtype=float)
                if (
                    x.ndim != 1
                    or y.ndim != 1
                    or len(x) != len(y)
                    or not np.isfinite(x).all()
                    or not np.isfinite(y).all()
                ):
                    return np.nan

                # Biased sample distance correlation uses separately centered
                # pairwise distances for each signal.
                distance_x = squareform(pdist(x[:, None]))
                distance_y = squareform(pdist(y[:, None]))
                centered_x = (
                    distance_x
                    - distance_x.mean(axis=0)
                    - distance_x.mean(axis=1)[:, None]
                    + distance_x.mean()
                )
                centered_y = (
                    distance_y
                    - distance_y.mean(axis=0)
                    - distance_y.mean(axis=1)[:, None]
                    + distance_y.mean()
                )
                variance_x = np.mean(centered_x**2)
                variance_y = np.mean(centered_y**2)
                if variance_x == 0 or variance_y == 0:
                    return np.nan
                correlation_squared = np.mean(centered_x * centered_y) / np.sqrt(
                    variance_x * variance_y
                )
                return float(np.sqrt(np.clip(correlation_squared, 0, 1)))
        return np.nan

    @staticmethod
    def covariance(signal1, signal2):
        if isinstance(signal1, (list, np.ndarray)) and isinstance(
            signal2, (list, np.ndarray)
        ):
            if len(signal1) > 1 and len(signal2) > 1:
                return np.cov(signal1, signal2)[0, 1]
        return np.nan

    @staticmethod
    def normalized_cross_correlation(signal1, signal2):
        if isinstance(signal1, (list, np.ndarray)) and isinstance(
            signal2, (list, np.ndarray)
        ):
            if len(signal1) > 1 and len(signal2) > 1:
                return np.correlate(
                    signal1 - np.mean(signal1), signal2 - np.mean(signal2)
                )[0] / (len(signal1) * np.std(signal1) * np.std(signal2))
        return np.nan

    @staticmethod
    def entropy_difference(signal1, signal2):
        if isinstance(signal1, (list, np.ndarray)) and isinstance(
            signal2, (list, np.ndarray)
        ):
            if len(signal1) > 1 and len(signal2) > 1:
                entropy1 = stats.entropy(signal1)
                entropy2 = stats.entropy(signal2)
                return np.abs(entropy1 - entropy2)
        return np.nan

    @staticmethod
    def kurtosis_ratio(signal1, signal2):
        if isinstance(signal1, (list, np.ndarray)) and isinstance(
            signal2, (list, np.ndarray)
        ):
            if len(signal1) > 1 and len(signal2) > 1:
                kurtosis1 = stats.kurtosis(signal1)
                kurtosis2 = stats.kurtosis(signal2)
                return np.abs(kurtosis1 / kurtosis2) if kurtosis2 != 0 else np.nan
        return np.nan

    @staticmethod
    def skewness_product(signal1, signal2):
        if isinstance(signal1, (list, np.ndarray)) and isinstance(
            signal2, (list, np.ndarray)
        ):
            if len(signal1) > 1 and len(signal2) > 1:
                skew1 = stats.skew(signal1)
                skew2 = stats.skew(signal2)
                return skew1 * skew2
        return np.nan

    @staticmethod
    def zero_crossings_difference(signal1, signal2):
        if isinstance(signal1, (list, np.ndarray)) and isinstance(
            signal2, (list, np.ndarray)
        ):
            if len(signal1) > 1 and len(signal2) > 1:
                zero_crossings1 = np.sum(np.diff(np.sign(signal1)) != 0)
                zero_crossings2 = np.sum(np.diff(np.sign(signal2)) != 0)
                return np.abs(zero_crossings1 - zero_crossings2)
        return np.nan

    @staticmethod
    def fft_peak_ratio(signal1, signal2):
        if isinstance(signal1, (list, np.ndarray)) and isinstance(
            signal2, (list, np.ndarray)
        ):
            if len(signal1) > 1 and len(signal2) > 1:
                fft1 = np.abs(np.fft.fft(signal1))
                fft2 = np.abs(np.fft.fft(signal2))
                peak1 = np.max(fft1[1:])  # Exclude DC component
                peak2 = np.max(fft2[1:])  # Exclude DC component
                return peak1 / peak2 if peak2 != 0 else np.nan
        return np.nan

    @staticmethod
    def fft_energy_ratio(signal1, signal2):
        if isinstance(signal1, (list, np.ndarray)) and isinstance(
            signal2, (list, np.ndarray)
        ):
            if len(signal1) > 1 and len(signal2) > 1:
                # Compute the FFT of the signals
                fft1 = np.fft.fft(signal1)
                fft2 = np.fft.fft(signal2)

                # Compute the energy in each frequency bin
                energy1 = np.sum(np.abs(fft1) ** 2)
                energy2 = np.sum(np.abs(fft2) ** 2)

                # Return the energy ratio
                return energy1 / energy2 if energy2 != 0 else np.nan
        return np.nan

    @staticmethod
    def histogram_intersection(signal1, signal2, bins=10):
        """Intersect probability histograms using shared bins for both profiles."""
        if isinstance(signal1, (list, np.ndarray)) and isinstance(
            signal2, (list, np.ndarray)
        ):
            if len(signal1) > 1 and len(signal2) > 1:
                x = np.asarray(signal1, dtype=float)
                y = np.asarray(signal2, dtype=float)
                if (
                    x.ndim != 1
                    or y.ndim != 1
                    or not np.isfinite(x).all()
                    or not np.isfinite(y).all()
                ):
                    return np.nan

                # Both histograms must describe the same intensity intervals.
                # Normalize each by its own sample count so different profile
                # lengths remain comparable and the score is symmetric.
                edges = np.histogram_bin_edges(np.concatenate((x, y)), bins=bins)
                hist1, _ = np.histogram(x, bins=edges)
                hist2, _ = np.histogram(y, bins=edges)
                count1 = hist1.sum()
                count2 = hist2.sum()
                if count1 == 0 or count2 == 0:
                    return np.nan
                return float(np.minimum(hist1 / count1, hist2 / count2).sum())
        return np.nan

    @staticmethod
    def cosine_similarity(signal1, signal2):
        if isinstance(signal1, (list, np.ndarray)) and isinstance(
            signal2, (list, np.ndarray)
        ):
            if len(signal1) > 1 and len(signal2) > 1:
                return np.dot(signal1, signal2) / (
                    np.linalg.norm(signal1) * np.linalg.norm(signal2)
                )
        return np.nan
