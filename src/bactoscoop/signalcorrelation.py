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

class SignalCorrelation:
    def __init__(self, df, channel1, channel2, feature, method_name):
        self.df = df
        self.channel1 = channel1
        self.channel2 = channel2
        self.feature = feature
        self.method_name = method_name
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
        feature1 = f"{self.channel1}_{self.feature}"
        feature2 = f"{self.channel2}_{self.feature}"

        signals1 = self.df[feature1]
        signals2 = self.df[feature2]

        results = []
        for signal1, signal2 in zip(signals1, signals2):
            if SignalCorrelation.is_valid_signal(
                signal1
            ) and SignalCorrelation.is_valid_signal(signal2):
                result = self.method(signal1, signal2)
                results.append(result)
            else:
                results.append(np.nan)  # Append NaN for invalid rows

        method_name = self.method.__name__
        new_feature_name = (
            f"{self.channel1}_{self.channel2}_{self.feature}_{method_name}"
        )
        self.df[new_feature_name] = results
        return self.df

    @staticmethod
    def is_valid_signal(signal):
        if isinstance(signal, (list, np.ndarray)):
            return all(pd.notna(val) for val in signal) and len(signal) > 0
        return pd.notna(signal)

    @staticmethod
    def ratio(signal1, signal2):
        if np.isscalar(signal1) and np.isscalar(signal2) and signal2 != 0:
            return signal1 / signal2
        return 0.

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
        if isinstance(signal1, (list, np.ndarray)) and isinstance(
            signal2, (list, np.ndarray)
        ):
            if len(signal1) > 1 and len(signal2) > 1:

                def dcov(X, Y):
                    X, Y = np.atleast_1d(X), np.atleast_1d(Y)
                    n = len(X)
                    XY = np.vstack((X, Y))
                    dXY = squareform(pdist(XY.T))
                    dX = squareform(pdist(X.reshape(n, 1)))
                    dY = squareform(pdist(Y.reshape(n, 1)))
                    return np.sqrt(
                        np.mean(dXY**2)
                        + np.mean(dX**2) * np.mean(dY**2)
                        - 2 * np.mean(dX * dY)
                    )

                dcor = dcov(signal1, signal2) / np.sqrt(
                    dcov(signal1, signal1) * dcov(signal2, signal2)
                )
                return dcor
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
        if isinstance(signal1, (list, np.ndarray)) and isinstance(
            signal2, (list, np.ndarray)
        ):
            if len(signal1) > 1 and len(signal2) > 1:
                hist1, _ = np.histogram(signal1, bins=bins)
                hist2, _ = np.histogram(signal2, bins=bins)
                return np.sum(np.minimum(hist1, hist2)) / np.sum(hist1)
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
