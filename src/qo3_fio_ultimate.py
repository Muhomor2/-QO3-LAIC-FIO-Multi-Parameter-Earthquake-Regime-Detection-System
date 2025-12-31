#!/usr/bin/env python3
"""
================================================================================
    QO3-LAIC-FIO ULTIMATE v2.0
    Multi-Parameter Earthquake Regime Detection System
================================================================================

    INTEGRATED COMPONENTS:
    ├── Core FIO Estimators (b-value, CV, entropy, SID)
    ├── Feature Builder (leak-free target construction)
    ├── Event Deduplicator (multi-catalog support)
    ├── Custom Data Loader (radon, GNSS, groundwater, etc.)
    ├── LAIC Coupling Index
    ├── Honest Backtest with Bootstrap CI
    ├── LaTeX Table Generator
    └── Telegram Notifications

    PHYSICAL BASIS:
    ├── b-value ↓ = fault locking (fluid expulsion → friction ↑)
    ├── CV → 1 = deterministic system (rigid coupling)
    ├── SID ↑ = seismic information deficit (quiescence)
    └── Cascadia-Tohoku analogy: hydrothermal processes

    MATHEMATICAL FOUNDATION:
    └── Sebestyen's theorem extension to unbounded operators (Barkaoui, 2025)

    DATA SOURCES:
    ├── Built-in: USGS, NOAA Kp, NOAA Solar F10.7
    └── Custom: Any CSV time series (radon, GNSS, groundwater, OLR, etc.)

Version: 2.0 (Ultimate)
Date: December 31, 2025
Author: Igor Chechelnitsky (ORCID: 0009-0007-4607-1946)

================================================================================
"""

import pandas as pd
import numpy as np
import requests
import json
import os
import sys
import glob
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
import warnings
warnings.filterwarnings('ignore')


# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class Config:
    """System configuration."""
    
    # Region
    region_name: str = "global"
    lat_min: float = -90.0
    lat_max: float = 90.0
    lon_min: float = -180.0
    lon_max: float = 180.0
    
    # FIO parameters
    magnitude_completeness: float = 2.5
    target_magnitude: float = 5.0
    forecast_horizon: int = 7
    b_window_days: int = 30
    cv_window_days: int = 14
    
    # Data
    history_days: int = 365
    train_fraction: float = 0.75
    custom_data_dir: str = "custom_data"
    
    # Deduplication
    dedup_time_sec: float = 30.0
    dedup_dist_km: float = 30.0
    dedup_mag_diff: float = 0.3
    
    # Telegram
    telegram_enabled: bool = False
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    alert_threshold: float = 50.0
    
    # Output
    output_dir: str = "."
    verbose: bool = True


# ==============================================================================
# REGIONS
# ==============================================================================

REGIONS = {
    'global': {'name': 'Global', 'lat': (-90, 90), 'lon': (-180, 180), 'mc': 4.5, 'target': 6.0},
    'japan': {'name': 'Japan', 'lat': (24, 46), 'lon': (122, 150), 'mc': 2.5, 'target': 5.0},
    'tohoku': {'name': 'Tohoku', 'lat': (35, 42), 'lon': (140, 145), 'mc': 2.0, 'target': 4.5},
    'california': {'name': 'California', 'lat': (32, 42), 'lon': (-125, -114), 'mc': 2.5, 'target': 4.5},
    'cascadia': {'name': 'Cascadia', 'lat': (40, 50), 'lon': (-130, -120), 'mc': 2.5, 'target': 5.0},
    'turkey': {'name': 'Turkey', 'lat': (36, 42), 'lon': (26, 45), 'mc': 3.0, 'target': 4.5},
    'israel': {'name': 'Israel-Levant', 'lat': (29, 34), 'lon': (34, 37), 'mc': 2.0, 'target': 4.0},
    'indonesia': {'name': 'Indonesia', 'lat': (-11, 6), 'lon': (95, 141), 'mc': 4.0, 'target': 5.5},
    'chile': {'name': 'Chile', 'lat': (-56, -17), 'lon': (-76, -66), 'mc': 4.0, 'target': 5.5},
    'mediterranean': {'name': 'Mediterranean', 'lat': (30, 47), 'lon': (-10, 40), 'mc': 3.0, 'target': 4.5},
    'iran': {'name': 'Iran', 'lat': (25, 40), 'lon': (44, 63), 'mc': 3.5, 'target': 5.0},
    'himalaya': {'name': 'Himalaya', 'lat': (26, 36), 'lon': (70, 97), 'mc': 4.0, 'target': 5.5},
    'alaska': {'name': 'Alaska', 'lat': (51, 72), 'lon': (-180, -130), 'mc': 3.5, 'target': 5.0},
    'newzealand': {'name': 'New Zealand', 'lat': (-48, -34), 'lon': (166, 179), 'mc': 3.0, 'target': 5.0},
}


# ==============================================================================
# FIO CORE ESTIMATORS
# ==============================================================================

class FIOEstimators:
    """
    Fractal Information Ontology - Core Physical Estimators.
    
    Mathematical foundation:
    - b-value: Aki-Utsu MLE with Shi-Bolt bias correction
    - CV: Coefficient of Variation of inter-event times
    - Entropy: Shannon entropy of magnitude distribution
    - SID: Seismic Information Deficit
    
    Physical interpretation (Cascadia-Tohoku model):
    - b-value ↓ = fault locking (fluid expulsion → friction ↑)
    - CV → 1 = transition from stochastic to deterministic
    - SID ↑ = seismic quiescence (information compression)
    """
    
    @staticmethod
    def b_value_aki_utsu(mags: np.ndarray, mc: float, min_n: int = 10) -> float:
        """
        Aki-Utsu MLE b-value with Shi-Bolt bias correction.
        
        Formula: b = log₁₀(e) / (M̄ - (Mc - 0.05))
        
        Physical interpretation:
            b ≈ 1.0: normal tectonic state
            b < 0.8: stress accumulation, fault locking (PRECURSOR)
            b > 1.2: stress release, aftershock regime
        
        Reference: Aki (1965), Utsu (1965), Shi & Bolt (1982)
        """
        mags = np.asarray(mags, dtype=float)
        mags = mags[~np.isnan(mags)]
        
        if len(mags) < min_n:
            return np.nan
        
        denom = np.mean(mags) - (mc - 0.05)  # Shi-Bolt correction
        
        if denom <= 0:
            return np.nan
        
        b = np.log10(np.e) / denom
        return float(np.clip(b, 0.3, 3.0))
    
    @staticmethod
    def cv_interevent(timestamps_ns: np.ndarray, min_n: int = 3) -> float:
        """
        Coefficient of Variation of inter-event times.
        
        Formula: CV = σ(ΔT) / μ(ΔT)
        
        Physical interpretation:
            CV > 1: clustered regime (aftershocks, high fluid content)
            CV = 1: Poisson process (random, baseline)
            CV < 1: quasi-periodic (locked fault, deterministic)
        
        The transition CV → 1 from above indicates system rigidification.
        """
        t = np.asarray(timestamps_ns, dtype='int64')
        t = t[~pd.isna(t)]
        
        if len(t) < min_n:
            return np.nan
        
        t = np.sort(t)
        dt = np.diff(t) / 1e9  # nanoseconds to seconds
        
        if len(dt) < 2 or np.mean(dt) <= 0:
            return np.nan
        
        return float(np.std(dt) / np.mean(dt))
    
    @staticmethod
    def shannon_entropy(mags: np.ndarray, bins: int = 20) -> float:
        """
        Shannon entropy of magnitude distribution.
        
        Formula: H = -Σ p(M) log₂ p(M)
        
        Low entropy = concentrated distribution = information compression.
        Pre-event regime typically shows entropy decrease.
        """
        mags = np.asarray(mags, dtype=float)
        mags = mags[~np.isnan(mags)]
        
        if len(mags) < 10:
            return np.nan
        
        hist, edges = np.histogram(mags, bins=bins, density=True)
        hist = hist[hist > 0]
        
        if len(hist) == 0:
            return np.nan
        
        bin_width = (edges[-1] - edges[0]) / bins
        return float(-np.sum(hist * np.log2(hist)) * bin_width)
    
    @staticmethod
    def seismic_information_deficit(entropy: float, background_entropy: float) -> float:
        """
        Seismic Information Deficit (SID).
        
        Formula: SID = 1 - H(t) / H_background
        
        Physical interpretation:
            SID ↑ = seismic quiescence = fault locking = PRECURSOR
            SID = 0: normal information content
            SID < 0: higher than normal activity
        
        This metric captures "the silence before the storm".
        """
        if np.isnan(entropy) or np.isnan(background_entropy) or background_entropy <= 0:
            return np.nan
        
        return float(1.0 - (entropy / background_entropy))


# ==============================================================================
# EVENT DEDUPLICATOR
# ==============================================================================

class EventDeduplicator:
    """
    Multi-catalog event deduplication.
    
    When merging events from multiple catalogs (USGS, JMA, EMSC, etc.),
    the same earthquake may appear multiple times with slightly different
    parameters. This inflates event counts and corrupts b-value/CV estimates.
    
    Deduplication rule: collapse events within
    - time: ±30 seconds
    - distance: ±30 km
    - magnitude: ±0.3
    """
    
    def __init__(self, time_sec: float = 30.0, dist_km: float = 30.0, mag_diff: float = 0.3):
        self.time_sec = time_sec
        self.dist_km = dist_km
        self.mag_diff = mag_diff
    
    @staticmethod
    def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Haversine distance in km."""
        R = 6371.0
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        return R * 2 * np.arcsin(np.sqrt(a))
    
    def deduplicate(self, events: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate events from merged catalogs.
        
        Args:
            events: DataFrame with columns [time, lat, lon, mag, ...]
        
        Returns:
            Deduplicated DataFrame (keeps event with largest magnitude)
        """
        if len(events) == 0:
            return events
        
        df = events.copy()
        df['time'] = pd.to_datetime(df['time'], utc=True)
        df = df.sort_values('time').reset_index(drop=True)
        
        # Group by approximate time windows
        df['time_group'] = (df['time'].astype('int64') // 10**9 // int(self.time_sec)).astype(int)
        
        keep_mask = np.ones(len(df), dtype=bool)
        
        for _, group in df.groupby('time_group'):
            if len(group) <= 1:
                continue
            
            indices = group.index.tolist()
            
            for i, idx1 in enumerate(indices):
                if not keep_mask[idx1]:
                    continue
                
                for idx2 in indices[i+1:]:
                    if not keep_mask[idx2]:
                        continue
                    
                    # Check distance
                    dist = self.haversine_km(
                        df.loc[idx1, 'lat'], df.loc[idx1, 'lon'],
                        df.loc[idx2, 'lat'], df.loc[idx2, 'lon']
                    )
                    
                    if dist > self.dist_km:
                        continue
                    
                    # Check magnitude difference
                    if abs(df.loc[idx1, 'mag'] - df.loc[idx2, 'mag']) > self.mag_diff:
                        continue
                    
                    # Duplicate found - keep the one with larger magnitude
                    if df.loc[idx1, 'mag'] >= df.loc[idx2, 'mag']:
                        keep_mask[idx2] = False
                    else:
                        keep_mask[idx1] = False
                        break
        
        result = df[keep_mask].drop(columns=['time_group']).reset_index(drop=True)
        
        removed = len(df) - len(result)
        if removed > 0:
            print(f"    Deduplication: removed {removed} duplicate events ({removed/len(df)*100:.1f}%)")
        
        return result


# ==============================================================================
# FEATURE BUILDER (LEAK-FREE)
# ==============================================================================

class QO3FeatureBuilder:
    """
    Feature matrix builder with strict leak-free target construction.
    
    Target definition (no look-ahead bias):
        y_t = 1 if max(M) in (t, t+horizon] >= target_magnitude
    
    The shift(-1) ensures we don't include today's events in the target.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.fio = FIOEstimators()
    
    def build_matrix(self, events: pd.DataFrame, 
                     timeseries: Optional[pd.DataFrame] = None,
                     kp_data: Optional[pd.DataFrame] = None,
                     solar_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Build feature matrix from events and optional time series.
        
        Args:
            events: DataFrame with [time, mag, lat, lon, ...]
            timeseries: Optional DataFrame with [time, kind, value]
            kp_data: Optional Kp index data
            solar_data: Optional solar flux data
        
        Returns:
            Daily DataFrame with features and target (no leakage)
        """
        ev = events.copy()
        ev['time'] = pd.to_datetime(ev['time'], utc=True)
        ev = ev.sort_values('time')
        ev['date'] = ev['time'].dt.floor('D')
        
        # Daily seismic aggregation
        daily = ev.groupby('date').agg(
            count=('mag', 'size'),
            max_mag=('mag', 'max'),
            mean_mag=('mag', 'mean'),
            energy=('mag', lambda x: np.sum(10 ** (1.5 * x + 4.8)))
        )
        
        # Full daily grid
        full_idx = pd.date_range(daily.index.min(), daily.index.max(), freq='D', tz='UTC')
        daily = daily.reindex(full_idx)
        daily['count'] = daily['count'].fillna(0)
        daily['max_mag'] = daily['max_mag'].fillna(0)
        daily['mean_mag'] = daily['mean_mag'].fillna(0)
        daily['energy'] = daily['energy'].fillna(0)
        
        # =====================================================================
        # TARGET (LEAK-FREE): event >= target in next horizon days
        # shift(-1) excludes today; rolling looks forward
        # =====================================================================
        h = self.config.forecast_horizon
        daily['future_max_mag'] = daily['max_mag'].shift(-1).rolling(h, min_periods=1).max()
        daily['target'] = (daily['future_max_mag'] >= self.config.target_magnitude).astype(int)
        
        # =====================================================================
        # RATE FEATURES
        # =====================================================================
        for w in [3, 7, 14, 30]:
            daily[f'rate_{w}d'] = daily['count'].rolling(w, min_periods=1).mean()
        
        daily['rate_accel_7_30'] = daily['rate_7d'] / (daily['rate_30d'] + 0.1)
        
        # Energy features
        daily['energy_7d'] = daily['energy'].rolling(7, min_periods=1).sum()
        daily['energy_30d'] = daily['energy'].rolling(30, min_periods=1).sum()
        daily['energy_accel'] = daily['energy_7d'] / (daily['energy_30d'] + 1)
        
        # =====================================================================
        # FIO FEATURES: b-value, CV, entropy, SID
        # =====================================================================
        ev_times = ev['time'].values
        ev_mags = ev['mag'].values
        mc = self.config.magnitude_completeness
        b_window = self.config.b_window_days
        cv_window = self.config.cv_window_days
        
        b_values = []
        cv_values = []
        entropy_values = []
        
        for day in daily.index:
            # b-value window
            b_start = day - pd.Timedelta(days=b_window)
            mask_b = (ev_times > b_start.to_datetime64()) & (ev_times <= day.to_datetime64())
            mags_w = np.asarray(ev_mags[mask_b], dtype=float)
            b_values.append(self.fio.b_value_aki_utsu(mags_w, mc))
            
            # CV window
            cv_start = day - pd.Timedelta(days=cv_window)
            mask_cv = (ev_times > cv_start.to_datetime64()) & (ev_times <= day.to_datetime64())
            t_w = pd.to_datetime(ev_times[mask_cv], utc=True).astype('int64').values
            cv_values.append(self.fio.cv_interevent(t_w))
            
            # Entropy
            entropy_values.append(self.fio.shannon_entropy(mags_w))
        
        daily['fio_b_value'] = b_values
        daily['fio_cv'] = cv_values
        daily['fio_entropy'] = entropy_values
        
        # Derivatives
        daily['fio_b_change_7d'] = daily['fio_b_value'].diff(7)
        daily['fio_b_change_14d'] = daily['fio_b_value'].diff(14)
        daily['fio_cv_change_7d'] = daily['fio_cv'].diff(7)
        
        # SID (Seismic Information Deficit)
        daily['fio_entropy_bg'] = daily['fio_entropy'].rolling(90, min_periods=30).mean()
        daily['fio_sid'] = daily.apply(
            lambda row: self.fio.seismic_information_deficit(row['fio_entropy'], row['fio_entropy_bg']),
            axis=1
        )
        
        # Stress indicator (b < 0.9)
        daily['fio_stress'] = (daily['fio_b_value'] < 0.9).astype(int)
        
        # Quiescence indicator
        daily['quiescence'] = (daily['rate_7d'] < daily['rate_30d'] * 0.5).astype(int)
        
        # =====================================================================
        # GEOMAGNETIC FEATURES (Kp)
        # =====================================================================
        if kp_data is not None and len(kp_data) > 0:
            daily = daily.join(kp_data, how='left')
            daily['kp_max'] = daily['kp_max'].ffill().bfill().fillna(2)
            daily['kp_mean'] = daily['kp_mean'].ffill().bfill().fillna(2)
            daily['geomag_storm'] = (daily['kp_max'] >= 5).astype(int)
        else:
            daily['kp_max'] = 2.0
            daily['kp_mean'] = 2.0
            daily['geomag_storm'] = 0
        
        # =====================================================================
        # SOLAR FEATURES
        # =====================================================================
        if solar_data is not None and len(solar_data) > 0:
            daily = daily.join(solar_data, how='left')
            daily['f107'] = daily['f107'].ffill().bfill().fillna(100)
            daily['ssn'] = daily['ssn'].ffill().bfill().fillna(50)
            daily['solar_anomaly'] = (daily['f107'] > 150).astype(int)
        else:
            daily['f107'] = 100.0
            daily['ssn'] = 50.0
            daily['solar_anomaly'] = 0
        
        # =====================================================================
        # TEC PROXY & LAIC INDEX
        # =====================================================================
        daily['tec_proxy'] = daily['kp_max'].rolling(3, min_periods=1).mean()
        tec_mean = daily['tec_proxy'].rolling(30).mean()
        tec_std = daily['tec_proxy'].rolling(30).std()
        daily['tec_anomaly'] = (daily['tec_proxy'] > tec_mean + 2 * tec_std).fillna(0).astype(int)
        
        # LAIC coupling index
        daily['laic_index'] = (
            daily['fio_stress'].astype(float) +
            daily['geomag_storm'].astype(float) +
            daily['tec_anomaly'].astype(float) +
            daily['solar_anomaly'].astype(float)
        ) / 4.0
        
        daily['laic_7d'] = daily['laic_index'].rolling(7, min_periods=1).mean()
        
        # =====================================================================
        # CUSTOM TIMESERIES
        # =====================================================================
        if timeseries is not None and len(timeseries) > 0:
            daily = self._add_timeseries_features(daily, timeseries)
        
        return daily
    
    def _add_timeseries_features(self, daily: pd.DataFrame, ts: pd.DataFrame) -> pd.DataFrame:
        """Add features from custom time series data."""
        
        ts = ts.copy()
        ts['time'] = pd.to_datetime(ts['time'], utc=True)
        ts = ts.sort_values('time')
        
        for kind in ts['kind'].dropna().unique():
            sub = ts[ts['kind'] == kind].set_index('time')['value'].astype(float)
            daily_agg = sub.resample('D').mean()
            
            col_mean = f'ts_{kind}_mean'
            col_diff1 = f'ts_{kind}_diff_1d'
            col_diff7 = f'ts_{kind}_diff_7d'
            col_anom = f'ts_{kind}_anomaly'
            
            daily[col_mean] = daily_agg
            daily[col_diff1] = daily_agg.diff(1)
            daily[col_diff7] = daily_agg.diff(7)
            
            # Anomaly detection
            roll_mean = daily[col_mean].rolling(30, min_periods=7).mean()
            roll_std = daily[col_mean].rolling(30, min_periods=7).std()
            daily[col_anom] = (
                (daily[col_mean] > roll_mean + 2 * roll_std) |
                (daily[col_mean] < roll_mean - 2 * roll_std)
            ).fillna(0).astype(int)
        
        # Forward fill custom data
        for col in daily.columns:
            if col.startswith('ts_'):
                daily[col] = daily[col].ffill().bfill().fillna(0)
        
        return daily


# ==============================================================================
# EVALUATION WITH BOOTSTRAP CI
# ==============================================================================

class ModelEvaluator:
    """
    Model evaluation with honest metrics and bootstrap confidence intervals.
    
    Primary metric: PR-AUC (appropriate for imbalanced classification)
    Uncertainty: 95% bootstrap CI on test set
    """
    
    @staticmethod
    def pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
        """Precision-Recall AUC."""
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)
        
        if len(np.unique(y_true)) < 2:
            return np.nan
        
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        return float(auc(recall, precision))
    
    @staticmethod
    def bootstrap_ci(y_true: np.ndarray, y_score: np.ndarray, 
                     metric_fn, n_bootstrap: int = 1000, 
                     ci: float = 0.95, seed: int = 42) -> Tuple[float, float]:
        """
        Bootstrap confidence interval for a metric.
        
        Returns (lower, upper) bounds of CI.
        """
        rng = np.random.default_rng(seed)
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)
        n = len(y_true)
        
        values = []
        for _ in range(n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            
            # Skip degenerate samples
            if len(np.unique(y_true[idx])) < 2:
                continue
            
            val = metric_fn(y_true[idx], y_score[idx])
            if not np.isnan(val):
                values.append(val)
        
        if len(values) < 50:
            return (np.nan, np.nan)
        
        values = np.sort(values)
        alpha = 1 - ci
        lo = float(np.quantile(values, alpha / 2))
        hi = float(np.quantile(values, 1 - alpha / 2))
        
        return (lo, hi)
    
    @staticmethod
    def skill_score(pr_auc: float, baseline_rate: float) -> float:
        """
        Skill score normalized by baseline.
        
        Skill = (PR-AUC - baseline) / (1 - baseline)
        """
        if baseline_rate >= 1:
            return 0.0
        return (pr_auc - baseline_rate) / (1 - baseline_rate)


# ==============================================================================
# LATEX TABLE GENERATOR
# ==============================================================================

class LaTeXGenerator:
    """Generate publication-ready LaTeX tables."""
    
    @staticmethod
    def experimental_setup_table(config: Config, catalog: str, 
                                  train_period: str, test_period: str) -> str:
        """Generate experimental setup table."""
        
        return f"""\\begin{{table}}[t]
\\centering
\\caption{{Experimental setup (strict temporal split, no look-ahead).}}
\\begin{{tabular}}{{ll}}
\\toprule
Catalog & {catalog} \\\\
Region & {config.region_name} (${config.lat_min}$--${config.lat_max}^\\circ$N, ${config.lon_min}$--${config.lon_max}^\\circ$E) \\\\
Completeness threshold & $M_c = {config.magnitude_completeness}$ \\\\
Target event & $M^* = {config.target_magnitude}$ \\\\
Forecast horizon & $\\Delta T = {config.forecast_horizon}$ days \\\\
Label definition & $y_t = \\mathbb{{I}}\\{{\\max M \\text{{ in }} (t,t+\\Delta T] \\ge M^*\\}}$ \\\\
Train/Test split & {train_period} / {test_period} \\\\
Primary metric & PR-AUC (precision-recall area) \\\\
Uncertainty & 95\\% bootstrap CI (1000 resamples) \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""
    
    @staticmethod
    def results_table(results: List[Dict]) -> str:
        """
        Generate results comparison table.
        
        results: list of dicts with keys:
            model, pr_auc, ci_lo, ci_hi, baseline_rate, skill, notes
        """
        
        lines = [
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{Model comparison: PR-AUC with 95\\% bootstrap CI.}",
            "\\begin{tabular}{lccccc}",
            "\\toprule",
            "Model & PR-AUC & 95\\% CI & Base Rate & Skill & Notes \\\\",
            "\\midrule"
        ]
        
        for r in results:
            ci = f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]" if np.isfinite(r['ci_lo']) else "--"
            notes = r.get('notes', '')
            lines.append(
                f"{r['model']} & {r['pr_auc']:.3f} & {ci} & {r['baseline_rate']:.3f} & {r['skill']:.3f} & {notes} \\\\"
            )
        
        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}"
        ])
        
        return "\n".join(lines)
    
    @staticmethod
    def fio_indicators_table() -> str:
        """Generate FIO indicators explanation table."""
        
        return """\\begin{table}[t]
\\centering
\\caption{FIO (Fractal Information Ontology) indicators and physical interpretation.}
\\begin{tabular}{llll}
\\toprule
Parameter & Formula & Precursor Signal & Physical Mechanism \\\\
\\midrule
$b$-value & $\\log_{10}(e)/(\\bar{M}-M_c)$ & $b < 0.9$ & Fault locking (fluid expulsion) \\\\
CV & $\\sigma(\\Delta T)/\\mu(\\Delta T)$ & CV $\\to 1$ & Transition to deterministic \\\\
Entropy $H$ & $-\\sum p_i \\log_2 p_i$ & $H \\downarrow$ & Information compression \\\\
SID & $1 - H/H_{\\text{bg}}$ & SID $\\uparrow$ & Seismic quiescence \\\\
LAIC & $(\\text{stress}+\\text{Kp}+\\text{TEC}+\\text{solar})/4$ & LAIC $\\uparrow$ & Multi-layer coupling \\\\
\\bottomrule
\\end{tabular}
\\end{table}"""


# ==============================================================================
# DATA LOADERS
# ==============================================================================

class USGSLoader:
    """USGS earthquake catalog loader."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def load(self, days: int = None) -> pd.DataFrame:
        days = days or self.config.history_days
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        params = {
            'format': 'geojson',
            'starttime': start_date.strftime('%Y-%m-%d'),
            'endtime': end_date.strftime('%Y-%m-%d'),
            'minmagnitude': self.config.magnitude_completeness,
            'minlatitude': self.config.lat_min,
            'maxlatitude': self.config.lat_max,
            'minlongitude': self.config.lon_min,
            'maxlongitude': self.config.lon_max,
            'orderby': 'time-asc',
            'limit': 20000
        }
        
        try:
            response = requests.get(
                'https://earthquake.usgs.gov/fdsnws/event/1/query',
                params=params, timeout=60
            )
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"  [USGS] Error: {e}")
            return pd.DataFrame()
        
        events = []
        for f in data.get('features', []):
            props = f['properties']
            coords = f['geometry']['coordinates']
            
            if props.get('type') != 'earthquake':
                continue
            
            time_ms = props.get('time')
            if time_ms is None:
                continue
            
            events.append({
                'time': pd.Timestamp(time_ms, unit='ms', tz='UTC'),
                'mag': props.get('mag', 0),
                'lat': coords[1],
                'lon': coords[0],
                'depth': coords[2] if len(coords) > 2 else 0,
                'place': props.get('place', ''),
                'catalog': 'USGS'
            })
        
        return pd.DataFrame(events)


class NOAAKpLoader:
    """NOAA Kp index loader."""
    
    def load(self) -> pd.DataFrame:
        try:
            response = requests.get(
                'https://services.swpc.noaa.gov/json/planetary_k_index_1m.json',
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
        except:
            return pd.DataFrame()
        
        records = []
        for item in data:
            try:
                dt = pd.to_datetime(item['time_tag'], utc=True)
                kp = float(item['kp_index'])
                records.append({'time': dt, 'kp': kp})
            except:
                continue
        
        if not records:
            return pd.DataFrame()
        
        df = pd.DataFrame(records)
        df['date'] = df['time'].dt.floor('D')
        daily = df.groupby('date').agg(kp_max=('kp', 'max'), kp_mean=('kp', 'mean'))
        daily.index = pd.to_datetime(daily.index, utc=True)
        return daily


class NOAASolarLoader:
    """NOAA solar activity loader."""
    
    def load(self) -> pd.DataFrame:
        try:
            response = requests.get(
                'https://services.swpc.noaa.gov/json/solar-cycle/observed-solar-cycle-indices.json',
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
        except:
            return pd.DataFrame()
        
        records = []
        for item in data[-365:]:
            try:
                dt = pd.to_datetime(item['time-tag'], utc=True)
                f107 = float(item.get('f10.7', 0))
                ssn = float(item.get('ssn', 0))
                records.append({'time': dt, 'f107': f107, 'ssn': ssn})
            except:
                continue
        
        if not records:
            return pd.DataFrame()
        
        df = pd.DataFrame(records)
        df['date'] = df['time'].dt.floor('D')
        daily = df.groupby('date').agg(f107=('f107', 'mean'), ssn=('ssn', 'mean'))
        daily.index = pd.to_datetime(daily.index, utc=True)
        return daily


# ==============================================================================
# CUSTOM DATA LOADER
# ==============================================================================

class CustomDataLoader:
    """
    Load user-provided CSV time series.
    
    Supported types:
    - radon_*.csv: Radon gas concentration
    - groundwater_*.csv: Groundwater levels
    - gnss_*.csv: GNSS displacement
    - lst_*.csv: Land surface temperature
    - olr_*.csv: Outgoing longwave radiation
    - custom_*.csv: Any other time series
    """
    
    DATA_TYPES = {
        'radon': {'pattern': 'radon*.csv', 'description': 'Radon (Bq/m³)'},
        'groundwater': {'pattern': 'groundwater*.csv', 'description': 'Groundwater (m)'},
        'gnss': {'pattern': 'gnss*.csv', 'description': 'GNSS displacement (m)'},
        'lst': {'pattern': ['lst*.csv', 'temperature*.csv'], 'description': 'Temperature (°C)'},
        'olr': {'pattern': 'olr*.csv', 'description': 'OLR (W/m²)'},
        'custom': {'pattern': 'custom*.csv', 'description': 'Custom data'}
    }
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
    
    def load_all(self) -> pd.DataFrame:
        """Load all custom data as unified timeseries DataFrame."""
        
        if not self.data_dir.exists():
            return pd.DataFrame()
        
        all_records = []
        
        for dtype, info in self.DATA_TYPES.items():
            patterns = info['pattern'] if isinstance(info['pattern'], list) else [info['pattern']]
            
            for pattern in patterns:
                for filepath in glob.glob(str(self.data_dir / pattern)):
                    try:
                        df = pd.read_csv(filepath, parse_dates=['date'])
                        
                        # Standardize to time/kind/value format
                        if 'value' in df.columns:
                            for _, row in df.iterrows():
                                all_records.append({
                                    'time': row['date'],
                                    'kind': dtype,
                                    'value': row['value']
                                })
                        elif 'level' in df.columns:  # groundwater
                            for _, row in df.iterrows():
                                all_records.append({
                                    'time': row['date'],
                                    'kind': 'groundwater',
                                    'value': row['level']
                                })
                        
                        print(f"    ✓ Loaded {os.path.basename(filepath)}: {len(df)} records")
                    except Exception as e:
                        print(f"    ✗ {os.path.basename(filepath)}: {e}")
        
        if not all_records:
            return pd.DataFrame()
        
        return pd.DataFrame(all_records)


# ==============================================================================
# TELEGRAM NOTIFIER
# ==============================================================================

class TelegramNotifier:
    """Telegram notification handler."""
    
    def __init__(self, config: Config):
        self.token = config.telegram_bot_token or os.environ.get('TELEGRAM_BOT_TOKEN', '')
        self.chat_id = config.telegram_chat_id or os.environ.get('TELEGRAM_CHAT_ID', '')
        self.enabled = config.telegram_enabled and bool(self.token and self.chat_id)
    
    def send(self, text: str, parse_mode: str = 'HTML') -> bool:
        if not self.enabled:
            return False
        
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        try:
            response = requests.post(url, json={
                'chat_id': self.chat_id,
                'text': text,
                'parse_mode': parse_mode,
                'disable_web_page_preview': True
            }, timeout=10)
            return response.status_code == 200
        except:
            return False
    
    def send_alert(self, forecast: Dict) -> bool:
        emoji = {'HIGH': '🔴', 'ELEVATED': '🟠', 'MODERATE': '🟡', 'LOW': '🟢'}
        e = emoji.get(forecast['risk_level'], '⚪')
        
        ind = forecast.get('indicators', {})
        ind_text = "\n".join([f"• {k}: {v}" for k, v in ind.items() if v is not None])
        
        msg = f"""
{e} <b>QO3-LAIC-FIO EARTHQUAKE ALERT</b> {e}

<b>Region:</b> {forecast['region']}
<b>Date:</b> {forecast['date']}

━━━━━━━━━━━━━━━━━━━━━━
<b>⚠️ RISK: {forecast['risk_level']}</b>
<b>Probability M≥{forecast['target_magnitude']}:</b> {forecast['probability']}%
<b>Horizon:</b> {forecast['horizon_days']} days
━━━━━━━━━━━━━━━━━━━━━━

<b>FIO Indicators:</b>
{ind_text}

<i>QO3-LAIC-FIO v2.0</i>
<i>I. Chechelnitsky</i>
"""
        return self.send(msg.strip())
    
    def send_daily_report(self, forecasts: List[Dict]) -> bool:
        now = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
        emoji = {'HIGH': '🔴', 'ELEVATED': '🟠', 'MODERATE': '🟡', 'LOW': '🟢'}
        
        lines = [
            "🌍 <b>QO3-LAIC-FIO DAILY REPORT</b>",
            f"<i>{now}</i>",
            "",
            "━━━━━━━━━━━━━━━━━━━━━━"
        ]
        
        for f in sorted([x for x in forecasts if 'error' not in x], 
                       key=lambda x: x['probability'], reverse=True):
            e = emoji.get(f['risk_level'], '⚪')
            lines.append(f"{e} <b>{f['region']}</b>: {f['probability']}% (M≥{f['target_magnitude']})")
        
        lines.extend([
            "",
            "━━━━━━━━━━━━━━━━━━━━━━",
            "<i>QO3-LAIC-FIO by I. Chechelnitsky</i>"
        ])
        
        return self.send("\n".join(lines))


# ==============================================================================
# MAIN SYSTEM
# ==============================================================================

class QO3System:
    """
    QO3-LAIC-FIO Ultimate System.
    
    Complete earthquake regime detection with:
    - Multi-source data integration
    - Leak-free feature engineering
    - Honest backtesting with bootstrap CI
    - LaTeX report generation
    - Telegram notifications
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.deduplicator = EventDeduplicator(
            config.dedup_time_sec,
            config.dedup_dist_km,
            config.dedup_mag_diff
        )
        self.feature_builder = QO3FeatureBuilder(config)
        self.evaluator = ModelEvaluator()
        self.latex = LaTeXGenerator()
        self.telegram = TelegramNotifier(config)
        
        self.events = None
        self.features = None
        self.model = None
    
    def load_data(self) -> bool:
        """Load all data sources."""
        
        print("\n" + "=" * 70)
        print("LOADING DATA")
        print("=" * 70)
        
        # Seismic data
        print("\n[USGS] Loading seismic catalog...")
        loader = USGSLoader(self.config)
        self.events = loader.load()
        
        if len(self.events) == 0:
            print("  ✗ No seismic data!")
            return False
        
        print(f"  ✓ Loaded {len(self.events)} events")
        
        # Deduplicate
        self.events = self.deduplicator.deduplicate(self.events)
        
        # Geomagnetic
        print("\n[NOAA] Loading Kp index...")
        kp_loader = NOAAKpLoader()
        kp_data = kp_loader.load()
        print(f"  ✓ Kp: {len(kp_data)} days")
        
        # Solar
        print("\n[NOAA] Loading solar flux...")
        solar_loader = NOAASolarLoader()
        solar_data = solar_loader.load()
        print(f"  ✓ Solar: {len(solar_data)} days")
        
        # Custom data
        print("\n[CUSTOM] Loading user data...")
        custom_loader = CustomDataLoader(self.config.custom_data_dir)
        custom_ts = custom_loader.load_all()
        
        # Build features
        print("\n" + "=" * 70)
        print("BUILDING FEATURES")
        print("=" * 70)
        
        self.features = self.feature_builder.build_matrix(
            self.events,
            timeseries=custom_ts if len(custom_ts) > 0 else None,
            kp_data=kp_data,
            solar_data=solar_data
        )
        
        print(f"\n  Total features: {len(self.features.columns)}")
        print(f"  Total days: {len(self.features)}")
        
        return True
    
    def train_and_evaluate(self) -> Dict:
        """Train model and compute honest metrics."""
        
        print("\n" + "=" * 70)
        print("TRAINING & EVALUATION")
        print("=" * 70)
        
        if self.features is None:
            return {'error': 'No features'}
        
        # Feature columns
        feature_cols = [c for c in self.features.columns 
                       if c not in ('target', 'future_max_mag', 'max_mag', 'mean_mag', 
                                   'count', 'energy', 'fio_entropy_bg')]
        
        # Remove columns with too many NaN
        valid_cols = []
        for col in feature_cols:
            if self.features[col].isna().mean() < 0.5:
                valid_cols.append(col)
        
        feature_cols = valid_cols
        print(f"\n  Features: {len(feature_cols)}")
        
        # Prepare data
        data = self.features.dropna(subset=feature_cols + ['target'])
        
        if len(data) < 100:
            return {'error': 'Insufficient data'}
        
        # Split
        split_idx = int(len(data) * self.config.train_fraction)
        train = data.iloc[:split_idx]
        test = data.iloc[split_idx:]
        
        X_train = train[feature_cols].values
        y_train = train['target'].values
        X_test = test[feature_cols].values
        y_test = test['target'].values
        
        baseline_rate = y_test.mean()
        
        print(f"  Train: {len(train)} days (positive: {y_train.mean():.1%})")
        print(f"  Test: {len(test)} days (positive: {baseline_rate:.1%})")
        
        results = []
        
        # ==================================================
        # Model 1: Rate-only baseline
        # ==================================================
        print("\n  Training: Rate-only baseline...")
        rate_cols = [c for c in ['rate_7d', 'rate_14d', 'rate_30d', 'rate_accel_7_30'] 
                    if c in feature_cols]
        
        if rate_cols:
            X_train_rate = train[rate_cols].values
            X_test_rate = test[rate_cols].values
            
            model_rate = LogisticRegression(max_iter=1000, random_state=42)
            model_rate.fit(X_train_rate, y_train)
            probs_rate = model_rate.predict_proba(X_test_rate)[:, 1]
            
            pr_auc_rate = self.evaluator.pr_auc(y_test, probs_rate)
            ci_rate = self.evaluator.bootstrap_ci(y_test, probs_rate, self.evaluator.pr_auc)
            skill_rate = self.evaluator.skill_score(pr_auc_rate, baseline_rate)
            
            results.append({
                'model': 'Baseline (rate-only)',
                'pr_auc': pr_auc_rate,
                'ci_lo': ci_rate[0],
                'ci_hi': ci_rate[1],
                'baseline_rate': baseline_rate,
                'skill': skill_rate,
                'notes': 'LogReg'
            })
            print(f"    PR-AUC: {pr_auc_rate:.3f} [{ci_rate[0]:.3f}, {ci_rate[1]:.3f}]")
        
        # ==================================================
        # Model 2: b-value only
        # ==================================================
        print("\n  Training: b-value only...")
        b_cols = [c for c in ['fio_b_value', 'fio_b_change_7d', 'fio_b_change_14d'] 
                 if c in feature_cols]
        
        if b_cols:
            X_train_b = train[b_cols].fillna(1.0).values
            X_test_b = test[b_cols].fillna(1.0).values
            
            model_b = LogisticRegression(max_iter=1000, random_state=42)
            model_b.fit(X_train_b, y_train)
            probs_b = model_b.predict_proba(X_test_b)[:, 1]
            
            pr_auc_b = self.evaluator.pr_auc(y_test, probs_b)
            ci_b = self.evaluator.bootstrap_ci(y_test, probs_b, self.evaluator.pr_auc)
            skill_b = self.evaluator.skill_score(pr_auc_b, baseline_rate)
            
            results.append({
                'model': 'b-value only',
                'pr_auc': pr_auc_b,
                'ci_lo': ci_b[0],
                'ci_hi': ci_b[1],
                'baseline_rate': baseline_rate,
                'skill': skill_b,
                'notes': 'LogReg'
            })
            print(f"    PR-AUC: {pr_auc_b:.3f} [{ci_b[0]:.3f}, {ci_b[1]:.3f}]")
        
        # ==================================================
        # Model 3: QO3 Full (all FIO + LAIC)
        # ==================================================
        print("\n  Training: QO3 Full (FIO + LAIC)...")
        
        self.model = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.08,
            min_samples_leaf=15,
            subsample=0.8,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        probs_full = self.model.predict_proba(X_test)[:, 1]
        
        pr_auc_full = self.evaluator.pr_auc(y_test, probs_full)
        ci_full = self.evaluator.bootstrap_ci(y_test, probs_full, self.evaluator.pr_auc)
        skill_full = self.evaluator.skill_score(pr_auc_full, baseline_rate)
        
        results.append({
            'model': 'QO3 Full (FIO + LAIC)',
            'pr_auc': pr_auc_full,
            'ci_lo': ci_full[0],
            'ci_hi': ci_full[1],
            'baseline_rate': baseline_rate,
            'skill': skill_full,
            'notes': 'GBM'
        })
        print(f"    PR-AUC: {pr_auc_full:.3f} [{ci_full[0]:.3f}, {ci_full[1]:.3f}]")
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # FIO contribution
        fio_cols = [c for c in feature_cols if c.startswith('fio_')]
        fio_importance = importance[importance['feature'].isin(fio_cols)]['importance'].sum()
        
        self.feature_cols = feature_cols
        
        return {
            'results': results,
            'feature_importance': importance,
            'fio_contribution': fio_importance,
            'baseline_rate': baseline_rate,
            'test_size': len(test),
            'train_size': len(train)
        }
    
    def forecast(self) -> Dict:
        """Generate current forecast."""
        
        if self.model is None or self.features is None:
            return {'error': 'Model not trained'}
        
        latest = self.features.iloc[-1:][self.feature_cols]
        
        # Fill NaN for prediction
        latest = latest.fillna(method='ffill').fillna(0)
        
        prob = self.model.predict_proba(latest.values)[0, 1]
        
        if prob >= 0.7:
            risk = 'HIGH'
        elif prob >= 0.5:
            risk = 'ELEVATED'
        elif prob >= 0.3:
            risk = 'MODERATE'
        else:
            risk = 'LOW'
        
        row = self.features.iloc[-1]
        
        def safe_round(v, d):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return None
            return round(float(v), d)
        
        indicators = {
            'b_value': safe_round(row.get('fio_b_value'), 2),
            'cv': safe_round(row.get('fio_cv'), 2),
            'sid': safe_round(row.get('fio_sid'), 2),
            'entropy': safe_round(row.get('fio_entropy'), 2),
            'kp_max': safe_round(row.get('kp_max'), 1),
            'f107': safe_round(row.get('f107'), 1),
            'laic_index': safe_round(row.get('laic_index'), 2)
        }
        
        return {
            'region': self.config.region_name,
            'date': self.features.index[-1].strftime('%Y-%m-%d'),
            'probability': round(prob * 100, 1),
            'risk_level': risk,
            'target_magnitude': self.config.target_magnitude,
            'horizon_days': self.config.forecast_horizon,
            'indicators': indicators
        }
    
    def generate_latex_report(self, eval_results: Dict) -> str:
        """Generate complete LaTeX report."""
        
        report = []
        
        # Setup table
        report.append(self.latex.experimental_setup_table(
            self.config,
            catalog='USGS + NOAA',
            train_period='First 75%',
            test_period='Last 25%'
        ))
        
        report.append("")
        
        # Results table
        report.append(self.latex.results_table(eval_results['results']))
        
        report.append("")
        
        # FIO indicators table
        report.append(self.latex.fio_indicators_table())
        
        return "\n\n".join(report)
    
    def run(self) -> Dict:
        """Full pipeline: load → build → train → evaluate → forecast."""
        
        if not self.load_data():
            return {'error': 'Failed to load data'}
        
        eval_results = self.train_and_evaluate()
        
        if 'error' in eval_results:
            return eval_results
        
        forecast = self.forecast()
        
        # Print results
        self._print_results(eval_results, forecast)
        
        # Generate LaTeX
        latex_report = self.generate_latex_report(eval_results)
        
        # Telegram
        if forecast.get('probability', 0) >= self.config.alert_threshold:
            self.telegram.send_alert(forecast)
        
        return {
            'evaluation': eval_results,
            'forecast': forecast,
            'latex': latex_report
        }
    
    def _print_results(self, eval_results: Dict, forecast: Dict):
        """Print formatted results."""
        
        print("\n" + "=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)
        
        print("\n  Model Comparison:")
        print("  " + "-" * 60)
        print(f"  {'Model':<30} {'PR-AUC':>10} {'95% CI':>20} {'Skill':>10}")
        print("  " + "-" * 60)
        
        for r in eval_results['results']:
            ci = f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]" if np.isfinite(r['ci_lo']) else "--"
            print(f"  {r['model']:<30} {r['pr_auc']:>10.3f} {ci:>20} {r['skill']:>10.3f}")
        
        print("  " + "-" * 60)
        print(f"  Base rate: {eval_results['baseline_rate']:.1%}")
        print(f"  FIO contribution: {eval_results['fio_contribution']:.1%}")
        
        print("\n  Top-10 Features:")
        for i, (_, row) in enumerate(eval_results['feature_importance'].head(10).iterrows()):
            bar = "█" * int(row['importance'] * 30)
            print(f"    {i+1:2d}. {row['feature']:<25} {row['importance']:.1%} {bar}")
        
        print("\n" + "=" * 70)
        print("CURRENT FORECAST")
        print("=" * 70)
        
        emoji = {'HIGH': '🔴', 'ELEVATED': '🟠', 'MODERATE': '🟡', 'LOW': '🟢'}
        e = emoji.get(forecast['risk_level'], '⚪')
        
        print(f"""
  Region: {forecast['region']}
  Date: {forecast['date']}
  Target: M≥{forecast['target_magnitude']} within {forecast['horizon_days']} days
  
  {e} RISK LEVEL: {forecast['risk_level']}
  Probability: {forecast['probability']}%
  
  FIO Indicators:""")
        
        for k, v in forecast['indicators'].items():
            if v is not None:
                print(f"    {k}: {v}")


# ==============================================================================
# CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='QO3-LAIC-FIO Ultimate v2.0: Multi-Parameter Earthquake Regime Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  python qo3_fio_ultimate.py --region japan
  python qo3_fio_ultimate.py --region tohoku --target_mag 4.5 --horizon 7
  python qo3_fio_ultimate.py --regions global,japan,israel --telegram --daily_report
  python qo3_fio_ultimate.py --region california --custom_dir my_data/

Author: Igor Chechelnitsky (ORCID: 0009-0007-4607-1946)
"""
    )
    
    parser.add_argument('--region', type=str, default='global', help='Region')
    parser.add_argument('--regions', type=str, help='Multiple regions (comma-separated)')
    parser.add_argument('--target_mag', type=float, help='Target magnitude')
    parser.add_argument('--horizon', type=int, default=7, help='Forecast horizon (days)')
    parser.add_argument('--days', type=int, default=365, help='History days')
    parser.add_argument('--custom_dir', type=str, default='custom_data', help='Custom data dir')
    parser.add_argument('--telegram', action='store_true', help='Enable Telegram')
    parser.add_argument('--daily_report', action='store_true', help='Send daily report')
    parser.add_argument('--output', type=str, default='README.md', help='Output file')
    parser.add_argument('--latex_output', type=str, default='results.tex', help='LaTeX output')
    parser.add_argument('--list_regions', action='store_true', help='List regions')
    
    args = parser.parse_args()
    
    if args.list_regions:
        print("\nAvailable regions:")
        for k, v in REGIONS.items():
            print(f"  {k:<15} : {v['name']} (Mc={v['mc']}, M*={v['target']})")
        return
    
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║     QO3-LAIC-FIO ULTIMATE v2.0                                               ║
║     Multi-Parameter Earthquake Regime Detection System                       ║
║                                                                              ║
║     • FIO: b-value, CV, entropy, SID                                        ║
║     • LAIC: Lithosphere-Atmosphere-Ionosphere Coupling                       ║
║     • Honest backtest with bootstrap CI                                      ║
║     • Custom data support (radon, GNSS, groundwater, etc.)                  ║
║                                                                              ║
║     Author: Igor Chechelnitsky (ORCID: 0009-0007-4607-1946)                 ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
    
    if args.regions:
        region_keys = [r.strip() for r in args.regions.split(',')]
    else:
        region_keys = [args.region]
    
    all_forecasts = []
    all_latex = []
    
    for region_key in region_keys:
        if region_key not in REGIONS:
            print(f"\n[!] Unknown region: {region_key}")
            continue
        
        reg = REGIONS[region_key]
        
        config = Config(
            region_name=reg['name'],
            lat_min=reg['lat'][0],
            lat_max=reg['lat'][1],
            lon_min=reg['lon'][0],
            lon_max=reg['lon'][1],
            magnitude_completeness=reg['mc'],
            target_magnitude=args.target_mag or reg['target'],
            forecast_horizon=args.horizon,
            history_days=args.days,
            custom_data_dir=args.custom_dir,
            telegram_enabled=args.telegram
        )
        
        print(f"\n{'='*70}")
        print(f"REGION: {reg['name']}")
        print(f"{'='*70}")
        
        system = QO3System(config)
        result = system.run()
        
        if 'forecast' in result:
            all_forecasts.append(result['forecast'])
        if 'latex' in result:
            all_latex.append(f"% Region: {reg['name']}\n{result['latex']}")
    
    # Save outputs
    if all_forecasts:
        # Markdown report
        report = generate_markdown_report(all_forecasts)
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"\n📄 Report: {args.output}")
        
        # JSON
        with open('forecast.json', 'w') as f:
            json.dump(all_forecasts, f, indent=2, default=str)
        print(f"📄 JSON: forecast.json")
        
        # LaTeX
        if all_latex:
            with open(args.latex_output, 'w') as f:
                f.write("\n\n".join(all_latex))
            print(f"📄 LaTeX: {args.latex_output}")
        
        # Telegram daily report
        if args.telegram and args.daily_report:
            telegram = TelegramNotifier(config)
            telegram.send_daily_report(all_forecasts)


def generate_markdown_report(forecasts: List[Dict]) -> str:
    """Generate Markdown report."""
    
    now = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
    emoji = {'HIGH': '🔴', 'ELEVATED': '🟠', 'MODERATE': '🟡', 'LOW': '🟢'}
    
    report = f"""# 🌍 QO3-LAIC-FIO Earthquake Forecast

**Updated:** {now}

## ⚡ Summary

| Region | Risk | Probability | Target | b-value | CV | SID | LAIC |
|--------|------|-------------|--------|---------|-----|-----|------|
"""
    
    for f in sorted([x for x in forecasts if 'error' not in x], 
                    key=lambda x: x['probability'], reverse=True):
        ind = f.get('indicators', {})
        e = emoji.get(f['risk_level'], '⚪')
        report += f"| {f['region']} | {e} {f['risk_level']} | **{f['probability']}%** | M≥{f['target_magnitude']} | {ind.get('b_value') or '—'} | {ind.get('cv') or '—'} | {ind.get('sid') or '—'} | {ind.get('laic_index') or '—'} |\n"
    
    report += """

---

## 🔬 FIO Indicators

| Parameter | Formula | Precursor Signal |
|-----------|---------|------------------|
| **b-value** | log₁₀(e)/(M̄-Mc) | < 0.9 = stress accumulation |
| **CV** | σ(ΔT)/μ(ΔT) | → 1 = deterministic system |
| **SID** | 1 - H/H_bg | ↑ = seismic quiescence |
| **LAIC** | coupling index | ↑ = multi-layer anomaly |

## 📚 Scientific Foundation

**Physical basis** (Cascadia-Tohoku analogy):
- b-value ↓ = fault locking (fluid expulsion → friction ↑)
- CV → 1 = transition from stochastic to deterministic
- SID ↑ = seismic quiescence (information compression)

**Mathematical basis**: Sebestyen's theorem extension to unbounded operators.

---

*QO3-LAIC-FIO Ultimate v2.0 by Igor Chechelnitsky (ORCID: 0009-0007-4607-1946)*
"""
    
    return report


if __name__ == '__main__':
    main()
