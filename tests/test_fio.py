#!/usr/bin/env python3
"""
Unit tests for QO3-LAIC-FIO

Run with: python -m pytest tests/
"""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, 'src')

from qo3_fio_ultimate import FIOEstimators, EventDeduplicator, ModelEvaluator


class TestFIOEstimators:
    """Tests for FIO indicator calculations."""
    
    def test_b_value_normal(self):
        """Test b-value for normal distribution."""
        fio = FIOEstimators()
        # Synthetic catalog with mean magnitude 3.0
        mags = np.random.normal(3.0, 0.5, 100)
        mags = mags[mags >= 2.0]
        
        b = fio.b_value_aki_utsu(mags, mc=2.0)
        
        # b should be positive and reasonable
        assert not np.isnan(b)
        assert 0.5 < b < 2.0
    
    def test_b_value_insufficient_data(self):
        """Test b-value with too few events."""
        fio = FIOEstimators()
        mags = np.array([3.0, 3.5, 4.0])
        
        b = fio.b_value_aki_utsu(mags, mc=2.5)
        
        assert np.isnan(b)
    
    def test_cv_poisson(self):
        """Test CV for Poisson-like process."""
        fio = FIOEstimators()
        # Exponential inter-event times (Poisson process)
        intervals = np.random.exponential(100, 50)
        timestamps = np.cumsum(intervals) * 1e9  # nanoseconds
        
        cv = fio.cv_interevent(timestamps)
        
        # Poisson process has CV ≈ 1
        assert not np.isnan(cv)
        assert 0.7 < cv < 1.5
    
    def test_cv_periodic(self):
        """Test CV for periodic process."""
        fio = FIOEstimators()
        # Perfectly periodic (CV → 0)
        timestamps = np.arange(0, 100, 1) * 86400 * 1e9  # daily, in ns
        
        cv = fio.cv_interevent(timestamps)
        
        assert not np.isnan(cv)
        assert cv < 0.1
    
    def test_entropy_uniform(self):
        """Test entropy for uniform distribution."""
        fio = FIOEstimators()
        # Uniform distribution has maximum entropy
        mags = np.random.uniform(2.0, 5.0, 100)
        
        H = fio.shannon_entropy(mags, bins=20)
        
        assert not np.isnan(H)
        assert H > 0
    
    def test_sid_quiescence(self):
        """Test SID for quiescent period."""
        fio = FIOEstimators()
        
        # Reduced entropy (quiescence)
        entropy = 1.0
        background = 2.0
        
        sid = fio.seismic_information_deficit(entropy, background)
        
        assert sid == 0.5  # (1 - 1/2) = 0.5


class TestDeduplicator:
    """Tests for event deduplication."""
    
    def test_no_duplicates(self):
        """Test with no duplicates."""
        dedup = EventDeduplicator()
        
        events = pd.DataFrame({
            'time': pd.to_datetime(['2025-01-01', '2025-01-02', '2025-01-03']),
            'lat': [35.0, 36.0, 37.0],
            'lon': [140.0, 141.0, 142.0],
            'mag': [3.0, 4.0, 5.0]
        })
        
        result = dedup.deduplicate(events)
        
        assert len(result) == 3
    
    def test_duplicate_removal(self):
        """Test duplicate removal."""
        dedup = EventDeduplicator(time_sec=60, dist_km=50, mag_diff=0.5)
        
        events = pd.DataFrame({
            'time': pd.to_datetime([
                '2025-01-01 00:00:00',
                '2025-01-01 00:00:30',  # Duplicate (30s later)
                '2025-01-02 00:00:00'
            ]),
            'lat': [35.0, 35.0, 36.0],
            'lon': [140.0, 140.0, 141.0],
            'mag': [4.0, 4.1, 5.0]
        })
        
        result = dedup.deduplicate(events)
        
        # Should keep the larger magnitude
        assert len(result) == 2
        assert 4.1 in result['mag'].values


class TestEvaluator:
    """Tests for model evaluation."""
    
    def test_pr_auc_perfect(self):
        """Test PR-AUC for perfect classifier."""
        evaluator = ModelEvaluator()
        
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        
        pr_auc = evaluator.pr_auc(y_true, y_score)
        
        assert pr_auc > 0.9
    
    def test_pr_auc_random(self):
        """Test PR-AUC for random classifier."""
        evaluator = ModelEvaluator()
        
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_score = np.random.random(100)
        
        pr_auc = evaluator.pr_auc(y_true, y_score)
        baseline = y_true.mean()
        
        # Random should be close to baseline
        assert abs(pr_auc - baseline) < 0.2
    
    def test_skill_score(self):
        """Test skill score calculation."""
        evaluator = ModelEvaluator()
        
        # Perfect skill
        skill = evaluator.skill_score(pr_auc=1.0, baseline_rate=0.5)
        assert skill == 1.0
        
        # No skill (same as baseline)
        skill = evaluator.skill_score(pr_auc=0.5, baseline_rate=0.5)
        assert skill == 0.0


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
