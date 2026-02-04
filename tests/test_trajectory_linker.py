"""
Tests for trajectory linking.
"""

import pytest
import numpy as np
import pandas as pd

from trajectory_linker import (
    link_trajectories,
    filter_trajectories,
    filter_trajectories_by_displacement,
    compute_drift,
    subtract_drift,
    link_and_filter,
)


@pytest.fixture
def simple_features():
    """Create simple features for testing."""
    # Create features that form clear trajectories
    data = []
    
    # Particle 1: stationary at (10, 10)
    for frame in range(10):
        data.append({'x': 10 + np.random.normal(0, 0.1),
                    'y': 10 + np.random.normal(0, 0.1),
                    'frame': frame})
    
    # Particle 2: moving from (20, 20) to (30, 30)
    for frame in range(10):
        data.append({'x': 20 + frame,
                    'y': 20 + frame,
                    'frame': frame})
    
    # Particle 3: short trajectory
    for frame in range(3):
        data.append({'x': 50,
                    'y': 50,
                    'frame': frame})
    
    return pd.DataFrame(data)


@pytest.fixture
def synthetic_trajectories():
    """Create synthetic trajectories for testing."""
    np.random.seed(42)
    data = []
    
    n_particles = 5
    n_frames = 20
    
    for particle_id in range(n_particles):
        start_x = np.random.uniform(10, 100)
        start_y = np.random.uniform(10, 100)
        
        for frame in range(n_frames):
            x = start_x + np.random.normal(0, 1)
            y = start_y + np.random.normal(0, 1)
            data.append({'x': x, 'y': y, 'frame': frame})
    
    return pd.DataFrame(data)


class TestLinkTrajectories:
    """Tests for trajectory linking."""
    
    def test_link_basic(self, simple_features):
        """Test basic trajectory linking."""
        trajectories = link_trajectories(simple_features, search_range=5.0)
        
        assert isinstance(trajectories, pd.DataFrame)
        assert 'particle' in trajectories.columns
        assert 'x' in trajectories.columns
        assert 'y' in trajectories.columns
        assert 'frame' in trajectories.columns
        
        # Should have multiple particles
        assert trajectories['particle'].nunique() >= 2
    
    def test_link_with_memory(self, simple_features):
        """Test linking with memory parameter."""
        trajectories = link_trajectories(
            simple_features,
            search_range=5.0,
            memory=2
        )
        
        assert 'particle' in trajectories.columns
        assert len(trajectories) == len(simple_features)
    
    def test_link_adaptive(self, simple_features):
        """Test adaptive linking."""
        trajectories = link_trajectories(
            simple_features,
            search_range=5.0,
            adaptive_stop=0.1,
            adaptive_step=0.95
        )
        
        assert 'particle' in trajectories.columns
    
    def test_search_range_effect(self, simple_features):
        """Test that search range affects linking."""
        # Large search range should link more
        traj_large = link_trajectories(simple_features, search_range=50.0)
        
        # Small search range may create more particles
        traj_small = link_trajectories(simple_features, search_range=1.0)
        
        assert traj_large['particle'].nunique() <= traj_small['particle'].nunique()
    
    def test_empty_features(self):
        """Test linking with empty features."""
        empty = pd.DataFrame(columns=['x', 'y', 'frame'])
        
        # Trackpy may raise an error with empty features
        try:
            trajectories = link_trajectories(empty, search_range=5.0)
            assert len(trajectories) == 0
        except (IndexError, ValueError):
            # Acceptable behavior for empty input
            pass


class TestFilterTrajectories:
    """Tests for trajectory filtering."""
    
    def test_filter_by_length(self):
        """Test filtering by trajectory length."""
        # Create trajectories of different lengths
        data = []
        
        # Short trajectory (length 3)
        for frame in range(3):
            data.append({'x': 10, 'y': 10, 'frame': frame, 'particle': 0})
        
        # Long trajectory (length 10)
        for frame in range(10):
            data.append({'x': 20, 'y': 20, 'frame': frame, 'particle': 1})
        
        trajectories = pd.DataFrame(data)
        
        # Filter out short trajectories
        filtered = filter_trajectories(trajectories, min_length=5)
        
        assert filtered['particle'].nunique() == 1
        assert all(filtered['particle'] == 1)
    
    def test_filter_max_length(self):
        """Test filtering by maximum length."""
        data = []
        
        # Short trajectory
        for frame in range(3):
            data.append({'x': 10, 'y': 10, 'frame': frame, 'particle': 0})
        
        # Long trajectory
        for frame in range(20):
            data.append({'x': 20, 'y': 20, 'frame': frame, 'particle': 1})
        
        trajectories = pd.DataFrame(data)
        
        # Filter out long trajectories
        filtered = filter_trajectories(
            trajectories,
            min_length=2,
            max_length=10
        )
        
        assert filtered['particle'].nunique() == 1
        assert all(filtered['particle'] == 0)
    
    def test_no_filtering(self):
        """Test that min_length=0 keeps all trajectories."""
        data = []
        for particle in range(3):
            for frame in range(5):
                data.append({'x': particle * 10, 'y': 10,
                           'frame': frame, 'particle': particle})
        
        trajectories = pd.DataFrame(data)
        filtered = filter_trajectories(trajectories, min_length=1)
        
        assert len(filtered) == len(trajectories)


class TestFilterTrajectoryByDisplacement:
    """Tests for displacement-based filtering."""
    
    def test_filter_stationary(self):
        """Test filtering stationary particles."""
        data = []
        
        # Stationary particle
        for frame in range(10):
            data.append({'x': 10, 'y': 10, 'frame': frame, 'particle': 0})
        
        # Moving particle
        for frame in range(10):
            data.append({'x': 10 + frame * 5, 'y': 10, 
                       'frame': frame, 'particle': 1})
        
        trajectories = pd.DataFrame(data)
        
        # Filter out moving particles
        filtered = filter_trajectories_by_displacement(
            trajectories,
            max_displacement=5.0
        )
        
        assert filtered['particle'].nunique() == 1
        assert all(filtered['particle'] == 0)
    
    def test_filter_moving(self):
        """Test keeping only moving particles."""
        data = []
        
        # Stationary
        for frame in range(10):
            data.append({'x': 10, 'y': 10, 'frame': frame, 'particle': 0})
        
        # Moving
        for frame in range(10):
            data.append({'x': 10 + frame * 5, 'y': 10,
                       'frame': frame, 'particle': 1})
        
        trajectories = pd.DataFrame(data)
        
        # Keep moving particles
        filtered = filter_trajectories_by_displacement(
            trajectories,
            max_displacement=100.0
        )
        
        # Both should pass
        assert filtered['particle'].nunique() == 2


class TestComputeDrift:
    """Tests for drift computation."""
    
    def test_compute_drift_no_drift(self):
        """Test drift computation with stationary particles."""
        data = []
        
        for particle in range(5):
            for frame in range(10):
                data.append({
                    'x': 10 + particle * 10 + np.random.normal(0, 0.1),
                    'y': 10 + particle * 10 + np.random.normal(0, 0.1),
                    'frame': frame,
                    'particle': particle
                })
        
        trajectories = pd.DataFrame(data)
        drift = compute_drift(trajectories)
        
        assert isinstance(drift, pd.DataFrame)
        # Drift may have 'frame' as index or column depending on trackpy version
        assert 'x' in drift.columns
        assert 'y' in drift.columns
        
        # Drift should be near zero
        assert abs(drift['x'].mean()) < 1.0
        assert abs(drift['y'].mean()) < 1.0
    
    def test_compute_drift_with_drift(self):
        """Test drift computation with systematic drift."""
        data = []
        
        for particle in range(5):
            for frame in range(10):
                # Add systematic drift in x direction
                drift_x = frame * 2.0
                data.append({
                    'x': 10 + particle * 10 + drift_x,
                    'y': 10 + particle * 10,
                    'frame': frame,
                    'particle': particle
                })
        
        trajectories = pd.DataFrame(data)
        drift = compute_drift(trajectories)
        
        # Drift in x should be positive and increasing
        assert drift['x'].iloc[-1] > drift['x'].iloc[0]


class TestSubtractDrift:
    """Tests for drift subtraction."""
    
    def test_subtract_drift_basic(self):
        """Test basic drift subtraction."""
        data = []
        
        for particle in range(3):
            for frame in range(10):
                data.append({
                    'x': 10 + particle * 10 + frame * 2.0,
                    'y': 10 + particle * 10,
                    'frame': frame,
                    'particle': particle
                })
        
        trajectories = pd.DataFrame(data)
        
        # Subtract drift
        corrected = subtract_drift(trajectories)
        
        assert isinstance(corrected, pd.DataFrame)
        assert len(corrected) == len(trajectories)
        assert 'x' in corrected.columns
        assert 'y' in corrected.columns
    
    def test_subtract_drift_reduces_motion(self):
        """Test that drift subtraction reduces overall motion."""
        data = []
        
        for particle in range(5):
            for frame in range(10):
                data.append({
                    'x': 10 + particle * 10 + frame * 5.0,  # Strong drift
                    'y': 10 + particle * 10,
                    'frame': frame,
                    'particle': particle
                })
        
        trajectories = pd.DataFrame(data)
        
        # Compute range before
        x_range_before = (trajectories.groupby('particle')['x'].max() - 
                         trajectories.groupby('particle')['x'].min()).mean()
        
        corrected = subtract_drift(trajectories)
        
        # Reset index if needed to avoid ambiguous column/index issue
        if 'particle' in corrected.index.names:
            corrected = corrected.reset_index()
        
        x_range_after = (corrected.groupby('particle')['x'].max() - 
                        corrected.groupby('particle')['x'].min()).mean()
        
        # Range should be reduced
        assert x_range_after < x_range_before


class TestLinkAndFilter:
    """Tests for combined link and filter."""
    
    def test_link_and_filter_basic(self, simple_features):
        """Test combined link and filter."""
        trajectories = link_and_filter(
            simple_features,
            search_range=5.0,
            memory=0,
            min_length=5
        )
        
        assert isinstance(trajectories, pd.DataFrame)
        assert 'particle' in trajectories.columns
        
        # Should have filtered out short trajectories
        for particle_id in trajectories['particle'].unique():
            traj_length = len(trajectories[trajectories['particle'] == particle_id])
            assert traj_length >= 5
    
    def test_link_and_filter_with_drift_subtraction(self, simple_features):
        """Test with drift subtraction enabled."""
        trajectories = link_and_filter(
            simple_features,
            search_range=5.0,
            min_length=5,
            subtract_drift_flag=True
        )
        
        assert isinstance(trajectories, pd.DataFrame)
        assert 'particle' in trajectories.columns
    
    def test_link_and_filter_empty(self):
        """Test with empty features."""
        empty = pd.DataFrame(columns=['x', 'y', 'frame'])
        
        # May raise error with empty input
        try:
            trajectories = link_and_filter(
                empty,
                search_range=5.0,
                min_length=5
            )
            assert len(trajectories) == 0
        except (IndexError, ValueError):
            # Acceptable for empty input
            pass


class TestSyntheticTrajectories:
    """Test with synthetic trajectory data."""
    
    def test_link_synthetic(self, synthetic_trajectories):
        """Test linking synthetic trajectories."""
        trajectories = link_trajectories(
            synthetic_trajectories,
            search_range=5.0
        )
        
        assert 'particle' in trajectories.columns
        assert trajectories['particle'].nunique() >= 3
    
    def test_filter_synthetic(self, synthetic_trajectories):
        """Test filtering synthetic trajectories."""
        # First link
        trajectories = link_trajectories(
            synthetic_trajectories,
            search_range=5.0
        )
        
        # Then filter
        filtered = filter_trajectories(trajectories, min_length=10)
        
        assert len(filtered) <= len(trajectories)
        
        # All remaining trajectories should be long enough
        for particle_id in filtered['particle'].unique():
            traj_len = len(filtered[filtered['particle'] == particle_id])
            assert traj_len >= 10
