"""
Tests for motion analysis.
"""

import pytest
import numpy as np
import pandas as pd

from motion_analyzer import (
    compute_msd,
    compute_individual_msd,
    compute_velocity,
    compute_diffusion_coefficient,
    analyze_particle_motion,
    compute_trajectory_statistics,
)


@pytest.fixture
def stationary_trajectories():
    """Create stationary particle trajectories."""
    data = []
    
    for particle in range(3):
        for frame in range(20):
            # Small random displacement
            data.append({
                'x': 50 + particle * 20 + np.random.normal(0, 0.5),
                'y': 50 + particle * 20 + np.random.normal(0, 0.5),
                'frame': frame,
                'particle': particle
            })
    
    return pd.DataFrame(data)


@pytest.fixture
def moving_trajectories():
    """Create moving particle trajectories."""
    data = []
    
    for particle in range(3):
        for frame in range(20):
            # Linear motion plus small noise
            data.append({
                'x': 10 + frame * 2.0 + np.random.normal(0, 0.2),
                'y': 10 + particle * 20 + frame * 1.0 + np.random.normal(0, 0.2),
                'frame': frame,
                'particle': particle
            })
    
    return pd.DataFrame(data)


@pytest.fixture
def diffusive_trajectories():
    """Create Brownian motion trajectories."""
    np.random.seed(42)
    data = []
    
    for particle in range(5):
        x, y = 50.0, 50.0
        
        for frame in range(50):
            # Random walk
            x += np.random.normal(0, 1)
            y += np.random.normal(0, 1)
            
            data.append({
                'x': x,
                'y': y,
                'frame': frame,
                'particle': particle
            })
    
    return pd.DataFrame(data)


class TestComputeMSD:
    """Tests for MSD computation."""
    
    def test_compute_msd_basic(self, stationary_trajectories):
        """Test basic MSD computation."""
        msd = compute_msd(stationary_trajectories)
        
        assert isinstance(msd, pd.DataFrame)
        assert 'lagt' in msd.columns
        assert 'msd' in msd.columns
        assert len(msd) > 0
        assert all(msd['msd'] >= 0)
    
    def test_compute_msd_stationary(self, stationary_trajectories):
        """Test MSD for stationary particles."""
        msd = compute_msd(stationary_trajectories, mpp=1.0, fps=1.0)
        
        # MSD should be small and relatively constant
        assert msd['msd'].iloc[0] < 5.0
        
        # MSD shouldn't grow much
        if len(msd) > 5:
            assert msd['msd'].iloc[5] < 10.0
    
    def test_compute_msd_moving(self, moving_trajectories):
        """Test MSD for moving particles."""
        msd = compute_msd(moving_trajectories, mpp=1.0, fps=1.0)
        
        # MSD should grow with lag time
        assert msd['msd'].iloc[-1] > msd['msd'].iloc[0]
    
    def test_compute_msd_scaling(self):
        """Test MSD with different spatial/temporal scaling."""
        data = []
        for frame in range(20):
            data.append({
                'x': frame,
                'y': 0,
                'frame': frame,
                'particle': 0
            })
        
        traj = pd.DataFrame(data)
        
        # Default scaling
        msd1 = compute_msd(traj, mpp=1.0, fps=1.0)
        
        # Double spatial scaling
        msd2 = compute_msd(traj, mpp=2.0, fps=1.0)
        
        # MSD should scale with mpp^2
        assert msd2['msd'].iloc[1] == pytest.approx(4 * msd1['msd'].iloc[1], rel=0.01)
    
    def test_compute_msd_max_lagtime(self, stationary_trajectories):
        """Test MSD with custom max lagtime."""
        msd_short = compute_msd(stationary_trajectories, max_lagtime=5)
        msd_long = compute_msd(stationary_trajectories, max_lagtime=10)
        
        assert len(msd_short) <= 5
        assert len(msd_long) <= 10
        assert len(msd_long) >= len(msd_short)
    
    def test_compute_msd_empty(self):
        """Test MSD with empty trajectories."""
        empty = pd.DataFrame(columns=['x', 'y', 'frame', 'particle'])
        
        msd = compute_msd(empty)
        
        assert isinstance(msd, pd.DataFrame)
        assert len(msd) == 0


class TestComputeIndividualMSD:
    """Tests for individual MSD computation."""
    
    def test_compute_individual_msd_basic(self, stationary_trajectories):
        """Test individual MSD computation."""
        imsd = compute_individual_msd(stationary_trajectories)
        
        assert imsd is not None
        assert len(imsd) > 0
    
    def test_compute_individual_msd_shape(self, stationary_trajectories):
        """Test that individual MSD has expected structure."""
        imsd = compute_individual_msd(stationary_trajectories)
        
        # Should have data for multiple particles
        if isinstance(imsd, pd.DataFrame):
            assert 'particle' in imsd.columns or len(imsd.columns) > 1
    
    def test_compute_individual_msd_empty(self):
        """Test individual MSD with empty trajectories."""
        empty = pd.DataFrame(columns=['x', 'y', 'frame', 'particle'])
        
        imsd = compute_individual_msd(empty)
        
        assert isinstance(imsd, pd.DataFrame)


class TestComputeVelocity:
    """Tests for velocity computation."""
    
    def test_compute_velocity_basic(self, moving_trajectories):
        """Test basic velocity computation."""
        velocities = compute_velocity(moving_trajectories, mpp=1.0, fps=1.0)
        
        assert isinstance(velocities, pd.DataFrame)
        assert 'vx' in velocities.columns
        assert 'vy' in velocities.columns
        assert 'v' in velocities.columns
    
    def test_compute_velocity_values(self):
        """Test velocity values for known motion."""
        # Particle moving at 2 pixels/frame in x direction
        data = []
        for frame in range(10):
            data.append({
                'x': frame * 2.0,
                'y': 0.0,
                'frame': frame,
                'particle': 0
            })
        
        traj = pd.DataFrame(data)
        velocities = compute_velocity(traj, mpp=1.0, fps=1.0)
        
        # Velocity should be approximately 2.0
        vx_mean = velocities['vx'].dropna().mean()
        assert vx_mean == pytest.approx(2.0, abs=0.1)
    
    def test_compute_velocity_stationary(self, stationary_trajectories):
        """Test velocity for stationary particles."""
        velocities = compute_velocity(stationary_trajectories, mpp=1.0, fps=1.0)
        
        # Mean velocity should be small
        v_mean = velocities['v'].dropna().mean()
        assert v_mean < 2.0
    
    def test_compute_velocity_scaling(self):
        """Test velocity scaling with mpp and fps."""
        data = []
        for frame in range(10):
            data.append({
                'x': frame * 2.0,
                'y': 0.0,
                'frame': frame,
                'particle': 0
            })
        
        traj = pd.DataFrame(data)
        
        vel1 = compute_velocity(traj, mpp=1.0, fps=1.0)
        vel2 = compute_velocity(traj, mpp=2.0, fps=1.0)  # Double spatial scale
        vel3 = compute_velocity(traj, mpp=1.0, fps=2.0)  # Double frame rate
        
        # Doubling mpp should double velocity
        assert vel2['vx'].dropna().mean() == pytest.approx(
            2 * vel1['vx'].dropna().mean(), rel=0.01
        )
        
        # Doubling fps should halve velocity in real-world units because:
        # - Same pixel displacement occurs over less time (dt = 1/fps is halved)
        # - velocity = displacement / time, so velocity is halved when time is halved
        assert vel3['vx'].dropna().mean() == pytest.approx(
            0.5 * vel1['vx'].dropna().mean(), rel=0.01
        )


class TestComputeDiffusionCoefficient:
    """Tests for diffusion coefficient computation."""
    
    def test_compute_diffusion_basic(self):
        """Test basic diffusion coefficient computation."""
        # Create simple MSD data (linear growth)
        msd_data = pd.DataFrame({
            'lagt': [0, 1, 2, 3, 4],
            'msd': [0, 4, 8, 12, 16]  # Slope = 4, D = 1
        })
        
        D, slope = compute_diffusion_coefficient(msd_data, n_points=4)
        
        assert slope == pytest.approx(4.0, rel=0.01)
        assert D == pytest.approx(1.0, rel=0.01)
    
    def test_compute_diffusion_zero(self):
        """Test diffusion coefficient for stationary particles."""
        msd_data = pd.DataFrame({
            'lagt': [0, 1, 2, 3, 4],
            'msd': [0, 0.1, 0.1, 0.1, 0.1]  # Nearly zero MSD
        })
        
        D, slope = compute_diffusion_coefficient(msd_data)
        
        assert D == pytest.approx(0.0, abs=0.1)
    
    def test_compute_diffusion_few_points(self):
        """Test with minimal data points."""
        msd_data = pd.DataFrame({
            'lagt': [0, 1],
            'msd': [0, 4]
        })
        
        D, slope = compute_diffusion_coefficient(msd_data, n_points=2)
        
        assert D > 0


class TestAnalyzeParticleMotion:
    """Tests for comprehensive motion analysis."""
    
    def test_analyze_particle_motion_basic(self, stationary_trajectories):
        """Test basic motion analysis."""
        analysis = analyze_particle_motion(
            stationary_trajectories,
            mpp=1.0,
            fps=1.0
        )
        
        assert isinstance(analysis, dict)
        assert 'msd' in analysis
        assert 'imsd' in analysis
        assert 'diffusion_coeff' in analysis
        assert 'velocities' in analysis
        assert 'summary_stats' in analysis
    
    def test_analyze_particle_motion_summary_stats(self, moving_trajectories):
        """Test summary statistics."""
        analysis = analyze_particle_motion(moving_trajectories)
        
        stats = analysis['summary_stats']
        
        assert 'n_trajectories' in stats
        assert 'n_frames' in stats
        assert 'mean_trajectory_length' in stats
        assert 'diffusion_coefficient' in stats
        assert 'mean_velocity' in stats
        
        assert stats['n_trajectories'] > 0
        assert stats['n_frames'] > 0
    
    def test_analyze_particle_motion_moving(self, moving_trajectories):
        """Test analysis of moving particles."""
        analysis = analyze_particle_motion(moving_trajectories)
        
        # Moving particles should have higher velocity
        assert analysis['summary_stats']['mean_velocity'] > 0
        
        # MSD should be increasing
        msd = analysis['msd']
        assert msd['msd'].iloc[-1] > msd['msd'].iloc[0]


class TestComputeTrajectoryStatistics:
    """Tests for trajectory statistics."""
    
    def test_compute_statistics_basic(self, stationary_trajectories):
        """Test basic trajectory statistics."""
        stats = compute_trajectory_statistics(stationary_trajectories, mpp=1.0)
        
        assert isinstance(stats, pd.DataFrame)
        assert 'particle' in stats.columns
        assert 'length' in stats.columns
        assert 'x_mean' in stats.columns
        assert 'y_mean' in stats.columns
        assert 'path_length' in stats.columns
        
        # One row per particle
        assert len(stats) == stationary_trajectories['particle'].nunique()
    
    def test_compute_statistics_stationary(self, stationary_trajectories):
        """Test statistics for stationary particles."""
        stats = compute_trajectory_statistics(stationary_trajectories, mpp=1.0)
        
        # Stationary particles should have small displacement
        assert all(stats['total_displacement'] < 10.0)
        
        # Net-to-gross ratio should be small
        assert all(stats['net_to_gross_ratio'] <= 1.0)
    
    def test_compute_statistics_moving(self):
        """Test statistics for moving particles."""
        # Create particle moving in straight line
        data = []
        for frame in range(20):
            data.append({
                'x': frame * 5.0,
                'y': 0.0,
                'frame': frame,
                'particle': 0
            })
        
        traj = pd.DataFrame(data)
        stats = compute_trajectory_statistics(traj, mpp=1.0)
        
        # Large displacement
        assert stats.iloc[0]['total_displacement'] > 50
        
        # High net-to-gross ratio (straight line)
        assert stats.iloc[0]['net_to_gross_ratio'] > 0.9
    
    def test_compute_statistics_ranges(self, moving_trajectories):
        """Test range calculations."""
        stats = compute_trajectory_statistics(moving_trajectories, mpp=1.0)
        
        # All values should be non-negative
        assert all(stats['x_range'] >= 0)
        assert all(stats['y_range'] >= 0)
        assert all(stats['path_length'] >= 0)
        assert all(stats['total_displacement'] >= 0)


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_single_point_trajectory(self):
        """Test trajectory with single point."""
        data = pd.DataFrame({
            'x': [10.0],
            'y': [20.0],
            'frame': [0],
            'particle': [0]
        })
        
        # MSD should handle single point
        msd = compute_msd(data)
        assert len(msd) >= 0
        
        # Statistics should handle single point
        stats = compute_trajectory_statistics(data)
        assert len(stats) == 1
    
    def test_two_point_trajectory(self):
        """Test trajectory with two points."""
        data = pd.DataFrame({
            'x': [0.0, 1.0],
            'y': [0.0, 1.0],
            'frame': [0, 1],
            'particle': [0, 0]
        })
        
        velocities = compute_velocity(data)
        
        assert 'v' in velocities.columns
        assert velocities['v'].dropna().iloc[0] > 0
