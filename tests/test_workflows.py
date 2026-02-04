"""
Tests for high-level workflow functions.
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

from workflows import (
    track_particles,
    analyze_motion,
    complete_pipeline,
    demo_with_synthetic_data,
    compare_methods_on_synthetic,
)
from data_loader import generate_synthetic_data


@pytest.fixture
def test_frames():
    """Generate test frames."""
    return generate_synthetic_data(
        n_spots=3,
        spot_width=2.0,
        signal=1000,
        noise_magnitude=50,
        n_frames=10
    )


class TestTrackParticles:
    """Tests for track_particles workflow."""
    
    @pytest.mark.parametrize("method", ['trackpy', 'gaussian', 'polynomial'])
    def test_track_particles_all_methods(self, test_frames, method):
        """Test tracking with all detection methods."""
        trajectories = track_particles(
            test_frames,
            method=method,
            search_range=10.0,
            min_length=3
        )
        
        assert isinstance(trajectories, pd.DataFrame)
        assert 'particle' in trajectories.columns
        assert 'x' in trajectories.columns
        assert 'y' in trajectories.columns
        assert 'frame' in trajectories.columns
    
    def test_track_particles_trackpy(self, test_frames):
        """Test tracking with trackpy method."""
        trajectories = track_particles(
            test_frames,
            method='trackpy',
            diameter=11,
            minmass=100,
            search_range=10.0,
            min_length=3
        )
        
        assert len(trajectories) > 0
        assert trajectories['particle'].nunique() >= 1
    
    def test_track_particles_gaussian(self, test_frames):
        """Test tracking with Gaussian fitting."""
        trajectories = track_particles(
            test_frames,
            method='gaussian',
            min_distance=10,
            threshold=100,
            search_range=10.0,
            min_length=3
        )
        
        assert isinstance(trajectories, pd.DataFrame)
    
    def test_track_particles_with_memory(self, test_frames):
        """Test tracking with memory parameter."""
        trajectories = track_particles(
            test_frames,
            method='trackpy',
            diameter=11,
            search_range=10.0,
            memory=2,
            min_length=3
        )
        
        assert 'particle' in trajectories.columns
    
    def test_track_particles_invalid_method(self, test_frames):
        """Test that invalid method raises error."""
        with pytest.raises(ValueError):
            track_particles(test_frames, method='invalid')
    
    def test_track_particles_empty_frames(self):
        """Test tracking with empty frames."""
        empty_frames = [np.zeros((100, 100)) for _ in range(5)]
        
        trajectories = track_particles(
            empty_frames,
            method='trackpy',
            diameter=11,
            minmass=100,
            min_length=1
        )
        
        # May return empty or very few trajectories
        assert isinstance(trajectories, pd.DataFrame)


class TestAnalyzeMotion:
    """Tests for analyze_motion workflow."""
    
    def test_analyze_motion_basic(self):
        """Test basic motion analysis."""
        # Create simple trajectories
        data = []
        for particle in range(3):
            for frame in range(20):
                data.append({
                    'x': 50 + particle * 20 + np.random.normal(0, 0.5),
                    'y': 50 + particle * 20 + np.random.normal(0, 0.5),
                    'frame': frame,
                    'particle': particle
                })
        
        trajectories = pd.DataFrame(data)
        
        analysis = analyze_motion(trajectories, mpp=1.0, fps=1.0)
        
        assert isinstance(analysis, dict)
        assert 'msd' in analysis
        assert 'summary_stats' in analysis
        assert 'diffusion_coeff' in analysis
    
    def test_analyze_motion_summary_stats(self):
        """Test that summary statistics are computed."""
        data = []
        for frame in range(10):
            data.append({
                'x': 10 + frame,
                'y': 10,
                'frame': frame,
                'particle': 0
            })
        
        trajectories = pd.DataFrame(data)
        analysis = analyze_motion(trajectories)
        
        stats = analysis['summary_stats']
        assert 'n_trajectories' in stats
        assert 'mean_trajectory_length' in stats
        assert 'diffusion_coefficient' in stats
        assert stats['n_trajectories'] == 1
    
    def test_analyze_motion_scaling(self):
        """Test motion analysis with different scaling."""
        data = []
        for frame in range(10):
            data.append({
                'x': frame,
                'y': 0,
                'frame': frame,
                'particle': 0
            })
        
        trajectories = pd.DataFrame(data)
        
        analysis1 = analyze_motion(trajectories, mpp=1.0, fps=1.0)
        analysis2 = analyze_motion(trajectories, mpp=2.0, fps=1.0)
        
        # MSD should scale with mpp^2
        msd1_val = analysis1['msd']['msd'].iloc[1]
        msd2_val = analysis2['msd']['msd'].iloc[1]
        
        assert msd2_val == pytest.approx(4 * msd1_val, rel=0.01)


class TestCompletePipeline:
    """Tests for complete_pipeline workflow."""
    
    def test_complete_pipeline_basic(self, test_frames):
        """Test complete pipeline without visualization."""
        trajectories, analysis = complete_pipeline(
            test_frames,
            method='trackpy',
            search_range=10.0,
            min_length=3,
            visualize=False
        )
        
        assert isinstance(trajectories, pd.DataFrame)
        assert isinstance(analysis, dict)
        assert 'msd' in analysis
        assert 'summary_stats' in analysis
    
    def test_complete_pipeline_with_params(self, test_frames):
        """Test complete pipeline with custom parameters."""
        trajectories, analysis = complete_pipeline(
            test_frames,
            method='gaussian',
            search_range=10.0,
            min_length=3,
            mpp=0.1,
            fps=10.0,
            visualize=False,
            min_distance=10,
            threshold=100
        )
        
        assert isinstance(trajectories, pd.DataFrame)
        assert isinstance(analysis, dict)
    
    def test_complete_pipeline_with_visualization(self, test_frames):
        """Test complete pipeline with visualization."""
        # This should not raise errors
        trajectories, analysis = complete_pipeline(
            test_frames,
            method='trackpy',
            search_range=10.0,
            min_length=3,
            visualize=True
        )
        
        plt.close('all')  # Clean up plots
        
        assert isinstance(trajectories, pd.DataFrame)
        assert isinstance(analysis, dict)


class TestDemoWithSyntheticData:
    """Tests for demo workflow."""
    
    def test_demo_basic(self):
        """Test basic demo workflow."""
        frames, trajectories = demo_with_synthetic_data(
            n_spots=3,
            n_frames=10,
            method='trackpy',
            visualize=False
        )
        
        assert len(frames) == 10
        assert isinstance(trajectories, pd.DataFrame)
        assert 'particle' in trajectories.columns
    
    def test_demo_with_visualization(self):
        """Test demo with visualization enabled."""
        frames, trajectories = demo_with_synthetic_data(
            n_spots=3,
            n_frames=10,
            method='gaussian',
            visualize=True
        )
        
        plt.close('all')  # Clean up plots
        
        assert len(frames) == 10
        assert isinstance(trajectories, pd.DataFrame)
    
    @pytest.mark.parametrize("method", ['trackpy', 'gaussian', 'polynomial'])
    def test_demo_all_methods(self, method):
        """Test demo with all detection methods."""
        frames, trajectories = demo_with_synthetic_data(
            n_spots=3,
            n_frames=5,
            method=method,
            visualize=False
        )
        
        assert len(frames) == 5
        assert isinstance(trajectories, pd.DataFrame)
    
    def test_demo_variable_parameters(self):
        """Test demo with variable parameters."""
        frames, trajectories = demo_with_synthetic_data(
            n_spots=5,
            spot_width=3.0,
            signal=2000,
            noise_magnitude=100,
            n_frames=15,
            method='trackpy',
            visualize=False
        )
        
        assert len(frames) == 15
        assert all(f.shape == (512, 512) for f in frames)


class TestCompareMethodsOnSynthetic:
    """Tests for method comparison workflow."""
    
    def test_compare_methods_basic(self):
        """Test basic method comparison."""
        comparison = compare_methods_on_synthetic(
            n_spots=3,
            n_frames=5,
            methods=['trackpy', 'gaussian']
        )
        
        assert isinstance(comparison, pd.DataFrame)
        assert 'method' in comparison.columns
        assert 'recall' in comparison.columns
        assert 'precision' in comparison.columns
        assert len(comparison) == 2
        
        plt.close('all')  # Clean up plots
    
    def test_compare_methods_all(self):
        """Test comparison of all methods."""
        comparison = compare_methods_on_synthetic(
            n_spots=3,
            n_frames=5,
            methods=None  # Default: all methods
        )
        
        plt.close('all')
        
        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 3  # trackpy, gaussian, polynomial
    
    def test_compare_methods_single(self):
        """Test comparison with single method."""
        comparison = compare_methods_on_synthetic(
            n_spots=3,
            n_frames=5,
            methods=['trackpy']
        )
        
        plt.close('all')
        
        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 1
    
    def test_compare_methods_metrics(self):
        """Test that comparison includes all metrics."""
        comparison = compare_methods_on_synthetic(
            n_spots=3,
            n_frames=5,
            methods=['trackpy', 'gaussian']
        )
        
        plt.close('all')
        
        required_columns = [
            'method', 'n_true', 'n_detected', 'n_matched',
            'recall', 'precision', 'f1_score', 'mean_error'
        ]
        
        for col in required_columns:
            assert col in comparison.columns
    
    def test_compare_methods_high_snr(self):
        """Test method comparison with high SNR."""
        comparison = compare_methods_on_synthetic(
            n_spots=3,
            spot_width=2.0,
            signal=5000,
            noise_magnitude=10,
            n_frames=5,
            methods=['trackpy', 'gaussian']
        )
        
        plt.close('all')
        
        # All methods should have good performance with high SNR
        assert all(comparison['recall'] > 0.5)


class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_full_workflow_integration(self):
        """Test complete workflow from data generation to analysis."""
        # Generate data
        frames = generate_synthetic_data(
            n_spots=4,
            spot_width=2.0,
            signal=1000,
            noise_magnitude=50,
            n_frames=15
        )
        
        # Track particles
        trajectories = track_particles(
            frames,
            method='gaussian',
            search_range=10.0,
            min_length=5
        )
        
        # Analyze motion
        analysis = analyze_motion(trajectories, mpp=0.1, fps=10.0)
        
        # Verify results
        assert len(trajectories) > 0
        assert trajectories['particle'].nunique() >= 1
        assert 'msd' in analysis
        assert len(analysis['msd']) > 0
    
    def test_workflow_with_filtering(self):
        """Test workflow with trajectory filtering."""
        frames = generate_synthetic_data(
            n_spots=5,
            n_frames=20
        )
        
        # Track with strict filtering
        trajectories = track_particles(
            frames,
            method='trackpy',
            search_range=10.0,
            min_length=10  # Strict minimum length
        )
        
        # All trajectories should be long enough
        if len(trajectories) > 0:
            for particle_id in trajectories['particle'].unique():
                traj_len = len(trajectories[trajectories['particle'] == particle_id])
                assert traj_len >= 10
    
    def test_multiple_methods_same_data(self):
        """Test tracking same data with multiple methods."""
        frames = generate_synthetic_data(n_spots=3, n_frames=10)
        
        methods = ['trackpy', 'gaussian', 'polynomial']
        results = {}
        
        for method in methods:
            traj = track_particles(
                frames,
                method=method,
                search_range=10.0,
                min_length=3
            )
            results[method] = traj
        
        # All methods should return valid DataFrames
        for method, traj in results.items():
            assert isinstance(traj, pd.DataFrame)
            assert 'particle' in traj.columns


class TestErrorHandling:
    """Tests for error handling in workflows."""
    
    def test_invalid_method_raises_error(self, test_frames):
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError):
            track_particles(test_frames, method='nonexistent')
    
    def test_empty_frames_handling(self):
        """Test handling of empty frames."""
        empty_frames = [np.zeros((100, 100)) for _ in range(5)]
        
        # Should not crash
        trajectories = track_particles(
            empty_frames,
            method='trackpy',
            diameter=11,
            minmass=100,
            min_length=1
        )
        
        assert isinstance(trajectories, pd.DataFrame)
    
    def test_single_frame_handling(self):
        """Test handling of single frame."""
        single_frame = [generate_synthetic_data(n_spots=3, n_frames=1)[0]]
        
        # Should not crash even with min_length requirement
        trajectories = track_particles(
            single_frame,
            method='trackpy',
            diameter=11,
            search_range=10.0,
            min_length=1
        )
        
        assert isinstance(trajectories, pd.DataFrame)
