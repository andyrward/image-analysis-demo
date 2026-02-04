"""
Smoke tests for visualization functions.

These tests verify that visualization functions run without errors,
but don't verify the actual visual output.
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

from visualizer import (
    plot_trajectories,
    annotate_frame,
    plot_msd,
    plot_gaussian_fit_surface,
    plot_polynomial_fit_surface,
    compare_detection_methods,
    plot_trajectory_overlay,
)
from data_loader import generate_synthetic_data


@pytest.fixture
def sample_trajectories():
    """Create sample trajectories for testing."""
    data = []
    
    # Create 3 trajectories
    for particle in range(3):
        for frame in range(20):
            x = 50 + particle * 50 + frame * 2 + np.random.normal(0, 1)
            y = 50 + particle * 50 + np.random.normal(0, 1)
            data.append({
                'x': x,
                'y': y,
                'frame': frame,
                'particle': particle
            })
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_msd():
    """Create sample MSD data."""
    return pd.DataFrame({
        'lagt': np.arange(10),
        'msd': np.arange(10) * 4 + np.random.normal(0, 0.5, 10)
    })


@pytest.fixture
def sample_image():
    """Create sample image."""
    return generate_synthetic_data(n_spots=3, n_frames=1)[0]


class TestPlotTrajectories:
    """Smoke tests for trajectory plotting."""
    
    def test_plot_trajectories_basic(self, sample_trajectories):
        """Test basic trajectory plotting."""
        fig, ax = plt.subplots()
        
        # Should not raise an error
        plot_trajectories(sample_trajectories, ax=ax)
        
        plt.close(fig)
    
    def test_plot_trajectories_no_ax(self, sample_trajectories):
        """Test plotting without providing axes."""
        ax = plot_trajectories(sample_trajectories)
        
        assert ax is not None
        plt.close('all')
    
    def test_plot_trajectories_colorby_particle(self, sample_trajectories):
        """Test coloring by particle."""
        fig, ax = plt.subplots()
        
        plot_trajectories(sample_trajectories, ax=ax, colorby='particle')
        
        plt.close(fig)
    
    def test_plot_trajectories_colorby_frame(self, sample_trajectories):
        """Test coloring by frame."""
        fig, ax = plt.subplots()
        
        plot_trajectories(sample_trajectories, ax=ax, colorby='frame')
        
        plt.close(fig)
    
    def test_plot_trajectories_subset(self, sample_trajectories):
        """Test plotting subset of particles."""
        fig, ax = plt.subplots()
        
        plot_trajectories(
            sample_trajectories,
            ax=ax,
            particle_ids=[0, 1]
        )
        
        plt.close(fig)
    
    def test_plot_trajectories_empty(self):
        """Test plotting empty trajectories."""
        empty = pd.DataFrame(columns=['x', 'y', 'frame', 'particle'])
        
        fig, ax = plt.subplots()
        plot_trajectories(empty, ax=ax)
        
        plt.close(fig)


class TestAnnotateFrame:
    """Smoke tests for frame annotation."""
    
    def test_annotate_frame_basic(self, sample_image, sample_trajectories):
        """Test basic frame annotation."""
        fig, ax = plt.subplots()
        
        annotate_frame(sample_image, sample_trajectories, frame_num=0, ax=ax)
        
        plt.close(fig)
    
    def test_annotate_frame_no_ax(self, sample_image, sample_trajectories):
        """Test annotation without providing axes."""
        ax = annotate_frame(sample_image, sample_trajectories, frame_num=0)
        
        assert ax is not None
        plt.close('all')
    
    def test_annotate_frame_no_labels(self, sample_image, sample_trajectories):
        """Test annotation without labels."""
        fig, ax = plt.subplots()
        
        annotate_frame(
            sample_image,
            sample_trajectories,
            frame_num=0,
            ax=ax,
            show_labels=False
        )
        
        plt.close(fig)
    
    def test_annotate_frame_custom_radius(self, sample_image, sample_trajectories):
        """Test annotation with custom circle radius."""
        fig, ax = plt.subplots()
        
        annotate_frame(
            sample_image,
            sample_trajectories,
            frame_num=0,
            ax=ax,
            circle_radius=10.0
        )
        
        plt.close(fig)
    
    def test_annotate_frame_no_features(self, sample_image):
        """Test annotation with no features."""
        empty_features = pd.DataFrame(columns=['x', 'y', 'frame'])
        
        fig, ax = plt.subplots()
        annotate_frame(sample_image, empty_features, frame_num=0, ax=ax)
        
        plt.close(fig)


class TestPlotMSD:
    """Smoke tests for MSD plotting."""
    
    def test_plot_msd_basic(self, sample_msd):
        """Test basic MSD plotting."""
        fig, ax = plt.subplots()
        
        plot_msd(sample_msd, ax=ax)
        
        plt.close(fig)
    
    def test_plot_msd_no_ax(self, sample_msd):
        """Test MSD plotting without axes."""
        ax = plot_msd(sample_msd)
        
        assert ax is not None
        plt.close('all')
    
    def test_plot_msd_with_fit(self, sample_msd):
        """Test MSD plotting with fit line."""
        fig, ax = plt.subplots()
        
        plot_msd(sample_msd, ax=ax, fit_line=True, n_fit_points=4)
        
        plt.close(fig)
    
    def test_plot_msd_loglog(self, sample_msd):
        """Test MSD plotting in log-log scale."""
        fig, ax = plt.subplots()
        
        plot_msd(sample_msd, ax=ax, loglog=True)
        
        plt.close(fig)
    
    def test_plot_msd_no_fit(self, sample_msd):
        """Test MSD plotting without fit line."""
        fig, ax = plt.subplots()
        
        plot_msd(sample_msd, ax=ax, fit_line=False)
        
        plt.close(fig)
    
    def test_plot_msd_few_points(self):
        """Test MSD plotting with few points."""
        msd = pd.DataFrame({
            'lagt': [0, 1, 2],
            'msd': [0, 1, 2]
        })
        
        fig, ax = plt.subplots()
        plot_msd(msd, ax=ax)
        
        plt.close(fig)


class TestPlotGaussianFitSurface:
    """Smoke tests for Gaussian fit visualization."""
    
    def test_plot_gaussian_fit_basic(self):
        """Test basic Gaussian fit surface plotting."""
        # Create test region and parameters
        size = 21
        y, x = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
        
        params = {
            'amplitude': 100,
            'x0': 10,
            'y0': 10,
            'sigma_x': 2,
            'sigma_y': 2,
            'offset': 10
        }
        
        # Create Gaussian region
        from fitting_utils import gaussian_2d
        region = gaussian_2d((x, y), **params).reshape(size, size)
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        plot_gaussian_fit_surface(region, params, ax=ax)
        
        plt.close(fig)
    
    def test_plot_gaussian_fit_no_ax(self):
        """Test Gaussian fit plotting without axes."""
        size = 21
        y, x = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
        
        params = {
            'amplitude': 100,
            'x0': 10,
            'y0': 10,
            'sigma_x': 2,
            'sigma_y': 2,
            'offset': 10
        }
        
        from fitting_utils import gaussian_2d
        region = gaussian_2d((x, y), **params).reshape(size, size)
        
        ax = plot_gaussian_fit_surface(region, params)
        
        assert ax is not None
        plt.close('all')


class TestPlotPolynomialFitSurface:
    """Smoke tests for polynomial fit visualization."""
    
    def test_plot_polynomial_fit_basic(self):
        """Test basic polynomial fit surface plotting."""
        size = 21
        y, x = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
        
        params = {
            'c0': 100,
            'c1': 0,
            'c2': 0,
            'c3': -1,
            'c4': 0,
            'c5': -1,
            'x_peak': 10,
            'y_peak': 10
        }
        
        from fitting_utils import polynomial_2d
        coeffs = [params['c0'], params['c1'], params['c2'],
                 params['c3'], params['c4'], params['c5']]
        region = polynomial_2d((x - 10, y - 10), *coeffs).reshape(size, size)
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        plot_polynomial_fit_surface(region, params, ax=ax)
        
        plt.close(fig)
    
    def test_plot_polynomial_fit_no_ax(self):
        """Test polynomial fit plotting without axes."""
        size = 21
        y, x = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
        
        params = {
            'c0': 100,
            'c1': 0,
            'c2': 0,
            'c3': -1,
            'c4': 0,
            'c5': -1,
            'x_peak': 10,
            'y_peak': 10
        }
        
        from fitting_utils import polynomial_2d
        coeffs = [params['c0'], params['c1'], params['c2'],
                 params['c3'], params['c4'], params['c5']]
        region = polynomial_2d((x - 10, y - 10), *coeffs).reshape(size, size)
        
        ax = plot_polynomial_fit_surface(region, params)
        
        assert ax is not None
        plt.close('all')


class TestCompareDetectionMethods:
    """Smoke tests for detection method comparison visualization."""
    
    def test_compare_detection_methods_basic(self, sample_image):
        """Test basic method comparison visualization."""
        # Create mock detections for multiple methods
        detections = {
            'method1': pd.DataFrame({
                'x': [50, 100, 150],
                'y': [50, 100, 150],
                'frame': [0, 0, 0]
            }),
            'method2': pd.DataFrame({
                'x': [55, 105, 155],
                'y': [55, 105, 155],
                'frame': [0, 0, 0]
            })
        }
        
        fig = compare_detection_methods(sample_image, detections, frame_num=0)
        
        assert fig is not None
        plt.close(fig)
    
    def test_compare_detection_methods_single(self, sample_image):
        """Test comparison with single method."""
        detections = {
            'method1': pd.DataFrame({
                'x': [50, 100],
                'y': [50, 100],
                'frame': [0, 0]
            })
        }
        
        fig = compare_detection_methods(sample_image, detections, frame_num=0)
        
        assert fig is not None
        plt.close(fig)
    
    def test_compare_detection_methods_custom_size(self, sample_image):
        """Test comparison with custom figure size."""
        detections = {
            'method1': pd.DataFrame({
                'x': [50],
                'y': [50],
                'frame': [0]
            }),
            'method2': pd.DataFrame({
                'x': [100],
                'y': [100],
                'frame': [0]
            })
        }
        
        fig = compare_detection_methods(
            sample_image,
            detections,
            frame_num=0,
            figsize=(20, 6)
        )
        
        assert fig is not None
        plt.close(fig)


class TestPlotTrajectoryOverlay:
    """Smoke tests for trajectory overlay visualization."""
    
    def test_plot_trajectory_overlay_basic(self, sample_image, sample_trajectories):
        """Test basic trajectory overlay."""
        fig, ax = plt.subplots()
        
        plot_trajectory_overlay(
            sample_image,
            sample_trajectories,
            frame_num=5,
            ax=ax
        )
        
        plt.close(fig)
    
    def test_plot_trajectory_overlay_no_ax(self, sample_image, sample_trajectories):
        """Test overlay without providing axes."""
        ax = plot_trajectory_overlay(
            sample_image,
            sample_trajectories,
            frame_num=5
        )
        
        assert ax is not None
        plt.close('all')
    
    def test_plot_trajectory_overlay_short_trail(self, sample_image, sample_trajectories):
        """Test overlay with short trail."""
        fig, ax = plt.subplots()
        
        plot_trajectory_overlay(
            sample_image,
            sample_trajectories,
            frame_num=5,
            trail_length=3,
            ax=ax
        )
        
        plt.close(fig)
    
    def test_plot_trajectory_overlay_long_trail(self, sample_image, sample_trajectories):
        """Test overlay with long trail."""
        fig, ax = plt.subplots()
        
        plot_trajectory_overlay(
            sample_image,
            sample_trajectories,
            frame_num=15,
            trail_length=20,
            ax=ax
        )
        
        plt.close(fig)
    
    def test_plot_trajectory_overlay_first_frame(self, sample_image, sample_trajectories):
        """Test overlay at first frame."""
        fig, ax = plt.subplots()
        
        plot_trajectory_overlay(
            sample_image,
            sample_trajectories,
            frame_num=0,
            ax=ax
        )
        
        plt.close(fig)


class TestVisualizationIntegration:
    """Integration tests for multiple visualizations."""
    
    def test_multiple_plots_same_figure(self, sample_trajectories, sample_msd, sample_image):
        """Test creating multiple plots in same figure."""
        fig = plt.figure(figsize=(15, 5))
        
        ax1 = plt.subplot(131)
        plot_trajectories(sample_trajectories, ax=ax1)
        
        ax2 = plt.subplot(132)
        plot_msd(sample_msd, ax=ax2)
        
        ax3 = plt.subplot(133)
        annotate_frame(sample_image, sample_trajectories, frame_num=0, ax=ax3)
        
        plt.close(fig)
    
    def test_sequential_plots(self, sample_trajectories):
        """Test creating multiple plots sequentially."""
        # First plot
        fig1, ax1 = plt.subplots()
        plot_trajectories(sample_trajectories, ax=ax1)
        plt.close(fig1)
        
        # Second plot
        fig2, ax2 = plt.subplots()
        plot_trajectories(sample_trajectories, ax=ax2, colorby='frame')
        plt.close(fig2)


class TestErrorHandling:
    """Test error handling in visualization functions."""
    
    def test_empty_trajectories_plot(self):
        """Test plotting empty trajectories."""
        empty = pd.DataFrame(columns=['x', 'y', 'frame', 'particle'])
        
        fig, ax = plt.subplots()
        
        # Should not raise an error
        plot_trajectories(empty, ax=ax)
        
        plt.close(fig)
    
    def test_empty_msd_plot(self):
        """Test plotting empty MSD."""
        empty_msd = pd.DataFrame(columns=['lagt', 'msd'])
        
        fig, ax = plt.subplots()
        
        # Should not raise an error
        plot_msd(empty_msd, ax=ax, fit_line=False)
        
        plt.close(fig)
    
    def test_single_point_trajectory(self):
        """Test plotting single-point trajectory."""
        single_point = pd.DataFrame({
            'x': [50],
            'y': [50],
            'frame': [0],
            'particle': [0]
        })
        
        fig, ax = plt.subplots()
        plot_trajectories(single_point, ax=ax)
        
        plt.close(fig)
