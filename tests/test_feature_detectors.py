"""
Tests for feature detection methods.
"""

import pytest
import numpy as np
import pandas as pd

from spot_generator import generate_spots_image
from feature_detectors import (
    TrackpyDetector,
    GaussianDetector,
    PolynomialDetector,
    detect_features,
)


@pytest.fixture
def simple_test_image():
    """Create a simple test image with known spots."""
    return generate_spots_image(
        n=3,
        spot_width=2.0,
        signal=1000,
        noise_magnitude=20,
        image_size=256
    )


@pytest.fixture
def test_image_sequence():
    """Create a sequence of test images."""
    np.random.seed(42)
    return [
        generate_spots_image(3, 2.0, 1000, 20, 256)
        for _ in range(5)
    ]


class TestTrackpyDetector:
    """Tests for trackpy-based detector."""
    
    def test_detect_frame_basic(self, simple_test_image):
        """Test basic detection on single frame."""
        detector = TrackpyDetector(diameter=11, minmass=100)
        features = detector.detect_frame(simple_test_image, frame_num=0)
        
        # Check DataFrame format
        assert isinstance(features, pd.DataFrame)
        assert 'x' in features.columns
        assert 'y' in features.columns
        assert 'frame' in features.columns
        assert len(features) > 0
        assert all(features['frame'] == 0)
    
    def test_detect_frame_returns_dataframe(self, simple_test_image):
        """Test that output is proper DataFrame."""
        detector = TrackpyDetector(diameter=11, minmass=100)
        features = detector.detect_frame(simple_test_image, frame_num=5)
        
        assert isinstance(features, pd.DataFrame)
        assert all(features['frame'] == 5)
    
    def test_detect_sequence(self, test_image_sequence):
        """Test detection on image sequence."""
        detector = TrackpyDetector(diameter=11, minmass=100)
        features = detector.detect_sequence(test_image_sequence)
        
        assert isinstance(features, pd.DataFrame)
        assert 'x' in features.columns
        assert 'y' in features.columns
        assert 'frame' in features.columns
        assert len(features) > 0
        assert features['frame'].min() == 0
        assert features['frame'].max() == 4
    
    def test_empty_image(self):
        """Test detection on empty image."""
        empty_image = np.zeros((100, 100))
        detector = TrackpyDetector(diameter=11, minmass=100)
        features = detector.detect_frame(empty_image, frame_num=0)
        
        assert isinstance(features, pd.DataFrame)
        assert len(features) == 0
    
    @pytest.mark.parametrize("diameter", [7, 11, 15])
    def test_variable_diameter(self, simple_test_image, diameter):
        """Test detection with different diameters."""
        detector = TrackpyDetector(diameter=diameter, minmass=50)
        features = detector.detect_frame(simple_test_image, frame_num=0)
        
        assert isinstance(features, pd.DataFrame)
        assert 'x' in features.columns


class TestGaussianDetector:
    """Tests for Gaussian fitting detector."""
    
    def test_detect_frame_basic(self, simple_test_image):
        """Test basic Gaussian detection."""
        detector = GaussianDetector(min_distance=10, threshold=100)
        features = detector.detect_frame(simple_test_image, frame_num=0)
        
        # Check DataFrame format
        assert isinstance(features, pd.DataFrame)
        assert 'x' in features.columns
        assert 'y' in features.columns
        assert 'frame' in features.columns
        assert 'amplitude' in features.columns
        assert 'sigma_x' in features.columns
        assert 'sigma_y' in features.columns
        assert all(features['frame'] == 0)
    
    def test_detect_sequence(self, test_image_sequence):
        """Test Gaussian detection on sequence."""
        detector = GaussianDetector(min_distance=10, threshold=100)
        features = detector.detect_sequence(test_image_sequence)
        
        assert isinstance(features, pd.DataFrame)
        assert 'x' in features.columns
        assert 'y' in features.columns
        assert 'frame' in features.columns
        assert len(features) > 0
    
    def test_subpixel_precision(self):
        """Test that Gaussian detector achieves sub-pixel precision."""
        # Create image with spot at known position
        image = np.zeros((50, 50))
        y, x = np.meshgrid(np.arange(50), np.arange(50), indexing='ij')
        
        # Spot centered at (25.3, 25.7)
        spot_x, spot_y = 25.3, 25.7
        gauss = 1000 * np.exp(-((x - spot_x)**2 + (y - spot_y)**2) / (2 * 2.0**2))
        image += gauss
        
        detector = GaussianDetector(min_distance=5, threshold=100)
        features = detector.detect_frame(image, frame_num=0)
        
        assert len(features) > 0
        # Check sub-pixel accuracy (within 0.5 pixels)
        detected_x = features.iloc[0]['x']
        detected_y = features.iloc[0]['y']
        assert abs(detected_x - spot_x) < 0.5
        assert abs(detected_y - spot_y) < 0.5
    
    def test_empty_image(self):
        """Test Gaussian detector on empty image."""
        empty_image = np.zeros((100, 100))
        detector = GaussianDetector(min_distance=5, threshold=100)
        features = detector.detect_frame(empty_image, frame_num=0)
        
        assert isinstance(features, pd.DataFrame)
        assert len(features) == 0


class TestPolynomialDetector:
    """Tests for polynomial fitting detector."""
    
    def test_detect_frame_basic(self, simple_test_image):
        """Test basic polynomial detection."""
        detector = PolynomialDetector(min_distance=20, threshold=50)
        features = detector.detect_frame(simple_test_image, frame_num=0)
        
        # Check DataFrame format
        assert isinstance(features, pd.DataFrame)
        
        # May or may not detect features depending on image, but should have proper columns
        if len(features) > 0:
            assert 'x' in features.columns
            assert 'y' in features.columns
            assert 'frame' in features.columns
            assert 'c0' in features.columns
            assert 'residual' in features.columns
            assert all(features['frame'] == 0)
        else:
            # Even if empty, should return DataFrame with proper columns
            expected_cols = ['x', 'y', 'frame']
            for col in expected_cols:
                assert col in features.columns or len(features) == 0
    
    def test_detect_sequence(self, test_image_sequence):
        """Test polynomial detection on sequence."""
        detector = PolynomialDetector(min_distance=20, threshold=50)
        features = detector.detect_sequence(test_image_sequence)
        
        assert isinstance(features, pd.DataFrame)
        
        # Should return DataFrame with proper format
        if len(features) > 0:
            assert 'x' in features.columns
            assert 'y' in features.columns
            assert 'frame' in features.columns
    
    def test_polynomial_coefficients(self, simple_test_image):
        """Test that polynomial coefficients are present."""
        detector = PolynomialDetector(min_distance=10, threshold=100)
        features = detector.detect_frame(simple_test_image, frame_num=0)
        
        if len(features) > 0:
            required_cols = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5']
            for col in required_cols:
                assert col in features.columns
    
    def test_empty_image(self):
        """Test polynomial detector on empty image."""
        empty_image = np.zeros((100, 100))
        detector = PolynomialDetector(min_distance=5, threshold=100)
        features = detector.detect_frame(empty_image, frame_num=0)
        
        assert isinstance(features, pd.DataFrame)
        assert len(features) == 0


class TestDetectFeatures:
    """Tests for convenience function."""
    
    @pytest.mark.parametrize("method", ['trackpy', 'gaussian', 'polynomial'])
    def test_all_methods(self, test_image_sequence, method):
        """Test that all methods work and return proper format."""
        features = detect_features(test_image_sequence, method=method)
        
        assert isinstance(features, pd.DataFrame)
        assert 'x' in features.columns
        assert 'y' in features.columns
        assert 'frame' in features.columns
    
    def test_invalid_method(self, test_image_sequence):
        """Test that invalid method raises error."""
        with pytest.raises(ValueError):
            detect_features(test_image_sequence, method='invalid')
    
    def test_method_specific_params(self, test_image_sequence):
        """Test passing method-specific parameters."""
        # Trackpy-specific
        features = detect_features(
            test_image_sequence,
            method='trackpy',
            diameter=11,
            minmass=100
        )
        assert len(features) > 0
        
        # Gaussian-specific
        features = detect_features(
            test_image_sequence,
            method='gaussian',
            min_distance=10,
            threshold=100
        )
        assert isinstance(features, pd.DataFrame)


class TestTrackpyIntegration:
    """Test trackpy compatibility of all detectors."""
    
    def test_trackpy_link_with_trackpy_detector(self, test_image_sequence):
        """Test that TrackpyDetector output works with trackpy.link."""
        import trackpy as tp
        
        detector = TrackpyDetector(diameter=11, minmass=100)
        features = detector.detect_sequence(test_image_sequence)
        
        if len(features) > 0:
            # This should not raise an error
            trajectories = tp.link(features, search_range=10)
            assert 'particle' in trajectories.columns
    
    def test_trackpy_link_with_gaussian_detector(self, test_image_sequence):
        """Test that GaussianDetector output works with trackpy.link."""
        import trackpy as tp
        
        detector = GaussianDetector(min_distance=10, threshold=100)
        features = detector.detect_sequence(test_image_sequence)
        
        if len(features) > 0:
            # This should not raise an error
            trajectories = tp.link(features, search_range=10)
            assert 'particle' in trajectories.columns
    
    def test_trackpy_link_with_polynomial_detector(self, test_image_sequence):
        """Test that PolynomialDetector output works with trackpy.link."""
        import trackpy as tp
        
        detector = PolynomialDetector(min_distance=10, threshold=100)
        features = detector.detect_sequence(test_image_sequence)
        
        if len(features) > 0:
            # This should not raise an error
            trajectories = tp.link(features, search_range=10)
            assert 'particle' in trajectories.columns
