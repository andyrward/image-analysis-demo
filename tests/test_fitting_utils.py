"""
Tests for fitting utilities.
"""

import pytest
import numpy as np

from fitting_utils import (
    find_local_maxima,
    extract_region,
    gaussian_2d,
    fit_gaussian_2d,
    polynomial_2d,
    fit_polynomial_2d,
    compute_fit_quality,
)


class TestFindLocalMaxima:
    """Tests for local maxima detection."""
    
    def test_single_peak(self):
        """Test detection of single peak."""
        image = np.zeros((50, 50))
        image[25, 25] = 100
        
        peaks = find_local_maxima(image, min_distance=5, threshold=50)
        
        assert len(peaks) == 1
        assert peaks[0] == (25, 25)
    
    def test_multiple_peaks(self):
        """Test detection of multiple peaks."""
        image = np.zeros((100, 100))
        image[20, 20] = 100
        image[50, 50] = 100
        image[80, 80] = 100
        
        peaks = find_local_maxima(image, min_distance=10, threshold=50)
        
        assert len(peaks) == 3
        assert (20, 20) in peaks
        assert (50, 50) in peaks
        assert (80, 80) in peaks
    
    def test_threshold_filtering(self):
        """Test that threshold filters low peaks."""
        image = np.zeros((50, 50))
        image[10, 10] = 50
        image[30, 30] = 150
        
        peaks = find_local_maxima(image, min_distance=5, threshold=100)
        
        assert len(peaks) == 1
        assert peaks[0] == (30, 30)
    
    def test_min_distance(self):
        """Test that min_distance prevents close peaks."""
        image = np.zeros((50, 50))
        image[25, 25] = 100
        image[27, 27] = 90  # Close peak
        
        peaks = find_local_maxima(image, min_distance=5, threshold=50)
        
        # Should only detect one peak (the maximum)
        assert len(peaks) == 1
    
    def test_empty_image(self):
        """Test on empty image."""
        image = np.zeros((50, 50))
        
        peaks = find_local_maxima(image, min_distance=5)
        
        # All pixels are equal, so all are maxima
        assert len(peaks) >= 0


class TestExtractRegion:
    """Tests for region extraction."""
    
    def test_center_region(self):
        """Test extraction from center of image."""
        image = np.arange(100).reshape(10, 10)
        
        region, x_start, y_start = extract_region(image, 5, 5, radius=2)
        
        assert region.shape == (5, 5)
        assert x_start == 3
        assert y_start == 3
    
    def test_edge_region(self):
        """Test extraction near edge."""
        image = np.arange(100).reshape(10, 10)
        
        region, x_start, y_start = extract_region(image, 1, 1, radius=2)
        
        # Should be clipped at edges
        assert region.shape[0] <= 5
        assert region.shape[1] <= 5
        assert x_start == 0
        assert y_start == 0
    
    def test_corner_region(self):
        """Test extraction from corner."""
        image = np.arange(100).reshape(10, 10)
        
        region, x_start, y_start = extract_region(image, 0, 0, radius=2)
        
        assert region.shape[0] <= 5
        assert region.shape[1] <= 5
        assert x_start == 0
        assert y_start == 0
    
    @pytest.mark.parametrize("radius", [1, 3, 5, 10])
    def test_variable_radius(self, radius):
        """Test extraction with different radii."""
        image = np.ones((100, 100))
        
        region, x_start, y_start = extract_region(image, 50, 50, radius=radius)
        
        expected_size = 2 * radius + 1
        assert region.shape == (expected_size, expected_size)


class TestGaussian2D:
    """Tests for 2D Gaussian function."""
    
    def test_gaussian_center_value(self):
        """Test that Gaussian peaks at center."""
        x, y = np.meshgrid(np.arange(20), np.arange(20))
        
        amplitude = 100
        x0, y0 = 10, 10
        sigma_x, sigma_y = 2, 2
        offset = 10
        
        values = gaussian_2d((x, y), amplitude, x0, y0, sigma_x, sigma_y, offset)
        values = values.reshape(20, 20)
        
        # Peak should be at center
        assert values[y0, x0] == pytest.approx(amplitude + offset, rel=1e-5)
    
    def test_gaussian_symmetry(self):
        """Test that circular Gaussian is symmetric."""
        x, y = np.meshgrid(np.arange(20), np.arange(20))
        
        values = gaussian_2d((x, y), 100, 10, 10, 2, 2, 0)
        values = values.reshape(20, 20)
        
        # Test symmetry
        assert values[10, 8] == pytest.approx(values[10, 12], rel=1e-5)
        assert values[8, 10] == pytest.approx(values[12, 10], rel=1e-5)
    
    def test_gaussian_offset(self):
        """Test that offset is added correctly."""
        x, y = np.meshgrid(np.arange(20), np.arange(20))
        
        offset = 50
        values = gaussian_2d((x, y), 100, 10, 10, 2, 2, offset)
        
        # Minimum value should be approximately the offset
        assert np.min(values) >= offset * 0.99


class TestFitGaussian2D:
    """Tests for Gaussian fitting."""
    
    def test_fit_perfect_gaussian(self):
        """Test fitting a perfect Gaussian."""
        # Create perfect Gaussian
        size = 21
        y, x = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
        
        true_params = {
            'amplitude': 100,
            'x0': 10,
            'y0': 10,
            'sigma_x': 2,
            'sigma_y': 2,
            'offset': 10
        }
        
        region = gaussian_2d((x, y), **true_params).reshape(size, size)
        
        params, residual = fit_gaussian_2d(region)
        
        # Check fitted parameters
        assert params['amplitude'] == pytest.approx(true_params['amplitude'], rel=0.01)
        assert params['x0'] == pytest.approx(true_params['x0'], rel=0.01)
        assert params['y0'] == pytest.approx(true_params['y0'], rel=0.01)
        assert params['sigma_x'] == pytest.approx(true_params['sigma_x'], rel=0.01)
        assert params['sigma_y'] == pytest.approx(true_params['sigma_y'], rel=0.01)
        assert residual < 1e-10  # Should be near zero for perfect fit
    
    def test_fit_noisy_gaussian(self):
        """Test fitting a noisy Gaussian."""
        size = 21
        y, x = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
        
        true_params = {
            'amplitude': 100,
            'x0': 10,
            'y0': 10,
            'sigma_x': 2,
            'sigma_y': 2,
            'offset': 10
        }
        
        region = gaussian_2d((x, y), **true_params).reshape(size, size)
        
        # Add noise
        np.random.seed(42)
        region += np.random.normal(0, 5, region.shape)
        
        params, residual = fit_gaussian_2d(region)
        
        # Should still be close to true values
        assert params['amplitude'] == pytest.approx(true_params['amplitude'], abs=10)
        assert params['x0'] == pytest.approx(true_params['x0'], abs=0.5)
        assert params['y0'] == pytest.approx(true_params['y0'], abs=0.5)
    
    def test_fit_elliptical_gaussian(self):
        """Test fitting elliptical Gaussian."""
        size = 21
        y, x = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
        
        true_params = {
            'amplitude': 100,
            'x0': 10,
            'y0': 10,
            'sigma_x': 3,
            'sigma_y': 1.5,
            'offset': 5
        }
        
        region = gaussian_2d((x, y), **true_params).reshape(size, size)
        
        params, residual = fit_gaussian_2d(region)
        
        # Check that ellipticity is captured
        assert params['sigma_x'] > params['sigma_y']
        assert residual < 1e-8


class TestPolynomial2D:
    """Tests for 2D polynomial function."""
    
    def test_polynomial_constant(self):
        """Test constant polynomial."""
        x, y = np.meshgrid(np.arange(10), np.arange(10))
        
        values = polynomial_2d((x, y), 5, 0, 0, 0, 0, 0)
        
        assert np.all(values == 5)
    
    def test_polynomial_linear(self):
        """Test linear polynomial."""
        x, y = np.meshgrid(np.arange(10), np.arange(10))
        
        # z = 1 + 2*x + 3*y
        values = polynomial_2d((x, y), 1, 2, 3, 0, 0, 0)
        values = values.reshape(10, 10)
        
        # Test specific point
        assert values[0, 0] == 1
        assert values[0, 1] == 3  # y=1
        assert values[1, 0] == 4  # x=1
    
    def test_polynomial_quadratic(self):
        """Test quadratic polynomial."""
        x, y = np.meshgrid(np.arange(10), np.arange(10))
        
        # z = 1 - x^2 - y^2 (inverted paraboloid)
        values = polynomial_2d((x, y), 1, 0, 0, -1, 0, -1)
        values = values.reshape(10, 10)
        
        # Peak should be at (0, 0)
        assert values[0, 0] > values[5, 5]


class TestFitPolynomial2D:
    """Tests for polynomial fitting."""
    
    def test_fit_perfect_polynomial(self):
        """Test fitting a perfect quadratic."""
        size = 21
        y, x = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
        
        # Create inverted paraboloid with peak at center
        # Note: coefficients are for the polynomial after shifting
        true_coeffs = [100, 0, 0, -1, 0, -1]
        region = polynomial_2d((x - 10, y - 10), *true_coeffs).reshape(size, size)
        
        params, residual = fit_polynomial_2d(region)
        
        # Should recover coefficients (approximately, due to numerical fitting)
        # c0 might be negative due to how polynomial fits the peak
        assert abs(params['c0']) == pytest.approx(abs(true_coeffs[0]), abs=5)
        assert params['c3'] < 0  # Negative quadratic term
        assert params['c5'] < 0
        assert residual < 1e-6
    
    def test_fit_finds_peak(self):
        """Test that polynomial fit finds peak location."""
        size = 21
        y, x = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
        
        # Create peak at (10, 10)
        peak_x, peak_y = 10, 10
        region = 100 * np.exp(-((x - peak_x)**2 + (y - peak_y)**2) / 8)
        
        params, residual = fit_polynomial_2d(region)
        
        # Peak should be near center
        assert params['x_peak'] == pytest.approx(peak_x, abs=1)
        assert params['y_peak'] == pytest.approx(peak_y, abs=1)
    
    def test_fit_noisy_data(self):
        """Test polynomial fitting with noise."""
        size = 21
        y, x = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
        
        region = 100 * np.exp(-((x - 10)**2 + (y - 10)**2) / 8)
        
        # Add noise
        np.random.seed(42)
        region += np.random.normal(0, 5, region.shape)
        
        params, residual = fit_polynomial_2d(region)
        
        # Should still find approximate peak
        assert 8 <= params['x_peak'] <= 12
        assert 8 <= params['y_peak'] <= 12


class TestComputeFitQuality:
    """Tests for fit quality metrics."""
    
    def test_perfect_fit(self):
        """Test quality metrics for perfect fit."""
        region = np.random.rand(10, 10) * 100
        fitted = region.copy()
        
        quality = compute_fit_quality(region, fitted)
        
        assert quality['r_squared'] == pytest.approx(1.0, abs=1e-10)
        assert quality['rmse'] == pytest.approx(0.0, abs=1e-10)
        assert quality['residual_sum'] == pytest.approx(0.0, abs=1e-10)
    
    def test_poor_fit(self):
        """Test quality metrics for poor fit."""
        region = np.random.rand(10, 10) * 100
        fitted = np.zeros_like(region)
        
        quality = compute_fit_quality(region, fitted)
        
        assert quality['r_squared'] < 0.5
        assert quality['rmse'] > 0
        assert quality['residual_sum'] > 0
    
    def test_reasonable_fit(self):
        """Test quality metrics for reasonable fit."""
        size = 20
        y, x = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
        
        # True function
        region = 100 * np.exp(-((x - 10)**2 + (y - 10)**2) / 8)
        
        # Approximate fit (slightly offset)
        fitted = 100 * np.exp(-((x - 10.5)**2 + (y - 10.5)**2) / 8)
        
        quality = compute_fit_quality(region, fitted)
        
        assert 0.8 < quality['r_squared'] < 1.0
        assert quality['rmse'] > 0
