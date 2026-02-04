"""
Shared utilities for Gaussian and Polynomial fitting.

Provides functions for local maxima detection, region extraction, and fitting.
"""

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.optimize import curve_fit
from typing import Tuple, List, Optional


def find_local_maxima(image: np.ndarray, 
                      min_distance: int = 5,
                      threshold: Optional[float] = None) -> List[Tuple[int, int]]:
    """
    Find local maxima in an image.
    
    Parameters
    ----------
    image : np.ndarray
        2D image array
    min_distance : int
        Minimum distance between peaks
    threshold : float, optional
        Minimum intensity threshold
        
    Returns
    -------
    list of tuple
        List of (x, y) coordinates of local maxima
    """
    # Apply maximum filter
    footprint_size = 2 * min_distance + 1
    local_max = maximum_filter(image, size=footprint_size) == image
    
    # Apply threshold if provided
    if threshold is not None:
        local_max = local_max & (image > threshold)
    
    # Find coordinates
    y_coords, x_coords = np.where(local_max)
    
    # Return as list of (x, y) tuples
    return list(zip(x_coords, y_coords))


def extract_region(image: np.ndarray, 
                   center_x: float, 
                   center_y: float, 
                   radius: int = 5) -> Tuple[np.ndarray, int, int]:
    """
    Extract a square region around a point.
    
    Parameters
    ----------
    image : np.ndarray
        2D image array
    center_x : float
        X coordinate of center
    center_y : float
        Y coordinate of center
    radius : int
        Radius of region (region will be (2*radius+1) x (2*radius+1))
        
    Returns
    -------
    region : np.ndarray
        Extracted region
    x_start : int
        X coordinate of top-left corner
    y_start : int
        Y coordinate of top-left corner
    """
    center_x = int(round(center_x))
    center_y = int(round(center_y))
    
    # Calculate bounds
    y_start = max(0, center_y - radius)
    y_end = min(image.shape[0], center_y + radius + 1)
    x_start = max(0, center_x - radius)
    x_end = min(image.shape[1], center_x + radius + 1)
    
    # Extract region
    region = image[y_start:y_end, x_start:x_end]
    
    return region, x_start, y_start


def gaussian_2d(xy: Tuple[np.ndarray, np.ndarray], 
                amplitude: float, 
                x0: float, 
                y0: float, 
                sigma_x: float, 
                sigma_y: float, 
                offset: float) -> np.ndarray:
    """
    2D Gaussian function.
    
    Parameters
    ----------
    xy : tuple of np.ndarray
        Meshgrid coordinates (x, y)
    amplitude : float
        Gaussian amplitude
    x0, y0 : float
        Center coordinates
    sigma_x, sigma_y : float
        Standard deviations
    offset : float
        Background offset
        
    Returns
    -------
    np.ndarray
        Flattened Gaussian values
    """
    x, y = xy
    gauss = amplitude * np.exp(
        -(((x - x0) ** 2) / (2 * sigma_x ** 2) + 
          ((y - y0) ** 2) / (2 * sigma_y ** 2))
    )
    return (gauss + offset).ravel()


def fit_gaussian_2d(region: np.ndarray, 
                    initial_guess: Optional[dict] = None) -> Tuple[dict, float]:
    """
    Fit a 2D Gaussian to a region.
    
    Parameters
    ----------
    region : np.ndarray
        2D image region to fit
    initial_guess : dict, optional
        Initial parameter guesses with keys:
        'amplitude', 'x0', 'y0', 'sigma_x', 'sigma_y', 'offset'
        
    Returns
    -------
    params : dict
        Fitted parameters
    residual : float
        Sum of squared residuals
    """
    # Create coordinate arrays
    y_size, x_size = region.shape
    y = np.arange(y_size)
    x = np.arange(x_size)
    x_grid, y_grid = np.meshgrid(x, y)
    
    # Initial parameter guess
    if initial_guess is None:
        amplitude = region.max() - region.min()
        x0 = x_size / 2
        y0 = y_size / 2
        sigma = 2.0
        offset = region.min()
        p0 = [amplitude, x0, y0, sigma, sigma, offset]
    else:
        p0 = [
            initial_guess.get('amplitude', region.max() - region.min()),
            initial_guess.get('x0', x_size / 2),
            initial_guess.get('y0', y_size / 2),
            initial_guess.get('sigma_x', 2.0),
            initial_guess.get('sigma_y', 2.0),
            initial_guess.get('offset', region.min())
        ]
    
    try:
        # Fit Gaussian
        popt, _ = curve_fit(
            gaussian_2d, 
            (x_grid, y_grid), 
            region.ravel(),
            p0=p0,
            maxfev=5000
        )
        
        # Calculate residual
        fitted = gaussian_2d((x_grid, y_grid), *popt).reshape(region.shape)
        residual = np.sum((region - fitted) ** 2)
        
        params = {
            'amplitude': popt[0],
            'x0': popt[1],
            'y0': popt[2],
            'sigma_x': popt[3],
            'sigma_y': popt[4],
            'offset': popt[5]
        }
        
        return params, residual
        
    except RuntimeError:
        # If fitting fails, return initial guess with high residual
        params = {
            'amplitude': p0[0],
            'x0': p0[1],
            'y0': p0[2],
            'sigma_x': p0[3],
            'sigma_y': p0[4],
            'offset': p0[5]
        }
        return params, float('inf')


def polynomial_2d(xy: Tuple[np.ndarray, np.ndarray], *coeffs) -> np.ndarray:
    """
    2D polynomial function (quadratic).
    
    f(x, y) = c0 + c1*x + c2*y + c3*x^2 + c4*x*y + c5*y^2
    
    Parameters
    ----------
    xy : tuple of np.ndarray
        Meshgrid coordinates (x, y)
    coeffs : float
        Polynomial coefficients [c0, c1, c2, c3, c4, c5]
        
    Returns
    -------
    np.ndarray
        Flattened polynomial values
    """
    x, y = xy
    c0, c1, c2, c3, c4, c5 = coeffs
    poly = c0 + c1*x + c2*y + c3*x**2 + c4*x*y + c5*y**2
    return poly.ravel()


def fit_polynomial_2d(region: np.ndarray, 
                      degree: int = 2) -> Tuple[dict, float]:
    """
    Fit a 2D polynomial to a region (quadratic by default).
    
    Parameters
    ----------
    region : np.ndarray
        2D image region to fit
    degree : int
        Polynomial degree (only 2 supported currently)
        
    Returns
    -------
    params : dict
        Fitted coefficients {'c0', 'c1', 'c2', 'c3', 'c4', 'c5'}
        Also includes 'x_peak', 'y_peak' (peak location)
    residual : float
        Sum of squared residuals
    """
    if degree != 2:
        raise ValueError("Only degree=2 (quadratic) is currently supported")
    
    # Create coordinate arrays
    y_size, x_size = region.shape
    y = np.arange(y_size)
    x = np.arange(x_size)
    x_grid, y_grid = np.meshgrid(x, y)
    
    # Initial parameter guess
    p0 = [region.mean(), 0, 0, -1, 0, -1]
    
    try:
        # Fit polynomial
        popt, _ = curve_fit(
            polynomial_2d,
            (x_grid, y_grid),
            region.ravel(),
            p0=p0,
            maxfev=5000
        )
        
        # Calculate residual
        fitted = polynomial_2d((x_grid, y_grid), *popt).reshape(region.shape)
        residual = np.sum((region - fitted) ** 2)
        
        # Find peak location by taking derivative
        # df/dx = c1 + 2*c3*x + c4*y = 0
        # df/dy = c2 + c4*x + 2*c5*y = 0
        # Solving for x, y
        c0, c1, c2, c3, c4, c5 = popt
        
        # Solve 2x2 system
        A = np.array([[2*c3, c4], [c4, 2*c5]])
        b = np.array([-c1, -c2])
        
        try:
            peak_coords = np.linalg.solve(A, b)
            x_peak, y_peak = peak_coords
        except np.linalg.LinAlgError:
            # If singular, use center
            x_peak, y_peak = x_size / 2, y_size / 2
        
        params = {
            'c0': c0, 'c1': c1, 'c2': c2,
            'c3': c3, 'c4': c4, 'c5': c5,
            'x_peak': x_peak,
            'y_peak': y_peak
        }
        
        return params, residual
        
    except RuntimeError:
        # If fitting fails, return center with high residual
        params = {
            'c0': p0[0], 'c1': p0[1], 'c2': p0[2],
            'c3': p0[3], 'c4': p0[4], 'c5': p0[5],
            'x_peak': x_size / 2,
            'y_peak': y_size / 2
        }
        return params, float('inf')


def compute_fit_quality(region: np.ndarray, 
                       fitted_values: np.ndarray) -> dict:
    """
    Compute quality metrics for a fit.
    
    Parameters
    ----------
    region : np.ndarray
        Original image region
    fitted_values : np.ndarray
        Fitted values
        
    Returns
    -------
    dict
        Quality metrics including:
        - 'r_squared': Coefficient of determination
        - 'rmse': Root mean squared error
        - 'residual_sum': Sum of squared residuals
    """
    residuals = region.ravel() - fitted_values.ravel()
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((region.ravel() - np.mean(region)) ** 2)
    
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    rmse = np.sqrt(ss_res / region.size)
    
    return {
        'r_squared': r_squared,
        'rmse': rmse,
        'residual_sum': ss_res
    }
