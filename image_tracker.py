"""
Image tracking module for spot detection and localization on 2D images.

This module provides a generic tracking function that supports multiple methods:
- pytrack: Weighted centroid tracking method
- gaussian: 2D Gaussian fit
- parabola: 2D Parabola fit
"""

import numpy as np
from scipy.optimize import curve_fit
from typing import Tuple, Dict, Any, Literal


TrackingMethod = Literal["pytrack", "gaussian", "parabola"]


def create_demo_image(size: int = 100, spot_center: Tuple[float, float] = None, 
                     spot_amplitude: float = 100.0, spot_sigma: float = 2.0) -> np.ndarray:
    """
    Create a demo 2D image with a single Gaussian spot.
    
    Args:
        size: Size of the square image (default: 100x100)
        spot_center: (x, y) position of the spot center. If None, defaults to image center.
        spot_amplitude: Amplitude of the Gaussian spot
        spot_sigma: Standard deviation of the Gaussian spot
        
    Returns:
        2D numpy array containing the image
    """
    if spot_center is None:
        spot_center = (size / 2.0, size / 2.0)
    
    # Create coordinate grids
    x = np.arange(size)
    y = np.arange(size)
    xx, yy = np.meshgrid(x, y)
    
    # Create Gaussian spot
    x0, y0 = spot_center
    image = spot_amplitude * np.exp(-((xx - x0)**2 + (yy - y0)**2) / (2 * spot_sigma**2))
    
    # Add small noise
    image += np.random.normal(0, 1, image.shape)
    
    return image


def gaussian_2d(coords: Tuple[np.ndarray, np.ndarray], amplitude: float, 
                x0: float, y0: float, sigma_x: float, sigma_y: float, 
                offset: float) -> np.ndarray:
    """
    2D Gaussian function for curve fitting.
    
    Args:
        coords: Tuple of (x, y) coordinate arrays
        amplitude: Peak amplitude
        x0, y0: Center position
        sigma_x, sigma_y: Standard deviations in x and y
        offset: Background offset
        
    Returns:
        Flattened array of function values
    """
    x, y = coords
    g = offset + amplitude * np.exp(-(((x - x0)**2 / (2 * sigma_x**2)) + 
                                       ((y - y0)**2 / (2 * sigma_y**2))))
    return g.ravel()


def parabola_2d(coords: Tuple[np.ndarray, np.ndarray], a: float, b: float, 
                c: float, d: float, e: float, f: float) -> np.ndarray:
    """
    2D Parabola function for curve fitting.
    
    The function is: f(x,y) = a*x^2 + b*y^2 + c*x + d*y + e*x*y + f
    
    Args:
        coords: Tuple of (x, y) coordinate arrays
        a, b, c, d, e, f: Parabola coefficients
        
    Returns:
        Flattened array of function values
    """
    x, y = coords
    p = a * x**2 + b * y**2 + c * x + d * y + e * x * y + f
    return p.ravel()


def track_spot_pytrack(image: np.ndarray, initial_guess: Tuple[float, float] = None) -> Dict[str, Any]:
    """
    Track a spot using the pytrack library.
    
    Note: This implementation uses a weighted centroid approach as the pytrack
    library may not be available or compatible.
    
    Args:
        image: 2D numpy array containing the image
        initial_guess: Initial (x, y) position guess
        
    Returns:
        Dictionary containing tracking results
    """
    try:
        # If no initial guess, use image center
        if initial_guess is None:
            initial_guess = (image.shape[1] / 2.0, image.shape[0] / 2.0)
        
        # Use weighted centroid approach for tracking
        # Find the centroid of the brightest region
        threshold = np.percentile(image, 90)
        mask = image > threshold
        
        if not np.any(mask):
            return {
                "method": "pytrack",
                "success": False,
                "error": "No bright spots found"
            }
        
        y_indices, x_indices = np.where(mask)
        weights = image[mask]
        
        x_center = np.average(x_indices, weights=weights)
        y_center = np.average(y_indices, weights=weights)
        
        return {
            "method": "pytrack",
            "success": True,
            "x": float(x_center),
            "y": float(y_center),
            "intensity": float(image[int(y_center), int(x_center)])
        }
    except Exception as e:
        return {
            "method": "pytrack",
            "success": False,
            "error": str(e)
        }


def track_spot_gaussian(image: np.ndarray, initial_guess: Tuple[float, float] = None, 
                       window_size: int = 20) -> Dict[str, Any]:
    """
    Track a spot using 2D Gaussian fitting.
    
    Args:
        image: 2D numpy array containing the image
        initial_guess: Initial (x, y) position guess
        window_size: Size of the window around the initial guess for fitting
        
    Returns:
        Dictionary containing tracking results
    """
    try:
        # If no initial guess, use brightest pixel
        if initial_guess is None:
            max_idx = np.unravel_index(np.argmax(image), image.shape)
            initial_guess = (max_idx[1], max_idx[0])  # (x, y)
        
        x0_init, y0_init = initial_guess
        
        # Extract region around the spot
        x_min = max(0, int(x0_init - window_size // 2))
        x_max = min(image.shape[1], int(x0_init + window_size // 2))
        y_min = max(0, int(y0_init - window_size // 2))
        y_max = min(image.shape[0], int(y0_init + window_size // 2))
        
        roi = image[y_min:y_max, x_min:x_max]
        
        # Create coordinate grids for the ROI
        x = np.arange(x_min, x_max)
        y = np.arange(y_min, y_max)
        xx, yy = np.meshgrid(x, y)
        
        # Initial parameter guesses
        amplitude_init = np.max(roi) - np.min(roi)
        offset_init = np.min(roi)
        sigma_init = 3.0
        
        initial_params = [amplitude_init, x0_init, y0_init, sigma_init, sigma_init, offset_init]
        
        # Fit the Gaussian
        popt, pcov = curve_fit(
            gaussian_2d, 
            (xx, yy), 
            roi.ravel(),
            p0=initial_params,
            maxfev=5000
        )
        
        amplitude, x0, y0, sigma_x, sigma_y, offset = popt
        
        # Calculate fit quality (R-squared)
        fitted = gaussian_2d((xx, yy), *popt).reshape(roi.shape)
        ss_res = np.sum((roi - fitted)**2)
        ss_tot = np.sum((roi - np.mean(roi))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        return {
            "method": "gaussian",
            "success": True,
            "x": float(x0),
            "y": float(y0),
            "amplitude": float(amplitude),
            "sigma_x": float(sigma_x),
            "sigma_y": float(sigma_y),
            "offset": float(offset),
            "r_squared": float(r_squared)
        }
    except Exception as e:
        return {
            "method": "gaussian",
            "success": False,
            "error": str(e)
        }


def track_spot_parabola(image: np.ndarray, initial_guess: Tuple[float, float] = None, 
                       window_size: int = 20) -> Dict[str, Any]:
    """
    Track a spot using 2D Parabola fitting.
    
    Args:
        image: 2D numpy array containing the image
        initial_guess: Initial (x, y) position guess
        window_size: Size of the window around the initial guess for fitting
        
    Returns:
        Dictionary containing tracking results
    """
    try:
        # If no initial guess, use brightest pixel
        if initial_guess is None:
            max_idx = np.unravel_index(np.argmax(image), image.shape)
            initial_guess = (max_idx[1], max_idx[0])  # (x, y)
        
        x0_init, y0_init = initial_guess
        
        # Extract region around the spot
        x_min = max(0, int(x0_init - window_size // 2))
        x_max = min(image.shape[1], int(x0_init + window_size // 2))
        y_min = max(0, int(y0_init - window_size // 2))
        y_max = min(image.shape[0], int(y0_init + window_size // 2))
        
        roi = image[y_min:y_max, x_min:x_max]
        
        # Create coordinate grids for the ROI
        x = np.arange(x_min, x_max)
        y = np.arange(y_min, y_max)
        xx, yy = np.meshgrid(x, y)
        
        # Initial parameter guesses for parabola
        initial_params = [-1.0, -1.0, 0.0, 0.0, 0.0, np.max(roi)]
        
        # Fit the parabola
        popt, pcov = curve_fit(
            parabola_2d, 
            (xx, yy), 
            roi.ravel(),
            p0=initial_params,
            maxfev=5000
        )
        
        a, b, c, d, e, f = popt
        
        # Find the maximum of the parabola
        # For a parabola z = ax^2 + by^2 + cx + dy + exy + f
        # The critical point is found by solving the system of equations:
        # dz/dx = 2ax + c + ey = 0
        # dz/dy = 2by + d + ex = 0
        # This gives us: 2ax + ey = -c and ex + 2by = -d
        
        det = 4 * a * b - e**2
        if abs(det) < 1e-10:
            raise ValueError("Parabola fitting failed: singular matrix")
        
        x_peak = (-2 * b * c + d * e) / det
        y_peak = (-2 * a * d + c * e) / det
        
        # Calculate fit quality (R-squared)
        fitted = parabola_2d((xx, yy), *popt).reshape(roi.shape)
        ss_res = np.sum((roi - fitted)**2)
        ss_tot = np.sum((roi - np.mean(roi))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        return {
            "method": "parabola",
            "success": True,
            "x": float(x_peak),
            "y": float(y_peak),
            "coefficients": {"a": float(a), "b": float(b), "c": float(c), 
                           "d": float(d), "e": float(e), "f": float(f)},
            "r_squared": float(r_squared)
        }
    except Exception as e:
        return {
            "method": "parabola",
            "success": False,
            "error": str(e)
        }


def track_spot(image: np.ndarray, method: TrackingMethod = "gaussian", 
              initial_guess: Tuple[float, float] = None, **kwargs) -> Dict[str, Any]:
    """
    Generic spot tracking function that supports multiple tracking methods.
    
    Args:
        image: 2D numpy array containing the image
        method: Tracking method to use ("pytrack", "gaussian", or "parabola")
        initial_guess: Initial (x, y) position guess for the spot
        **kwargs: Additional method-specific parameters
        
    Returns:
        Dictionary containing tracking results with at least:
        - method: str - The tracking method used
        - success: bool - Whether tracking was successful
        - x, y: float - Tracked position (if successful)
        - Additional method-specific results
        
    Raises:
        ValueError: If an unsupported method is specified
    """
    if method == "pytrack":
        return track_spot_pytrack(image, initial_guess)
    elif method == "gaussian":
        return track_spot_gaussian(image, initial_guess, **kwargs)
    elif method == "parabola":
        return track_spot_parabola(image, initial_guess, **kwargs)
    else:
        raise ValueError(f"Unsupported tracking method: {method}. "
                        f"Supported methods are: 'pytrack', 'gaussian', 'parabola'")
