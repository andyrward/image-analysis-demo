"""
Feature detection methods for particle tracking.

All detectors return pandas DataFrames with ['x', 'y', 'frame'] columns
for trackpy compatibility.
"""

import numpy as np
import pandas as pd
import trackpy as tp
from typing import List, Optional, Union
from fitting_utils import (
    find_local_maxima, extract_region, 
    fit_gaussian_2d, fit_polynomial_2d
)


class TrackpyDetector:
    """Detector using trackpy's built-in feature detection."""
    
    def __init__(self, diameter: int = 11, minmass: float = 100.0, 
                 separation: Optional[float] = None):
        """
        Initialize trackpy detector.
        
        Parameters
        ----------
        diameter : int
            Feature diameter (should be odd)
        minmass : float
            Minimum integrated intensity
        separation : float, optional
            Minimum separation between features
        """
        self.diameter = diameter
        self.minmass = minmass
        self.separation = separation if separation is not None else diameter
        
    def detect_frame(self, image: np.ndarray, frame_num: int = 0) -> pd.DataFrame:
        """
        Detect features in a single frame.
        
        Parameters
        ----------
        image : np.ndarray
            2D image array
        frame_num : int
            Frame number
            
        Returns
        -------
        pd.DataFrame
            DataFrame with columns ['x', 'y', 'frame', ...other trackpy columns]
        """
        features = tp.locate(
            image,
            diameter=self.diameter,
            minmass=self.minmass,
            separation=self.separation
        )
        features['frame'] = frame_num
        return features
    
    def detect_sequence(self, frames: Union[List[np.ndarray], object]) -> pd.DataFrame:
        """
        Detect features in a sequence of frames.
        
        Parameters
        ----------
        frames : list of np.ndarray or video object
            Image sequence
            
        Returns
        -------
        pd.DataFrame
            DataFrame with columns ['x', 'y', 'frame', ...]
        """
        all_features = []
        for frame_num, frame in enumerate(frames):
            features = self.detect_frame(np.array(frame), frame_num)
            all_features.append(features)
        
        if all_features:
            return pd.concat(all_features, ignore_index=True)
        else:
            return pd.DataFrame(columns=['x', 'y', 'frame'])


class GaussianDetector:
    """Detector using 2D Gaussian fitting for sub-pixel accuracy."""
    
    def __init__(self, 
                 min_distance: int = 5,
                 threshold: Optional[float] = None,
                 fit_radius: int = 5,
                 max_residual: float = 1e6):
        """
        Initialize Gaussian detector.
        
        Parameters
        ----------
        min_distance : int
            Minimum distance between peaks for initial detection
        threshold : float, optional
            Minimum intensity threshold
        fit_radius : int
            Radius of region to extract for fitting
        max_residual : float
            Maximum allowed residual for accepting fit
        """
        self.min_distance = min_distance
        self.threshold = threshold
        self.fit_radius = fit_radius
        self.max_residual = max_residual
        
    def detect_frame(self, image: np.ndarray, frame_num: int = 0) -> pd.DataFrame:
        """
        Detect features in a single frame using Gaussian fitting.
        
        Parameters
        ----------
        image : np.ndarray
            2D image array
        frame_num : int
            Frame number
            
        Returns
        -------
        pd.DataFrame
            DataFrame with columns ['x', 'y', 'frame', 'amplitude', 'sigma_x', 
            'sigma_y', 'offset', 'residual']
        """
        # Find initial peak positions
        peaks = find_local_maxima(image, self.min_distance, self.threshold)
        
        results = []
        for peak_x, peak_y in peaks:
            # Extract region around peak
            region, x_start, y_start = extract_region(
                image, peak_x, peak_y, self.fit_radius
            )
            
            if region.size == 0:
                continue
            
            # Fit Gaussian
            params, residual = fit_gaussian_2d(region)
            
            # Accept fit if residual is acceptable
            if residual <= self.max_residual:
                # Convert local coordinates to global
                x_global = x_start + params['x0']
                y_global = y_start + params['y0']
                
                results.append({
                    'x': x_global,
                    'y': y_global,
                    'frame': frame_num,
                    'amplitude': params['amplitude'],
                    'sigma_x': params['sigma_x'],
                    'sigma_y': params['sigma_y'],
                    'offset': params['offset'],
                    'residual': residual
                })
        
        return pd.DataFrame(results)
    
    def detect_sequence(self, frames: Union[List[np.ndarray], object]) -> pd.DataFrame:
        """
        Detect features in a sequence of frames.
        
        Parameters
        ----------
        frames : list of np.ndarray or video object
            Image sequence
            
        Returns
        -------
        pd.DataFrame
            DataFrame with columns ['x', 'y', 'frame', ...]
        """
        all_features = []
        for frame_num, frame in enumerate(frames):
            features = self.detect_frame(np.array(frame), frame_num)
            all_features.append(features)
        
        if all_features:
            return pd.concat(all_features, ignore_index=True)
        else:
            return pd.DataFrame(columns=['x', 'y', 'frame'])


class PolynomialDetector:
    """Detector using 2D polynomial fitting for peak location."""
    
    def __init__(self, 
                 min_distance: int = 5,
                 threshold: Optional[float] = None,
                 fit_radius: int = 5,
                 max_residual: float = 1e6):
        """
        Initialize polynomial detector.
        
        Parameters
        ----------
        min_distance : int
            Minimum distance between peaks for initial detection
        threshold : float, optional
            Minimum intensity threshold
        fit_radius : int
            Radius of region to extract for fitting
        max_residual : float
            Maximum allowed residual for accepting fit
        """
        self.min_distance = min_distance
        self.threshold = threshold
        self.fit_radius = fit_radius
        self.max_residual = max_residual
        
    def detect_frame(self, image: np.ndarray, frame_num: int = 0) -> pd.DataFrame:
        """
        Detect features in a single frame using polynomial fitting.
        
        Parameters
        ----------
        image : np.ndarray
            2D image array
        frame_num : int
            Frame number
            
        Returns
        -------
        pd.DataFrame
            DataFrame with columns ['x', 'y', 'frame', 'c0', ..., 'c5', 'residual']
        """
        # Find initial peak positions
        peaks = find_local_maxima(image, self.min_distance, self.threshold)
        
        results = []
        for peak_x, peak_y in peaks:
            # Extract region around peak
            region, x_start, y_start = extract_region(
                image, peak_x, peak_y, self.fit_radius
            )
            
            if region.size == 0:
                continue
            
            # Fit polynomial
            params, residual = fit_polynomial_2d(region)
            
            # Accept fit if residual is acceptable and peak is within region
            if (residual <= self.max_residual and 
                0 <= params['x_peak'] < region.shape[1] and
                0 <= params['y_peak'] < region.shape[0]):
                
                # Convert local coordinates to global
                x_global = x_start + params['x_peak']
                y_global = y_start + params['y_peak']
                
                results.append({
                    'x': x_global,
                    'y': y_global,
                    'frame': frame_num,
                    'c0': params['c0'],
                    'c1': params['c1'],
                    'c2': params['c2'],
                    'c3': params['c3'],
                    'c4': params['c4'],
                    'c5': params['c5'],
                    'residual': residual
                })
        
        return pd.DataFrame(results)
    
    def detect_sequence(self, frames: Union[List[np.ndarray], object]) -> pd.DataFrame:
        """
        Detect features in a sequence of frames.
        
        Parameters
        ----------
        frames : list of np.ndarray or video object
            Image sequence
            
        Returns
        -------
        pd.DataFrame
            DataFrame with columns ['x', 'y', 'frame', ...]
        """
        all_features = []
        for frame_num, frame in enumerate(frames):
            features = self.detect_frame(np.array(frame), frame_num)
            all_features.append(features)
        
        if all_features:
            return pd.concat(all_features, ignore_index=True)
        else:
            return pd.DataFrame(columns=['x', 'y', 'frame'])


def detect_features(frames: Union[List[np.ndarray], object],
                   method: str = 'trackpy',
                   **kwargs) -> pd.DataFrame:
    """
    Convenience function to detect features using specified method.
    
    Parameters
    ----------
    frames : list of np.ndarray or video object
        Image sequence
    method : str
        Detection method: 'trackpy', 'gaussian', or 'polynomial'
    **kwargs
        Method-specific parameters
        
    Returns
    -------
    pd.DataFrame
        Detected features with columns ['x', 'y', 'frame', ...]
    """
    if method == 'trackpy':
        detector = TrackpyDetector(**kwargs)
    elif method == 'gaussian':
        detector = GaussianDetector(**kwargs)
    elif method == 'polynomial':
        detector = PolynomialDetector(**kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return detector.detect_sequence(frames)
