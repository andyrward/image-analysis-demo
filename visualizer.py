"""
Visualization utilities for particle tracking.

Provides comprehensive plotting functions including trajectories, MSD,
and 3D surface plots for fit validation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, List, Tuple
from fitting_utils import gaussian_2d, polynomial_2d


def plot_trajectories(trajectories: pd.DataFrame,
                     ax: Optional[plt.Axes] = None,
                     colorby: str = 'particle',
                     particle_ids: Optional[List[int]] = None,
                     **kwargs) -> plt.Axes:
    """
    Plot particle trajectories.
    
    Parameters
    ----------
    trajectories : pd.DataFrame
        Linked trajectories with 'particle' column
    ax : plt.Axes, optional
        Axes to plot on
    colorby : str
        Color trajectories by 'particle' or 'frame'
    particle_ids : list, optional
        Specific particle IDs to plot
    **kwargs
        Additional arguments for plot
        
    Returns
    -------
    plt.Axes
        Axes with plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    # Reset index to avoid ambiguity
    trajectories = trajectories.reset_index(drop=True)
    
    # Filter particles if specified
    if particle_ids is not None:
        trajectories = trajectories[trajectories['particle'].isin(particle_ids)]
    
    # Plot each trajectory
    for particle_id, particle_indices in trajectories.groupby('particle').groups.items():
        particle_data = trajectories.loc[particle_indices].sort_values('frame')
        
        if colorby == 'particle':
            ax.plot(particle_data['x'], particle_data['y'], '-', 
                   alpha=0.7, **kwargs)
        elif colorby == 'frame':
            scatter = ax.scatter(particle_data['x'], particle_data['y'], 
                               c=particle_data['frame'], cmap='viridis',
                               alpha=0.7, **kwargs)
    
    if colorby == 'frame':
        plt.colorbar(scatter, ax=ax, label='Frame')
    
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_title('Particle Trajectories')
    ax.set_aspect('equal')
    
    return ax


def annotate_frame(image: np.ndarray,
                  features: pd.DataFrame,
                  frame_num: int,
                  ax: Optional[plt.Axes] = None,
                  circle_radius: float = 5.0,
                  show_labels: bool = True,
                  **kwargs) -> plt.Axes:
    """
    Annotate detected features on a frame.
    
    Parameters
    ----------
    image : np.ndarray
        Image to annotate
    features : pd.DataFrame
        Detected features (can include 'particle' column)
    frame_num : int
        Frame number to display
    ax : plt.Axes, optional
        Axes to plot on
    circle_radius : float
        Radius of circles marking features
    show_labels : bool
        Whether to show particle IDs
    **kwargs
        Additional arguments for Circle patches
        
    Returns
    -------
    plt.Axes
        Axes with annotated image
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    
    # Display image
    ax.imshow(image, cmap='gray', origin='lower')
    
    # Reset index to avoid ambiguity
    features = features.reset_index(drop=True)
    
    # Filter features for this frame
    frame_features = features[features['frame'] == frame_num]
    
    # Add circles for each feature
    for idx, row in frame_features.iterrows():
        circle = Circle((row['x'], row['y']), circle_radius, 
                       fill=False, color='red', linewidth=2, **kwargs)
        ax.add_patch(circle)
        
        # Add label if particle ID exists
        if show_labels and 'particle' in row:
            ax.text(row['x'] + circle_radius + 2, row['y'], 
                   str(int(row['particle'])),
                   color='yellow', fontsize=8, fontweight='bold')
    
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(0, image.shape[0])
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_title(f'Frame {frame_num}: {len(frame_features)} features detected')
    
    return ax


def plot_msd(msd_data: pd.DataFrame,
            ax: Optional[plt.Axes] = None,
            fit_line: bool = True,
            n_fit_points: int = 4,
            loglog: bool = False,
            **kwargs) -> plt.Axes:
    """
    Plot mean squared displacement.
    
    Parameters
    ----------
    msd_data : pd.DataFrame
        MSD data with columns ['lagt', 'msd']
    ax : plt.Axes, optional
        Axes to plot on
    fit_line : bool
        Whether to add linear fit line
    n_fit_points : int
        Number of initial points for fit
    loglog : bool
        Whether to use log-log scale
    **kwargs
        Additional arguments for plot
        
    Returns
    -------
    plt.Axes
        Axes with MSD plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot MSD
    if loglog:
        ax.loglog(msd_data['lagt'], msd_data['msd'], 'o-', **kwargs)
    else:
        ax.plot(msd_data['lagt'], msd_data['msd'], 'o-', **kwargs)
    
    # Add fit line
    if fit_line and len(msd_data) >= n_fit_points:
        fit_data = msd_data.head(n_fit_points)
        slope = np.polyfit(fit_data['lagt'], fit_data['msd'], 1)[0]
        D = slope / 4.0
        
        # Plot fit line
        fit_x = np.linspace(0, fit_data['lagt'].max(), 100)
        fit_y = slope * fit_x
        ax.plot(fit_x, fit_y, '--', color='red', 
               label=f'Fit: D = {D:.4f}')
        ax.legend()
    
    ax.set_xlabel('Lag time')
    ax.set_ylabel('MSD')
    ax.set_title('Mean Squared Displacement')
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_gaussian_fit_surface(region: np.ndarray,
                              params: dict,
                              ax: Optional[Axes3D] = None,
                              title: str = 'Gaussian Fit') -> Axes3D:
    """
    Plot 3D surface of Gaussian fit.
    
    Parameters
    ----------
    region : np.ndarray
        Original image region
    params : dict
        Gaussian fit parameters
    ax : Axes3D, optional
        3D axes to plot on
    title : str
        Plot title
        
    Returns
    -------
    Axes3D
        3D axes with surface plots
    """
    if ax is None:
        fig = plt.figure(figsize=(12, 5))
        ax = fig.add_subplot(121, projection='3d')
    
    # Create coordinate grids
    y_size, x_size = region.shape
    y = np.arange(y_size)
    x = np.arange(x_size)
    x_grid, y_grid = np.meshgrid(x, y)
    
    # Compute fitted surface
    fitted = gaussian_2d((x_grid, y_grid),
                        params['amplitude'],
                        params['x0'],
                        params['y0'],
                        params['sigma_x'],
                        params['sigma_y'],
                        params['offset']).reshape(region.shape)
    
    # Plot original data
    ax.plot_surface(x_grid, y_grid, region, cmap='viridis', alpha=0.6)
    
    # Plot fitted surface
    ax.plot_wireframe(x_grid, y_grid, fitted, color='red', alpha=0.8, 
                     linewidth=1)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Intensity')
    ax.set_title(title)
    
    return ax


def plot_polynomial_fit_surface(region: np.ndarray,
                                params: dict,
                                ax: Optional[Axes3D] = None,
                                title: str = 'Polynomial Fit') -> Axes3D:
    """
    Plot 3D surface of polynomial fit.
    
    Parameters
    ----------
    region : np.ndarray
        Original image region
    params : dict
        Polynomial fit parameters
    ax : Axes3D, optional
        3D axes to plot on
    title : str
        Plot title
        
    Returns
    -------
    Axes3D
        3D axes with surface plots
    """
    if ax is None:
        fig = plt.figure(figsize=(12, 5))
        ax = fig.add_subplot(121, projection='3d')
    
    # Create coordinate grids
    y_size, x_size = region.shape
    y = np.arange(y_size)
    x = np.arange(x_size)
    x_grid, y_grid = np.meshgrid(x, y)
    
    # Compute fitted surface
    coeffs = [params['c0'], params['c1'], params['c2'],
              params['c3'], params['c4'], params['c5']]
    fitted = polynomial_2d((x_grid, y_grid), *coeffs).reshape(region.shape)
    
    # Plot original data
    ax.plot_surface(x_grid, y_grid, region, cmap='viridis', alpha=0.6)
    
    # Plot fitted surface
    ax.plot_wireframe(x_grid, y_grid, fitted, color='red', alpha=0.8,
                     linewidth=1)
    
    # Mark peak location
    if 0 <= params['x_peak'] < x_size and 0 <= params['y_peak'] < y_size:
        peak_z = region[int(params['y_peak']), int(params['x_peak'])]
        ax.scatter([params['x_peak']], [params['y_peak']], [peak_z],
                  color='yellow', s=100, marker='*')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Intensity')
    ax.set_title(title)
    
    return ax


def compare_detection_methods(image: np.ndarray,
                              detections: dict,
                              frame_num: int = 0,
                              figsize: Tuple[int, int] = (15, 5)) -> plt.Figure:
    """
    Compare results from different detection methods on same frame.
    
    Parameters
    ----------
    image : np.ndarray
        Image to analyze
    detections : dict
        Dictionary of {method_name: features_df}
    frame_num : int
        Frame number
    figsize : tuple
        Figure size
        
    Returns
    -------
    plt.Figure
        Figure with comparison plots
    """
    n_methods = len(detections)
    fig, axes = plt.subplots(1, n_methods, figsize=figsize)
    
    if n_methods == 1:
        axes = [axes]
    
    for ax, (method_name, features) in zip(axes, detections.items()):
        annotate_frame(image, features, frame_num, ax=ax)
        ax.set_title(f'{method_name}\n({len(features[features["frame"] == frame_num])} features)')
    
    plt.tight_layout()
    return fig


def plot_trajectory_overlay(image: np.ndarray,
                           trajectories: pd.DataFrame,
                           frame_num: int,
                           trail_length: int = 10,
                           ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot trajectories overlaid on a frame with trailing history.
    
    Parameters
    ----------
    image : np.ndarray
        Frame to display
    trajectories : pd.DataFrame
        Linked trajectories
    frame_num : int
        Current frame number
    trail_length : int
        Number of past positions to show
    ax : plt.Axes, optional
        Axes to plot on
        
    Returns
    -------
    plt.Axes
        Axes with overlay
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    
    # Display image
    ax.imshow(image, cmap='gray', origin='lower')
    
    # Reset index to avoid ambiguity
    trajectories = trajectories.reset_index(drop=True)
    
    # Get relevant frames
    start_frame = max(0, frame_num - trail_length)
    relevant_data = trajectories[
        (trajectories['frame'] >= start_frame) & 
        (trajectories['frame'] <= frame_num)
    ]
    
    # Plot each trajectory
    for particle_id, particle_indices in relevant_data.groupby('particle').groups.items():
        particle_data = relevant_data.loc[particle_indices].sort_values('frame')
        
        # Draw trail with fading alpha
        for i in range(len(particle_data) - 1):
            alpha = (i + 1) / len(particle_data) * 0.8
            ax.plot(particle_data['x'].iloc[i:i+2], 
                   particle_data['y'].iloc[i:i+2],
                   'b-', alpha=alpha, linewidth=2)
        
        # Mark current position
        current = particle_data[particle_data['frame'] == frame_num]
        if len(current) > 0:
            ax.plot(current['x'], current['y'], 'ro', markersize=8)
            ax.text(current['x'].iloc[0] + 5, current['y'].iloc[0],
                   str(int(particle_id)), color='yellow', fontweight='bold')
    
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(0, image.shape[0])
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_title(f'Frame {frame_num}')
    
    return ax
