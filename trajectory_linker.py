"""
Trajectory linking utilities - thin wrappers around trackpy functions.

Links detected features across frames into trajectories.
"""

import pandas as pd
import trackpy as tp
from typing import Optional


def link_trajectories(features: pd.DataFrame,
                     search_range: float,
                     memory: int = 0,
                     adaptive_stop: Optional[float] = None,
                     adaptive_step: float = 0.95,
                     **kwargs) -> pd.DataFrame:
    """
    Link features into trajectories using trackpy.
    
    Parameters
    ----------
    features : pd.DataFrame
        Detected features with columns ['x', 'y', 'frame']
    search_range : float
        Maximum displacement between frames
    memory : int
        Number of frames a particle can disappear
    adaptive_stop : float, optional
        If provided, adaptive search is used
    adaptive_step : float
        Adaptive search step size
    **kwargs
        Additional arguments passed to trackpy.link or trackpy.link_df
        
    Returns
    -------
    pd.DataFrame
        Linked trajectories with added 'particle' column
    """
    if adaptive_stop is not None:
        # Use adaptive linking
        trajectories = tp.link(
            features,
            search_range=search_range,
            memory=memory,
            adaptive_stop=adaptive_stop,
            adaptive_step=adaptive_step,
            **kwargs
        )
    else:
        # Use standard linking
        trajectories = tp.link(
            features,
            search_range=search_range,
            memory=memory,
            **kwargs
        )
    
    return trajectories


def filter_trajectories(trajectories: pd.DataFrame,
                       min_length: int = 10,
                       max_length: Optional[int] = None) -> pd.DataFrame:
    """
    Filter trajectories by length.
    
    Parameters
    ----------
    trajectories : pd.DataFrame
        Linked trajectories with 'particle' column
    min_length : int
        Minimum trajectory length
    max_length : int, optional
        Maximum trajectory length
        
    Returns
    -------
    pd.DataFrame
        Filtered trajectories
    """
    filtered = tp.filter_stubs(trajectories, threshold=min_length)
    
    if max_length is not None:
        # Count trajectory lengths
        trajectory_lengths = filtered.groupby('particle').size()
        valid_particles = trajectory_lengths[trajectory_lengths <= max_length].index
        filtered = filtered[filtered['particle'].isin(valid_particles)]
    
    return filtered


def filter_trajectories_by_displacement(trajectories: pd.DataFrame,
                                        max_displacement: float) -> pd.DataFrame:
    """
    Filter trajectories by maximum total displacement.
    
    Parameters
    ----------
    trajectories : pd.DataFrame
        Linked trajectories with 'particle' column
    max_displacement : float
        Maximum allowed displacement
        
    Returns
    -------
    pd.DataFrame
        Filtered trajectories
    """
    def compute_max_displacement(traj):
        if len(traj) < 2:
            return 0.0
        x_range = traj['x'].max() - traj['x'].min()
        y_range = traj['y'].max() - traj['y'].min()
        return (x_range**2 + y_range**2)**0.5
    
    # Compute displacement for each trajectory
    displacements = trajectories.groupby('particle').apply(compute_max_displacement)
    valid_particles = displacements[displacements <= max_displacement].index
    
    return trajectories[trajectories['particle'].isin(valid_particles)]


def compute_drift(trajectories: pd.DataFrame) -> pd.DataFrame:
    """
    Compute ensemble drift (mean displacement per frame).
    
    Parameters
    ----------
    trajectories : pd.DataFrame
        Linked trajectories
        
    Returns
    -------
    pd.DataFrame
        Drift with columns ['frame', 'x', 'y']
    """
    return tp.compute_drift(trajectories)


def subtract_drift(trajectories: pd.DataFrame,
                  drift: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Subtract drift from trajectories.
    
    Parameters
    ----------
    trajectories : pd.DataFrame
        Linked trajectories
    drift : pd.DataFrame, optional
        Pre-computed drift. If None, drift is computed automatically.
        
    Returns
    -------
    pd.DataFrame
        Trajectories with drift subtracted
    """
    if drift is None:
        drift = compute_drift(trajectories)
    
    return tp.subtract_drift(trajectories, drift)


def link_and_filter(features: pd.DataFrame,
                   search_range: float,
                   memory: int = 0,
                   min_length: int = 10,
                   subtract_drift_flag: bool = False) -> pd.DataFrame:
    """
    Convenience function to link and filter in one step.
    
    Parameters
    ----------
    features : pd.DataFrame
        Detected features
    search_range : float
        Maximum displacement between frames
    memory : int
        Number of frames a particle can disappear
    min_length : int
        Minimum trajectory length
    subtract_drift_flag : bool
        Whether to subtract ensemble drift
        
    Returns
    -------
    pd.DataFrame
        Linked and filtered trajectories
    """
    # Link
    trajectories = link_trajectories(features, search_range, memory)
    
    # Filter
    trajectories = filter_trajectories(trajectories, min_length)
    
    # Subtract drift if requested
    if subtract_drift_flag:
        trajectories = subtract_drift(trajectories)
    
    return trajectories
