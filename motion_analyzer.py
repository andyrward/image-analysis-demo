"""
Motion analysis utilities - thin wrappers around trackpy motion functions.

Analyzes particle motion including MSD, velocity, and other statistics.
"""

import pandas as pd
import numpy as np
import trackpy as tp
from typing import Optional, Tuple


def compute_msd(trajectories: pd.DataFrame,
                mpp: float = 1.0,
                fps: float = 1.0,
                max_lagtime: Optional[int] = None) -> pd.DataFrame:
    """
    Compute mean squared displacement (MSD).
    
    Parameters
    ----------
    trajectories : pd.DataFrame
        Linked trajectories with 'particle' column
    mpp : float
        Microns per pixel (for spatial scaling)
    fps : float
        Frames per second (for temporal scaling)
    max_lagtime : int, optional
        Maximum lag time in frames
        
    Returns
    -------
    pd.DataFrame
        MSD with columns ['lagt', 'msd']
    """
    if len(trajectories) == 0:
        return pd.DataFrame(columns=['lagt', 'msd'])
    
    # Determine max_lagtime if not provided
    if max_lagtime is None:
        max_traj_length = trajectories.groupby('particle').size().max()
        max_lagtime = max(1, int(max_traj_length / 2))
    
    msd = tp.emsd(trajectories, mpp=mpp, fps=fps, max_lagtime=max_lagtime)
    
    # Convert Series to DataFrame
    if isinstance(msd, pd.Series):
        msd_df = pd.DataFrame({'lagt': msd.index, 'msd': msd.values})
    else:
        msd_df = msd
    
    return msd_df


def compute_individual_msd(trajectories: pd.DataFrame,
                          mpp: float = 1.0,
                          fps: float = 1.0,
                          max_lagtime: Optional[int] = None) -> pd.DataFrame:
    """
    Compute MSD for each individual trajectory.
    
    Parameters
    ----------
    trajectories : pd.DataFrame
        Linked trajectories with 'particle' column
    mpp : float
        Microns per pixel
    fps : float
        Frames per second
    max_lagtime : int, optional
        Maximum lag time in frames
        
    Returns
    -------
    pd.DataFrame
        Individual MSDs (may be Series or DataFrame depending on trackpy version)
    """
    if len(trajectories) == 0:
        return pd.DataFrame(columns=['lagt', 'particle'])
    
    # Determine max_lagtime if not provided
    if max_lagtime is None:
        max_traj_length = trajectories.groupby('particle').size().max()
        max_lagtime = max(1, int(max_traj_length / 2))
    
    imsd = tp.imsd(trajectories, mpp=mpp, fps=fps, max_lagtime=max_lagtime)
    return imsd


def compute_velocity(trajectories: pd.DataFrame,
                    mpp: float = 1.0,
                    fps: float = 1.0) -> pd.DataFrame:
    """
    Compute velocity for each trajectory point.
    
    Parameters
    ----------
    trajectories : pd.DataFrame
        Linked trajectories
    mpp : float
        Microns per pixel
    fps : float
        Frames per second
        
    Returns
    -------
    pd.DataFrame
        Trajectories with added velocity columns ['vx', 'vy', 'v']
    """
    result = trajectories.copy()
    
    # Reset index to avoid ambiguity
    result = result.reset_index(drop=True)
    
    # Initialize velocity columns
    result['vx'] = np.nan
    result['vy'] = np.nan
    result['v'] = np.nan
    
    # Group by particle
    for particle_id, particle_indices in result.groupby('particle').groups.items():
        particle_data = result.loc[particle_indices].sort_values('frame')
        
        if len(particle_data) < 2:
            continue
        
        # Compute displacements
        dx = particle_data['x'].diff() * mpp
        dy = particle_data['y'].diff() * mpp
        dt = particle_data['frame'].diff() / fps
        
        # Compute velocities
        vx = dx / dt
        vy = dy / dt
        v = np.sqrt(vx**2 + vy**2)
        
        # Store in result
        result.loc[particle_data.index, 'vx'] = vx.values
        result.loc[particle_data.index, 'vy'] = vy.values
        result.loc[particle_data.index, 'v'] = v.values
    
    return result


def compute_diffusion_coefficient(msd_data: pd.DataFrame,
                                  n_points: int = 4) -> Tuple[float, float]:
    """
    Compute diffusion coefficient from MSD data.
    
    Fits MSD = 4*D*t to the first n_points of the MSD curve.
    
    Parameters
    ----------
    msd_data : pd.DataFrame
        MSD data from compute_msd
    n_points : int
        Number of initial points to use for fitting
        
    Returns
    -------
    D : float
        Diffusion coefficient
    slope : float
        Slope of MSD vs time fit
    """
    # Use first n_points
    data = msd_data.head(n_points)
    
    if len(data) < 2:
        return 0.0, 0.0
    
    # Linear fit: MSD = slope * t
    # D = slope / 4 for 2D
    lagt = data['lagt'].values
    msd_vals = data['msd'].values
    
    # Simple linear regression
    slope = np.polyfit(lagt, msd_vals, 1)[0]
    D = slope / 4.0
    
    return D, slope


def analyze_particle_motion(trajectories: pd.DataFrame,
                           mpp: float = 1.0,
                           fps: float = 1.0) -> dict:
    """
    Comprehensive motion analysis for all trajectories.
    
    Parameters
    ----------
    trajectories : pd.DataFrame
        Linked trajectories
    mpp : float
        Microns per pixel
    fps : float
        Frames per second
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'msd': Ensemble MSD
        - 'imsd': Individual MSDs
        - 'diffusion_coeff': Diffusion coefficient
        - 'velocities': Trajectories with velocity information
        - 'summary_stats': Summary statistics
    """
    # Compute MSD
    msd = compute_msd(trajectories, mpp=mpp, fps=fps)
    imsd = compute_individual_msd(trajectories, mpp=mpp, fps=fps)
    
    # Compute diffusion coefficient
    D, slope = compute_diffusion_coefficient(msd)
    
    # Compute velocities
    velocities = compute_velocity(trajectories, mpp=mpp, fps=fps)
    
    # Compute summary statistics
    n_trajectories = trajectories['particle'].nunique()
    n_frames = trajectories['frame'].nunique()
    mean_traj_length = trajectories.groupby('particle').size().mean()
    
    summary_stats = {
        'n_trajectories': n_trajectories,
        'n_frames': n_frames,
        'mean_trajectory_length': mean_traj_length,
        'diffusion_coefficient': D,
        'msd_slope': slope
    }
    
    if 'v' in velocities.columns:
        summary_stats['mean_velocity'] = velocities['v'].mean()
        summary_stats['std_velocity'] = velocities['v'].std()
    
    return {
        'msd': msd,
        'imsd': imsd,
        'diffusion_coeff': D,
        'velocities': velocities,
        'summary_stats': summary_stats
    }


def compute_trajectory_statistics(trajectories: pd.DataFrame,
                                 mpp: float = 1.0) -> pd.DataFrame:
    """
    Compute statistics for each trajectory.
    
    Parameters
    ----------
    trajectories : pd.DataFrame
        Linked trajectories
    mpp : float
        Microns per pixel
        
    Returns
    -------
    pd.DataFrame
        Statistics with one row per particle
    """
    stats = []
    
    for particle_id, particle_data in trajectories.groupby('particle'):
        # Sort by frame
        particle_data = particle_data.sort_values('frame')
        
        # Basic stats
        length = len(particle_data)
        x_vals = particle_data['x'].values * mpp
        y_vals = particle_data['y'].values * mpp
        
        # Compute metrics
        x_range = x_vals.max() - x_vals.min()
        y_range = y_vals.max() - y_vals.min()
        total_displacement = np.sqrt((x_vals[-1] - x_vals[0])**2 + 
                                    (y_vals[-1] - y_vals[0])**2)
        
        # Path length
        path_length = 0.0
        for i in range(1, len(x_vals)):
            dx = x_vals[i] - x_vals[i-1]
            dy = y_vals[i] - y_vals[i-1]
            path_length += np.sqrt(dx**2 + dy**2)
        
        # Net-to-gross displacement ratio
        net_to_gross = total_displacement / path_length if path_length > 0 else 0
        
        stats.append({
            'particle': particle_id,
            'length': length,
            'x_mean': x_vals.mean(),
            'y_mean': y_vals.mean(),
            'x_range': x_range,
            'y_range': y_range,
            'total_displacement': total_displacement,
            'path_length': path_length,
            'net_to_gross_ratio': net_to_gross
        })
    
    return pd.DataFrame(stats)
