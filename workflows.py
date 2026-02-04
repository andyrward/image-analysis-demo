"""
High-level workflow functions for particle tracking.

Provides convenience APIs for common particle tracking workflows.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Union, List, Dict, Tuple
from pathlib import Path

from data_loader import (
    load_video, load_image_sequence, generate_synthetic_data,
    SyntheticDataGenerator
)
from feature_detectors import (
    TrackpyDetector, GaussianDetector, PolynomialDetector
)
from trajectory_linker import link_and_filter, subtract_drift
from motion_analyzer import analyze_particle_motion
from visualizer import (
    plot_trajectories, plot_msd, annotate_frame,
    compare_detection_methods, plot_trajectory_overlay
)
from accuracy import compare_methods_accuracy


def track_particles(frames: Union[List[np.ndarray], object],
                   method: str = 'trackpy',
                   search_range: float = 5.0,
                   memory: int = 0,
                   min_length: int = 10,
                   subtract_drift_flag: bool = False,
                   **detector_kwargs) -> pd.DataFrame:
    """
    Complete particle tracking workflow: detect + link + filter.
    
    Parameters
    ----------
    frames : list of np.ndarray or video object
        Image sequence
    method : str
        Detection method: 'trackpy', 'gaussian', or 'polynomial'
    search_range : float
        Maximum displacement between frames
    memory : int
        Number of frames a particle can disappear
    min_length : int
        Minimum trajectory length
    subtract_drift_flag : bool
        Whether to subtract ensemble drift
    **detector_kwargs
        Method-specific detection parameters
        
    Returns
    -------
    pd.DataFrame
        Linked and filtered trajectories
    """
    # Select detector
    if method == 'trackpy':
        detector = TrackpyDetector(**detector_kwargs)
    elif method == 'gaussian':
        detector = GaussianDetector(**detector_kwargs)
    elif method == 'polynomial':
        detector = PolynomialDetector(**detector_kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Detect features
    print(f"Detecting features using {method}...")
    features = detector.detect_sequence(frames)
    print(f"Detected {len(features)} features across {features['frame'].nunique()} frames")
    
    # Link trajectories
    print("Linking trajectories...")
    trajectories = link_and_filter(
        features,
        search_range=search_range,
        memory=memory,
        min_length=min_length,
        subtract_drift_flag=subtract_drift_flag
    )
    print(f"Found {trajectories['particle'].nunique()} trajectories")
    
    return trajectories


def analyze_motion(trajectories: pd.DataFrame,
                  mpp: float = 1.0,
                  fps: float = 1.0) -> Dict:
    """
    Analyze particle motion and compute statistics.
    
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
        Motion analysis results
    """
    print("Analyzing particle motion...")
    results = analyze_particle_motion(trajectories, mpp=mpp, fps=fps)
    
    print("\nMotion Analysis Summary:")
    print(f"  Number of trajectories: {results['summary_stats']['n_trajectories']}")
    print(f"  Mean trajectory length: {results['summary_stats']['mean_trajectory_length']:.1f} frames")
    print(f"  Diffusion coefficient: {results['summary_stats']['diffusion_coefficient']:.4f}")
    
    return results


def complete_pipeline(frames: Union[List[np.ndarray], object],
                     method: str = 'trackpy',
                     search_range: float = 5.0,
                     min_length: int = 10,
                     mpp: float = 1.0,
                     fps: float = 1.0,
                     visualize: bool = True,
                     **detector_kwargs) -> Tuple[pd.DataFrame, Dict]:
    """
    Complete particle tracking and analysis pipeline.
    
    Parameters
    ----------
    frames : list of np.ndarray or video object
        Image sequence
    method : str
        Detection method
    search_range : float
        Maximum displacement between frames
    min_length : int
        Minimum trajectory length
    mpp : float
        Microns per pixel
    fps : float
        Frames per second
    visualize : bool
        Whether to create visualization plots
    **detector_kwargs
        Detection parameters
        
    Returns
    -------
    trajectories : pd.DataFrame
        Linked trajectories
    analysis : dict
        Motion analysis results
    """
    # Track particles
    trajectories = track_particles(
        frames,
        method=method,
        search_range=search_range,
        min_length=min_length,
        **detector_kwargs
    )
    
    # Analyze motion
    analysis = analyze_motion(trajectories, mpp=mpp, fps=fps)
    
    # Visualize if requested
    if visualize:
        quick_viz_results(trajectories, analysis, frames)
    
    return trajectories, analysis


def quick_viz_results(trajectories: pd.DataFrame,
                     analysis: Dict,
                     frames: Optional[Union[List[np.ndarray], object]] = None):
    """
    Quick visualization of tracking results.
    
    Parameters
    ----------
    trajectories : pd.DataFrame
        Linked trajectories
    analysis : dict
        Motion analysis results
    frames : list of np.ndarray or video object, optional
        Original frames for overlay
    """
    fig = plt.figure(figsize=(15, 5))
    
    # Plot trajectories
    ax1 = plt.subplot(131)
    plot_trajectories(trajectories, ax=ax1)
    
    # Plot MSD
    ax2 = plt.subplot(132)
    plot_msd(analysis['msd'], ax=ax2)
    
    # Plot annotated frame if available
    if frames is not None:
        ax3 = plt.subplot(133)
        frame_num = 0
        annotate_frame(np.array(frames[frame_num]), trajectories, frame_num, ax=ax3)
    
    plt.tight_layout()
    plt.show()


def quick_viz_trajectories(trajectories: pd.DataFrame,
                          n_trajectories: Optional[int] = None):
    """
    Quick visualization of trajectories.
    
    Parameters
    ----------
    trajectories : pd.DataFrame
        Linked trajectories
    n_trajectories : int, optional
        Number of random trajectories to plot (default: all)
    """
    particle_ids = trajectories['particle'].unique()
    
    if n_trajectories is not None and len(particle_ids) > n_trajectories:
        particle_ids = np.random.choice(particle_ids, n_trajectories, replace=False)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    plot_trajectories(trajectories, ax=ax, particle_ids=particle_ids)
    plt.show()


def quick_viz_msd(analysis: Dict):
    """
    Quick visualization of MSD.
    
    Parameters
    ----------
    analysis : dict
        Motion analysis results containing 'msd'
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_msd(analysis['msd'], ax=ax)
    plt.show()


def demo_with_synthetic_data(n_spots: int = 5,
                            spot_width: float = 2.0,
                            signal: float = 1000.0,
                            noise_magnitude: float = 50.0,
                            n_frames: int = 10,
                            method: str = 'gaussian',
                            visualize: bool = True) -> Tuple[List[np.ndarray], pd.DataFrame]:
    """
    Run a demo tracking workflow with synthetic data.
    
    Parameters
    ----------
    n_spots : int
        Number of spots per dimension
    spot_width : float
        Width of Gaussian spots
    signal : float
        Spot intensity
    noise_magnitude : float
        Noise level
    n_frames : int
        Number of frames
    method : str
        Detection method
    visualize : bool
        Whether to show plots
        
    Returns
    -------
    frames : list
        Generated frames
    trajectories : pd.DataFrame
        Detected trajectories
    """
    print(f"Generating {n_frames} synthetic frames with {n_spots}x{n_spots} spots...")
    
    # Generate data
    frames = generate_synthetic_data(
        n_spots=n_spots,
        spot_width=spot_width,
        signal=signal,
        noise_magnitude=noise_magnitude,
        n_frames=n_frames
    )
    
    # Track particles
    trajectories = track_particles(
        frames,
        method=method,
        search_range=10.0,
        min_length=3
    )
    
    if visualize and len(trajectories) > 0:
        # Show first frame with detections
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        annotate_frame(frames[0], trajectories, 0, ax=ax1)
        plot_trajectories(trajectories, ax=ax2)
        
        plt.tight_layout()
        plt.show()
    
    return frames, trajectories


def compare_methods_on_synthetic(n_spots: int = 5,
                                spot_width: float = 2.0,
                                signal: float = 1000.0,
                                noise_magnitude: float = 50.0,
                                n_frames: int = 5,
                                methods: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Compare detection methods on synthetic data with ground truth.
    
    Parameters
    ----------
    n_spots : int
        Number of spots per dimension
    spot_width : float
        Width of Gaussian spots
    signal : float
        Spot intensity
    noise_magnitude : float
        Noise level
    n_frames : int
        Number of frames
    methods : list of str, optional
        Methods to compare (default: all three)
        
    Returns
    -------
    pd.DataFrame
        Comparison results
    """
    if methods is None:
        methods = ['trackpy', 'gaussian', 'polynomial']
    
    print(f"Comparing {len(methods)} detection methods on synthetic data...")
    
    # Generate data with known ground truth
    generator = SyntheticDataGenerator(
        n_spots=n_spots,
        spot_width=spot_width,
        signal=signal,
        noise_magnitude=noise_magnitude
    )
    frames, ground_truth = generator.generate_static_spots(n_frames)
    
    # Run each detector
    detections_dict = {}
    for method in methods:
        print(f"\nTesting {method}...")
        if method == 'trackpy':
            detector = TrackpyDetector(diameter=11, minmass=100)
        elif method == 'gaussian':
            detector = GaussianDetector(min_distance=5, threshold=100)
        elif method == 'polynomial':
            detector = PolynomialDetector(min_distance=5, threshold=100)
        
        detections = detector.detect_sequence(frames)
        detections_dict[method] = detections
        print(f"  Detected {len(detections[detections['frame'] == 0])} features in frame 0")
    
    # Compare accuracy
    comparison = compare_methods_accuracy(detections_dict, ground_truth, frame_num=0)
    
    print("\n" + "="*70)
    print("COMPARISON RESULTS:")
    print("="*70)
    print(comparison.to_string(index=False))
    
    # Visualize comparison
    fig = compare_detection_methods(frames[0], detections_dict, frame_num=0)
    plt.suptitle('Detection Method Comparison', fontsize=14, y=1.02)
    plt.show()
    
    return comparison


def load_and_track(path: Union[str, Path],
                  method: str = 'trackpy',
                  search_range: float = 5.0,
                  min_length: int = 10,
                  **detector_kwargs) -> pd.DataFrame:
    """
    Load video/images and perform tracking.
    
    Parameters
    ----------
    path : str or Path
        Path to video file or image pattern
    method : str
        Detection method
    search_range : float
        Maximum displacement
    min_length : int
        Minimum trajectory length
    **detector_kwargs
        Detection parameters
        
    Returns
    -------
    pd.DataFrame
        Tracked trajectories
    """
    path = Path(path)
    
    # Determine if it's a video or image sequence
    if path.is_file():
        print(f"Loading video: {path}")
        video = load_video(path)
        frames = video
    else:
        print(f"Loading image sequence: {path}")
        frames = load_image_sequence(str(path))
    
    # Track
    trajectories = track_particles(
        frames,
        method=method,
        search_range=search_range,
        min_length=min_length,
        **detector_kwargs
    )
    
    return trajectories


def save_results(trajectories: pd.DataFrame,
                analysis: Dict,
                output_dir: Union[str, Path]):
    """
    Save tracking and analysis results to files.
    
    Parameters
    ----------
    trajectories : pd.DataFrame
        Tracked trajectories
    analysis : dict
        Motion analysis results
    output_dir : str or Path
        Output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save trajectories
    trajectories.to_csv(output_dir / 'trajectories.csv', index=False)
    print(f"Saved trajectories to {output_dir / 'trajectories.csv'}")
    
    # Save MSD
    analysis['msd'].to_csv(output_dir / 'msd.csv', index=False)
    print(f"Saved MSD to {output_dir / 'msd.csv'}")
    
    # Save summary statistics
    with open(output_dir / 'summary_stats.json', 'w') as f:
        json.dump(analysis['summary_stats'], f, indent=2)
    print(f"Saved summary statistics to {output_dir / 'summary_stats.json'}")
