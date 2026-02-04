"""
Example workflows demonstrating particle tracking library usage.

These examples show how to use the library for various tracking scenarios.
"""

import numpy as np
import matplotlib.pyplot as plt
from data_loader import generate_synthetic_data, SyntheticDataGenerator
from feature_detectors import TrackpyDetector, GaussianDetector, PolynomialDetector
from trajectory_linker import link_and_filter
from motion_analyzer import analyze_particle_motion, compute_msd
from visualizer import (
    plot_trajectories, annotate_frame, plot_msd,
    plot_gaussian_fit_surface, compare_detection_methods
)
from workflows import (
    track_particles, complete_pipeline,
    compare_methods_on_synthetic, demo_with_synthetic_data
)
from accuracy import compute_detection_accuracy, compare_methods_accuracy
from fitting_utils import extract_region, fit_gaussian_2d, fit_polynomial_2d


def example_basic_tracking_trackpy():
    """
    Example 1: Basic particle tracking using trackpy detector.
    
    Demonstrates the simplest workflow: generate data, detect, link, visualize.
    """
    print("="*70)
    print("Example 1: Basic Tracking with Trackpy")
    print("="*70)
    
    # Generate synthetic data
    print("\n1. Generating synthetic data...")
    frames = generate_synthetic_data(
        n_spots=5,
        spot_width=2.0,
        signal=1000.0,
        noise_magnitude=50.0,
        n_frames=20
    )
    print(f"   Generated {len(frames)} frames of size {frames[0].shape}")
    
    # Detect features
    print("\n2. Detecting features with trackpy...")
    detector = TrackpyDetector(diameter=11, minmass=100)
    features = detector.detect_sequence(frames)
    print(f"   Detected {len(features)} total features")
    print(f"   Features per frame: {len(features) / len(frames):.1f}")
    
    # Link trajectories
    print("\n3. Linking trajectories...")
    trajectories = link_and_filter(
        features,
        search_range=10.0,
        memory=2,
        min_length=10
    )
    print(f"   Found {trajectories['particle'].nunique()} trajectories")
    
    # Visualize
    print("\n4. Visualizing results...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot trajectories
    plot_trajectories(trajectories, ax=axes[0])
    
    # Annotate first frame
    annotate_frame(frames[0], trajectories, frame_num=0, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig('/tmp/example1_trackpy.png', dpi=100, bbox_inches='tight')
    print("   Saved plot to /tmp/example1_trackpy.png")
    plt.close()
    
    print("\n✓ Example 1 complete!\n")
    return trajectories


def example_gaussian_tracking():
    """
    Example 2: High-precision tracking using Gaussian fitting.
    
    Demonstrates Gaussian fitting for sub-pixel accuracy with 3D visualization.
    """
    print("="*70)
    print("Example 2: High-Precision Tracking with Gaussian Fitting")
    print("="*70)
    
    # Generate synthetic data
    print("\n1. Generating high-quality synthetic data...")
    frames = generate_synthetic_data(
        n_spots=4,
        spot_width=2.5,
        signal=2000.0,
        noise_magnitude=30.0,
        n_frames=15
    )
    
    # Detect with Gaussian fitting
    print("\n2. Detecting features with Gaussian fitting...")
    detector = GaussianDetector(
        min_distance=10,
        threshold=200.0,
        fit_radius=6
    )
    features = detector.detect_sequence(frames)
    print(f"   Detected {len(features)} features")
    print(f"   Mean amplitude: {features['amplitude'].mean():.1f}")
    print(f"   Mean sigma: {(features['sigma_x'].mean() + features['sigma_y'].mean())/2:.2f}")
    
    # Link trajectories
    print("\n3. Linking trajectories...")
    trajectories = link_and_filter(features, search_range=8.0, min_length=10)
    print(f"   Found {trajectories['particle'].nunique()} trajectories")
    
    # Show 3D fit for one feature
    print("\n4. Visualizing Gaussian fit in 3D...")
    first_feature = features.iloc[0]
    region, x_start, y_start = extract_region(
        frames[0],
        first_feature['x'],
        first_feature['y'],
        radius=6
    )
    
    params = {
        'amplitude': first_feature['amplitude'],
        'x0': first_feature['x'] - x_start,
        'y0': first_feature['y'] - y_start,
        'sigma_x': first_feature['sigma_x'],
        'sigma_y': first_feature['sigma_y'],
        'offset': first_feature['offset']
    }
    
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111, projection='3d')
    plot_gaussian_fit_surface(region, params, ax=ax, title='Gaussian Fit to Particle')
    plt.savefig('/tmp/example2_gaussian_fit.png', dpi=100, bbox_inches='tight')
    print("   Saved 3D plot to /tmp/example2_gaussian_fit.png")
    plt.close()
    
    # Plot trajectories
    fig, ax = plt.subplots(figsize=(10, 10))
    plot_trajectories(trajectories, ax=ax)
    plt.savefig('/tmp/example2_trajectories.png', dpi=100, bbox_inches='tight')
    print("   Saved trajectories to /tmp/example2_trajectories.png")
    plt.close()
    
    print("\n✓ Example 2 complete!\n")
    return trajectories


def example_polynomial_tracking():
    """
    Example 3: Particle tracking using polynomial fitting.
    
    Demonstrates polynomial peak finding with quality assessment.
    """
    print("="*70)
    print("Example 3: Tracking with Polynomial Fitting")
    print("="*70)
    
    # Generate synthetic data
    print("\n1. Generating synthetic data...")
    frames = generate_synthetic_data(
        n_spots=5,
        spot_width=3.0,
        signal=1500.0,
        noise_magnitude=40.0,
        n_frames=15
    )
    
    # Detect with polynomial fitting
    print("\n2. Detecting features with polynomial fitting...")
    detector = PolynomialDetector(
        min_distance=8,
        threshold=150.0,
        fit_radius=5
    )
    features = detector.detect_sequence(frames)
    print(f"   Detected {len(features)} features")
    print(f"   Mean residual: {features['residual'].mean():.1f}")
    
    # Link trajectories
    print("\n3. Linking trajectories...")
    trajectories = link_and_filter(features, search_range=10.0, min_length=8)
    print(f"   Found {trajectories['particle'].nunique()} trajectories")
    
    # Show polynomial fit example
    print("\n4. Visualizing polynomial fit...")
    first_feature = features.iloc[0]
    region, x_start, y_start = extract_region(
        frames[0],
        first_feature['x'],
        first_feature['y'],
        radius=5
    )
    
    params = {
        'c0': first_feature['c0'], 'c1': first_feature['c1'],
        'c2': first_feature['c2'], 'c3': first_feature['c3'],
        'c4': first_feature['c4'], 'c5': first_feature['c5'],
        'x_peak': first_feature['x'] - x_start,
        'y_peak': first_feature['y'] - y_start
    }
    
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111, projection='3d')
    from visualizer import plot_polynomial_fit_surface
    plot_polynomial_fit_surface(region, params, ax=ax, title='Polynomial Fit to Particle')
    plt.savefig('/tmp/example3_polynomial_fit.png', dpi=100, bbox_inches='tight')
    print("   Saved 3D plot to /tmp/example3_polynomial_fit.png")
    plt.close()
    
    print("\n✓ Example 3 complete!\n")
    return trajectories


def example_compare_detectors():
    """
    Example 4: Compare all three detection methods.
    
    Side-by-side comparison of trackpy, Gaussian, and polynomial detectors.
    """
    print("="*70)
    print("Example 4: Comparing Detection Methods")
    print("="*70)
    
    # Generate synthetic data with known ground truth
    print("\n1. Generating synthetic data with ground truth...")
    generator = SyntheticDataGenerator(
        n_spots=5,
        spot_width=2.5,
        signal=1200.0,
        noise_magnitude=40.0
    )
    frames, ground_truth = generator.generate_static_spots(n_frames=5)
    print(f"   Generated {len(frames)} frames with {len(ground_truth)} spots")
    
    # Run all three detectors
    print("\n2. Running all detection methods...")
    detectors = {
        'Trackpy': TrackpyDetector(diameter=11, minmass=100),
        'Gaussian': GaussianDetector(min_distance=8, threshold=150),
        'Polynomial': PolynomialDetector(min_distance=8, threshold=150)
    }
    
    detections_dict = {}
    for name, detector in detectors.items():
        print(f"   Running {name}...")
        detections = detector.detect_sequence(frames)
        detections_dict[name] = detections
        print(f"     Detected {len(detections[detections['frame'] == 0])} features in frame 0")
    
    # Compare accuracy
    print("\n3. Computing accuracy metrics...")
    comparison = compare_methods_accuracy(detections_dict, ground_truth, frame_num=0)
    print("\n" + comparison.to_string(index=False))
    
    # Visualize comparison
    print("\n4. Creating comparison visualization...")
    fig = compare_detection_methods(frames[0], detections_dict, frame_num=0)
    plt.suptitle(f'Method Comparison: {len(ground_truth)} ground truth spots', 
                 fontsize=14, y=1.02)
    plt.savefig('/tmp/example4_comparison.png', dpi=100, bbox_inches='tight')
    print("   Saved comparison to /tmp/example4_comparison.png")
    plt.close()
    
    print("\n✓ Example 4 complete!\n")
    return comparison


def example_msd_analysis():
    """
    Example 5: Motion analysis with MSD calculation.
    
    Demonstrates full motion analysis including MSD and diffusion coefficient.
    """
    print("="*70)
    print("Example 5: Mean Squared Displacement Analysis")
    print("="*70)
    
    # Generate longer sequence for better statistics
    print("\n1. Generating long sequence...")
    frames = generate_synthetic_data(
        n_spots=6,
        spot_width=2.0,
        signal=1500.0,
        noise_magnitude=45.0,
        n_frames=50
    )
    
    # Track particles
    print("\n2. Tracking particles...")
    trajectories = track_particles(
        frames,
        method='gaussian',
        search_range=10.0,
        min_length=30,
        min_distance=8,
        threshold=150
    )
    
    # Analyze motion
    print("\n3. Analyzing motion...")
    analysis = analyze_particle_motion(trajectories, mpp=0.16, fps=10.0)
    
    print("\n   Motion Statistics:")
    print(f"     Trajectories: {analysis['summary_stats']['n_trajectories']}")
    print(f"     Mean length: {analysis['summary_stats']['mean_trajectory_length']:.1f} frames")
    print(f"     Diffusion coeff: {analysis['summary_stats']['diffusion_coefficient']:.4f} μm²/s")
    
    # Plot MSD
    print("\n4. Plotting MSD...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Linear plot
    plot_msd(analysis['msd'], ax=axes[0], fit_line=True)
    
    # Log-log plot
    plot_msd(analysis['msd'], ax=axes[1], fit_line=False, loglog=True)
    axes[1].set_title('MSD (log-log scale)')
    
    plt.tight_layout()
    plt.savefig('/tmp/example5_msd.png', dpi=100, bbox_inches='tight')
    print("   Saved MSD plot to /tmp/example5_msd.png")
    plt.close()
    
    print("\n✓ Example 5 complete!\n")
    return analysis


def example_synthetic_data_demo():
    """
    Example 6: Complete demo using high-level API.
    
    Shows the simplest way to get results using workflow functions.
    """
    print("="*70)
    print("Example 6: High-Level API Demo")
    print("="*70)
    
    # Use the high-level demo function
    print("\n1. Running demo with synthetic data...")
    frames, trajectories = demo_with_synthetic_data(
        n_spots=5,
        spot_width=2.5,
        signal=1500.0,
        noise_magnitude=35.0,
        n_frames=25,
        method='gaussian',
        visualize=False  # We'll make our own plots
    )
    
    # Run complete analysis pipeline
    print("\n2. Running complete analysis pipeline...")
    trajectories, analysis = complete_pipeline(
        frames,
        method='gaussian',
        search_range=10.0,
        min_length=15,
        mpp=0.16,
        fps=10.0,
        visualize=False
    )
    
    # Create comprehensive visualization
    print("\n3. Creating comprehensive visualization...")
    fig = plt.figure(figsize=(15, 10))
    
    # Trajectories
    ax1 = plt.subplot(2, 2, 1)
    plot_trajectories(trajectories, ax=ax1)
    
    # MSD
    ax2 = plt.subplot(2, 2, 2)
    plot_msd(analysis['msd'], ax=ax2)
    
    # First frame annotated
    ax3 = plt.subplot(2, 2, 3)
    annotate_frame(frames[0], trajectories, 0, ax=ax3)
    
    # Last frame annotated
    ax4 = plt.subplot(2, 2, 4)
    annotate_frame(frames[-1], trajectories, len(frames)-1, ax=ax4)
    
    plt.tight_layout()
    plt.savefig('/tmp/example6_complete.png', dpi=100, bbox_inches='tight')
    print("   Saved complete analysis to /tmp/example6_complete.png")
    plt.close()
    
    print("\n✓ Example 6 complete!\n")
    return trajectories, analysis


def run_all_examples():
    """Run all example workflows."""
    print("\n" + "="*70)
    print("RUNNING ALL EXAMPLES")
    print("="*70 + "\n")
    
    try:
        example_basic_tracking_trackpy()
        example_gaussian_tracking()
        example_polynomial_tracking()
        example_compare_detectors()
        example_msd_analysis()
        example_synthetic_data_demo()
        
        print("\n" + "="*70)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\nGenerated files:")
        print("  /tmp/example1_trackpy.png")
        print("  /tmp/example2_gaussian_fit.png")
        print("  /tmp/example2_trajectories.png")
        print("  /tmp/example3_polynomial_fit.png")
        print("  /tmp/example4_comparison.png")
        print("  /tmp/example5_msd.png")
        print("  /tmp/example6_complete.png")
        print()
        
    except Exception as e:
        print(f"\n✗ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    # Run all examples when script is executed
    run_all_examples()
