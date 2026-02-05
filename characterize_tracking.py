"""
Characterization script for tracking algorithms.

This script tests tracking methods (trackpy, gaussian, parabola) on a single particle
that moves incrementally across frames to understand pixel bias and accuracy.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from image_tracker import create_demo_image, track_spot


def create_particle_image(position, image_size=100, diameter=4, amplitude=100.0):
    """
    Create an image with a single particle.
    
    Args:
        position: (x, y) position of the particle center
        image_size: Size of the square image
        diameter: Particle diameter in pixels (4 pixels corresponds to sigma ~ 2)
        amplitude: Peak amplitude of the Gaussian
        
    Returns:
        2D numpy array containing the image
    """
    # Convert diameter to sigma (FWHM = 2.355 * sigma, diameter ~ 2*FWHM)
    # For diameter = 4, we use sigma = 2 as a reasonable approximation
    sigma = diameter / 2.0
    
    image = create_demo_image(
        size=image_size,
        spot_center=position,
        spot_amplitude=amplitude,
        spot_sigma=sigma
    )
    
    return image


def track_moving_particle(n_steps=201, step_size=0.01, start_x=50.5, start_y=50.5):
    """
    Track a particle moving in small increments using multiple methods.
    
    Args:
        n_steps: Number of frames to generate (default 201)
        step_size: Distance to move particle each step in pixels (default 0.01)
        start_x: Starting x position (default 50.5)
        start_y: Starting y position (default 50.5)
        
    Returns:
        Dictionary containing:
        - 'true_positions': Array of true positions
        - 'trackpy_results': List of tracking results
        - 'gaussian_results': List of tracking results
        - 'parabola_results': List of tracking results
    """
    print(f"Generating {n_steps} frames with particle moving {step_size} pixels/step")
    print(f"Starting position: ({start_x}, {start_y})")
    print(f"Total movement: {(n_steps - 1) * step_size} pixels")
    
    # Initialize results storage
    true_positions = []
    methods = ['trackpy', 'gaussian', 'parabola']
    results = {method: [] for method in methods}
    
    # Generate frames and track
    for i in range(n_steps):
        # Calculate current position (moving in x direction)
        current_x = start_x + i * step_size
        current_y = start_y  # y stays constant
        true_positions.append((current_x, current_y))
        
        # Create image
        image = create_particle_image(
            position=(current_x, current_y),
            image_size=100,
            diameter=4,
            amplitude=100.0
        )
        
        # Track with each method
        for method in methods:
            try:
                # Set method-specific parameters
                if method == 'trackpy':
                    # Trackpy needs diameter parameter
                    result = track_spot(
                        image,
                        method=method,
                        initial_guess=(current_x, current_y),
                        diameter=9  # Use 9 pixels for trackpy (should be odd)
                    )
                else:
                    result = track_spot(
                        image,
                        method=method,
                        initial_guess=(current_x, current_y),
                        window_size=20
                    )
                results[method].append(result)
            except Exception as e:
                # Store failed result
                results[method].append({
                    'success': False,
                    'error': str(e),
                    'method': method
                })
        
        # Progress indicator
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Processed frame {i + 1}/{n_steps}")
    
    return {
        'true_positions': np.array(true_positions),
        'trackpy_results': results['trackpy'],
        'gaussian_results': results['gaussian'],
        'parabola_results': results['parabola']
    }


def extract_tracked_positions(results_list):
    """
    Extract x, y positions from tracking results.
    
    Args:
        results_list: List of tracking result dictionaries
        
    Returns:
        Tuple of (x_positions, y_positions, success_mask)
    """
    x_positions = []
    y_positions = []
    success_mask = []
    
    for result in results_list:
        if result.get('success', False):
            x_positions.append(result['x'])
            y_positions.append(result['y'])
            success_mask.append(True)
        else:
            x_positions.append(np.nan)
            y_positions.append(np.nan)
            success_mask.append(False)
    
    return np.array(x_positions), np.array(y_positions), np.array(success_mask)


def compute_bias_statistics(tracked_positions, true_positions, method_name):
    """
    Compute bias statistics for a tracking method.
    
    Args:
        tracked_positions: Tuple of (x_tracked, y_tracked)
        true_positions: Array of true (x, y) positions
        method_name: Name of the tracking method
        
    Returns:
        Dictionary of statistics
    """
    x_tracked, y_tracked = tracked_positions
    
    # Remove NaN values
    valid_mask = ~(np.isnan(x_tracked) | np.isnan(y_tracked))
    x_tracked_valid = x_tracked[valid_mask]
    y_tracked_valid = y_tracked[valid_mask]
    true_pos_valid = true_positions[valid_mask]
    
    if len(x_tracked_valid) == 0:
        return {
            'method': method_name,
            'success_rate': 0.0,
            'x_bias': np.nan,
            'y_bias': np.nan,
            'x_std': np.nan,
            'y_std': np.nan,
            'mean_error': np.nan
        }
    
    # Compute errors
    x_errors = x_tracked_valid - true_pos_valid[:, 0]
    y_errors = y_tracked_valid - true_pos_valid[:, 1]
    
    # Compute statistics
    stats = {
        'method': method_name,
        'success_rate': len(x_tracked_valid) / len(tracked_positions[0]),
        'x_bias': np.mean(x_errors),
        'y_bias': np.mean(y_errors),
        'x_std': np.std(x_errors),
        'y_std': np.std(y_errors),
        'mean_error': np.sqrt(np.mean(x_errors**2 + y_errors**2))
    }
    
    return stats


def plot_tracking_results(data, output_dir):
    """
    Create comprehensive plots of tracking results.
    
    Args:
        data: Dictionary containing tracking data
        output_dir: Path to save plots
    """
    true_positions = data['true_positions']
    true_x = true_positions[:, 0]
    true_y = true_positions[:, 1]
    
    # Extract tracked positions
    trackpy_x, trackpy_y, trackpy_success = extract_tracked_positions(data['trackpy_results'])
    gaussian_x, gaussian_y, gaussian_success = extract_tracked_positions(data['gaussian_results'])
    parabola_x, parabola_y, parabola_success = extract_tracked_positions(data['parabola_results'])
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Tracking Method Characterization: Single Particle Moving 0.01 px/step', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: X position tracking
    ax1 = axes[0, 0]
    ax1.plot(true_x, trackpy_x, 'o', label='Trackpy', alpha=0.6, markersize=3)
    ax1.plot(true_x, gaussian_x, 's', label='Gaussian', alpha=0.6, markersize=3)
    ax1.plot(true_x, parabola_x, '^', label='Parabola', alpha=0.6, markersize=3)
    ax1.plot([true_x.min(), true_x.max()], [true_x.min(), true_x.max()], 
             'k--', label='Perfect tracking', linewidth=2)
    ax1.set_xlabel('True X Position (pixels)', fontsize=11)
    ax1.set_ylabel('Tracked X Position (pixels)', fontsize=11)
    ax1.set_title('X Position Tracking', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Y position tracking
    ax2 = axes[0, 1]
    ax2.plot(true_y, trackpy_y, 'o', label='Trackpy', alpha=0.6, markersize=3)
    ax2.plot(true_y, gaussian_y, 's', label='Gaussian', alpha=0.6, markersize=3)
    ax2.plot(true_y, parabola_y, '^', label='Parabola', alpha=0.6, markersize=3)
    ax2.plot([true_y.min(), true_y.max()], [true_y.min(), true_y.max()], 
             'k--', label='Perfect tracking', linewidth=2)
    ax2.set_xlabel('True Y Position (pixels)', fontsize=11)
    ax2.set_ylabel('Tracked Y Position (pixels)', fontsize=11)
    ax2.set_title('Y Position Tracking', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: X position error vs true X
    ax3 = axes[1, 0]
    ax3.plot(true_x, trackpy_x - true_x, 'o', label='Trackpy', alpha=0.6, markersize=3)
    ax3.plot(true_x, gaussian_x - true_x, 's', label='Gaussian', alpha=0.6, markersize=3)
    ax3.plot(true_x, parabola_x - true_x, '^', label='Parabola', alpha=0.6, markersize=3)
    ax3.axhline(y=0, color='k', linestyle='--', linewidth=2, label='Zero error')
    ax3.set_xlabel('True X Position (pixels)', fontsize=11)
    ax3.set_ylabel('X Position Error (pixels)', fontsize=11)
    ax3.set_title('X Position Error', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Y position error vs true X
    ax4 = axes[1, 1]
    ax4.plot(true_x, trackpy_y - true_y, 'o', label='Trackpy', alpha=0.6, markersize=3)
    ax4.plot(true_x, gaussian_y - true_y, 's', label='Gaussian', alpha=0.6, markersize=3)
    ax4.plot(true_x, parabola_y - true_y, '^', label='Parabola', alpha=0.6, markersize=3)
    ax4.axhline(y=0, color='k', linestyle='--', linewidth=2, label='Zero error')
    ax4.set_xlabel('True X Position (pixels)', fontsize=11)
    ax4.set_ylabel('Y Position Error (pixels)', fontsize=11)
    ax4.set_title('Y Position Error', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_file = output_dir / 'tracking_characterization.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Saved plot: {output_file}")
    
    plt.close()


def plot_bias_statistics(statistics, output_dir):
    """
    Create bar plots of bias statistics.
    
    Args:
        statistics: List of statistics dictionaries
        output_dir: Path to save plots
    """
    methods = [s['method'] for s in statistics]
    x_bias = [s['x_bias'] for s in statistics]
    y_bias = [s['y_bias'] for s in statistics]
    x_std = [s['x_std'] for s in statistics]
    y_std = [s['y_std'] for s in statistics]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Pixel Bias Statistics', fontsize=14, fontweight='bold')
    
    # Plot 1: Bias
    ax1 = axes[0]
    x_pos = np.arange(len(methods))
    width = 0.35
    
    ax1.bar(x_pos - width/2, x_bias, width, label='X Bias', alpha=0.8)
    ax1.bar(x_pos + width/2, y_bias, width, label='Y Bias', alpha=0.8)
    ax1.set_ylabel('Mean Error (pixels)', fontsize=11)
    ax1.set_xlabel('Tracking Method', fontsize=11)
    ax1.set_title('Mean Bias', fontsize=12, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(methods)
    ax1.legend()
    ax1.axhline(y=0, color='k', linestyle='--', linewidth=1)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Standard deviation
    ax2 = axes[1]
    ax2.bar(x_pos - width/2, x_std, width, label='X Std Dev', alpha=0.8)
    ax2.bar(x_pos + width/2, y_std, width, label='Y Std Dev', alpha=0.8)
    ax2.set_ylabel('Standard Deviation (pixels)', fontsize=11)
    ax2.set_xlabel('Tracking Method', fontsize=11)
    ax2.set_title('Error Variability', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(methods)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save figure
    output_file = output_dir / 'bias_statistics.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Saved plot: {output_file}")
    
    plt.close()


def print_statistics_summary(statistics):
    """
    Print a formatted table of statistics.
    
    Args:
        statistics: List of statistics dictionaries
    """
    print("\n" + "=" * 90)
    print("PIXEL BIAS STATISTICS SUMMARY")
    print("=" * 90)
    print(f"{'Method':<12} {'Success Rate':<14} {'X Bias':<12} {'Y Bias':<12} "
          f"{'X Std':<12} {'Y Std':<12} {'Mean Error':<12}")
    print("-" * 90)
    
    for stat in statistics:
        print(f"{stat['method'].capitalize():<12} "
              f"{stat['success_rate']:<14.2%} "
              f"{stat['x_bias']:<12.6f} "
              f"{stat['y_bias']:<12.6f} "
              f"{stat['x_std']:<12.6f} "
              f"{stat['y_std']:<12.6f} "
              f"{stat['mean_error']:<12.6f}")
    
    print("=" * 90 + "\n")


def main():
    """Main characterization function."""
    print("=" * 90)
    print("TRACKING METHOD CHARACTERIZATION")
    print("=" * 90)
    print("\nTest Configuration:")
    print("  - Single particle with diameter 4 pixels")
    print("  - Starting position: (50.5, 50.5)")
    print("  - Movement: 0.01 pixel/step in X direction")
    print("  - Number of steps: 201 (total movement: 2 pixels)")
    print("  - Tracking methods: trackpy, gaussian, parabola")
    print()
    
    # Create output directory
    output_dir = Path("characterization_output")
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir.absolute()}\n")
    
    # Track moving particle
    print("=" * 90)
    print("STEP 1: Tracking particle across 201 frames")
    print("=" * 90)
    data = track_moving_particle(n_steps=201, step_size=0.01, start_x=50.5, start_y=50.5)
    print("✓ Tracking complete\n")
    
    # Compute statistics
    print("=" * 90)
    print("STEP 2: Computing bias statistics")
    print("=" * 90)
    
    trackpy_positions = extract_tracked_positions(data['trackpy_results'])[:2]
    gaussian_positions = extract_tracked_positions(data['gaussian_results'])[:2]
    parabola_positions = extract_tracked_positions(data['parabola_results'])[:2]
    
    statistics = [
        compute_bias_statistics(trackpy_positions, data['true_positions'], 'trackpy'),
        compute_bias_statistics(gaussian_positions, data['true_positions'], 'gaussian'),
        compute_bias_statistics(parabola_positions, data['true_positions'], 'parabola')
    ]
    
    print_statistics_summary(statistics)
    
    # Generate plots
    print("=" * 90)
    print("STEP 3: Generating plots")
    print("=" * 90)
    plot_tracking_results(data, output_dir)
    plot_bias_statistics(statistics, output_dir)
    print("✓ All plots generated\n")
    
    print("=" * 90)
    print("CHARACTERIZATION COMPLETE")
    print("=" * 90)
    print(f"\nResults saved to: {output_dir.absolute()}")
    print("  - tracking_characterization.png: Main tracking plots")
    print("  - bias_statistics.png: Bias and variability statistics")
    print()
    
    return data, statistics


if __name__ == "__main__":
    data, statistics = main()
