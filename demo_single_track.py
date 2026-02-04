"""
Demo script for single spot tracking with quantitative outputs and visualization.

This script demonstrates tracking a single spot on a 100x100 image using all
available tracking methods, provides quantitative metrics, and generates plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from image_tracker import create_demo_image, track_spot
from pathlib import Path


def create_output_dir():
    """Create output directory for plots."""
    output_dir = Path("demo_output")
    output_dir.mkdir(exist_ok=True)
    return output_dir


def plot_tracking_results(image, results, true_position, output_dir):
    """
    Create visualization plots for tracking results.
    
    Args:
        image: The test image
        results: Dictionary of tracking results by method
        true_position: True (x, y) position of the spot
        output_dir: Directory to save plots
    """
    # Filter successful results
    successful_results = {k: v for k, v in results.items() if v.get("success", False)}
    
    if not successful_results:
        print("No successful tracking results to plot.")
        return
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # Plot 1: Image with tracked positions
    ax1 = plt.subplot(2, 3, 1)
    im = ax1.imshow(image, cmap='hot', origin='lower')
    plt.colorbar(im, ax=ax1, label='Intensity')
    ax1.set_title('Image with Tracked Positions', fontsize=12, fontweight='bold')
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)')
    
    # Plot true position
    ax1.plot(true_position[0], true_position[1], 'g+', markersize=15, 
             markeredgewidth=2, label='True Position')
    
    # Plot tracked positions
    colors = {'gaussian': 'cyan', 'parabola': 'yellow', 'pytrack': 'magenta'}
    markers = {'gaussian': 'x', 'parabola': '^', 'pytrack': 'o'}
    
    for method, result in successful_results.items():
        color = colors.get(method, 'white')
        marker = markers.get(method, 's')
        ax1.plot(result['x'], result['y'], marker, color=color, 
                markersize=10, markeredgewidth=2, 
                label=f'{method.capitalize()}')
    
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Zoomed view around spot
    ax2 = plt.subplot(2, 3, 2)
    x_center, y_center = true_position
    zoom_size = 15
    x_min = max(0, int(x_center - zoom_size))
    x_max = min(image.shape[1], int(x_center + zoom_size))
    y_min = max(0, int(y_center - zoom_size))
    y_max = min(image.shape[0], int(y_center + zoom_size))
    
    zoomed = image[y_min:y_max, x_min:x_max]
    im2 = ax2.imshow(zoomed, cmap='hot', origin='lower', 
                     extent=[x_min, x_max, y_min, y_max])
    plt.colorbar(im2, ax=ax2, label='Intensity')
    ax2.set_title('Zoomed View (±15 pixels)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('X (pixels)')
    ax2.set_ylabel('Y (pixels)')
    
    # Plot positions on zoomed view
    ax2.plot(true_position[0], true_position[1], 'g+', markersize=12, 
             markeredgewidth=2, label='True')
    
    for method, result in successful_results.items():
        color = colors.get(method, 'white')
        marker = markers.get(method, 's')
        ax2.plot(result['x'], result['y'], marker, color=color, 
                markersize=8, markeredgewidth=1.5)
    
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Position error comparison
    ax3 = plt.subplot(2, 3, 3)
    methods_list = list(successful_results.keys())
    errors = []
    for method in methods_list:
        result = successful_results[method]
        error = np.sqrt((result['x'] - true_position[0])**2 + 
                       (result['y'] - true_position[1])**2)
        errors.append(error)
    
    bars = ax3.bar(range(len(methods_list)), errors, 
                   color=[colors.get(m, 'gray') for m in methods_list],
                   edgecolor='black', linewidth=1.5)
    ax3.set_xticks(range(len(methods_list)))
    ax3.set_xticklabels([m.capitalize() for m in methods_list], rotation=45)
    ax3.set_ylabel('Position Error (pixels)', fontweight='bold')
    ax3.set_title('Tracking Accuracy Comparison', fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, error) in enumerate(zip(bars, errors)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{error:.4f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Plot 4: X and Y error components
    ax4 = plt.subplot(2, 3, 4)
    x_errors = [successful_results[m]['x'] - true_position[0] for m in methods_list]
    y_errors = [successful_results[m]['y'] - true_position[1] for m in methods_list]
    
    x_pos = np.arange(len(methods_list))
    width = 0.35
    
    bars1 = ax4.bar(x_pos - width/2, x_errors, width, label='X Error', 
                    color='steelblue', edgecolor='black')
    bars2 = ax4.bar(x_pos + width/2, y_errors, width, label='Y Error', 
                    color='coral', edgecolor='black')
    
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([m.capitalize() for m in methods_list], rotation=45)
    ax4.set_ylabel('Error (pixels)', fontweight='bold')
    ax4.set_title('X and Y Position Errors', fontsize=12, fontweight='bold')
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    # Plot 5: R² comparison (for methods that have it)
    ax5 = plt.subplot(2, 3, 5)
    r2_methods = [m for m in methods_list if 'r_squared' in successful_results[m]]
    if r2_methods:
        r2_values = [successful_results[m]['r_squared'] for m in r2_methods]
        bars = ax5.bar(range(len(r2_methods)), r2_values,
                      color=[colors.get(m, 'gray') for m in r2_methods],
                      edgecolor='black', linewidth=1.5)
        ax5.set_xticks(range(len(r2_methods)))
        ax5.set_xticklabels([m.capitalize() for m in r2_methods], rotation=45)
        ax5.set_ylabel('R² Score', fontweight='bold')
        ax5.set_title('Goodness of Fit (R²)', fontsize=12, fontweight='bold')
        ax5.set_ylim([0, 1.05])
        ax5.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, r2 in zip(bars, r2_values):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{r2:.4f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    else:
        ax5.text(0.5, 0.5, 'No R² data available', 
                ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Goodness of Fit (R²)', fontsize=12, fontweight='bold')
    
    # Plot 6: Summary statistics table
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('tight')
    ax6.axis('off')
    
    # Prepare table data
    table_data = [['Method', 'X Pos', 'Y Pos', 'Error (px)', 'R²']]
    for method in methods_list:
        result = successful_results[method]
        error = np.sqrt((result['x'] - true_position[0])**2 + 
                       (result['y'] - true_position[1])**2)
        r2_str = f"{result['r_squared']:.4f}" if 'r_squared' in result else 'N/A'
        table_data.append([
            method.capitalize(),
            f"{result['x']:.4f}",
            f"{result['y']:.4f}",
            f"{error:.4f}",
            r2_str
        ])
    
    table = ax6.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.18, 0.18, 0.18, 0.18, 0.18])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header row
    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(5):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    ax6.set_title('Quantitative Results Summary', fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save figure
    output_file = output_dir / 'tracking_results.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {output_file}")
    
    return fig


def print_quantitative_summary(results, true_position):
    """Print detailed quantitative summary of tracking results."""
    
    print("\n" + "=" * 80)
    print("QUANTITATIVE TRACKING RESULTS SUMMARY")
    print("=" * 80)
    
    successful_results = {k: v for k, v in results.items() if v.get("success", False)}
    
    if not successful_results:
        print("No successful tracking results.")
        return
    
    print(f"\nTrue Position: ({true_position[0]:.4f}, {true_position[1]:.4f})")
    print("\n" + "-" * 80)
    print(f"{'Method':<12} {'X Position':<12} {'Y Position':<12} {'X Error':<12} "
          f"{'Y Error':<12} {'Total Error':<12}")
    print("-" * 80)
    
    errors_dict = {}
    for method, result in successful_results.items():
        x_err = result['x'] - true_position[0]
        y_err = result['y'] - true_position[1]
        total_err = np.sqrt(x_err**2 + y_err**2)
        errors_dict[method] = total_err
        
        print(f"{method.capitalize():<12} {result['x']:<12.4f} {result['y']:<12.4f} "
              f"{x_err:<12.4f} {y_err:<12.4f} {total_err:<12.4f}")
    
    print("-" * 80)
    
    # Print additional metrics
    print("\nAdditional Metrics:")
    print("-" * 80)
    
    for method, result in successful_results.items():
        print(f"\n{method.upper()}:")
        if 'r_squared' in result:
            print(f"  R² Score: {result['r_squared']:.6f}")
        if 'amplitude' in result:
            print(f"  Fitted Amplitude: {result['amplitude']:.4f}")
        if 'sigma_x' in result and 'sigma_y' in result:
            print(f"  Fitted Sigma: ({result['sigma_x']:.4f}, {result['sigma_y']:.4f})")
        if 'intensity' in result:
            print(f"  Spot Intensity: {result['intensity']:.4f}")
    
    # Print ranking
    print("\n" + "=" * 80)
    print("ACCURACY RANKING (Best to Worst)")
    print("=" * 80)
    
    sorted_methods = sorted(errors_dict.items(), key=lambda x: x[1])
    for rank, (method, error) in enumerate(sorted_methods, 1):
        print(f"{rank}. {method.capitalize():<12} - Error: {error:.6f} pixels")
    
    print("=" * 80 + "\n")


def demo_single_spot_tracking():
    """Main demo function for single spot tracking."""
    
    print("=" * 80)
    print("SINGLE SPOT TRACKING DEMO")
    print("=" * 80)
    print("\nThis demo creates a 100x100 test image with a single Gaussian spot")
    print("and tracks it using all available methods.")
    print()
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create output directory
    output_dir = create_output_dir()
    print(f"Output directory: {output_dir.absolute()}")
    
    # Create test image
    true_position = (50.0, 50.0)
    print(f"\nCreating test image with spot at {true_position}...")
    image = create_demo_image(
        size=100, 
        spot_center=true_position,
        spot_amplitude=100.0,
        spot_sigma=2.0
    )
    
    print(f"  Image shape: {image.shape}")
    print(f"  Image range: [{image.min():.2f}, {image.max():.2f}]")
    print(f"  Mean intensity: {image.mean():.2f}")
    print(f"  Std deviation: {image.std():.2f}")
    
    # Track with all methods
    print("\n" + "-" * 80)
    print("TRACKING WITH ALL METHODS")
    print("-" * 80)
    
    methods = ["gaussian", "parabola", "pytrack"]
    results = {}
    
    for method in methods:
        print(f"\nTesting {method.upper()} method...")
        try:
            result = track_spot(image, method=method, initial_guess=true_position)
            results[method] = result
            
            if result["success"]:
                error = np.sqrt((result['x'] - true_position[0])**2 + 
                              (result['y'] - true_position[1])**2)
                print(f"  ✓ Success! Position: ({result['x']:.4f}, {result['y']:.4f})")
                print(f"    Error: {error:.6f} pixels")
            else:
                print(f"  ✗ Failed: {result.get('error', 'Unknown error')}")
        except Exception as e:
            print(f"  ✗ Exception: {str(e)}")
            results[method] = {"success": False, "error": str(e)}
    
    # Print quantitative summary
    print_quantitative_summary(results, true_position)
    
    # Generate plots
    print("Generating visualization plots...")
    fig = plot_tracking_results(image, results, true_position, output_dir)
    
    if fig:
        print("\n✓ Visualization complete! Check the plots for detailed analysis.")
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETED SUCCESSFULLY")
    print("=" * 80)
    
    return results, image


if __name__ == "__main__":
    results, image = demo_single_spot_tracking()
    
    # Show plots (optional - comment out if running in non-interactive environment)
    print("\nDisplaying plots... (close plot window to exit)")
    plt.show()
