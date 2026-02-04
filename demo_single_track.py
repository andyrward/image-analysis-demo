"""
Demo script for single spot tracking with quantitative outputs and interactive visualization.

This script demonstrates tracking a single spot on a 100x100 image using all
available tracking methods, provides quantitative metrics, and generates interactive plots.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from image_tracker import create_demo_image, track_spot
from pathlib import Path


def create_output_dir():
    """Create output directory for plots."""
    output_dir = Path("demo_output")
    output_dir.mkdir(exist_ok=True)
    return output_dir


def plot_tracking_results(image, results, true_position, output_dir):
    """
    Create interactive visualization plots for tracking results using Plotly.
    
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
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Image with Tracked Positions',
            'Zoomed View (±15 pixels)',
            'Position Error Comparison',
            'X and Y Error Components',
            'Goodness of Fit (R²)',
            'Error Scatter Plot'
        ),
        specs=[
            [{"type": "heatmap"}, {"type": "heatmap"}],
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "scatter"}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.12
    )
    
    # Plot 1: Full image with tracked positions
    fig.add_trace(
        go.Heatmap(
            z=image,
            colorscale='Hot',
            showscale=True,
            colorbar=dict(x=0.46, len=0.25, y=0.87),
            hovertemplate='X: %{x}<br>Y: %{y}<br>Intensity: %{z:.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Add true position
    fig.add_trace(
        go.Scatter(
            x=[true_position[0]], y=[true_position[1]],
            mode='markers',
            marker=dict(symbol='x', size=15, color='lime', line=dict(width=2)),
            name='True Position',
            hovertemplate='True: (%{x:.2f}, %{y:.2f})<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Add tracked positions
    colors_map = {'gaussian': 'cyan', 'parabola': 'yellow', 'pytrack': 'magenta'}
    markers_map = {'gaussian': 'x', 'parabola': 'triangle-up', 'pytrack': 'circle'}
    
    for method, result in successful_results.items():
        fig.add_trace(
            go.Scatter(
                x=[result['x']], y=[result['y']],
                mode='markers',
                marker=dict(
                    symbol=markers_map.get(method, 'square'),
                    size=12,
                    color=colors_map.get(method, 'white'),
                    line=dict(width=2, color='black')
                ),
                name=method.capitalize(),
                hovertemplate=f'{method.capitalize()}: (%{{x:.4f}}, %{{y:.4f}})<extra></extra>'
            ),
            row=1, col=1
        )
    
    # Plot 2: Zoomed view
    x_center, y_center = true_position
    zoom_size = 15
    x_min = max(0, int(x_center - zoom_size))
    x_max = min(image.shape[1], int(x_center + zoom_size))
    y_min = max(0, int(y_center - zoom_size))
    y_max = min(image.shape[0], int(y_center + zoom_size))
    
    zoomed = image[y_min:y_max, x_min:x_max]
    
    fig.add_trace(
        go.Heatmap(
            z=zoomed,
            x=list(range(x_min, x_max)),
            y=list(range(y_min, y_max)),
            colorscale='Hot',
            showscale=True,
            colorbar=dict(x=1.0, len=0.25, y=0.87),
            hovertemplate='X: %{x}<br>Y: %{y}<br>Intensity: %{z:.2f}<extra></extra>'
        ),
        row=1, col=2
    )
    
    # Add markers to zoomed view
    fig.add_trace(
        go.Scatter(
            x=[true_position[0]], y=[true_position[1]],
            mode='markers',
            marker=dict(symbol='x', size=12, color='lime', line=dict(width=2)),
            showlegend=False,
            hovertemplate='True: (%{x:.2f}, %{y:.2f})<extra></extra>'
        ),
        row=1, col=2
    )
    
    for method, result in successful_results.items():
        fig.add_trace(
            go.Scatter(
                x=[result['x']], y=[result['y']],
                mode='markers',
                marker=dict(
                    symbol=markers_map.get(method, 'square'),
                    size=10,
                    color=colors_map.get(method, 'white'),
                    line=dict(width=2, color='black')
                ),
                showlegend=False,
                hovertemplate=f'{method.capitalize()}: (%{{x:.4f}}, %{{y:.4f}})<extra></extra>'
            ),
            row=1, col=2
        )
    
    # Plot 3: Position error comparison
    methods_list = list(successful_results.keys())
    errors = []
    for method in methods_list:
        result = successful_results[method]
        error = np.sqrt((result['x'] - true_position[0])**2 + 
                       (result['y'] - true_position[1])**2)
        errors.append(error)
    
    fig.add_trace(
        go.Bar(
            x=[m.capitalize() for m in methods_list],
            y=errors,
            marker=dict(
                color=[colors_map.get(m, 'gray') for m in methods_list],
                line=dict(color='black', width=2)
            ),
            text=[f'{e:.4f}' for e in errors],
            textposition='outside',
            showlegend=False,
            hovertemplate='%{x}<br>Error: %{y:.6f} px<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Plot 4: X and Y error components
    x_errors = [successful_results[m]['x'] - true_position[0] for m in methods_list]
    y_errors = [successful_results[m]['y'] - true_position[1] for m in methods_list]
    
    fig.add_trace(
        go.Bar(
            x=[m.capitalize() for m in methods_list],
            y=x_errors,
            name='X Error',
            marker=dict(color='steelblue', line=dict(color='black', width=1.5)),
            hovertemplate='%{x}<br>X Error: %{y:.4f} px<extra></extra>'
        ),
        row=2, col=2
    )
    
    fig.add_trace(
        go.Bar(
            x=[m.capitalize() for m in methods_list],
            y=y_errors,
            name='Y Error',
            marker=dict(color='coral', line=dict(color='black', width=1.5)),
            hovertemplate='%{x}<br>Y Error: %{y:.4f} px<extra></extra>'
        ),
        row=2, col=2
    )
    
    # Plot 5: R² comparison
    r2_methods = [m for m in methods_list if 'r_squared' in successful_results[m]]
    if r2_methods:
        r2_values = [successful_results[m]['r_squared'] for m in r2_methods]
        fig.add_trace(
            go.Bar(
                x=[m.capitalize() for m in r2_methods],
                y=r2_values,
                marker=dict(
                    color=[colors_map.get(m, 'gray') for m in r2_methods],
                    line=dict(color='black', width=2)
                ),
                text=[f'{r:.4f}' for r in r2_values],
                textposition='outside',
                showlegend=False,
                hovertemplate='%{x}<br>R²: %{y:.6f}<extra></extra>'
            ),
            row=3, col=1
        )
    
    # Plot 6: Error scatter plot
    for method in methods_list:
        result = successful_results[method]
        x_err = result['x'] - true_position[0]
        y_err = result['y'] - true_position[1]
        
        fig.add_trace(
            go.Scatter(
                x=[x_err], y=[y_err],
                mode='markers',
                marker=dict(
                    symbol=markers_map.get(method, 'square'),
                    size=15,
                    color=colors_map.get(method, 'white'),
                    line=dict(width=2, color='black')
                ),
                name=f'{method.capitalize()} Error',
                showlegend=False,
                hovertemplate=f'{method.capitalize()}<br>X Error: %{{x:.4f}}<br>Y Error: %{{y:.4f}}<extra></extra>'
            ),
            row=3, col=2
        )
    
    # Add perfect position marker
    fig.add_trace(
        go.Scatter(
            x=[0], y=[0],
            mode='markers',
            marker=dict(symbol='x', size=20, color='lime', line=dict(width=3)),
            name='Perfect (0,0)',
            showlegend=False,
            hovertemplate='Perfect: (0, 0)<extra></extra>'
        ),
        row=3, col=2
    )
    
    # Add reference lines to scatter plot
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=2)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", row=3, col=2)
    
    # Update layout
    fig.update_xaxes(title_text="X (pixels)", row=1, col=1)
    fig.update_yaxes(title_text="Y (pixels)", row=1, col=1)
    fig.update_xaxes(title_text="X (pixels)", row=1, col=2)
    fig.update_yaxes(title_text="Y (pixels)", row=1, col=2)
    fig.update_xaxes(title_text="Method", row=2, col=1)
    fig.update_yaxes(title_text="Position Error (pixels)", row=2, col=1)
    fig.update_xaxes(title_text="Method", row=2, col=2)
    fig.update_yaxes(title_text="Error (pixels)", row=2, col=2)
    fig.update_xaxes(title_text="Method", row=3, col=1)
    fig.update_yaxes(title_text="R² Score", row=3, col=1)
    fig.update_xaxes(title_text="X Error (pixels)", row=3, col=2)
    fig.update_yaxes(title_text="Y Error (pixels)", row=3, col=2)
    
    # Update overall layout
    fig.update_layout(
        title_text="<b>Interactive Tracking Results Analysis</b>",
        title_x=0.5,
        title_font_size=20,
        height=1200,
        width=1400,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode='closest'
    )
    
    # Save as interactive HTML
    output_file = output_dir / 'tracking_results_interactive.html'
    fig.write_html(str(output_file))
    print(f"\n✓ Interactive plot saved to: {output_file}")
    
    # Also save as static image
    try:
        static_file = output_dir / 'tracking_results.png'
        fig.write_image(str(static_file), width=1400, height=1200)
        print(f"✓ Static plot saved to: {static_file}")
    except Exception as e:
        print(f"  Note: Could not save static image (kaleido may not be installed): {e}")
    
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
    
    print("\nDemo completed! Open the HTML file in your browser for interactive visualization.")
    print("Note: The interactive plot allows you to:")
    print("  - Hover over points to see exact values")
    print("  - Zoom in/out on any subplot")
    print("  - Pan around the images")
    print("  - Toggle traces on/off by clicking legend items")
