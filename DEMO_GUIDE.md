# Demo Usage Guide

## Quick Start

### 1. Install Dependencies
```bash
pip install numpy scipy pillow plotly pandas jupyter kaleido
```

### 2. Run Basic Demo
```bash
python demo_quick.py
```
This provides a quick overview of all tracking methods.

### 3. Run Comprehensive Demo (Interactive Plots)
```bash
python demo_single_track.py
```
This generates:
- `demo_output/tracking_results_interactive.html` - Interactive Plotly visualization
- `demo_output/tracking_results.png` - Static image export
- Detailed console output with quantitative metrics

### 4. Run Jupyter Notebook (Best for Exploration)
```bash
jupyter notebook demo_tracking.ipynb
```
This provides an interactive, step-by-step analysis with inline visualizations.

## Output Files

### Interactive HTML (`tracking_results_interactive.html`)
Open in your browser for:
- **Hover** to see exact pixel values
- **Zoom** by drawing a box or scrolling
- **Pan** by clicking and dragging
- **Toggle** methods on/off via legend
- **Export** to PNG using camera icon

### Static PNG (`tracking_results.png`)
High-resolution static image showing all 6 analysis plots.

## What's Included in the Visualizations

1. **Image with Tracked Positions**: Full 100x100 image with all method overlays
2. **Zoomed View**: ±15 pixel region around the spot
3. **Position Error Comparison**: Bar chart ranking accuracy
4. **X and Y Error Components**: Breakdown of errors in each dimension
5. **Goodness of Fit (R²)**: Quality metrics for curve-fitting methods
6. **Error Scatter Plot**: Visual representation of accuracy

## Quantitative Metrics Provided

- **Position (X, Y)**: Tracked coordinates for each method
- **Error (X, Y, Total)**: Deviation from true position
- **R² Score**: Goodness of fit (for Gaussian and Parabola)
- **Fitted Parameters**: Amplitude, sigma, etc. (method-specific)
- **Accuracy Ranking**: Methods sorted by performance

## Expected Results

On a 100x100 image with centered Gaussian spot:
- **Gaussian Method**: ~0.01 px error (sub-pixel accuracy)
- **Parabola Method**: ~0.4 px error
- **Pytrack Method**: ~0.4 px error

## Tips

1. **For presentations**: Use the interactive HTML file
2. **For reports**: Use the PNG export
3. **For learning**: Use the Jupyter notebook
4. **For quick tests**: Use demo_quick.py

## Customization

You can modify tracking parameters in the demo scripts:
- `spot_amplitude`: Brightness of the spot
- `spot_sigma`: Width of the Gaussian
- `spot_center`: Position of the spot
- `window_size`: Region size for fitting algorithms

Enjoy exploring the tracking capabilities! 🎯
