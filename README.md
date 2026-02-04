# Image Analysis Demo - Particle Tracking

A flexible Python library for tracking particles in microscopy images and video, featuring multiple detection methods and comprehensive analysis tools.

## Features

### Multiple Detection Methods
- **Trackpy (Crocker-Grier)**: Fast, proven centroid-based detection
- **Gaussian Fitting**: Sub-pixel precision via 2D Gaussian fitting with 3D surface visualization
- **Polynomial Fitting**: Surface fitting for complex particle shapes with 3D visualization

All methods produce compatible output for unified trajectory linking and analysis.

### Trajectory Linking
Uses trackpy's robust trajectory linking algorithm (works with all detection methods).

### Motion Analysis
- Mean squared displacement (MSD)
- Velocity autocorrelation
- Drift analysis
- Individual and ensemble statistics

### Rich Visualizations
- Trajectory plots and animations
- 3D surface plots for Gaussian/polynomial fits
- Contour plots and residual analysis
- Method comparison tools
- Ground truth overlay for validation

### Synthetic Data Generation
Built-in synthetic data generation (`spot_generator.py`) with known ground truth for:
- Testing and validation
- Algorithm comparison
- Accuracy measurement

## Quick Start

```python
import workflows as wf

# Complete pipeline in one call
trajectories, results = wf.complete_pipeline(
    'video.avi',
    method='gaussian',
    detector_params={'sigma': 2.0, 'threshold': 100},
    link_params={'search_range': 5, 'memory': 3},
    mpp=0.16,
    fps=30
)

# Generate all plots
wf.generate_all_plots('video.avi', trajectories, results,
                      output_dir='results/', frame_shape=(512, 512))
```

## Using Synthetic Data

```python
# Test with synthetic data (known ground truth)
trajectories, truth, metrics = wf.demo_with_synthetic_data(
    scenario='brownian',
    method='gaussian',
    detector_params={'sigma': 2.0, 'threshold': 100}
)

print(f"Mean localization error: {metrics['mean_error']:.2f} pixels")
print(f"Detection rate: {metrics['detection_rate']:.1%}")
```

## Comparing Methods

```python
# Compare all detection methods
wf.quick_compare_methods(
    'video.avi',
    trackpy_params={'diameter': 11, 'minmass': 100},
    gaussian_params={'sigma': 2.0, 'threshold': 100},
    polynomial_params={'window_size': 7, 'degree': 2, 'threshold': 100},
    save_path='comparison.png'
)
```

## Surface Plot Visualizations

```python
# Visualize Gaussian fit in 3D
wf.quick_viz_fit_surface(
    'video.avi',
    frame_idx=0,
    particle_idx=0,
    method='gaussian',
    save_path='gaussian_fit_3d.png'
)
```

## Installation

```bash
pip install trackpy pims pandas scipy matplotlib numpy pillow
```

Or with the pyproject.toml:

```bash
pip install -e .
```

## Project Structure

```
image-analysis-demo/
├── spot_generator.py      # Synthetic data generation (existing)
├── data_loader.py         # Load videos and synthetic data
├── feature_detectors.py   # Multiple detection methods
├── fitting_utils.py       # Gaussian/polynomial fitting utilities
├── trajectory_linker.py   # Link features into trajectories
├── motion_analyzer.py     # MSD and motion analysis
├── visualizer.py          # All visualization functions
├── accuracy.py            # Ground truth comparison
├── workflows.py           # High-level convenience API
├── examples.py            # Complete workflow examples
└── tests/                 # Comprehensive test suite
    ├── test_data_loader.py
    ├── test_feature_detectors.py
    ├── test_fitting_utils.py
    ├── test_trajectory_linker.py
    ├── test_motion_analyzer.py
    ├── test_accuracy.py
    ├── test_workflows.py
    └── test_visualizations.py
```

## Detailed Usage Examples

### Basic Particle Tracking

```python
import workflows as wf

# Track particles using trackpy method (fastest)
trajectories = wf.track_particles(
    'video.avi',
    method='trackpy',
    detector_params={'diameter': 11, 'minmass': 100},
    link_params={'search_range': 5, 'memory': 3},
    filter_min_length=10
)

print(f"Tracked {trajectories['particle'].nunique()} particles")
```

### Motion Analysis

```python
import workflows as wf

# Complete motion analysis
results = wf.analyze_motion(
    trajectories,
    mpp=0.16,  # microns per pixel
    fps=30,    # frames per second
    max_lagtime=100
)

# Plot MSD
wf.quick_viz_msd(results['msd'], save_path='msd_plot.png')
```

### Detection Method Comparison

```python
import workflows as wf

# Compare accuracy of all methods on synthetic data
comparison = wf.compare_methods_on_synthetic(
    scenario='brownian',
    methods=['trackpy', 'gaussian', 'polynomial']
)

for method, metrics in comparison.items():
    print(f"{method}: error={metrics['mean_error']:.2f} px, "
          f"detection_rate={metrics['detection_rate']:.1%}")
```

### Custom Workflow

```python
import data_loader as dl
import feature_detectors as fd
import trajectory_linker as tl
import motion_analyzer as ma
import visualizer as viz

# Load data
frames = dl.load_video('video.avi')

# Detect particles with Gaussian fitting
detector = fd.GaussianDetector()
features = detector.detect_batch(frames, sigma=2.0, threshold=100)

# Link trajectories
trajectories = tl.link_trajectories(features, search_range=5, memory=3)
trajectories = tl.filter_trajectories(trajectories, min_length=10)

# Analyze motion
msd = ma.compute_msd(trajectories, mpp=0.16, fps=30)

# Visualize
viz.plot_trajectories(trajectories, frame_shape=frames[0].shape)
viz.plot_msd(msd)
```

### Working with Synthetic Data

```python
import data_loader as dl
import accuracy as acc
from spot_generator import generate_spots_image

# Generate synthetic data with known ground truth
image = generate_spots_image(n=5, spot_width=3.0, signal=100.0, 
                            noise_magnitude=5.0, image_size=512)

# Or use the data loader wrapper
synthetic_data = dl.load_synthetic_data(
    scenario='brownian',
    n_spots=25,
    spot_width=3.0,
    signal=100.0,
    noise_magnitude=5.0
)

# Track and compare with ground truth
# (ground truth positions stored in synthetic_data['ground_truth'])
```

### 3D Surface Visualization

```python
import visualizer as viz
import data_loader as dl
import feature_detectors as fd

# Load frame
frames = dl.load_video('video.avi')
frame = frames[0]

# Detect with Gaussian fitting
detector = fd.GaussianDetector()
features = detector.detect_single(frame, sigma=2.0, threshold=100)

# Get fit parameters for first particle
position = (features.iloc[0]['x'], features.iloc[0]['y'])
fit_params = detector.get_fit_params(0)  # Get stored fit parameters

# Create 3D surface plot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(121, projection='3d')
viz.plot_gaussian_fit_surface(frame, position, fit_params, window_size=7, ax=ax)

ax2 = fig.add_subplot(122)
viz.plot_gaussian_fit_contours(frame, position, fit_params, window_size=7)
plt.show()
```

## Running Examples

The library includes complete working examples:

```python
import examples as ex

# Run basic tracking example
ex.example_basic_tracking_trackpy()

# Run Gaussian fitting example with 3D visualization
ex.example_gaussian_tracking()

# Compare all detection methods
ex.example_compare_detectors()

# Complete MSD analysis workflow
ex.example_msd_analysis()

# Synthetic data demo with ground truth validation
ex.example_synthetic_data_demo()

# Full demo with all visualizations
ex.example_with_visualizations()
```

## Running Tests

```bash
# Install pytest
pip install pytest

# Run all tests
pytest tests/

# Run specific test module
pytest tests/test_feature_detectors.py

# Run with verbose output
pytest tests/ -v

# Run tests matching a pattern
pytest tests/ -k "gaussian"
```

## API Reference

### data_loader Module

- `load_video(video_path)` - Load video using pims
- `load_image_sequence(directory, pattern='*.tif')` - Load series of images
- `load_synthetic_data(scenario='brownian', **kwargs)` - Generate synthetic data
- `get_frame(source, frame_idx)` - Get single frame from video source

### feature_detectors Module

All detectors inherit from `BaseDetector` and return pandas DataFrames with columns:
- Required: `['x', 'y', 'frame']`
- Optional: `['mass', 'size', 'ecc', 'signal', 'raw_mass', 'ep']`

Classes:
- `TrackpyDetector()` - Crocker-Grier algorithm (wraps trackpy.locate/batch)
- `GaussianDetector()` - 2D Gaussian fitting for sub-pixel localization
- `PolynomialDetector()` - 2D polynomial surface fitting

Methods:
- `detect_single(frame, **params)` - Detect in single frame
- `detect_batch(frames, **params)` - Detect in multiple frames

### fitting_utils Module

- `find_local_maxima(frame, threshold, min_separation)` - Find local maxima
- `extract_region(frame, center, window_size)` - Extract region around point
- `fit_gaussian_2d(image_region)` - Fit 2D Gaussian to image region
- `fit_polynomial_2d(image_region, degree=2)` - Fit 2D polynomial surface
- `compute_fit_quality(observed, fitted)` - Calculate fit quality metrics

### trajectory_linker Module

- `link_trajectories(features_df, search_range=5, memory=0, **kwargs)` - Link features
- `filter_trajectories(trajectories, min_length=10)` - Remove short trajectories
- `filter_drift(trajectories)` - Subtract overall drift
- `get_trajectory_stats(trajectories)` - Get trajectory statistics

### motion_analyzer Module

- `compute_msd(trajectories, mpp=1.0, fps=1.0, max_lagtime=100)` - Mean squared displacement
- `compute_emsd(trajectories, mpp=1.0, fps=1.0, max_lagtime=100)` - Ensemble MSD
- `compute_imsd(trajectories, mpp=1.0, fps=1.0, max_lagtime=100)` - Individual particle MSDs
- `compute_velocity_autocorr(trajectories, mpp=1.0, fps=1.0)` - Velocity autocorrelation
- `analyze_drift(trajectories)` - Compute drift parameters

### visualizer Module

Standard visualizations:
- `plot_trajectories(trajectories, frame_shape, **kwargs)` - Plot all trajectories
- `annotate_frame(frame, features, **kwargs)` - Overlay detected particles
- `plot_msd(msd_data, log_scale=True)` - Plot MSD vs lag time

3D surface plots:
- `plot_gaussian_fit_surface(frame, position, fit_params, window_size=7, ax=None)` - 3D Gaussian fit
- `plot_polynomial_fit_surface(frame, position, poly_coeffs, degree=2, window_size=7, ax=None)` - 3D polynomial fit
- `plot_gaussian_fit_contours(frame, position, fit_params, window_size=7)` - 2D contour plot
- `plot_polynomial_fit_contours(frame, position, poly_coeffs, window_size=7)` - 2D contour plot

Comparison visualizations:
- `compare_detection_methods(frame, detectors_dict, params_dict)` - Side-by-side comparison
- `plot_detection_diagnostics(features, frame_shape)` - Diagnostic plots
- `plot_linking_diagnostics(trajectories)` - Trajectory diagnostics

Ground truth visualizations:
- `plot_ground_truth_overlay(frame, ground_truth_frame, detected_features)` - Overlay ground truth
- `plot_error_distribution(errors)` - Histogram of localization errors
- `plot_method_accuracy_comparison(accuracy_dict)` - Bar chart comparing accuracy

### accuracy Module

- `compute_localization_error(detected, ground_truth, max_distance=5)` - Calculate position errors
- `compare_trajectories(tracked, ground_truth)` - Compare tracked vs ground truth trajectories
- `compute_detection_metrics(detected, ground_truth, max_distance=5)` - Compute detection metrics
- `plot_accuracy_results(metrics)` - Visualize accuracy metrics

### workflows Module

High-level functions:
- `track_particles(video_path, method='trackpy', detector_params=None, link_params=None, filter_min_length=10)`
- `analyze_motion(trajectories, mpp=1.0, fps=1.0, max_lagtime=100)`
- `complete_pipeline(video_path, method='trackpy', detector_params=None, link_params=None, mpp=1.0, fps=1.0, output_dir=None)`

Quick visualization functions:
- `quick_viz_trajectories(trajectories, frame_shape, save_path=None, figsize=(10, 8), **kwargs)`
- `quick_viz_msd(msd_data, save_path=None, log_scale=True, figsize=(8, 6))`
- `quick_viz_detection(video_path, frame_idx=0, method='trackpy', detector_params=None, save_path=None, figsize=(10, 8))`
- `quick_viz_fit_surface(video_path, frame_idx=0, particle_idx=0, method='gaussian', detector_params=None, save_path=None, figsize=(12, 5))`
- `quick_compare_methods(video_path, frame_idx=0, trackpy_params=None, gaussian_params=None, polynomial_params=None, save_path=None, figsize=(15, 5))`

Workflow functions:
- `generate_all_plots(video_path, trajectories, analysis_results, output_dir, frame_shape, show_fit_examples=True, n_fit_examples=3)`
- `demo_with_synthetic_data(scenario='brownian', method='trackpy', detector_params=None, link_params=None)`
- `compare_methods_on_synthetic(scenario='brownian', methods=['trackpy', 'gaussian', 'polynomial'])`
- `print_analysis_summary(analysis_results)`

## Requirements

- Python >= 3.9
- numpy >= 2.0.2
- pillow >= 11.3.0
- matplotlib >= 3.8.0
- scipy >= 1.11.0
- pandas >= 2.0.0
- trackpy >= 0.6.0
- pims >= 0.6.0

## Based on Trackpy Tutorial

This implementation extends the official trackpy walkthrough:
https://soft-matter.github.io/trackpy/v0.7/tutorial/walkthrough.html

## License

See LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
