# Quick Start Guide

## Installation
All dependencies are already in pyproject.toml:
```bash
pip install numpy pillow jupyter matplotlib scipy pandas trackpy pims
```

## Basic Usage

### 1. Simple Tracking with Synthetic Data
```python
from workflows import demo_with_synthetic_data

# Run a quick demo
frames, trajectories = demo_with_synthetic_data(
    n_spots=5,
    spot_width=2.0,
    signal=1500,
    noise_magnitude=40,
    n_frames=20,
    method='gaussian',
    visualize=True
)
```

### 2. Complete Pipeline
```python
from workflows import complete_pipeline
from data_loader import generate_synthetic_data

# Generate data
frames = generate_synthetic_data(n_spots=5, n_frames=20)

# Run complete analysis
trajectories, analysis = complete_pipeline(
    frames,
    method='gaussian',      # or 'trackpy' or 'polynomial'
    search_range=10.0,
    min_length=10,
    mpp=0.16,              # microns per pixel
    fps=10.0,              # frames per second
    visualize=True
)

print(f"Diffusion coefficient: {analysis['summary_stats']['diffusion_coefficient']:.4f}")
```

### 3. Compare Detection Methods
```python
from workflows import compare_methods_on_synthetic

comparison = compare_methods_on_synthetic(
    n_spots=5,
    n_frames=10,
    methods=['trackpy', 'gaussian', 'polynomial']
)

print(comparison)
```

### 4. Custom Workflow
```python
from data_loader import generate_synthetic_data
from feature_detectors import GaussianDetector
from trajectory_linker import link_and_filter
from motion_analyzer import analyze_particle_motion
from visualizer import plot_trajectories, plot_msd
import matplotlib.pyplot as plt

# Generate data
frames = generate_synthetic_data(n_spots=5, n_frames=20)

# Detect features
detector = GaussianDetector(min_distance=10, threshold=200)
features = detector.detect_sequence(frames)

# Link trajectories
trajectories = link_and_filter(features, search_range=10, min_length=10)

# Analyze motion
analysis = analyze_particle_motion(trajectories, mpp=0.16, fps=10.0)

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
plot_trajectories(trajectories, ax=ax1)
plot_msd(analysis['msd'], ax=ax2)
plt.show()
```

## Running Examples

```python
# Run individual examples
from examples import (
    example_basic_tracking_trackpy,
    example_gaussian_tracking,
    example_compare_detectors,
    example_msd_analysis
)

# Run specific example
example_gaussian_tracking()

# Run all examples
from examples import run_all_examples
run_all_examples()
```

## Detection Methods

### TrackpyDetector
Best for: Fast, general-purpose tracking
```python
from feature_detectors import TrackpyDetector

detector = TrackpyDetector(diameter=11, minmass=100)
features = detector.detect_sequence(frames)
```

### GaussianDetector
Best for: High-precision sub-pixel accuracy
```python
from feature_detectors import GaussianDetector

detector = GaussianDetector(
    min_distance=10,
    threshold=200,
    fit_radius=6
)
features = detector.detect_sequence(frames)
```

### PolynomialDetector
Best for: Alternative sub-pixel method
```python
from feature_detectors import PolynomialDetector

detector = PolynomialDetector(
    min_distance=10,
    threshold=200,
    fit_radius=5
)
features = detector.detect_sequence(frames)
```

## 3D Visualization

```python
from visualizer import plot_gaussian_fit_surface
from fitting_utils import extract_region
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Assuming you have features from GaussianDetector
first_feature = features.iloc[0]
region, x, y = extract_region(frames[0], first_feature['x'], first_feature['y'], radius=6)

params = {
    'amplitude': first_feature['amplitude'],
    'x0': first_feature['x'] - x,
    'y0': first_feature['y'] - y,
    'sigma_x': first_feature['sigma_x'],
    'sigma_y': first_feature['sigma_y'],
    'offset': first_feature['offset']
}

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plot_gaussian_fit_surface(region, params, ax=ax)
plt.show()
```

## Loading Real Data

### Video Files
```python
from workflows import load_and_track

trajectories = load_and_track(
    'path/to/video.avi',
    method='gaussian',
    search_range=10.0,
    min_length=10
)
```

### Image Sequences
```python
from data_loader import load_image_sequence
from workflows import track_particles

frames = load_image_sequence('path/to/images/*.png')
trajectories = track_particles(frames, method='gaussian')
```

## Saving Results

```python
from workflows import save_results

save_results(trajectories, analysis, output_dir='./results')
# Saves:
#   - trajectories.csv
#   - msd.csv
#   - summary_stats.json
```

## Module Reference

- **data_loader** - Load data (videos, sequences, synthetic)
- **fitting_utils** - Gaussian/polynomial fitting utilities
- **feature_detectors** - Detect particles (3 methods)
- **trajectory_linker** - Link features into trajectories
- **motion_analyzer** - Analyze motion (MSD, velocity, etc.)
- **visualizer** - Plot results (2D/3D)
- **accuracy** - Validate with ground truth
- **workflows** - High-level convenience functions
- **examples** - Complete demonstrations

For detailed documentation, see MODULE_SUMMARY.md
