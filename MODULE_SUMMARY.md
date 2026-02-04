# Particle Tracking Library - Module Summary

This document summarizes the 9 modules created for the particle tracking library.

## Module Overview

### 1. **data_loader.py**
Data loading utilities for particle tracking.

**Key Classes:**
- `VideoLoader` - Load video files using PIMS
- `ImageSequenceLoader` - Load image sequences from directories
- `SyntheticDataGenerator` - Generate synthetic test data with known ground truth

**Key Functions:**
- `load_video()` - Load video files
- `load_image_sequence()` - Load image sequences
- `generate_synthetic_data()` - Quick synthetic data generation

**Features:**
- Support for video files via PIMS
- Image sequence loading with patterns
- Synthetic data generation using spot_generator.py
- Ground truth position tracking for validation

---

### 2. **fitting_utils.py**
Shared utilities for Gaussian and Polynomial fitting.

**Key Functions:**
- `find_local_maxima()` - Detect local maxima in images
- `extract_region()` - Extract square regions around points
- `fit_gaussian_2d()` - Fit 2D Gaussian to image regions
- `fit_polynomial_2d()` - Fit 2D polynomial (quadratic) to image regions
- `compute_fit_quality()` - Calculate R², RMSE, and residuals

**Features:**
- Sub-pixel accuracy through curve fitting
- Robust fitting with error handling
- Quality metrics for fit assessment

---

### 3. **feature_detectors.py**
Multiple detection methods for particle tracking.

**Key Classes:**
- `TrackpyDetector` - Uses trackpy's built-in detection
- `GaussianDetector` - 2D Gaussian fitting for sub-pixel accuracy
- `PolynomialDetector` - 2D polynomial fitting for peak detection

**Key Function:**
- `detect_features()` - Convenience function for any method

**Features:**
- All detectors return pandas DataFrames with ['x', 'y', 'frame'] columns
- Trackpy compatibility guaranteed
- Single frame and sequence detection
- Method-specific parameters for fine-tuning

---

### 4. **trajectory_linker.py**
Thin wrappers around trackpy linking functions.

**Key Functions:**
- `link_trajectories()` - Link features across frames
- `filter_trajectories()` - Filter by length or displacement
- `compute_drift()` - Calculate ensemble drift
- `subtract_drift()` - Remove drift from trajectories
- `link_and_filter()` - Combined convenience function

**Features:**
- Minimal reimplementation - wraps trackpy
- Adaptive search support
- Memory parameter for disappearing particles
- Drift correction capabilities

---

### 5. **motion_analyzer.py**
Thin wrappers around trackpy motion analysis functions.

**Key Functions:**
- `compute_msd()` - Mean squared displacement
- `compute_individual_msd()` - Per-trajectory MSD
- `compute_velocity()` - Velocity calculation
- `compute_diffusion_coefficient()` - Diffusion from MSD
- `analyze_particle_motion()` - Comprehensive analysis
- `compute_trajectory_statistics()` - Per-trajectory statistics

**Features:**
- Automatic handling of trackpy Series/DataFrame outputs
- Velocity computation with proper temporal scaling
- Summary statistics generation
- Path length and net displacement metrics

---

### 6. **visualizer.py**
Comprehensive visualization tools.

**Key Functions:**
- `plot_trajectories()` - 2D trajectory plots
- `annotate_frame()` - Mark detected features on images
- `plot_msd()` - MSD plots with optional fitting
- `plot_gaussian_fit_surface()` - **3D surface plot** for Gaussian fits
- `plot_polynomial_fit_surface()` - **3D surface plot** for polynomial fits
- `compare_detection_methods()` - Side-by-side method comparison
- `plot_trajectory_overlay()` - Trajectories with trail history

**Features:**
- 3D visualization using matplotlib's Axes3D
- Surface plots for fit validation
- Colormap support for trajectories
- Annotated frames with particle IDs
- MSD plotting with diffusion coefficient estimation

---

### 7. **accuracy.py**
Ground truth comparison and accuracy assessment.

**Key Functions:**
- `match_detections_to_ground_truth()` - Match detected to true positions
- `compute_detection_accuracy()` - Recall, precision, F1 score
- `compare_methods_accuracy()` - Compare multiple detectors
- `compute_subpixel_accuracy()` - Sub-pixel error statistics
- `validate_with_synthetic_data()` - Full validation pipeline

**Features:**
- Uses spot_generator for ground truth
- Position error metrics (mean, std, bias)
- Detection quality metrics
- Method comparison tables

---

### 8. **workflows.py**
High-level convenience API for common workflows.

**Key Functions:**
- `track_particles()` - Complete detect + link + filter workflow
- `analyze_motion()` - Motion analysis with MSD
- `complete_pipeline()` - Full tracking and analysis
- `quick_viz_*()` - Quick visualization functions
- `demo_with_synthetic_data()` - Demo workflow with synthetic data
- `compare_methods_on_synthetic()` - Method comparison with ground truth
- `load_and_track()` - Load file and track in one call
- `save_results()` - Save trajectories and analysis to files

**Features:**
- One-line tracking solutions
- Automatic visualization options
- Built-in method comparison
- Result saving to CSV/JSON

---

### 9. **examples.py**
Complete workflow demonstrations.

**Example Functions:**
- `example_basic_tracking_trackpy()` - Basic trackpy workflow
- `example_gaussian_tracking()` - High-precision Gaussian tracking with 3D viz
- `example_polynomial_tracking()` - Polynomial fitting workflow
- `example_compare_detectors()` - Compare all three methods
- `example_msd_analysis()` - Full motion analysis with MSD
- `example_synthetic_data_demo()` - High-level API demonstration
- `run_all_examples()` - Execute all examples

**Features:**
- Self-contained examples
- Progressive complexity
- Visualization for each example
- Comparison and validation examples

---

## Key Design Decisions

1. **DataFrame Consistency**: All detectors return pandas DataFrames with ['x', 'y', 'frame'] columns for trackpy compatibility

2. **Thin Wrappers**: trajectory_linker.py and motion_analyzer.py are thin wrappers around trackpy functions rather than reimplementations

3. **3D Visualization**: Uses matplotlib's Axes3D from mpl_toolkits.mplot3d for surface plots

4. **Ground Truth Integration**: Uses existing spot_generator.py (never modified) for validation

5. **Modular Design**: Each module has a clear, single responsibility

6. **Progressive API**: From low-level (detectors) to high-level (workflows) to examples

## Usage Example

```python
# Simple workflow
from workflows import complete_pipeline
from data_loader import generate_synthetic_data

frames = generate_synthetic_data(n_spots=5, n_frames=20)
trajectories, analysis = complete_pipeline(
    frames,
    method='gaussian',
    search_range=10.0,
    min_length=10,
    mpp=0.16,
    fps=10.0,
    visualize=True
)

# Method comparison
from workflows import compare_methods_on_synthetic

comparison = compare_methods_on_synthetic(
    n_spots=5,
    n_frames=10,
    methods=['trackpy', 'gaussian', 'polynomial']
)
```

## Testing

All modules have been tested and validated:
- ✓ Data loading and synthetic generation
- ✓ Fitting utilities (Gaussian and polynomial)
- ✓ All three detection methods
- ✓ Trajectory linking and filtering
- ✓ Motion analysis and MSD computation
- ✓ 2D and 3D visualizations
- ✓ Accuracy assessment with ground truth
- ✓ Complete workflows
- ✓ Example demonstrations

Test outputs saved to /tmp/:
- test_trajectories.png
- test_annotated.png
- test_msd.png
- test_gaussian_3d.png
