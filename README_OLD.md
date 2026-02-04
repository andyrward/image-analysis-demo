# Image Analysis Demo

This repository contains Python code for generating and analyzing synthetic images with spots.

## Quick Start

### Using the Jupyter Notebook (Recommended)

The primary way to explore and use this project is through the **interactive Jupyter notebook**:

```bash
# Install dependencies
pip install numpy pillow jupyter matplotlib

# Start Jupyter
jupyter notebook image_analysis_demo.ipynb
```

The notebook (`image_analysis_demo.ipynb`) provides:
- Interactive demonstrations with different parameters
- Visual comparisons of noise levels and spot widths
- Example analysis and image statistics
- Ready-to-run code cells for experimentation

### Using the Python Module

The core functionality is in `spot_generator.py`, which exports the `generate_spots_image()` function:

```python
from spot_generator import generate_spots_image
import numpy as np

# Generate a 5x5 grid of spots
image = generate_spots_image(
    n=5,                    # 5x5 grid
    spot_width=3.0,         # Gaussian sigma in pixels
    signal=100.0,           # Peak intensity
    noise_magnitude=5.0,    # Noise level
    image_size=512          # Image size in pixels
)
```

## Features

### Spot Image Generation

The `generate_spots_image()` function creates synthetic images with NxN spots placed on a grid with random sub-pixel coordinates. This is useful for testing image analysis algorithms.

**Parameters:**
- `n` (int): Number of spots in each dimension (creates NxN spots total)
- `spot_width` (float): Width (sigma) of the Gaussian spots
- `signal` (float): Intensity/amplitude of the spots
- `noise_magnitude` (float): Standard deviation of the Gaussian noise added to the image
- `image_size` (int, optional): Size of the output image in pixels (default: 512)

**Key Features:**
- Spots are placed on a regular grid pattern
- Each spot has a random sub-pixel offset (between -0.5 and 0.5 pixels) for realistic positioning
- Spots are modeled as 2D Gaussians
- Gaussian noise is added to simulate realistic imaging conditions

## Installation

Install the required dependencies:

```bash
pip install numpy pillow jupyter matplotlib
```

Or if using `uv`:

```bash
uv sync
```

## Use Cases

These synthetic images are ideal for:
- Developing and testing spot detection algorithms
- Validating sub-pixel localization methods
- Benchmarking image analysis pipelines
- Training machine learning models
- Testing noise reduction techniques

## Requirements

- Python >= 3.9
- numpy >= 2.0.2
- pillow >= 11.3.0
- jupyter >= 1.0.0
- matplotlib >= 3.8.0

A Python library for tracking spots on 2D images using multiple tracking methods.

## Features

This library provides a generic image tracking function that supports multiple methods:

- **Gaussian Fit**: 2D Gaussian curve fitting for high-precision tracking
- **Parabola Fit**: 2D Parabola curve fitting for spot localization
- **PyTrack**: Weighted centroid tracking method

## Installation

```bash
pip install numpy scipy pillow plotly pandas jupyter
```

## Usage

### Basic Example

```python
from image_tracker import create_demo_image, track_spot

# Create a demo 100x100 image with a spot at the center
image = create_demo_image(size=100, spot_center=(50.0, 50.0))

# Track the spot using Gaussian fitting (most accurate)
result = track_spot(image, method="gaussian")
print(f"Position: x={result['x']:.2f}, y={result['y']:.2f}")
```

### Running the Demo

```bash
# Quick demo
python demo_quick.py

# Comprehensive testing with interactive Plotly visualizations
python demo_single_track.py

# Interactive Jupyter notebook with plots
jupyter notebook demo_tracking.ipynb
```

The demo scripts generate:
- **Quantitative metrics** comparing all tracking methods
- **Interactive Plotly visualizations** (hover, zoom, pan)
- **HTML output** with interactive plots
- **PNG exports** of the visualizations

## API Reference

### `track_spot(image, method="gaussian", initial_guess=None, **kwargs)`

Generic spot tracking function.

**Parameters:**
- `image`: 2D numpy array containing the image
- `method`: Tracking method ("gaussian", "parabola", or "pytrack")
- `initial_guess`: Optional (x, y) initial position
- `**kwargs`: Method-specific parameters (e.g., `window_size` for fitting methods)

**Returns:** Dictionary with tracking results including `x`, `y` position and method-specific details.

### `create_demo_image(size=100, spot_center=None, spot_amplitude=100.0, spot_sigma=2.0)`

Creates a test image with a Gaussian spot.

**Parameters:**
- `size`: Image size (creates square image)
- `spot_center`: (x, y) position of the spot (defaults to center)
- `spot_amplitude`: Peak intensity of the spot
- `spot_sigma`: Width of the Gaussian spot

## Performance

Test results on a 100x100 image with a centered spot at (50, 50):

| Method   | Position Error | R² Score |
|----------|---------------|----------|
| Gaussian | 0.0103 px     | 0.996    |
| Parabola | 0.4091 px     | 0.276    |
| PyTrack  | 0.4457 px     | N/A      |

**Gaussian fitting** provides the highest accuracy for symmetric Gaussian-like spots.
