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

