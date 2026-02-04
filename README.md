# Image Analysis Demo

A Python library for tracking spots on 2D images using multiple tracking methods.

## Features

This library provides a generic image tracking function that supports multiple methods:

- **Gaussian Fit**: 2D Gaussian curve fitting for high-precision tracking
- **Parabola Fit**: 2D Parabola curve fitting for spot localization
- **PyTrack**: Weighted centroid tracking method

## Installation

```bash
pip install numpy scipy pillow
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
python main.py

# Comprehensive testing
python test_tracking.py
```

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
