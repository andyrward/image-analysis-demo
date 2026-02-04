# Image Analysis Demo

This repository contains Python code for generating and analyzing images with spots.

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
pip install numpy pillow
```

Or if using `uv`:

```bash
uv sync
```

## Usage

Run the demo:

```bash
python main.py
```

This will generate a sample image with 5x5 spots and save it as `spots_output.png`.

### Using the Function

```python
from main import generate_spots_image
import numpy as np
from PIL import Image

# Generate image with 5x5 spots
n = 5
spot_width = 3.0  # Gaussian sigma
signal = 100.0     # Spot intensity
noise_magnitude = 5.0  # Noise level

image_array = generate_spots_image(n, spot_width, signal, noise_magnitude)

# Save as image
if image_array.max() > image_array.min():
    image_normalized = (image_array - image_array.min()) / (image_array.max() - image_array.min()) * 255
else:
    image_normalized = np.full_like(image_array, 128)
image_normalized = image_normalized.astype(np.uint8)
img = Image.fromarray(image_normalized)
img.save("output.png")
```

## Requirements

- Python >= 3.9
- numpy >= 2.0.2
- pillow >= 11.3.0
