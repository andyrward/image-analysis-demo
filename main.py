import numpy as np


def generate_spots_image(n, spot_width, signal, noise_magnitude, image_size=512):
    """
    Generate an image with NxN spots at random sub-pixel coordinates.
    
    Parameters:
    -----------
    n : int
        Number of spots in each dimension (creates NxN spots total)
    spot_width : float
        Width (sigma) of the Gaussian spots
    signal : float
        Intensity/amplitude of the spots
    noise_magnitude : float
        Standard deviation of the Gaussian noise added to the image
    image_size : int, optional
        Size of the output image in pixels (default: 512)
    
    Returns:
    --------
    numpy.ndarray
        2D array representing the generated image
    """
    # Create empty image
    image = np.zeros((image_size, image_size), dtype=np.float64)
    
    # Calculate grid spacing
    grid_spacing = image_size / (n + 1)
    
    # Create coordinate arrays for the entire image
    y_coords, x_coords = np.meshgrid(
        np.arange(image_size), 
        np.arange(image_size), 
        indexing='ij'
    )
    
    # Generate NxN spots
    for i in range(n):
        for j in range(n):
            # Calculate grid position (i=row, j=column)
            grid_y = (i + 1) * grid_spacing
            grid_x = (j + 1) * grid_spacing
            
            # Add random sub-pixel offset (between -0.5 and 0.5 pixels)
            random_offset_y = np.random.uniform(-0.5, 0.5)
            random_offset_x = np.random.uniform(-0.5, 0.5)
            
            # Final spot position
            spot_y = grid_y + random_offset_y
            spot_x = grid_x + random_offset_x
            
            # Generate 2D Gaussian spot
            gaussian = signal * np.exp(
                -((x_coords - spot_x)**2 + (y_coords - spot_y)**2) / (2 * spot_width**2)
            )
            
            # Add spot to image
            image += gaussian
    
    # Add Gaussian noise
    if noise_magnitude > 0:
        noise = np.random.normal(0, noise_magnitude, image.shape)
        image += noise
    
    # Ensure non-negative values
    image = np.maximum(image, 0)
    
    return image

