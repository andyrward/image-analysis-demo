"""
Data loading utilities for particle tracking.

Supports loading videos, image sequences, and generating synthetic data.
"""

import numpy as np
import pims
from pathlib import Path
from typing import Union, Optional, Tuple
from spot_generator import generate_spots_image


class VideoLoader:
    """Load video files using PIMS."""
    
    def __init__(self, path: Union[str, Path]):
        """
        Initialize video loader.
        
        Parameters
        ----------
        path : str or Path
            Path to video file
        """
        self.path = Path(path)
        self.video = pims.open(str(self.path))
        
    def __len__(self):
        return len(self.video)
    
    def __getitem__(self, idx):
        return self.video[idx]
    
    def get_frame(self, idx: int) -> np.ndarray:
        """Get a single frame as numpy array."""
        return np.array(self.video[idx])
    
    @property
    def shape(self):
        """Get video shape (frames, height, width)."""
        return (len(self.video), self.video.frame_shape[0], self.video.frame_shape[1])


class ImageSequenceLoader:
    """Load sequence of images from a directory or pattern."""
    
    def __init__(self, path_pattern: str):
        """
        Initialize image sequence loader.
        
        Parameters
        ----------
        path_pattern : str
            Path pattern for images (e.g., 'frames/*.png' or 'frame_*.tif')
        """
        self.images = pims.ImageSequence(path_pattern)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx]
    
    def get_frame(self, idx: int) -> np.ndarray:
        """Get a single frame as numpy array."""
        return np.array(self.images[idx])
    
    @property
    def shape(self):
        """Get sequence shape (frames, height, width)."""
        return (len(self.images), self.images.frame_shape[0], self.images.frame_shape[1])


class SyntheticDataGenerator:
    """Generate synthetic particle tracking data using spot_generator."""
    
    def __init__(self, 
                 n_spots: int = 5,
                 spot_width: float = 2.0,
                 signal: float = 1000.0,
                 noise_magnitude: float = 50.0,
                 image_size: int = 512):
        """
        Initialize synthetic data generator.
        
        Parameters
        ----------
        n_spots : int
            Number of spots per dimension (creates n x n spots)
        spot_width : float
            Width (sigma) of Gaussian spots
        signal : float
            Intensity/amplitude of spots
        noise_magnitude : float
            Standard deviation of noise
        image_size : int
            Size of image in pixels
        """
        self.n_spots = n_spots
        self.spot_width = spot_width
        self.signal = signal
        self.noise_magnitude = noise_magnitude
        self.image_size = image_size
        
    def generate_frame(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Generate a single synthetic frame.
        
        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility
            
        Returns
        -------
        np.ndarray
            Generated image
        """
        if seed is not None:
            np.random.seed(seed)
        
        return generate_spots_image(
            n=self.n_spots,
            spot_width=self.spot_width,
            signal=self.signal,
            noise_magnitude=self.noise_magnitude,
            image_size=self.image_size
        )
    
    def generate_sequence(self, n_frames: int, 
                         seed: Optional[int] = None) -> list:
        """
        Generate a sequence of synthetic frames.
        
        Parameters
        ----------
        n_frames : int
            Number of frames to generate
        seed : int, optional
            Random seed for reproducibility
            
        Returns
        -------
        list
            List of generated images
        """
        frames = []
        for i in range(n_frames):
            frame_seed = seed + i if seed is not None else None
            frames.append(self.generate_frame(seed=frame_seed))
        return frames
    
    def generate_static_spots(self, n_frames: int) -> Tuple[list, np.ndarray]:
        """
        Generate frames with static spot positions (for tracking validation).
        
        Parameters
        ----------
        n_frames : int
            Number of frames to generate
            
        Returns
        -------
        frames : list
            List of generated images with same spot positions
        positions : np.ndarray
            Array of shape (n_spots*n_spots, 2) with (x, y) positions
        """
        # Generate spot positions once
        np.random.seed(42)
        grid_spacing = self.image_size / (self.n_spots + 1)
        positions = []
        
        for i in range(self.n_spots):
            for j in range(self.n_spots):
                grid_y = (i + 1) * grid_spacing
                grid_x = (j + 1) * grid_spacing
                
                random_offset_y = np.random.uniform(-0.5, 0.5)
                random_offset_x = np.random.uniform(-0.5, 0.5)
                
                spot_y = grid_y + random_offset_y
                spot_x = grid_x + random_offset_x
                
                positions.append([spot_x, spot_y])
        
        positions = np.array(positions)
        
        # Generate frames with same positions but different noise
        frames = []
        for frame_idx in range(n_frames):
            image = np.zeros((self.image_size, self.image_size), dtype=np.float64)
            y_coords, x_coords = np.meshgrid(
                np.arange(self.image_size), 
                np.arange(self.image_size), 
                indexing='ij'
            )
            
            for spot_x, spot_y in positions:
                gaussian = self.signal * np.exp(
                    -((x_coords - spot_x)**2 + (y_coords - spot_y)**2) / 
                    (2 * self.spot_width**2)
                )
                image += gaussian
            
            # Add different noise each frame
            if self.noise_magnitude > 0:
                noise = np.random.normal(0, self.noise_magnitude, image.shape)
                image += noise
            
            image = np.maximum(image, 0)
            frames.append(image)
        
        return frames, positions


def load_video(path: Union[str, Path]) -> VideoLoader:
    """
    Load a video file.
    
    Parameters
    ----------
    path : str or Path
        Path to video file
        
    Returns
    -------
    VideoLoader
        Video loader object
    """
    return VideoLoader(path)


def load_image_sequence(path_pattern: str) -> ImageSequenceLoader:
    """
    Load an image sequence.
    
    Parameters
    ----------
    path_pattern : str
        Path pattern for images
        
    Returns
    -------
    ImageSequenceLoader
        Image sequence loader object
    """
    return ImageSequenceLoader(path_pattern)


def generate_synthetic_data(n_spots: int = 5,
                           spot_width: float = 2.0,
                           signal: float = 1000.0,
                           noise_magnitude: float = 50.0,
                           image_size: int = 512,
                           n_frames: int = 10) -> list:
    """
    Generate synthetic particle tracking data.
    
    Parameters
    ----------
    n_spots : int
        Number of spots per dimension
    spot_width : float
        Width of Gaussian spots
    signal : float
        Spot intensity
    noise_magnitude : float
        Noise standard deviation
    image_size : int
        Image size in pixels
    n_frames : int
        Number of frames to generate
        
    Returns
    -------
    list
        List of generated images
    """
    generator = SyntheticDataGenerator(
        n_spots=n_spots,
        spot_width=spot_width,
        signal=signal,
        noise_magnitude=noise_magnitude,
        image_size=image_size
    )
    return generator.generate_sequence(n_frames, seed=42)
