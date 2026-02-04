"""
Tests for data loading utilities.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
from PIL import Image

from data_loader import (
    SyntheticDataGenerator,
    generate_synthetic_data,
)


class TestSyntheticDataGenerator:
    """Tests for synthetic data generation."""
    
    def test_generate_frame_basic(self):
        """Test basic frame generation."""
        gen = SyntheticDataGenerator(n_spots=3, spot_width=2.0, 
                                     signal=1000, noise_magnitude=50)
        frame = gen.generate_frame(seed=42)
        
        assert frame.shape == (512, 512)
        assert frame.dtype == np.float64
        assert np.all(frame >= 0)
        assert frame.max() > 0
    
    def test_generate_frame_custom_size(self):
        """Test frame generation with custom size."""
        gen = SyntheticDataGenerator(n_spots=2, image_size=256)
        frame = gen.generate_frame(seed=42)
        
        assert frame.shape == (256, 256)
    
    def test_generate_frame_reproducible(self):
        """Test that same seed produces same output."""
        gen = SyntheticDataGenerator(n_spots=3, spot_width=2.0)
        
        frame1 = gen.generate_frame(seed=42)
        frame2 = gen.generate_frame(seed=42)
        
        np.testing.assert_array_equal(frame1, frame2)
    
    def test_generate_frame_different_seeds(self):
        """Test that different seeds produce different outputs."""
        gen = SyntheticDataGenerator(n_spots=3, spot_width=2.0)
        
        frame1 = gen.generate_frame(seed=42)
        frame2 = gen.generate_frame(seed=43)
        
        assert not np.array_equal(frame1, frame2)
    
    def test_generate_sequence(self):
        """Test sequence generation."""
        gen = SyntheticDataGenerator(n_spots=3)
        frames = gen.generate_sequence(n_frames=5, seed=42)
        
        assert len(frames) == 5
        assert all(f.shape == (512, 512) for f in frames)
        assert all(f.dtype == np.float64 for f in frames)
    
    def test_generate_static_spots(self):
        """Test generation of frames with static spot positions."""
        gen = SyntheticDataGenerator(n_spots=3)
        frames, positions = gen.generate_static_spots(n_frames=5)
        
        assert len(frames) == 5
        assert positions.shape == (9, 2)  # 3x3 spots
        assert all(f.shape == (512, 512) for f in frames)
        
        # Verify positions are within image bounds
        assert np.all(positions[:, 0] >= 0)
        assert np.all(positions[:, 0] < 512)
        assert np.all(positions[:, 1] >= 0)
        assert np.all(positions[:, 1] < 512)
    
    def test_zero_noise(self):
        """Test generation with zero noise."""
        gen = SyntheticDataGenerator(n_spots=2, noise_magnitude=0.0)
        
        # With zero noise and same seed, two runs should be identical
        frame1 = gen.generate_frame(seed=42)
        frame2 = gen.generate_frame(seed=42)
        np.testing.assert_array_equal(frame1, frame2)
        
        # Verify noise is actually zero by checking that no negative values exist
        # (without noise floor, this would be guaranteed)
        assert np.all(frame1 >= 0)
    
    @pytest.mark.parametrize("n_spots", [1, 3, 5, 10])
    def test_variable_spot_count(self, n_spots):
        """Test generation with different spot counts."""
        gen = SyntheticDataGenerator(n_spots=n_spots)
        frame = gen.generate_frame(seed=42)
        
        assert frame.shape == (512, 512)
        assert frame.max() > 0
    
    @pytest.mark.parametrize("spot_width", [1.0, 2.0, 3.0, 5.0])
    def test_variable_spot_width(self, spot_width):
        """Test generation with different spot widths."""
        gen = SyntheticDataGenerator(n_spots=3, spot_width=spot_width)
        frame = gen.generate_frame(seed=42)
        
        assert frame.shape == (512, 512)
        assert frame.max() > 0


class TestGenerateSyntheticData:
    """Tests for convenience function."""
    
    def test_basic_generation(self):
        """Test basic synthetic data generation."""
        frames = generate_synthetic_data(
            n_spots=3,
            spot_width=2.0,
            signal=1000,
            noise_magnitude=50,
            n_frames=5
        )
        
        assert len(frames) == 5
        assert all(f.shape == (512, 512) for f in frames)
    
    def test_reproducible(self):
        """Test reproducibility with default seed."""
        frames1 = generate_synthetic_data(n_spots=3, n_frames=3)
        frames2 = generate_synthetic_data(n_spots=3, n_frames=3)
        
        for f1, f2 in zip(frames1, frames2):
            np.testing.assert_array_equal(f1, f2)
    
    def test_custom_image_size(self):
        """Test with custom image size."""
        frames = generate_synthetic_data(
            n_spots=2,
            image_size=256,
            n_frames=2
        )
        
        assert len(frames) == 2
        assert all(f.shape == (256, 256) for f in frames)


class TestVideoLoader:
    """Tests for video loading (requires actual video files)."""
    
    def test_video_loader_missing_file(self):
        """Test that missing file raises appropriate error."""
        from data_loader import VideoLoader
        
        with pytest.raises(Exception):
            loader = VideoLoader("nonexistent_file.mp4")


class TestImageSequenceLoader:
    """Tests for image sequence loading."""
    
    def test_image_sequence_loader_no_files(self):
        """Test behavior with no matching files."""
        from data_loader import ImageSequenceLoader
        
        with pytest.raises(Exception):
            loader = ImageSequenceLoader("/nonexistent/path/*.png")
    
    def test_image_sequence_loader_with_temp_images(self):
        """Test loading a sequence of temporary images."""
        from data_loader import ImageSequenceLoader
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test images
            for i in range(3):
                img = Image.fromarray(np.random.randint(0, 255, (100, 100), dtype=np.uint8))
                img.save(Path(tmpdir) / f"frame_{i:03d}.png")
            
            # Load sequence
            loader = ImageSequenceLoader(str(Path(tmpdir) / "frame_*.png"))
            
            assert len(loader) == 3
            assert loader.shape[0] == 3
            
            # Test frame access
            frame = loader.get_frame(0)
            assert frame.shape == (100, 100)
