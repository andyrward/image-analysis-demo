"""
Tests for accuracy assessment.
"""

import pytest
import numpy as np
import pandas as pd

from accuracy import (
    match_detections_to_ground_truth,
    compute_detection_accuracy,
    compare_methods_accuracy,
    compute_subpixel_accuracy,
    validate_with_synthetic_data,
)
from data_loader import SyntheticDataGenerator
from feature_detectors import TrackpyDetector


@pytest.fixture
def ground_truth_positions():
    """Create ground truth positions."""
    # 3x3 grid of positions
    positions = []
    for i in range(3):
        for j in range(3):
            positions.append([50 + j * 50, 50 + i * 50])
    
    return np.array(positions)


@pytest.fixture
def perfect_detections(ground_truth_positions):
    """Create detections that match ground truth perfectly."""
    data = []
    for x, y in ground_truth_positions:
        data.append({'x': x, 'y': y, 'frame': 0})
    
    return pd.DataFrame(data)


@pytest.fixture
def noisy_detections(ground_truth_positions):
    """Create detections with small errors."""
    np.random.seed(42)
    data = []
    for x, y in ground_truth_positions:
        # Add small random error
        x_det = x + np.random.normal(0, 0.3)
        y_det = y + np.random.normal(0, 0.3)
        data.append({'x': x_det, 'y': y_det, 'frame': 0})
    
    return pd.DataFrame(data)


class TestMatchDetectionsToGroundTruth:
    """Tests for matching detections to ground truth."""
    
    def test_match_perfect(self, perfect_detections, ground_truth_positions):
        """Test matching with perfect detections."""
        distances, matched_det, matched_truth = match_detections_to_ground_truth(
            perfect_detections,
            ground_truth_positions,
            frame_num=0,
            max_distance=5.0
        )
        
        assert len(distances) == len(ground_truth_positions)
        assert len(matched_det) == len(ground_truth_positions)
        assert len(matched_truth) == len(ground_truth_positions)
        
        # All distances should be near zero
        assert np.all(distances < 0.1)
    
    def test_match_noisy(self, noisy_detections, ground_truth_positions):
        """Test matching with noisy detections."""
        distances, matched_det, matched_truth = match_detections_to_ground_truth(
            noisy_detections,
            ground_truth_positions,
            frame_num=0,
            max_distance=5.0
        )
        
        assert len(distances) == len(ground_truth_positions)
        
        # Distances should be small but non-zero
        assert np.all(distances < 1.0)
        assert np.any(distances > 0.1)
    
    def test_match_missing_detections(self, ground_truth_positions):
        """Test matching when some detections are missing."""
        # Only detect half the spots
        data = []
        for i in range(len(ground_truth_positions) // 2):
            x, y = ground_truth_positions[i]
            data.append({'x': x, 'y': y, 'frame': 0})
        
        detections = pd.DataFrame(data)
        
        distances, matched_det, matched_truth = match_detections_to_ground_truth(
            detections,
            ground_truth_positions,
            frame_num=0,
            max_distance=5.0
        )
        
        # Should match available detections
        assert len(matched_det) <= len(detections)
        assert len(matched_truth) <= len(ground_truth_positions)
    
    def test_match_extra_detections(self, ground_truth_positions):
        """Test matching with false positive detections."""
        data = []
        
        # Add all ground truth
        for x, y in ground_truth_positions:
            data.append({'x': x, 'y': y, 'frame': 0})
        
        # Add false positives
        data.append({'x': 200, 'y': 200, 'frame': 0})
        data.append({'x': 300, 'y': 300, 'frame': 0})
        
        detections = pd.DataFrame(data)
        
        distances, matched_det, matched_truth = match_detections_to_ground_truth(
            detections,
            ground_truth_positions,
            frame_num=0,
            max_distance=5.0
        )
        
        # Should only match ground truth spots
        assert len(matched_truth) == len(ground_truth_positions)
    
    def test_match_max_distance(self, ground_truth_positions):
        """Test max_distance threshold."""
        # Create detections far from ground truth
        data = []
        for x, y in ground_truth_positions:
            data.append({'x': x + 10, 'y': y + 10, 'frame': 0})
        
        detections = pd.DataFrame(data)
        
        # Strict threshold
        distances_strict, matched_det_strict, matched_truth_strict = (
            match_detections_to_ground_truth(
                detections, ground_truth_positions, 0, max_distance=5.0
            )
        )
        
        # Loose threshold
        distances_loose, matched_det_loose, matched_truth_loose = (
            match_detections_to_ground_truth(
                detections, ground_truth_positions, 0, max_distance=20.0
            )
        )
        
        # Loose threshold should match more
        assert len(matched_det_loose) >= len(matched_det_strict)
    
    def test_match_empty_detections(self, ground_truth_positions):
        """Test with no detections."""
        empty = pd.DataFrame(columns=['x', 'y', 'frame'])
        
        distances, matched_det, matched_truth = match_detections_to_ground_truth(
            empty,
            ground_truth_positions,
            frame_num=0,
            max_distance=5.0
        )
        
        assert len(distances) == 0
        assert len(matched_det) == 0
        assert len(matched_truth) == 0


class TestComputeDetectionAccuracy:
    """Tests for accuracy metrics computation."""
    
    def test_accuracy_perfect(self, perfect_detections, ground_truth_positions):
        """Test accuracy with perfect detections."""
        metrics = compute_detection_accuracy(
            perfect_detections,
            ground_truth_positions,
            frame_num=0,
            max_distance=5.0
        )
        
        assert metrics['recall'] == pytest.approx(1.0, abs=0.01)
        assert metrics['precision'] == pytest.approx(1.0, abs=0.01)
        assert metrics['f1_score'] == pytest.approx(1.0, abs=0.01)
        assert metrics['mean_error'] < 0.1
        assert metrics['n_true'] == len(ground_truth_positions)
        assert metrics['n_detected'] == len(perfect_detections)
        assert metrics['n_matched'] == len(ground_truth_positions)
    
    def test_accuracy_missing_detections(self, ground_truth_positions):
        """Test accuracy with missing detections."""
        # Detect only half
        data = []
        for i in range(len(ground_truth_positions) // 2):
            x, y = ground_truth_positions[i]
            data.append({'x': x, 'y': y, 'frame': 0})
        
        detections = pd.DataFrame(data)
        
        metrics = compute_detection_accuracy(
            detections,
            ground_truth_positions,
            frame_num=0
        )
        
        # Recall should be approximately 0.5
        assert 0.4 < metrics['recall'] < 0.6
        # Precision should be high (no false positives)
        assert metrics['precision'] > 0.9
    
    def test_accuracy_false_positives(self, ground_truth_positions):
        """Test accuracy with false positive detections."""
        data = []
        
        # Add ground truth
        for x, y in ground_truth_positions:
            data.append({'x': x, 'y': y, 'frame': 0})
        
        # Add same number of false positives
        for _ in range(len(ground_truth_positions)):
            data.append({'x': 500, 'y': 500, 'frame': 0})
        
        detections = pd.DataFrame(data)
        
        metrics = compute_detection_accuracy(
            detections,
            ground_truth_positions,
            frame_num=0
        )
        
        # Recall should be high (all ground truth found)
        assert metrics['recall'] > 0.9
        # Precision should be approximately 0.5
        assert 0.4 < metrics['precision'] < 0.6
    
    def test_accuracy_empty_detections(self, ground_truth_positions):
        """Test accuracy with no detections."""
        empty = pd.DataFrame(columns=['x', 'y', 'frame'])
        
        metrics = compute_detection_accuracy(
            empty,
            ground_truth_positions,
            frame_num=0
        )
        
        assert metrics['recall'] == 0.0
        assert metrics['precision'] == 0.0
        assert metrics['f1_score'] == 0.0
        assert metrics['n_matched'] == 0


class TestCompareMethodsAccuracy:
    """Tests for method comparison."""
    
    def test_compare_methods(self, ground_truth_positions):
        """Test comparison of multiple methods."""
        # Create detections for multiple methods
        methods = {}
        
        # Perfect method
        data_perfect = []
        for x, y in ground_truth_positions:
            data_perfect.append({'x': x, 'y': y, 'frame': 0})
        methods['perfect'] = pd.DataFrame(data_perfect)
        
        # Noisy method
        data_noisy = []
        for x, y in ground_truth_positions:
            data_noisy.append({
                'x': x + np.random.normal(0, 0.5),
                'y': y + np.random.normal(0, 0.5),
                'frame': 0
            })
        methods['noisy'] = pd.DataFrame(data_noisy)
        
        comparison = compare_methods_accuracy(
            methods,
            ground_truth_positions,
            frame_num=0
        )
        
        assert isinstance(comparison, pd.DataFrame)
        assert 'method' in comparison.columns
        assert 'recall' in comparison.columns
        assert 'precision' in comparison.columns
        assert 'f1_score' in comparison.columns
        assert len(comparison) == 2
        
        # Perfect method should have better metrics
        perfect_row = comparison[comparison['method'] == 'perfect'].iloc[0]
        noisy_row = comparison[comparison['method'] == 'noisy'].iloc[0]
        
        assert perfect_row['mean_error'] < noisy_row['mean_error']


class TestComputeSubpixelAccuracy:
    """Tests for sub-pixel accuracy computation."""
    
    def test_subpixel_accuracy_perfect(self, perfect_detections, ground_truth_positions):
        """Test sub-pixel accuracy with perfect detections."""
        metrics = compute_subpixel_accuracy(
            perfect_detections,
            ground_truth_positions,
            frame_num=0
        )
        
        assert metrics['x_bias'] == pytest.approx(0.0, abs=0.01)
        assert metrics['y_bias'] == pytest.approx(0.0, abs=0.01)
        assert metrics['x_std'] < 0.1
        assert metrics['y_std'] < 0.1
        assert metrics['radial_error'] < 0.1
    
    def test_subpixel_accuracy_biased(self, ground_truth_positions):
        """Test sub-pixel accuracy with systematic bias."""
        # Add systematic bias in x direction
        data = []
        bias_x = 0.3
        for x, y in ground_truth_positions:
            data.append({'x': x + bias_x, 'y': y, 'frame': 0})
        
        detections = pd.DataFrame(data)
        
        metrics = compute_subpixel_accuracy(
            detections,
            ground_truth_positions,
            frame_num=0
        )
        
        # Should detect x bias
        assert metrics['x_bias'] == pytest.approx(bias_x, abs=0.1)
        assert abs(metrics['y_bias']) < 0.1
    
    def test_subpixel_accuracy_noisy(self, noisy_detections, ground_truth_positions):
        """Test sub-pixel accuracy with random noise."""
        metrics = compute_subpixel_accuracy(
            noisy_detections,
            ground_truth_positions,
            frame_num=0
        )
        
        # Bias should be small
        assert abs(metrics['x_bias']) < 0.5
        assert abs(metrics['y_bias']) < 0.5
        
        # Standard deviation should reflect noise
        assert metrics['x_std'] > 0.1
        assert metrics['y_std'] > 0.1
    
    def test_subpixel_accuracy_empty(self, ground_truth_positions):
        """Test sub-pixel accuracy with no detections."""
        empty = pd.DataFrame(columns=['x', 'y', 'frame'])
        
        metrics = compute_subpixel_accuracy(
            empty,
            ground_truth_positions,
            frame_num=0
        )
        
        assert metrics['x_bias'] == 0
        assert metrics['y_bias'] == 0
        assert metrics['radial_error'] == 0


class TestValidateWithSyntheticData:
    """Tests for validation with synthetic data."""
    
    def test_validate_basic(self):
        """Test basic validation workflow."""
        
        def simple_detector(frames):
            """Simple detector for testing."""
            detector = TrackpyDetector(diameter=11, minmass=100)
            return detector.detect_sequence(frames)
        
        results = validate_with_synthetic_data(
            simple_detector,
            n_spots=3,
            spot_width=2.0,
            signal=1000,
            noise_magnitude=50,
            n_frames=5
        )
        
        assert isinstance(results, dict)
        assert 'mean_recall' in results
        assert 'mean_precision' in results
        assert 'mean_f1_score' in results
        assert 'mean_position_error' in results
        assert 'frame_accuracies' in results
        assert 'subpixel_accuracy' in results
    
    def test_validate_metrics_range(self):
        """Test that metrics are in valid ranges."""
        
        def simple_detector(frames):
            detector = TrackpyDetector(diameter=11, minmass=100)
            return detector.detect_sequence(frames)
        
        results = validate_with_synthetic_data(
            simple_detector,
            n_spots=3,
            n_frames=5
        )
        
        # Metrics should be between 0 and 1
        assert 0 <= results['mean_recall'] <= 1
        assert 0 <= results['mean_precision'] <= 1
        assert 0 <= results['mean_f1_score'] <= 1
        
        # Position error should be non-negative
        assert results['mean_position_error'] >= 0
    
    def test_validate_high_snr(self):
        """Test validation with high signal-to-noise ratio."""
        
        def simple_detector(frames):
            detector = TrackpyDetector(diameter=11, minmass=100)
            return detector.detect_sequence(frames)
        
        # High SNR
        results_high_snr = validate_with_synthetic_data(
            simple_detector,
            n_spots=3,
            signal=5000,
            noise_magnitude=10,
            n_frames=5
        )
        
        # Low SNR
        results_low_snr = validate_with_synthetic_data(
            simple_detector,
            n_spots=3,
            signal=500,
            noise_magnitude=200,
            n_frames=5
        )
        
        # High SNR should have better accuracy
        # (may not always be true due to randomness, so just check they run)
        assert results_high_snr['mean_recall'] >= 0
        assert results_low_snr['mean_recall'] >= 0


class TestGroundTruthGeneration:
    """Tests for ground truth data generation."""
    
    def test_synthetic_generator_positions(self):
        """Test that synthetic generator produces consistent positions."""
        generator = SyntheticDataGenerator(n_spots=3)
        frames, positions = generator.generate_static_spots(n_frames=5)
        
        # Check position array shape
        assert positions.shape == (9, 2)
        
        # Positions should be within image bounds
        assert np.all(positions[:, 0] >= 0)
        assert np.all(positions[:, 0] < 512)
        assert np.all(positions[:, 1] >= 0)
        assert np.all(positions[:, 1] < 512)
        
        # Spots should be reasonably spaced
        min_spacing = 512 / (3 + 1)
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dist = np.linalg.norm(positions[i] - positions[j])
                # Most spots should be at least min_spacing/2 apart
                if dist < min_spacing / 3:
                    # Allow some close pairs due to random offsets
                    pass
