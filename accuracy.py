"""
Accuracy assessment utilities using ground truth data.

Compares detected positions with ground truth from synthetic data.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
from scipy.spatial.distance import cdist
from data_loader import SyntheticDataGenerator


def match_detections_to_ground_truth(detections: pd.DataFrame,
                                    ground_truth: np.ndarray,
                                    frame_num: int,
                                    max_distance: float = 5.0) -> Tuple[np.ndarray, List[int], List[int]]:
    """
    Match detected features to ground truth positions.
    
    Parameters
    ----------
    detections : pd.DataFrame
        Detected features with columns ['x', 'y', 'frame']
    ground_truth : np.ndarray
        Array of shape (n_spots, 2) with ground truth (x, y) positions
    frame_num : int
        Frame number to analyze
    max_distance : float
        Maximum distance for a match to be considered valid
        
    Returns
    -------
    distances : np.ndarray
        Distances between matched pairs
    matched_detections : list
        Indices of matched detections
    matched_truth : list
        Indices of matched ground truth spots
    """
    # Get detections for this frame
    frame_detections = detections[detections['frame'] == frame_num]
    
    if len(frame_detections) == 0:
        return np.array([]), [], []
    
    # Extract positions
    detected_pos = frame_detections[['x', 'y']].values
    
    # Compute pairwise distances
    distances_matrix = cdist(detected_pos, ground_truth)
    
    # Greedy matching: assign each ground truth to nearest detection
    matched_detections = []
    matched_truth = []
    distances = []
    
    used_detections = set()
    
    for truth_idx in range(len(ground_truth)):
        # Find nearest detection
        dists_to_truth = distances_matrix[:, truth_idx]
        min_idx = np.argmin(dists_to_truth)
        min_dist = dists_to_truth[min_idx]
        
        # Accept if within threshold and not already used
        if min_dist <= max_distance and min_idx not in used_detections:
            matched_detections.append(min_idx)
            matched_truth.append(truth_idx)
            distances.append(min_dist)
            used_detections.add(min_idx)
    
    return np.array(distances), matched_detections, matched_truth


def compute_detection_accuracy(detections: pd.DataFrame,
                              ground_truth: np.ndarray,
                              frame_num: int = 0,
                              max_distance: float = 5.0) -> Dict:
    """
    Compute detection accuracy metrics.
    
    Parameters
    ----------
    detections : pd.DataFrame
        Detected features
    ground_truth : np.ndarray
        Ground truth positions
    frame_num : int
        Frame number to analyze
    max_distance : float
        Maximum distance for match
        
    Returns
    -------
    dict
        Metrics including:
        - 'n_true': Number of ground truth spots
        - 'n_detected': Number of detections
        - 'n_matched': Number of correct matches
        - 'recall': Fraction of ground truth spots found
        - 'precision': Fraction of detections that are correct
        - 'f1_score': Harmonic mean of precision and recall
        - 'mean_error': Mean position error for matched pairs
        - 'std_error': Standard deviation of position errors
    """
    distances, matched_det, matched_truth = match_detections_to_ground_truth(
        detections, ground_truth, frame_num, max_distance
    )
    
    n_true = len(ground_truth)
    n_detected = len(detections[detections['frame'] == frame_num])
    n_matched = len(matched_det)
    
    recall = n_matched / n_true if n_true > 0 else 0
    precision = n_matched / n_detected if n_detected > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    mean_error = np.mean(distances) if len(distances) > 0 else 0
    std_error = np.std(distances) if len(distances) > 0 else 0
    
    return {
        'n_true': n_true,
        'n_detected': n_detected,
        'n_matched': n_matched,
        'recall': recall,
        'precision': precision,
        'f1_score': f1_score,
        'mean_error': mean_error,
        'std_error': std_error
    }


def compare_methods_accuracy(detections_dict: Dict[str, pd.DataFrame],
                            ground_truth: np.ndarray,
                            frame_num: int = 0,
                            max_distance: float = 5.0) -> pd.DataFrame:
    """
    Compare accuracy of multiple detection methods.
    
    Parameters
    ----------
    detections_dict : dict
        Dictionary of {method_name: detections_df}
    ground_truth : np.ndarray
        Ground truth positions
    frame_num : int
        Frame number to analyze
    max_distance : float
        Maximum distance for match
        
    Returns
    -------
    pd.DataFrame
        Comparison table with one row per method
    """
    results = []
    
    for method_name, detections in detections_dict.items():
        metrics = compute_detection_accuracy(
            detections, ground_truth, frame_num, max_distance
        )
        metrics['method'] = method_name
        results.append(metrics)
    
    df = pd.DataFrame(results)
    df = df[['method', 'n_true', 'n_detected', 'n_matched', 
             'recall', 'precision', 'f1_score', 'mean_error', 'std_error']]
    
    return df


def compute_subpixel_accuracy(detections: pd.DataFrame,
                             ground_truth: np.ndarray,
                             frame_num: int = 0,
                             max_distance: float = 5.0) -> Dict:
    """
    Compute sub-pixel accuracy statistics.
    
    Parameters
    ----------
    detections : pd.DataFrame
        Detected features with sub-pixel positions
    ground_truth : np.ndarray
        Ground truth positions
    frame_num : int
        Frame number to analyze
    max_distance : float
        Maximum distance for match
        
    Returns
    -------
    dict
        Sub-pixel accuracy metrics including:
        - 'x_bias': Mean error in x direction
        - 'y_bias': Mean error in y direction
        - 'x_std': Standard deviation of x errors
        - 'y_std': Standard deviation of y errors
        - 'radial_error': Mean radial error
        - 'errors_x': List of x errors
        - 'errors_y': List of y errors
    """
    distances, matched_det, matched_truth = match_detections_to_ground_truth(
        detections, ground_truth, frame_num, max_distance
    )
    
    if len(matched_det) == 0:
        return {
            'x_bias': 0, 'y_bias': 0,
            'x_std': 0, 'y_std': 0,
            'radial_error': 0,
            'errors_x': [], 'errors_y': []
        }
    
    # Get matched positions
    frame_detections = detections[detections['frame'] == frame_num]
    detected_pos = frame_detections[['x', 'y']].values
    
    detected_matched = detected_pos[matched_det]
    truth_matched = ground_truth[matched_truth]
    
    # Compute errors
    errors = detected_matched - truth_matched
    errors_x = errors[:, 0]
    errors_y = errors[:, 1]
    
    return {
        'x_bias': np.mean(errors_x),
        'y_bias': np.mean(errors_y),
        'x_std': np.std(errors_x),
        'y_std': np.std(errors_y),
        'radial_error': np.mean(distances),
        'errors_x': errors_x.tolist(),
        'errors_y': errors_y.tolist()
    }


def validate_with_synthetic_data(detector_function,
                                n_spots: int = 5,
                                spot_width: float = 2.0,
                                signal: float = 1000.0,
                                noise_magnitude: float = 50.0,
                                n_frames: int = 10,
                                **detector_kwargs) -> Dict:
    """
    Validate a detector using synthetic data with known ground truth.
    
    Parameters
    ----------
    detector_function : callable
        Function that takes frames and returns detections DataFrame
    n_spots : int
        Number of spots per dimension
    spot_width : float
        Width of Gaussian spots
    signal : float
        Spot intensity
    noise_magnitude : float
        Noise level
    n_frames : int
        Number of frames to test
    **detector_kwargs
        Additional arguments for detector
        
    Returns
    -------
    dict
        Validation results including accuracy metrics
    """
    # Generate synthetic data with known positions
    generator = SyntheticDataGenerator(
        n_spots=n_spots,
        spot_width=spot_width,
        signal=signal,
        noise_magnitude=noise_magnitude
    )
    
    frames, ground_truth = generator.generate_static_spots(n_frames)
    
    # Run detection
    detections = detector_function(frames, **detector_kwargs)
    
    # Compute accuracy for each frame
    frame_accuracies = []
    for frame_num in range(n_frames):
        accuracy = compute_detection_accuracy(
            detections, ground_truth, frame_num
        )
        accuracy['frame'] = frame_num
        frame_accuracies.append(accuracy)
    
    # Compute overall statistics
    df_accuracies = pd.DataFrame(frame_accuracies)
    
    overall = {
        'mean_recall': df_accuracies['recall'].mean(),
        'mean_precision': df_accuracies['precision'].mean(),
        'mean_f1_score': df_accuracies['f1_score'].mean(),
        'mean_position_error': df_accuracies['mean_error'].mean(),
        'frame_accuracies': df_accuracies
    }
    
    # Compute sub-pixel accuracy for first frame
    subpixel = compute_subpixel_accuracy(detections, ground_truth, frame_num=0)
    overall['subpixel_accuracy'] = subpixel
    
    return overall
