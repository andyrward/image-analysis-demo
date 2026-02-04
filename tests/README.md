# Test Suite for Image Analysis Demo

This directory contains comprehensive tests for all modules in the image-analysis-demo project.

## Test Files

1. **test_data_loader.py** - Tests for video/image loading and synthetic data generation
   - SyntheticDataGenerator functionality
   - VideoLoader and ImageSequenceLoader
   - Frame generation and reproducibility

2. **test_feature_detectors.py** - Tests for all detection methods
   - TrackpyDetector
   - GaussianDetector  
   - PolynomialDetector
   - DataFrame format validation
   - Trackpy integration compatibility

3. **test_fitting_utils.py** - Tests for Gaussian and polynomial fitting
   - Local maxima detection
   - Region extraction
   - 2D Gaussian fitting
   - 2D polynomial fitting
   - Fit quality metrics

4. **test_trajectory_linker.py** - Tests for trajectory linking
   - Feature linking with trackpy
   - Trajectory filtering (by length, displacement)
   - Drift computation and subtraction
   - Synthetic trajectory generation

5. **test_motion_analyzer.py** - Tests for motion analysis
   - MSD (Mean Squared Displacement) computation
   - Individual MSD calculation
   - Velocity computation
   - Diffusion coefficient calculation
   - Trajectory statistics

6. **test_accuracy.py** - Tests for ground truth comparison
   - Detection-to-ground-truth matching
   - Accuracy metrics (recall, precision, F1)
   - Sub-pixel accuracy assessment
   - Method comparison
   - Validation with synthetic data

7. **test_workflows.py** - Integration tests for high-level workflows
   - Complete tracking pipeline
   - Motion analysis workflows
   - Demo functions
   - Method comparison on synthetic data

8. **test_visualizations.py** - Smoke tests for visualization functions
   - Trajectory plotting
   - Frame annotation
   - MSD plotting
   - 3D surface plots for fits
   - Method comparison visualization

## Running Tests

### Run all tests:
```bash
pytest tests/
```

### Run specific test file:
```bash
pytest tests/test_feature_detectors.py
```

### Run with verbose output:
```bash
pytest tests/ -v
```

### Run specific test class or method:
```bash
pytest tests/test_feature_detectors.py::TestTrackpyDetector::test_detect_frame_basic
```

### Run tests matching a pattern:
```bash
pytest tests/ -k "gaussian"
```

### Generate coverage report:
```bash
pytest tests/ --cov=. --cov-report=html
```

## Test Statistics

- **Total Tests**: 199
- **Test Coverage**: All major modules and functions
- **Parametrized Tests**: Multiple test methods use pytest.mark.parametrize for comprehensive coverage
- **Fixtures**: Shared test data through pytest fixtures

## Test Categories

### Unit Tests
- Individual function testing
- Input validation
- Edge case handling
- Error conditions

### Integration Tests
- Complete workflows
- Module interactions
- End-to-end pipelines

### Smoke Tests
- Visualization functions (ensure no crashes)
- Plot generation without visual verification

## Requirements

- pytest
- numpy
- pandas
- scipy
- matplotlib
- trackpy
- pims
- pillow

## Notes

- Visualization tests use matplotlib's 'Agg' backend to avoid requiring display
- Some tests use synthetic data for reproducibility
- Tests include both success and failure cases
- Trackpy integration is validated for all detector outputs
