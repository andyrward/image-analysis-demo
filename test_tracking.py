"""
Demo script to test the image tracking functionality.

This script creates a demo 100x100 image with a single spot in the center
and tests all available tracking methods.
"""

import numpy as np
from image_tracker import create_demo_image, track_spot


def test_tracking_methods():
    """Test all tracking methods on a demo image."""
    
    print("=" * 70)
    print("Image Tracking Demo - Testing All Methods")
    print("=" * 70)
    print()
    
    # Create a demo image with a spot at the center
    print("Creating demo 100x100 image with a spot at center (50, 50)...")
    image = create_demo_image(size=100, spot_center=(50.0, 50.0), 
                             spot_amplitude=100.0, spot_sigma=2.0)
    print(f"Image shape: {image.shape}")
    print(f"Image min/max values: {image.min():.2f} / {image.max():.2f}")
    print()
    
    # Test each tracking method
    methods = ["pytrack", "gaussian", "parabola"]
    
    results = {}
    
    for method in methods:
        print("-" * 70)
        print(f"Testing method: {method}")
        print("-" * 70)
        
        try:
            result = track_spot(image, method=method, initial_guess=(50.0, 50.0))
            results[method] = result
            
            if result["success"]:
                print(f"✓ Tracking successful!")
                print(f"  Position: x={result['x']:.4f}, y={result['y']:.4f}")
                print(f"  Error from true center: dx={result['x'] - 50.0:.4f}, dy={result['y'] - 50.0:.4f}")
                print(f"  Distance error: {np.sqrt((result['x'] - 50.0)**2 + (result['y'] - 50.0)**2):.4f} pixels")
                
                # Print method-specific information
                if method == "gaussian":
                    print(f"  Amplitude: {result['amplitude']:.2f}")
                    print(f"  Sigma: ({result['sigma_x']:.4f}, {result['sigma_y']:.4f})")
                    print(f"  R²: {result['r_squared']:.6f}")
                elif method == "parabola":
                    print(f"  R²: {result['r_squared']:.6f}")
                elif method == "pytrack":
                    print(f"  Intensity: {result['intensity']:.2f}")
            else:
                print(f"✗ Tracking failed!")
                print(f"  Error: {result.get('error', 'Unknown error')}")
        except Exception as e:
            print(f"✗ Exception occurred!")
            print(f"  Error: {str(e)}")
            results[method] = {"success": False, "error": str(e)}
        
        print()
    
    # Summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    successful_methods = [m for m, r in results.items() if r.get("success", False)]
    print(f"Successful methods: {len(successful_methods)}/{len(methods)}")
    
    if successful_methods:
        print("\nPosition comparison:")
        print(f"  True center: (50.0000, 50.0000)")
        for method in successful_methods:
            r = results[method]
            print(f"  {method:10s}: ({r['x']:7.4f}, {r['y']:7.4f}) - "
                  f"error: {np.sqrt((r['x'] - 50.0)**2 + (r['y'] - 50.0)**2):.4f} px")
    print()
    
    return results


def test_tracking_with_offset_spot():
    """Test tracking with a spot not at the center."""
    
    print("=" * 70)
    print("Additional Test: Off-center Spot")
    print("=" * 70)
    print()
    
    # Create a demo image with a spot at (30, 60)
    spot_pos = (30.0, 60.0)
    print(f"Creating demo image with spot at {spot_pos}...")
    image = create_demo_image(size=100, spot_center=spot_pos, 
                             spot_amplitude=100.0, spot_sigma=2.5)
    print()
    
    # Test with Gaussian method (usually most accurate)
    result = track_spot(image, method="gaussian")
    
    if result["success"]:
        print(f"✓ Gaussian tracking successful!")
        print(f"  True position: ({spot_pos[0]:.4f}, {spot_pos[1]:.4f})")
        print(f"  Found position: ({result['x']:.4f}, {result['y']:.4f})")
        error = np.sqrt((result['x'] - spot_pos[0])**2 + (result['y'] - spot_pos[1])**2)
        print(f"  Position error: {error:.4f} pixels")
    else:
        print(f"✗ Tracking failed: {result.get('error', 'Unknown error')}")
    
    print()


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run main test
    results = test_tracking_methods()
    
    # Run additional test
    test_tracking_with_offset_spot()
    
    print("Demo completed!")
