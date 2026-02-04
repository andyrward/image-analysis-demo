"""
Quick demo script for the image-analysis-demo application.

This provides a fast overview of all tracking methods without detailed analysis.
For comprehensive results with interactive plots, use demo_single_track.py
"""

import numpy as np
from image_tracker import create_demo_image, track_spot


def quick_demo():
    """Quick demo function showing all tracking methods at a glance."""
    print("=" * 70)
    print("Image Analysis Demo - Spot Tracking")
    print("=" * 70)
    print()
    
    # Create a demo image
    print("Creating demo 100x100 image with a centered spot...")
    image = create_demo_image(size=100, spot_center=(50.0, 50.0))
    print(f"Image created: {image.shape}")
    print()
    
    # Track the spot using different methods
    print("Tracking spot with different methods:")
    print()
    
    for method in ["gaussian", "parabola", "pytrack"]:
        result = track_spot(image, method=method)
        if result["success"]:
            print(f"  {method:10s}: x={result['x']:.2f}, y={result['y']:.2f}")
        else:
            print(f"  {method:10s}: Failed - {result.get('error', 'Unknown error')}")
    
    print()
    print("For detailed testing with interactive plots:")
    print("  - Python script: python demo_single_track.py")
    print("  - Jupyter notebook: jupyter notebook demo_tracking.ipynb")
    print()


if __name__ == "__main__":
    quick_demo()
