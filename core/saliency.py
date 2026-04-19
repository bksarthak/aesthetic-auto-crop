import cv2
import numpy as np

def compute_saliency_map(proxy_rgb: np.ndarray, threshold: float = 0.5) -> tuple[np.ndarray, list[tuple]]:
    """
    Computes the spectral residual saliency map of an image to find 'heat zones'.
    Returns the raw saliency map (float32) and a list of target coordinates (x, y)
    that exceed the given threshold.
    """
    # OpenCV saliency expects BGR
    bgr = cv2.cvtColor(proxy_rgb, cv2.COLOR_RGB2BGR)
    
    # Initialize the saliency algorithm
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    
    # Compute the saliency map
    success, saliency_map = saliency.computeSaliency(bgr)
    
    if not success:
        raise ValueError("Failed to compute Spectral Residual Saliency Map")
        
    # Saliency map is returned as float32 in range [0, 1]
    # Normalize just in case
    saliency_map = cv2.normalize(saliency_map, None, 0, 1, cv2.NORM_MINMAX)
    
    # Find coordinates where saliency > threshold
    hot_zones = np.where(saliency_map >= threshold)
    
    # hot_zones is (y_coords, x_coords)
    # We want a list of (x, y) coordinates to feed as point prompts to SAM 2
    points = []
    
    # Randomly subsample if there are too many hot pixels, to avoid memory explosion
    num_points = len(hot_zones[0])
    if num_points > 0:
        max_points = 50
        indices = np.random.choice(num_points, min(num_points, max_points), replace=False)
        for i in indices:
            y = int(hot_zones[0][i])
            x = int(hot_zones[1][i])
            points.append((x, y))
            
    return saliency_map, points
