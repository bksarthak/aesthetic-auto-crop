import cv2
import numpy as np
from pathlib import Path

def generate_proxy(image_path: Path, max_edge: int = 1024) -> tuple[np.ndarray, float]:
    """
    Ingests a high-resolution RAW or JPG and returns a memory-safe proxy numpy array 
    for GPU inference, alongside the scale factor relative to the original image.
    """
    path_str = str(image_path)
    ext = image_path.suffix.lower()
    
    # Decode image
    if ext in ['.dng', '.arw', '.cr3', '.raw']:
        try:
            import rawpy
            with rawpy.imread(path_str) as raw:
                # Use half_size=True if possible for parsing speed (it cuts the Bayer matrix lookup time)
                # However, half_size doesn't perfectly scale by 0.5, it just parses the CFA differently
                rgb = raw.postprocess(use_camera_wb=True, half_size=True)
        except ImportError:
            raise ImportError("rawpy is required to process RAW files. Install it via `pip install rawpy`")
    else:
        # Standard JPG/PNG flow
        bgr = cv2.imread(path_str)
        if bgr is None:
            raise ValueError(f"cv2 could not read {path_str}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        
    original_h, original_w = rgb.shape[:2]
    
    # Calculate scale proxy
    longest_edge = max(original_h, original_w)
    if longest_edge > max_edge:
        scale = max_edge / longest_edge
        new_w = int(original_w * scale)
        new_h = int(original_h * scale)
        proxy = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        actual_scale = new_w / original_w  # precise scale representation
    else:
        proxy = rgb
        actual_scale = 1.0
        
    return proxy, actual_scale
