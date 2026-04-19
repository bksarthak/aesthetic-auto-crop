from typing import Optional

def calculate_rule_of_thirds_crop(
    image_shape: tuple, 
    mask_bbox: tuple, 
    target_ratio: str = '4:5'
) -> Optional[tuple]:
    """
    Calculates a cinematic crop that aligns the center of the subject mask 
    onto a Rule of Thirds intersection within the target aspect ratio.
    
    Args:
        image_shape: (height, width) of the image proxy
        mask_bbox: (x, y, w, h) bounding box of the segmented subject
        target_ratio: '4:5' or '1:1'
        
    Returns:
        (x, y, w, h) crop coordinates, or None if crop exceeds boundaries
    """
    img_h, img_w = image_shape
    mx, my, mw, mh = mask_bbox
    
    # Centroid of the mask (the 'subject')
    cx = mx + mw / 2
    cy = my + mh / 2
    
    # Parse target ratio (width / height)
    if target_ratio == '4:5':
        aspect = 4.0 / 5.0
    elif target_ratio == '1:1':
        aspect = 1.0
    elif target_ratio == '16:9':
        aspect = 16.0 / 9.0
    else:
        raise ValueError(f"Unsupported target ratio: {target_ratio}")
        
    # First, figure out the target crop width/height based on mask size.
    # We want the subject to take up roughly a certain proportion of the final crop.
    # We'll make the final crop 2x to 3x larger than the subject.
    crop_h = mh * 2.5
    crop_w = crop_h * aspect
    
    # If crop is bigger than image, scale it down
    if crop_w > img_w or crop_h > img_h:
        scale = min(img_w / crop_w, img_h / crop_h)
        crop_w *= scale
        crop_h *= scale
        
    # We want cx, cy to sit on one of the 4 Rule of Thirds intersections defined by the new crop box.
    # Intersections are at 1/3 and 2/3 of the crop width and height.
    # Let's test all 4 intersections and pick the one that keeps the crop fully inside the source image.
    
    intersections = [
        (crop_w / 3, crop_h / 3),       # Top-Left
        (crop_w * 2 / 3, crop_h / 3),   # Top-Right
        (crop_w / 3, crop_h * 2 / 3),   # Bottom-Left
        (crop_w * 2 / 3, crop_h * 2 / 3)# Bottom-Right
    ]
    
    best_crop = None
    best_dist = float('inf')
    
    for (ix, iy) in intersections:
        # If the intersection sits at (ix, iy) relative to the top-left of the crop,
        # then the top-left of the crop (crop_x, crop_y) absolute coordinates are:
        crop_x = cx - ix
        crop_y = cy - iy
        
        # Check bounds
        if crop_x >= 0 and crop_y >= 0 and (crop_x + crop_w) <= img_w and (crop_y + crop_h) <= img_h:
            # Pick the crop that places the box closest to the center of the original image
            # Extraneous but adds to aesthetic safety (less edge distortion)
            dist_to_center = ((crop_x + crop_w/2) - img_w/2)**2 + ((crop_y + crop_h/2) - img_h/2)**2
            if dist_to_center < best_dist:
                best_dist = dist_to_center
                best_crop = (int(crop_x), int(crop_y), int(crop_w), int(crop_h))
                
    if best_crop is None:
        # Fallback: Just center the crop if all Rule of Thirds options break boundaries
        crop_x = cx - crop_w / 2
        crop_y = cy - crop_h / 2
        # Clamp bounds
        crop_x = max(0, min(crop_x, img_w - crop_w))
        crop_y = max(0, min(crop_y, img_h - crop_h))
        return (int(crop_x), int(crop_y), int(crop_w), int(crop_h))
        
    return best_crop
