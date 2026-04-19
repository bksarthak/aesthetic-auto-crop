from core.geometry import calculate_rule_of_thirds_crop

def test_calculate_rule_of_thirds_boundary_safe():
    # Mock proxy image of 1024x1024
    image_shape = (1024, 1024)
    # Mock SAM mask of center subject (x, y, w, h)
    mask_bbox = (400, 400, 100, 100) 
    
    crop = calculate_rule_of_thirds_crop(image_shape, mask_bbox, target_ratio='4:5')
    
    assert crop is not None
    assert len(crop) == 4
    
    x, y, w, h = crop
    # Guard against negative coordinate crops (OOB memory errors)
    assert x >= 0 and y >= 0
    # Guard against bounding boxes exceeding the image boundaries
    assert (x + w) <= 1024
    assert (y + h) <= 1024

def test_calculate_rule_of_thirds_ratio_enforced():
    image_shape = (2000, 2000)
    mask_bbox = (100, 100, 50, 50) 
    
    crop = calculate_rule_of_thirds_crop(image_shape, mask_bbox, target_ratio='1:1')
    
    assert crop is not None
    x, y, w, h = crop
    
    # 1:1 Aspect Ratio Assertion
    assert w == h
