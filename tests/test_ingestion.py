import numpy as np
from PIL import Image
from core.ingestion import generate_proxy

def test_ingestion_proxy_downscaling(tmp_path):
    # Simulate a massive 4000x3000 Raw/JPG export
    mock_img_path = tmp_path / "mock_massive.jpg"
    img_array = np.random.randint(0, 255, (3000, 4000, 3), dtype=np.uint8)
    Image.fromarray(img_array).save(mock_img_path)
    
    # Our engine should aggressively compress this to a proxy
    proxy, scale = generate_proxy(mock_img_path, max_edge=1024)
    
    # Assert Memory Safety (Longest edge MUST be threshold)
    assert max(proxy.shape[:2]) == 1024
    
    # Assert coordinate scale factor logic is precise for upscaling later
    expected_scale = 1024 / 4000
    assert abs(scale - expected_scale) < 0.001
