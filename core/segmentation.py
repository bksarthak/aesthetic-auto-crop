import numpy as np

def generate_masks(proxy_rgb: np.ndarray, point_coords: list[tuple]) -> list[tuple[int, int, int, int]]:
    """
    Feeds Saliency heat points into SAM 2 to generate subject masks.
    Explicitly manages VRAM by clearing CUDA cache after prediction.
    Returns a list of bounding boxes (x, y, w, h) for the extracted masks.
    """
    if not point_coords:
        return []
        
    try:
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError:
        raise ImportError("SAM 2 is strictly required for accurate composition framing. Please install it via: pip install -r requirements.txt")

    # Dynamic device selection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        
    # Turnkey Checkpoint Downloader (Low touch for photographers)
    import os
    import urllib.request
    
    os.makedirs("checkpoint", exist_ok=True)
    sam2_checkpoint = "checkpoint/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
    
    EXPECTED_SHA256 = "7442e4e9b732a508f80e141e7c2913437a3610ee0c77381a66658c3a445df87b"
    
    if not os.path.exists(sam2_checkpoint):
        print("First run detected: Downloading 5GB SAM-2 weights automatically (this will only happen once)...")
        url = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
        urllib.request.urlretrieve(url, sam2_checkpoint)
        
    # Security: SHA-256 Checksum Validation against ML Supply Chain Poisoning
    import hashlib
    print("Validating PyTorch .pt checksum integrity...")
    sha256_hash = hashlib.sha256()
    with open(sam2_checkpoint, "rb") as f:
        for byte_block in iter(lambda: f.read(16384), b""):
            sha256_hash.update(byte_block)
            
    if sha256_hash.hexdigest() != EXPECTED_SHA256:
        os.remove(sam2_checkpoint)
        raise RuntimeError(f"SECURITY ERROR: SAM 2 Checksum Validation Failed! Hash mismatch. The file {sam2_checkpoint} was instantly deleted.")
    
    # Load SAM 2
    try:
        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
        predictor = SAM2ImagePredictor(sam2_model)
    except Exception as e:
        print(f"Failed to load SAM2 Model: {e}")
        return []

    # Ensure 3-channel RGB for Monochrom/Grayscale sensors before PyTorch ingest
    if proxy_rgb.ndim == 2:
        proxy_rgb = np.stack((proxy_rgb,)*3, axis=-1)
    elif proxy_rgb.ndim == 3 and proxy_rgb.shape[-1] == 1:
        proxy_rgb = np.concatenate([proxy_rgb]*3, axis=-1)
        
    # Map image to VRAM
    predictor.set_image(proxy_rgb)
    
    bboxes = []
    height, width = proxy_rgb.shape[:2]
    total_area = height * width
        
    # Generate a distinct mask for each salient point individually to prevent shape collapse
    for x, y in point_coords:
        pts = np.array([[x, y]])
        lbls = np.array([1])
        
        masks, scores, _ = predictor.predict(
            point_coords=pts,
            point_labels=lbls,
            multimask_output=True,
        )
        
        # Extract the highest confidence mask from the multimask outputs
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        
        y_indices, x_indices = np.where(best_mask > 0)
        if len(x_indices) == 0:
            continue
            
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        w_box = x_max - x_min
        h_box = y_max - y_min
        
        mask_area = w_box * h_box
        # VRAM / Quality Thresholding: Discard masks that are too small (<5%) or too large (>50%)
        if mask_area < total_area * 0.05 or mask_area > total_area * 0.5:
            continue
            
        bboxes.append((int(x_min), int(y_min), int(w_box), int(h_box)))
        
    # IMMEDIATE VRAM RELEASE: Non-negotiable for 24GB cards doing batch iteration
    del predictor
    del sam2_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return bboxes
