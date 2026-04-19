import numpy as np
from PIL import Image
from .base import AestheticScorer

class OpenCLIPScorer(AestheticScorer):
    def __init__(self):
        self.mock_mode = False
        try:
            import torch
            import open_clip
            self.torch = torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print("Loading OpenCLIP Model into VRAM...")
            self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', device=self.device)
            self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
        except ImportError:
            print("Warning: open_clip or torch missing. Mocking OpenCLIP evaluation for architectural dry-run.")
            self.mock_mode = True
        
    def score_crops(self, crops: list[np.ndarray], prompt: str) -> list[dict]:
        if not crops:
            return []
            
        if self.mock_mode:
            import random
            return [{"score": random.uniform(70.0, 99.0), "rationale": "MOCK SCORE - Requires torch/open_clip"} for _ in crops]
            
        text = self.tokenizer([prompt]).to(self.device)
        results = []
        
        with self.torch.no_grad(), self.torch.autocast(device_type=self.device):
            text_features = self.model.encode_text(text)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            for crop in crops:
                # Convert np array (H, W, C) to PIL Image
                image_pil = Image.fromarray(crop)
                image = self.preprocess(image_pil).unsqueeze(0).to(self.device)
                
                image_features = self.model.encode_image(image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                
                # Cosine similarity
                similarity = (100.0 * image_features @ text_features.T).item()
                
                results.append({
                    "score": float(similarity),
                    "rationale": "Scored via absolute OpenCLIP Zero-Shot Cosine Similarity"
                })
                
        if self.device == "cuda":
            self.torch.cuda.empty_cache()
            
        return results
