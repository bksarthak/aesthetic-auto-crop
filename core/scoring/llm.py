import os
import base64
import cv2
import json
import numpy as np
from pydantic import BaseModel
from .base import AestheticScorer

# ==========================================
# CLOUD SCORING CONFIGURATION
# ==========================================
DEFAULT_GEMINI_MODEL = "gemini-3.1-pro-preview"

class AlchemyResult(BaseModel):
    alchemy_score: int
    artistic_rationale: str

class VisionLLMScorer(AestheticScorer):
    def __init__(self, provider: str = "gemini", model_override: str = None):
        self.provider = provider
        self.model_name = model_override if model_override else DEFAULT_GEMINI_MODEL
        
        if self.provider == "gemini":
            try:
                from google import genai
            except ImportError:
                raise ImportError("google-genai is required. Install via pip.")
                
            self.api_key = os.getenv("GEMINI_API_KEY")
            if not self.api_key:
                raise ValueError("GEMINI_API_KEY missing from .env")
            self.client = genai.Client(api_key=self.api_key)
        else:
            raise NotImplementedError(f"Provider {provider} not currently supported. Defaulting to gemini.")

    def score_crops(self, crops: list[np.ndarray], prompt: str) -> list[dict]:
        from PIL import Image
        results = []
        for crop in crops:
            pil_image = Image.fromarray(crop)
            
            # The prompt combines user aesthetic prompt with our extraction format
            system_instruction = (
                "You are an expert Photography Art Director specializing in 'Urban Alchemy'. "
                f"Evaluate this crop against the vision: '{prompt}'\n"
                "Provide a JSON response with 'alchemy_score' (integer 1-100) and 'artistic_rationale' (string)."
            )
            
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[
                        pil_image,
                        system_instruction
                    ],
                    config={
                        'response_mime_type': 'application/json',
                        'temperature': 0.2
                    }
                )
                
                data = json.loads(response.text)
                results.append({
                    "score": float(data.get("alchemy_score", 0)),
                    "rationale": data.get("artistic_rationale", "No rationale provided")
                })
                
            except Exception as e:
                print(f"Error scoring with LLM: {e}")
                results.append({"score": 0.0, "rationale": "API Error"})
                
        return results
