from abc import ABC, abstractmethod
import numpy as np

class AestheticScorer(ABC):
    @abstractmethod
    def score_crops(self, crops: list[np.ndarray], prompt: str) -> list[dict]:
        """
        Given a list of crop numpy arrays, score them against an aesthetic prompt. 
        Returns a list of dicts: {"score": float, "rationale": str}
        """
        pass
