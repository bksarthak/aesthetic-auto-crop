import argparse
import os
import json
from pathlib import Path
from dotenv import load_dotenv

# Ensure core is loadable
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.ingestion import generate_proxy
from core.saliency import compute_saliency_map
from core.segmentation import generate_masks
from core.geometry import calculate_rule_of_thirds_crop

def main():
    parser = argparse.ArgumentParser(description="The Alchemist's Eye: Sub-Frame Discovery Engine")
    parser.add_argument("input_dir", type=str, help="Directory containing RAW/JPG images")
    parser.add_argument("--engine", choices=['local', 'cloud'], default='local', 
                        help="Aesthetic scoring engine (local=OpenCLIP, cloud=LLM Vision API)")
    parser.add_argument("--prompt", type=str,
                        default="A masterpiece street photograph, cinematic composition, minimalist urban alchemy.",
                        help="The creative vision or aesthetic prompt to grade against.")
    
    args = parser.parse_args()
    
    input_path = Path(args.input_dir)
    if not input_path.exists() or not input_path.is_dir():
        print(f"Error: {input_path} is not a valid directory.")
        return

    scorer = None
    if args.engine == 'cloud':
        load_dotenv()
        if not os.getenv("GEMINI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
            print("Warning: Running --engine cloud but no vision API keys found in .env")
        from core.scoring.llm import VisionLLMScorer
        scorer = VisionLLMScorer(provider="gemini")
    else:
        from core.scoring.clip import OpenCLIPScorer
        scorer = OpenCLIPScorer()

    print(f"Starting The Alchemist's Eye...")
    print(f"Target Directory: {input_path}")
    print(f"Aesthetic Engine: {args.engine.upper()}")
    
    # Check for valid images
    valid_exts = {'.raw', '.arw', '.cr3', '.dng', '.jpg', '.jpeg', '.png'}
    images = [p for p in input_path.rglob("*") if p.suffix.lower() in valid_exts and 'discoveries' not in str(p)]
    
    if not images:
        print("No valid images found in the target directory.")
        return
        
    print(f"Found {len(images)} potential images for analysis.")
    
    # Setup Output
    output_dir = input_path / "discoveries"
    output_dir.mkdir(exist_ok=True)
    metadata_file = output_dir / "metadata.json"
    
    metadata = {}
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

    prompt = args.prompt
    
    for img_path in images:
        print(f"Analyzing {img_path.name}...")
        
        # 1. Ingest
        proxy, scale = generate_proxy(img_path)
        
        # 2. Saliency
        saliency_map, points = compute_saliency_map(proxy, threshold=0.6)
        
        # 3. Segmentation (SAM 2)
        bboxes = generate_masks(proxy, points)
        
        # 4. Filter & Crop Generator
        candidate_crops = []
        candidate_coords = []
        for bbox in bboxes:
            crop_coords = calculate_rule_of_thirds_crop(proxy.shape[:2], bbox, target_ratio='4:5')
            if crop_coords:
                x, y, w, h = crop_coords
                # Extract crop from proxy for scoring
                crop_arr = proxy[y:y+h, x:x+w]
                candidate_crops.append(crop_arr)
                candidate_coords.append(crop_coords)
                
        # 5. Score Candidates
        if candidate_crops:
            scores = scorer.score_crops(candidate_crops, prompt)
            
            # 6. Rank and Take Top 1
            scored_candidates = list(zip(scores, candidate_coords))
            scored_candidates.sort(key=lambda x: x[0]['score'], reverse=True)
            
            best_score_data, best_coords = scored_candidates[0]
            
            # Map coordinates back to original size resolution
            orig_x = int(best_coords[0] * (1/scale))
            orig_y = int(best_coords[1] * (1/scale))
            orig_w = int(best_coords[2] * (1/scale))
            orig_h = int(best_coords[3] * (1/scale))
            
            print(f"  -> Best Crop Score: {best_score_data['score']:.2f}")
            print(f"  -> Rationale: {best_score_data['rationale']}")
            
            # Save Metadata
            metadata[img_path.name] = {
                "alchemy_score": best_score_data['score'],
                "rationale": best_score_data['rationale'],
                "crop_coords_proxy": best_coords,
                "crop_coords_original": (orig_x, orig_y, orig_w, orig_h),
                "engine": args.engine
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=4)
        else:
            print(f"  -> No valid structural crops found.")

if __name__ == "__main__":
    main()
