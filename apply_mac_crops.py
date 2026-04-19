#!/usr/bin/env python3
import json
import os
import subprocess
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Apply 'Urban Alchemy' metadata crops natively using macOS sips.")
    parser.add_argument("target_dir", type=str, help="Directory containing the original raw files and discoveries/metadata.json")
    parser.add_argument("--score-threshold", type=int, default=50, help="Only crop images scored above this threshold.")
    args = parser.parse_args()

    base_dir = Path(args.target_dir)
    meta_file = base_dir / "discoveries" / "metadata.json"
    
    if not meta_file.exists():
        print(f"Error: metadata.json not found at {meta_file}.")
        return

    with open(meta_file, 'r') as f:
        data = json.load(f)

    print(f"Scanning {len(data)} evaluated photographs...")

    for filename, info in data.items():
        score = int(info.get("alchemy_score", 0))
        if score < args.score_threshold:
            continue
            
        coords = info.get("crop_coords_original")
        if not coords:
            continue
            
        x, y, w, h = coords
        input_file = (base_dir / filename).resolve()
        
        # Security: Prevent path traversal from malicious metadata.json keys
        if not input_file.is_relative_to(base_dir.resolve()):
            print(f"Security Error: Path traversal attempt blocked for {filename}")
            continue
            
        out_name = f"alchemy_crop_{score}_{Path(filename).stem}.jpg"
        output_file = base_dir / "discoveries" / out_name
        
        if not input_file.exists():
            print(f"Warning: Original file {filename} no longer exists. Skipping.")
            continue

        # MacOS native 'sips' uses CoreImage. It reads RAW/DNG natively!
        # sips --cropToHeightWidth [H] [W] --cropOffset [Y] [X] format jpeg
        cmd = [
            "sips",
            "-s", "format", "jpeg",
            "--cropToHeightWidth", str(h), str(w),
            "--cropOffset", str(y), str(x),
            str(input_file),
            "--out", str(output_file)
        ]
        
        print(f"[*] Native Crop Processing -> {filename} (Score: {score})")
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Open the physical output directly on the user's screen in Apple Preview
        if output_file.exists():
            subprocess.run(["open", "-a", "Preview", str(output_file)])

if __name__ == "__main__":
    main()
