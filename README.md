<div align="center">

# 👁️‍🗨️ The Alchemist's Eye
> *Finding the hidden beauty, duality, and "Equivalent Exchange" within the urban chaos.*

[![Built via Antigravity](https://img.shields.io/badge/Built_via-Antigravity-8A2BE2?style=for-the-badge&logo=google-gemini)](https://deepmind.google/technologies/gemini/)
[![Model](https://img.shields.io/badge/Model-Gemini_3.1_Pro_High-blue?style=for-the-badge&logo=googlebard)](https://deepmind.google/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA_Optimized-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)

</div>

---

## 📖 What is this?
**The Alchemist's Eye** is an automated, localized ML pipeline designed for street photographers and archivists. It acts as a programmatic art director—ingesting massive directories of high-resolution `.dng`/`.cr3`/`.raw` photography, hunting for compelling subjects, and mathematically extracting perfectly composed sub-frames (crops) at cinematic aspect ratios.

It answers one simple question: *Hidden inside this messy 45-Megapixel shot, is there a minimalist masterpiece waiting to be found?*

---

## 🏗️ Architecture & Data Flow
The Alchemist's Eye utilizes a strictly VRAM-controlled, multi-stage proxy architecture to avoid Out-Of-Memory (OOM) errors common in naive ML image pipelines.

1. **Proxy Ingestion**: Massive RAW files are opened via `rawpy`. Instead of pulling 15GB of raw sensor data into memory, the engine natively downscales the Bayer matrix into a lightweight `1024px` NumPy proxy strictly in memory.
2. **Saliency Mapping**: OpenCV's Spectral Residual algorithms scan the proxy for visual "heat"—areas of dense texture or high contrast.
3. **Segmentation Scalpel**: Meta's **Segment Anything Model (SAM 2)** isolates the heated coordinates, generating precise semantic borders around subjects. Masks >50% or <5% of the frame are dynamically discarded. `torch.cuda.empty_cache()` is violently executed.
4. **Cinematic Reframing**: Valid bounding boxes are computationally expanded to a perfect `4:5` or `1:1` ratio. The engine shifts the absolute coordinates so the subject lands dead-center on a structural **Rule-of-Thirds intersection**.
5. **Aesthetic Scoring**: The proxy crop is evaluated either locally via absolute mathematical similarities (OpenCLIP), or dynamically via an LLM.
6. **Final Extraction**: The winning proxy coordinates are mapped back to the original full-frame RAW to generate pristine crops.

---

## 🔒 Security, Privacy & Threat Model (Design Decisions)
Designed specifically for Information Security professionals protecting unreleased NDA photography, this architecture was built around strict threat-modeling principles.

### Why certain decisions were made:
- **Zero-Trust EXIF Stripping**: We chose to isolate the proxy ingestion locally *before* bridging to external APIs. Why? Because `.DNG` files house sensitive GPS coordinates, datetimes, and hardware serial numbers. By natively hashing logic into memory-safe `numpy` arrays locally, our code physically burns the metadata layer in RAM, guaranteeing zero EXIF/IP leakage.
- **ML Supply-Chain Hardening**: PyTorch `.pt` files heavily utilize Python's underlying `pickle` module, making them vulnerable to Insecure Unpickling via compromised CDNs. During the auto-download phase, the script forces explicit **SHA-256 Checksum Verification** of the 5GB SAM-2 weights, instantly trapping and deleting tampered payloads natively.
- **Sandboxed Execution Logic**: Upstream dependencies in `requirements.txt` are bound strictly to immutable Git commit hashes tracking SAM-2, defeating Man-in-the-Middle dependency mutations.
- **Path Traversal Shields**: Sub-crop execution scripts like `apply_mac_crops.py` force explicit `.resolve().is_relative_to()` constraints, shielding the root Host machine from JSON-deserialization attacks (preventing malicious string traversals).

### Deployment Tradeoffs (Security vs Usability)
- **Deployment A: Bare-Metal Local**: Maximum Privacy, Maximum Friction. Protects intellectual property entirely offline, but forces the user to fight local Python 3.10 requirements and MacOS/PyTorch compilation barriers.
- **Deployment B: Docker / Colima**: High Privacy, Medium Friction. Encapsulates dependency hell in heavily permissioned sandboxes, but limits access to Host GPUs based on the OS structure and demands rigorous 16GB daemon memory limits.
- **Deployment C: Google Colab**: Lowest Privacy, Lowest Friction. Bypasses all local hardware limits perfectly via free Cloud GPUs, but shifts trust boundaries, requiring the user to expose their RAW catalog to Google Drive mounts provisionally. 
- **Deployment D: SaaS WASM Platform (Our Master Roadmap)**: Implementing Client-Side WebAssembly (WASM) will offer maximum usability and total privacy by running proxy stripping directly inside the unprivileged browser sandbox before offloading anonymized boundaries to the cloud.

---

## 🚀 Deployment & Installation Options

### ⚠️ Minimum System Requirements
Due to the immense size of the SAM-2 neural network, your environment must be structurally equipped:
- **Bare-Metal Installs**: **Python 3.10+ is strictly required**. Native MacOS Python 3.9 environments will hard-crash against Meta's repository constraints.
- **Docker Environments**: Your daemon MUST be provisioned with at least **16GB of RAM** alongside 4 CPU cores (e.g., `colima start --cpu 4 --memory 16`). Attempting defaults will trigger `Exit Code: 137` (Out of Memory Kernel Kill).

### Option 1: The Frictionless Cloud (Google Colab)
For photographers lacking massive local GPU hardware, we have decoupled execution natively to the cloud.
1. Open the included `UrbanAlchemy.ipynb` file in [Google Colab](https://colab.research.google.com/).
2. Hit **Run All**. The engine will default to analyzing the included `demo_dataset/` directory, processing two high-fidelity street photography masterpieces without you needing to authenticate anything!
3. **Analyze your own photos:** To map your own massive `.DNG` directory natively, securely paste your Gemini key in the variables, uncomment the Google Drive cell, and point `TARGET_DIRECTORY` to your catalog folder!

### Option 2: Turnkey Execution (Docker)
The easiest way to run the pipeline offline dynamically, shielding MacOS from OS conflicts.

```bash
# 1. Build the engine
docker build -t alchemist-eye .

# 2. Run the pipeline against a directory (mapping volume mounts securely)
docker run --rm \
    --gpus all \
    -v /absolute/path/to/photos:/data \
    -v $(pwd)/.env:/app/.env \
    -v $(pwd)/checkpoint:/app/checkpoint \
    alchemist-eye /data --engine cloud
```

### Option 3: Bare-Metal Deep Learning (Advanced)
Best for offline homelabs processing thousands of files natively without VM emulation bottlenecks.

```bash
# 1. Install Dependencies (Mandates natively running 3.10+)
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 🧠 Configuration: The Scoring Engines
Once installed, execute the pipeline by pointing it mapped to your local Target Directory.

### Mode A: The Local Intelligence (Zero-Trust Offline)
Utilizes `open_clip` locally. Fully offline logic.
```bash
python discover.py /Volumes/Photos/2026/Street --engine local
```

### Mode B: The Generative Critic (Gemini / Anthropic)
Requires a `.env` API Key. This provides dynamic conversational curation and high-end composition awareness.
```bash
python discover.py /Volumes/Photos/2026/Street --engine cloud --prompt "Moody, high-contrast, film-noir street portraiture."
```

---

## 📦 Post-Processing (Native MacOS Hooks)
When execution breaks execution, a `discoveries/` folder and `metadata.json` map is instantly generated.

We included a frictionless MacOS Script to seamlessly extract your master shots:
```bash
python apply_mac_crops.py /path/to/directory --score-threshold 85
```
*(This hook parses the ML boundary matrices and leverages built-in Apple `sips` / CoreImage Native toolchains dynamically to read massive `.DNG` raw files, execute the slice without needing Adobe Lightroom loaded, and launch the masterpiece right into your MacOS Preview screen.)*

---

## ✨ The Art of The Possible (Built with Antigravity)
This entire architecture—from the VRAM memory management paradigms to the Abstract Base Class designs for the Strategy Pattern—was collaboratively designed, engineered, and scaffolded natively using **Antigravity** paired with **Gemini 3.1 Pro (High)**. 

This represents the bleeding edge of the AI workflow: you define the "Why" and philosophical parameters (*Urban Alchemy*), and the agent handles the massively complex security, Python logic, and deterministic deployment architecture ("The How") entirely on its own.
