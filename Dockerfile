FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies for OpenCV (libgl1) and git for SAM-2
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependencies
COPY requirements.txt .

# Install Python dependencies (SAM-2 pull takes place here)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the pipeline code
COPY . .

# Set discover.py as the entrypoint so docker acts like a CLI binary
ENTRYPOINT ["python", "discover.py"]
CMD ["--help"]
