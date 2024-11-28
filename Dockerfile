# Use NVIDIA CUDA base image
FROM nvidia/cuda:12.6.2-cudnn-runtime-ubi9

# Install required system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6 \
    && apt-get clean

# Install Python libraries
RUN pip install --no-cache-dir \
    opencv-python-headless \
    opencv-python \
    numpy \
    scipy \
    matplotlib \
    ffmpeg-python \
    opencv-contrib-python-headless[extra]

# Set up working directory
WORKDIR /app

# Copy processing script to the container
COPY process_video_cuda.py /app/process_video.py

# Set the entrypoint
ENTRYPOINT ["python3", "/app/process_video.py"]
