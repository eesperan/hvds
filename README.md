# Heatmap Vibration Detection System

## Hardware Requirements

### Video Capture Hardware

#### Camera

A high-frame-rate camera for capturing vibration (minimum 60 FPS; ideal 120+ FPS).
**Examples:** GoPro Hero, DSLR (e.g., Canon EOS), or a smartphone with high-FPS recording capability.

#### Tripod or Mount

To ensure the camera remains stable during video capture.

#### Lighting: Consistent and adequate lighting for clear video capture.

#### Vibration Source

Ensure the object being evaluated is stationary, except for the vibrations you’re testing.

### Hardware Requirements

| Component     | Requirement                                                          |
| ------------- | -------------------------------------------------------------------- |
| Processor     | Quad-core or higher (Intel i5/i7, AMD Ryzen 5/7, or equivalent)      |
| RAM           | Minimum 8GB (16GB recommended for smoother processing)               |
| Graphics Card | Optional: For faster rendering (e.g., NVIDIA GTX/RTX series)         |
| Storage       | SSD with at least 50GB free for video files and processing           |
| Resolution    | 1080p or higher for better detail (e.g., 1920x1080 pixels)           |
| Frame Rate    | At least 60 FPS to detect rapid vibrations                           |
| File Format   | Use uncompressed or lightly compressed formats like MP4, MOV, or AVI |
| Duration      | 10–30 seconds for testing; adjust as needed based on processing time |

### Software Requirements

#### Required Software

| Software      | Purpose                                              |
| ------------- | ---------------------------------------------------- |
| Python 3.x    | For scripting and automation                         |
| OpenCV        | Motion detection, optical flow, and video processing |
| NumPy & SciPy | Numerical computations and FFT                       |
| Matplotlib    | Heat map generation                                  |
| FFmpeg        | Video encoding/decoding                              |

#### Optional Software

| Software      | Purpose                                                           |
| ------------- | ----------------------------------------------------------------- |
| Blender       | For advanced video overlay and rendering (if desired)             |
| MATLAB/Octave | For advanced signal analysis (alternative to SciPy)               |
| ImageJ        | Free GUI tool for motion analysis (if scripting is not preferred) |

## Installation Steps

### Docker

A `Dockerfile` is provided to make the installation process easier. Once built (`docker build -t hvds .`), the Docker image can be invoked as follows:

```
docker run --rm -it -v <local dir with videos>:/videos hvds <filename of video to process>
```

This will mount your local directory in the container, and direct the HVDS script to locate the filename provided and run it through the processing steps. The output is written to the same mounted directory.

### Manual

- Install Python 3.x
- Install Libraries: `pip install opencv-python numpy scipy matplotlib ffmpeg-python`
- Install FFmpeg (from your package manager, or ffmpeg.org)
  - Be sure to add FFmpeg to your system PATH.

## Workflow

### Video Capture

- Mount your camera on a tripod or other stable surface, unimpacted by the vibrations of the object you're evaluating
- Ensure consistent lighting to minimize shadows or noise.
- Start recording while simulating or introducing vibrations
- Save the video to your computer.

### Processing Steps

This is a general overview of how the data is processed by the process_video.py script:

#### Extract Frames

Use FFmpeg or OpenCV to split the video into frames:
bash
Copy code
ffmpeg -i input*video.mp4 -vf fps=30 frames/frame*%04d.png
Downscale (Optional):
Resize frames to reduce processing time:
python
Copy code
import cv2
for i in range(frame*count):
frame = cv2.imread(f'frames/frame*{i:04d}.png')
resized = cv2.resize(frame, (640, 360)) # Adjust resolution
cv2.imwrite(f'resized*frames/frame*{i:04d}.png', resized)
C. Motion/Vibration Analysis

Compute Optical Flow:

Use OpenCV to detect frame-by-frame motion:
python
Copy code
import cv2
import numpy as np

prev_frame = cv2.imread('frames/frame_0001.png', cv2.IMREAD_GRAYSCALE)
hsv = np.zeros_like(cv2.imread('frames/frame_0001.png'))
hsv[..., 1] = 255

for i in range(2, frame*count + 1):
curr_frame = cv2.imread(f'frames/frame*{i:04d}.png', cv2.IMREAD*GRAYSCALE)
flow = cv2.calcOpticalFlowFarneback(prev_frame, curr_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
hsv[..., 0] = ang \* 180 / np.pi / 2
hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
cv2.imwrite(f'optical_flow/frame*{i:04d}.png', rgb_flow)
prev_frame = curr_frame
Vibration Analysis (FFT):

Apply Fourier Transform to detect vibration intensity at each pixel:
python
Copy code
from scipy.fftpack import fft
vibration_data = []

for i in range(frame*count):
frame = cv2.imread(f'frames/frame*{i:04d}.png', cv2.IMREAD_GRAYSCALE)
vibration_data.append(np.mean(np.abs(fft(frame.flatten()))))

vibration_heatmap = np.array(vibration_data).reshape((height, width))
D. Generate Heat Map

Create Heat Map:

Use Matplotlib to visualize vibration data:
python
Copy code
import matplotlib.pyplot as plt

heatmap = cv2.applyColorMap((vibration_heatmap \* 255).astype(np.uint8), cv2.COLORMAP_JET)
plt.imshow(heatmap)
plt.savefig('heatmap_overlay.png')
Overlay Heat Map on Video:

Blend the heat map with the original video frames:
python
Copy code
for i in range(frame*count):
frame = cv2.imread(f'frames/frame*{i:04d}.png')
heatmap = cv2.imread(f'heatmaps/frame*{i:04d}.png')
overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)
cv2.imwrite(f'output_frames/frame*{i:04d}.png', overlay)
E. Reconstruct Video

Combine frames back into a video using FFmpeg:
bash
Copy code
ffmpeg -framerate 30 -i output*frames/frame*%04d.png -c:v libx264 -pix_fmt yuv420p output_video.mp4
F. Display and Analysis

Play the video and review areas with high vibration intensity.
Use markers in the heat map to indicate regions of concern.
