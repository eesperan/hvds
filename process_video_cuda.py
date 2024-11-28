#! /usr/bin/env python3
import os     # Using os to create directories
import sys    # Exit codes
import cv2    # OpenCV library
import numpy as np              # Numpy library for array processing
import matplotlib.pyplot as plt # Matplotlib for plotting
from scipy.fftpack import fft   # Scipy library for Fourier Transforms
import ffmpeg                   # FFmpeg for video processing
import tqdm                     # tqdm for progress bar

# Constants
EXTRACT_FPS = 120 # The framerate that the input video is processed at
EXPORT_FPS = 30   # The framerate of the final processed video

## Function: extract_frames
#
# Extract frames from the video. The function takes in the video path,
# the output directory, and the frames per second (fps) to extract.
# A higher framerate is recommended for better motion analysis, but
# will increase the memory and time required to process the video.
def extract_frames(video_path, output_dir, fps=EXTRACT_FPS):
    os.makedirs(output_dir, exist_ok=True)                                    # Create output directory if it doesn't exist
    cap = cv2.VideoCapture(video_path)                                        # Create a VideoCapture object (`cap`) to read frames from the video file specified by `video_path`
    frame_count = 0                                                           # Initialize frame count starting at 0
    success, frame = cap.read()                                               # Read the first frame from the video. `success` will be true if the frame was read successfully, and `frame` will be the actual frame.
    while success:                                                            # Continue reading frames, until there are no more frames to read
        frame_file = os.path.join(output_dir, f"frame_{frame_count:04d}.png") # Create a file path for the current frame
        cv2.imwrite(frame_file, frame)                                        # Write the current frame to the file
        success, frame = cap.read()                                           # Read the next frame from the video
        frame_count += 1                                                      # Increment the frame count
    cap.release()                                                             # Release the VideoCapture object (`cap`) to free up resources.
    return frame_count                                                        # Return the total number of frames read. This tells the other functions the number of frames to process.

## Function: compute_optical_flow
#
# Compute optical flow on the GPU. "Optical flow" is a technique used to detect and measure the motion of objects in a video. 
def compute_optical_flow(input_dir, frame_count, output_dir):                                # Compute optical flow on the GPU. "Optical flow" is a technique used to detect and measure the motion of objects in a video.
    os.makedirs(output_dir, exist_ok=True)                                                   # Create output directory if it doesn't exist. It should already exist, but we'll check anyway.
    prev_frame = cv2.imread(os.path.join(input_dir, "frame_0000.png"), cv2.IMREAD_GRAYSCALE) # Read the first frame from the input directory as a grayscale image (we don't need color).
    prev_frame_gpu = cv2.cuda_GpuMat()                                                       # Create a GPU-accelerated matrix (`prev_frame_gpu`) to hold the previous frame.
    prev_frame_gpu.upload(prev_frame)                                                        # Upload the previous frame to the GPU.

    # Create CUDA optical flow object
    optical_flow = cv2.cuda_FarnebackOpticalFlow.create(5, 0.5, False, 15, 3, 5, 1.1, 0)             # Compute a dense optical flow using the Gunnar Farneback's algorithm.
    for i in range(1, frame_count):                                                                  # Loop through the remaining frames (as determined by `frame_count`)
        # Read the next frame (i) from the input directory as a grayscale image.
        # We use grayscale (single channel) instead of color (3 channels) since optical flow
        # only needs intensity information, not color. This reduces memory usage and processing time.
        # The frame number is zero-padded to 4 digits (e.g. 0001, 0002) to maintain proper ordering.
        curr_frame = cv2.imread(os.path.join(input_dir, f"frame_{i:04d}.png"), cv2.IMREAD_GRAYSCALE)

        # Create a GPU-accelerated matrix (`curr_frame_gpu`) to represent the current frame.
        curr_frame_gpu = cv2.cuda_GpuMat()
        curr_frame_gpu.upload(curr_frame) # Upload the current frame to the GPU.

        # Compute optical flow on GPU
        flow_gpu = optical_flow.calc(prev_frame_gpu, curr_frame_gpu, None) # Compare the previous frame to the current frame and store the result in `flow_gpu`.
        flow = flow_gpu.download()                                         # Download the result from the GPU back to the CPU.

        # Convert flow to HSV (hue, saturation, value) for visualization
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])                         # Convert the flow vectors to polar coordinates (magnitude and angle). The `flow[..., 0]` and `flow[..., 1]` are the x and y components of the flow vectors.
        hsv = np.zeros_like(cv2.imread(os.path.join(input_dir, f"frame_{i:04d}.png"))) # Create a blank image with the same dimensions as the input frame (zeros_like creates an array of zeros with the same shape and type as a given array).
        hsv[..., 0] = ang * 180 / np.pi / 2                                            # Set the hue channel to the angle of the flow vectors, scaled to 0-180 degrees. The `hsv[..., 0]` is the hue channel.
        hsv[..., 1] = 255                                                              # Set the saturation channel to 255 (fully saturated). The `hsv[..., 1]` is the saturation channel.
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)                # Set the value channel to the magnitude of the flow vectors, scaled to 0-255. The `hsv[..., 2]` is the value channel.
        bgr_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)                                # Convert the HSV image to a BGR image (OpenCV uses BGR instead of RGB).

        flow_file = os.path.join(output_dir, f"frame_{i:04d}.png")                     # Prepare a file path for the current frame.
        cv2.imwrite(flow_file, bgr_flow)                                               # Write the current BGR flow image to the file.

        prev_frame_gpu = curr_frame_gpu                                                # Update the previous frame to the current frame to prepare for the next iteration.

## Function: generate_heatmap
#
# Generate a heatmap using the GPU. The heatmap is a visual representation of the frequency content of the video frames.
def generate_heatmap(input_dir, frame_count, output_dir):      # Read the input directory, frame count, and output directory.
    import cupy as cp                                          # Use CuPy (a GPU-accelerated array library) for GPU-accelerated array processing
    os.makedirs(output_dir, exist_ok=True)                     # Create the output directory if it doesn't exist (again, it should already exist, but we'll check anyway)

    for i in range(frame_count):                               # Loop through the frames as determined by `frame_count`
        frame = cv2.imread(os.path.join(input_dir, f"frame_{i:04d}.png"), cv2.IMREAD_GRAYSCALE) # Read the current frame from the input directory as a grayscale image, similar to what we did for the optical flow function.
        frame_gpu = cp.asarray(frame)                          # Use CuPy to transfer the frame to the GPU
        fft_data_gpu = cp.abs(cp.fft.fft(frame_gpu.flatten())) # Perform the Fourier Transform for this frame on the GPU
        fft_data = cp.asnumpy(fft_data_gpu)                    # Transfer the result back to the CPU

        heatmap_data = fft_data.reshape(frame.shape)[:frame.shape[0], :frame.shape[1]]       # Reshape the data to match the frame dimensions, which is necessary for the next step.
        heatmap = cv2.applyColorMap((heatmap_data * 255).astype(np.uint8), cv2.COLORMAP_JET) # Apply a color map to the heatmap data and convert it to an 8-bit unsigned integer format.
        heatmap_file = os.path.join(output_dir, f"frame_{i:04d}.png")                        # Prepare a file path for the current frame.
        cv2.imwrite(heatmap_file, heatmap)                                                   # Write the current heatmap image to the output file.

## Function: overlay_heatmap
#
# Overlay the heatmap onto the corresponding frames. This function takes in the frames directory, heatmap directory, output directory, and frame count.
# It then loops through the frames, reads each frame and heatmap, and overlays the heatmap onto the frame. The result is a series of frames with the heatmap overlayed.
def overlay_heatmap(frames_dir, heatmap_dir, output_dir, frame_count):
    os.makedirs(output_dir, exist_ok=True)                                      # Create the output directory if it doesn't exist (again, it should already exist, but we'll check anyway)
    for i in range(frame_count):                                                # Loop through the frames as determined by `frame_count`
        frame = cv2.imread(os.path.join(frames_dir, f"frame_{i:04d}.png"))      # Read the current frame from the frames directory
        heatmap = cv2.imread(os.path.join(heatmap_dir, f"frame_{i:04d}.png"))   # Read the current heatmap from the heatmap directory
        
        # Overlay the heatmap onto the frame using a weighted sum. The `0.6` and `0.4` are the weights for the frame and heatmap, respectively.
        # The `0` is the gamma value, which is used to adjust the brightness of the overlay. The end result is a brighter base frame,
        # with the heatmap overlayed.
        overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)                  
        output_file = os.path.join(output_dir, f"frame_{i:04d}.png")            # Prepare a file path for the current frame.
        cv2.imwrite(output_file, overlay)                                       # Write the current merged frame+heatmap image to the output file.

## Function: assemble_video
#
# Assemble the frames into a video. This function takes in the frames directory, output video path, and 
# the frames per second (fps) to use. For this step, we only need 30 fps.
def assemble_video(frames_dir, output_video, fps=EXPORT_FPS):
    (
        ffmpeg
        .input(f"{frames_dir}/frame_%04d.png", framerate=fps)      # The format string is different than the one used in the other functions, because ffmpeg requires it.
        .output(output_video, vcodec="libx264", pix_fmt="yuv420p") # yuv420p is the color space used by OpenCV.
        .run()
    )

## Function: main
#
# The main function is the entry point for the script. It takes in the video filename, and processes the video through the steps outlined above.    
def main():
    if len(sys.argv) != 2:
        print("Usage: process_video.py <video filename>")
        sys.exit(1)

    video_filename = sys.argv[1]
    if not os.path.exists(f"/videos/{video_filename}"):
        print(f"Error: File '{video_filename}' not found in /videos")
        sys.exit(1)

    input_video_path = f"/videos/{video_filename}"
    output_video_path = f"/videos/{os.path.splitext(video_filename)[0]}-processed.mp4"

    # Define directories
    frames_dir = "/tmp/frames"
    optical_flow_dir = "/tmp/optical_flow"
    heatmap_dir = "/tmp/heatmap"
    overlay_dir = "/tmp/overlay"

    # Process video
    print("Extracting frames...")
    frame_count = extract_frames(input_video_path, frames_dir)

    print("Computing optical flow...")
    compute_optical_flow(frames_dir, frame_count, optical_flow_dir)

    print("Generating heatmap...")
    generate_heatmap(frames_dir, frame_count, heatmap_dir)

    print("Overlaying heatmap onto frames...")
    overlay_heatmap(frames_dir, heatmap_dir, overlay_dir, frame_count)

    print("Assembling final video...")
    assemble_video(overlay_dir, output_video_path)

    print(f"Processing complete. Output saved to {output_video_path}")

if __name__ == "__main__":
    main()

