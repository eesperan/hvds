import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import ffmpeg

def extract_frames(video_path, output_dir, fps=30):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    success, frame = cap.read()
    while success:
        frame_file = os.path.join(output_dir, f"frame_{frame_count:04d}.png")
        cv2.imwrite(frame_file, frame)
        success, frame = cap.read()
        frame_count += 1
    cap.release()
    return frame_count

def compute_optical_flow(input_dir, frame_count, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    prev_frame = cv2.imread(os.path.join(input_dir, "frame_0000.png"), cv2.IMREAD_GRAYSCALE)
    hsv = np.zeros_like(cv2.imread(os.path.join(input_dir, "frame_0000.png")))
    hsv[..., 1] = 255

    for i in range(1, frame_count):
        curr_frame = cv2.imread(os.path.join(input_dir, f"frame_{i:04d}.png"), cv2.IMREAD_GRAYSCALE)
        flow = cv2.calcOpticalFlowFarneback(prev_frame, curr_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        flow_file = os.path.join(output_dir, f"frame_{i:04d}.png")
        cv2.imwrite(flow_file, rgb_flow)
        prev_frame = curr_frame

def generate_heatmap(input_dir, frame_count, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for i in range(frame_count):
        frame = cv2.imread(os.path.join(input_dir, f"frame_{i:04d}.png"), cv2.IMREAD_GRAYSCALE)
        fft_data = np.abs(fft(frame.flatten()))
        heatmap_data = fft_data.reshape(frame.shape)[:frame.shape[0], :frame.shape[1]]
        heatmap = cv2.applyColorMap((heatmap_data * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_file = os.path.join(output_dir, f"frame_{i:04d}.png")
        cv2.imwrite(heatmap_file, heatmap)

def overlay_heatmap(frames_dir, heatmap_dir, output_dir, frame_count):
    os.makedirs(output_dir, exist_ok=True)
    for i in range(frame_count):
        frame = cv2.imread(os.path.join(frames_dir, f"frame_{i:04d}.png"))
        heatmap = cv2.imread(os.path.join(heatmap_dir, f"frame_{i:04d}.png"))
        overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)
        output_file = os.path.join(output_dir, f"frame_{i:04d}.png")
        cv2.imwrite(output_file, overlay)

def assemble_video(frames_dir, output_video, fps=30):
    (
        ffmpeg
        .input(f"{frames_dir}/frame_%04d.png", framerate=fps)
        .output(output_video, vcodec="libx264", pix_fmt="yuv420p")
        .run()
    )

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

