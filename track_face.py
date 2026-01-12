#!/usr/bin/env python3
"""
Facial Landmark Tracking Script
Uses MediaPipe Face Mesh to extract 468 landmark points from video frames.
"""

import cv2
import mediapipe as mp
import json
from pathlib import Path


def track_face(video_path: str, output_path: str):
    """
    Process video and extract facial landmarks using MediaPipe Face Mesh.

    Args:
        video_path: Path to input video file
        output_path: Path to output JSON file for landmark data
    """
    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh

    # Configure Face Mesh with high accuracy settings
    # static_image_mode=False: Optimized for video processing
    # max_num_faces=1: Track one face (Sarah's face)
    # refine_landmarks=True: Include iris landmarks for better accuracy
    # min_detection_confidence=0.5: Balance between speed and accuracy
    # min_tracking_confidence=0.5: Smooth tracking across frames
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Open video file
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    # Get video properties
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Processing video: {video_path}")
    print(f"Total frames: {total_frames}")
    print(f"FPS: {fps}")
    print(f"Resolution: {width}x{height}")
    print("-" * 50)

    # Store all frame data
    face_data = {
        "video_info": {
            "fps": fps,
            "width": width,
            "height": height,
            "total_frames": total_frames
        },
        "frames": []
    }

    frame_count = 0
    frames_with_face = 0

    # Process each frame
    while video.isOpened():
        success, frame = video.read()

        if not success:
            break

        # Convert BGR to RGB (MediaPipe uses RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame with Face Mesh
        results = face_mesh.process(rgb_frame)

        # Extract landmarks if face is detected
        frame_data = {
            "frame_number": frame_count,
            "timestamp": frame_count / fps if fps > 0 else 0,
            "face_detected": False,
            "landmarks": []
        }

        if results.multi_face_landmarks:
            # Get first (and only) face
            face_landmarks = results.multi_face_landmarks[0]

            # Extract all 468 landmark points
            landmarks = []
            for landmark in face_landmarks.landmark:
                # Store normalized coordinates (0-1 range) and visibility
                landmarks.append({
                    "x": landmark.x,
                    "y": landmark.y,
                    "z": landmark.z,  # Depth information
                    "visibility": landmark.visibility if hasattr(landmark, 'visibility') else None
                })

            frame_data["face_detected"] = True
            frame_data["landmarks"] = landmarks
            frames_with_face += 1

        face_data["frames"].append(frame_data)

        # Progress indicator
        frame_count += 1
        if frame_count % 30 == 0:  # Print every 30 frames (~1 second at 30fps)
            progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
            print(f"Processing frame {frame_count}/{total_frames} ({progress:.1f}%) - "
                  f"Faces detected: {frames_with_face}/{frame_count}")

    # Cleanup
    video.release()
    face_mesh.close()

    # Save to JSON file
    print("-" * 50)
    print(f"Saving landmark data to {output_path}...")

    with open(output_path, 'w') as f:
        json.dump(face_data, f, indent=2)

    # Print summary
    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    detection_rate = (frames_with_face / frame_count * 100) if frame_count > 0 else 0

    print(f"✓ Processing complete!")
    print(f"  Total frames processed: {frame_count}")
    print(f"  Frames with face detected: {frames_with_face} ({detection_rate:.1f}%)")
    print(f"  Landmarks per frame: 468")
    print(f"  Output file size: {file_size_mb:.2f} MB")
    print(f"  Output saved to: {output_path}")


if __name__ == "__main__":
    # Configuration
    INPUT_VIDEO = "input_video.mp4"
    OUTPUT_JSON = "face_data.json"

    try:
        track_face(INPUT_VIDEO, OUTPUT_JSON)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
