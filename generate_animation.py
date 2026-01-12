#!/usr/bin/env python3
"""
Animation Generation Script
Uses trained model to generate facial landmarks from audio input.
"""

import numpy as np
import torch
import torch.nn as nn
import librosa
import json
import cv2
from scipy import interpolate
from pathlib import Path
import argparse


class LipSyncMLP(nn.Module):
    """
    Multi-Layer Perceptron for audio-driven facial animation.
    Must match the architecture used during training.
    """

    def __init__(self, input_size=13, hidden_sizes=[256, 512, 512], output_size=1404, dropout=0.2):
        super(LipSyncMLP, self).__init__()

        layers = []

        # Input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.LeakyReLU(0.2))  # LeakyReLU prevents dead neurons
        layers.append(nn.Dropout(dropout))

        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.LeakyReLU(0.2))  # LeakyReLU prevents dead neurons
            layers.append(nn.Dropout(dropout))

        # Output layer (no activation - we'll use normalized outputs)
        layers.append(nn.Linear(hidden_sizes[-1], output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def load_model(model_path, device='cuda'):
    """
    Load trained model from checkpoint.

    Args:
        model_path: Path to saved model (.pth file)
        device: Device to load model on ('cuda' or 'cpu')

    Returns:
        Tuple of (model, preprocessing_params)
        - model: Loaded model in evaluation mode
        - preprocessing_params: Dictionary with scaler and normalization info
    """
    print(f"Loading model from {model_path}...")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Get model configuration
    if 'model_config' in checkpoint:
        config = checkpoint['model_config']
        input_size = config['input_size']
        hidden_sizes = config['hidden_sizes']
        output_size = config['output_size']
        dropout = config['dropout']
    else:
        # Default configuration (if not saved in checkpoint)
        input_size = 13
        hidden_sizes = [256, 512, 512]
        output_size = 1404
        dropout = 0.2

    # Create model
    model = LipSyncMLP(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
        dropout=dropout
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()  # Set to evaluation mode

    print(f"  ✓ Model loaded successfully!")
    print(f"  Architecture: {input_size} → {hidden_sizes} → {output_size}")

    if 'best_val_loss' in checkpoint:
        print(f"  Best validation loss: {checkpoint['best_val_loss']:.6f}")

    # Load preprocessing parameters
    preprocessing_params = {
        'mfcc_mean': checkpoint.get('mfcc_mean', None),
        'mfcc_std': checkpoint.get('mfcc_std', None),
        'landmarks_min': checkpoint.get('landmarks_min', -1.0),
        'landmarks_max': checkpoint.get('landmarks_max', 1.0),
        'epsilon': checkpoint.get('epsilon', 1e-8),
    }

    if preprocessing_params['mfcc_mean'] is not None:
        print(f"  ✓ Preprocessing parameters loaded")
        print(f"    MFCC mean shape: {preprocessing_params['mfcc_mean'].shape}")
        print(f"    MFCC std shape: {preprocessing_params['mfcc_std'].shape}")
    else:
        print(f"  ⚠ Warning: No preprocessing parameters found (using defaults)")

    return model, preprocessing_params


def extract_mfcc_from_audio(audio_path, target_frames=None, fps=30, n_mfcc=13):
    """
    Extract and interpolate MFCC features from audio.

    Args:
        audio_path: Path to audio file
        target_frames: Target number of frames (if None, calculated from audio duration)
        fps: Target frames per second
        n_mfcc: Number of MFCC coefficients

    Returns:
        Interpolated MFCC features and metadata
    """
    print(f"Loading audio from {audio_path}...")

    # Load audio
    audio, sample_rate = librosa.load(audio_path, sr=None)
    duration = len(audio) / sample_rate

    print(f"  Audio duration: {duration:.2f} seconds")
    print(f"  Sample rate: {sample_rate} Hz")

    # Calculate target frames if not provided
    if target_frames is None:
        target_frames = int(duration * fps)

    print(f"  Target frames: {target_frames} ({fps} fps)")
    print()

    # Extract MFCC features
    print(f"Extracting MFCC features (n_mfcc={n_mfcc})...")
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    mfcc = mfcc.T  # Transpose to (time, features)

    print(f"  Raw MFCC shape: {mfcc.shape}")

    # Interpolate to target frame count
    print(f"Interpolating to {target_frames} frames...")

    original_length = mfcc.shape[0]
    num_features = mfcc.shape[1]

    original_indices = np.linspace(0, 1, original_length)
    target_indices = np.linspace(0, 1, target_frames)

    interpolated_mfcc = np.zeros((target_frames, num_features))

    for i in range(num_features):
        f = interpolate.interp1d(original_indices, mfcc[:, i], kind='linear')
        interpolated_mfcc[:, i] = f(target_indices)

    print(f"  Interpolated MFCC shape: {interpolated_mfcc.shape}")
    print()

    return interpolated_mfcc, {
        'fps': fps,
        'duration': duration,
        'total_frames': target_frames,
        'sample_rate': sample_rate
    }


def predict_landmarks(model, mfcc_features, preprocessing_params, device='cuda', batch_size=32):
    """
    Predict facial landmarks from MFCC features using trained model.

    Args:
        model: Trained PyTorch model
        mfcc_features: MFCC features array (num_frames, n_mfcc)
        preprocessing_params: Dictionary with scaler and normalization parameters
        device: Device for inference
        batch_size: Batch size for inference

    Returns:
        Predicted landmarks array (num_frames, 468, 3)
    """
    print("Preprocessing MFCC features...")

    # Apply standardization to MFCC features (same as training)
    mfcc_mean = preprocessing_params['mfcc_mean']
    mfcc_std = preprocessing_params['mfcc_std']
    epsilon = preprocessing_params['epsilon']

    print(f"  MFCC features before standardization: min={mfcc_features.min():.4f}, max={mfcc_features.max():.4f}, mean={mfcc_features.mean():.4f}")

    if mfcc_mean is not None and mfcc_std is not None:
        # Manually standardize: (X - mean) / std
        mfcc_standardized = (mfcc_features - mfcc_mean) / mfcc_std
        print(f"  ✓ MFCC standardized using saved mean/std")
        print(f"    After standardization: min={mfcc_standardized.min():.4f}, max={mfcc_standardized.max():.4f}, mean={mfcc_standardized.mean():.4f}, std={mfcc_standardized.std():.4f}")
    else:
        mfcc_standardized = mfcc_features
        print(f"  ⚠ Warning: No scaler parameters, using raw MFCC")

    print("Running inference...")

    num_frames = mfcc_standardized.shape[0]
    predictions = []

    model.eval()
    with torch.no_grad():
        # Process in batches for memory efficiency
        for i in range(0, num_frames, batch_size):
            batch_end = min(i + batch_size, num_frames)
            batch_mfcc = mfcc_standardized[i:batch_end]

            # Convert to tensor and move to device
            batch_tensor = torch.FloatTensor(batch_mfcc).to(device)

            # Predict
            batch_predictions = model(batch_tensor)

            # Move back to CPU and convert to numpy
            predictions.append(batch_predictions.cpu().numpy())

            if (i // batch_size + 1) % 10 == 0:
                print(f"  Processed {batch_end}/{num_frames} frames...")

    # Concatenate all batches
    predictions = np.concatenate(predictions, axis=0)

    print(f"  ✓ Inference complete!")
    print(f"  Predictions shape: {predictions.shape}")

    # Denormalize predictions from [-1, 1] back to [0, 1]
    print("Denormalizing predictions...")

    print(f"  Predictions before denormalization: min={predictions.min():.4f}, max={predictions.max():.4f}, mean={predictions.mean():.4f}")

    # Predictions are in [-1, 1] range, convert back to [0, 1]
    # Formula: (normalized + 1) / 2
    predictions_denormalized = (predictions + 1.0) / 2.0

    # Clamp to [0, 1] range to handle any numerical issues
    predictions_denormalized = np.clip(predictions_denormalized, 0.0, 1.0)

    print(f"  ✓ Denormalized to [0, 1] range")
    print(f"  After denormalization: min={predictions_denormalized.min():.4f}, max={predictions_denormalized.max():.4f}, mean={predictions_denormalized.mean():.4f}")
    print()

    # Reshape from (num_frames, 1404) to (num_frames, 468, 3)
    landmarks = predictions_denormalized.reshape(num_frames, 468, 3)

    # DEBUG: Print first 5 coordinates to verify non-zero predictions
    print("=" * 70)
    print("DEBUG: First 5 predicted landmark coordinates (frame 0):")
    print("=" * 70)
    for i in range(min(5, landmarks.shape[1])):
        x, y, z = landmarks[0, i, 0], landmarks[0, i, 1], landmarks[0, i, 2]
        print(f"  Landmark {i}: x={x:.6f}, y={y:.6f}, z={z:.6f}")
    print("=" * 70)
    print()

    # Check if all predictions are zeros or same value
    if np.allclose(predictions_denormalized, 0.0):
        print("⚠️  WARNING: All predictions are ZERO! Model may have collapsed.")
    elif np.allclose(predictions_denormalized, 0.5):
        print("⚠️  WARNING: All predictions are 0.5! Model may have collapsed to mean.")
    elif predictions_denormalized.std() < 0.001:
        print(f"⚠️  WARNING: Very low variance (std={predictions_denormalized.std():.6f}). Model may have collapsed.")
    else:
        print(f"✓ Predictions look good! Variance: {predictions_denormalized.std():.6f}")
    print()

    return landmarks


def save_landmarks_json(landmarks, metadata, output_path):
    """
    Save predicted landmarks to JSON file in same format as face_data.json.

    Args:
        landmarks: Landmarks array (num_frames, 468, 3)
        metadata: Video metadata dictionary
        output_path: Output JSON file path
    """
    print(f"Saving landmarks to {output_path}...")

    num_frames = landmarks.shape[0]
    fps = metadata['fps']

    # Create JSON structure
    face_data = {
        "video_info": {
            "fps": fps,
            "width": 1920,  # Default resolution
            "height": 1080,
            "total_frames": num_frames
        },
        "frames": []
    }

    # Add each frame
    for frame_idx in range(num_frames):
        frame_landmarks = []

        for landmark_idx in range(468):
            frame_landmarks.append({
                "x": float(landmarks[frame_idx, landmark_idx, 0]),
                "y": float(landmarks[frame_idx, landmark_idx, 1]),
                "z": float(landmarks[frame_idx, landmark_idx, 2]),
                "visibility": None
            })

        frame_data = {
            "frame_number": frame_idx,
            "timestamp": frame_idx / fps if fps > 0 else 0,
            "face_detected": True,
            "landmarks": frame_landmarks
        }

        face_data["frames"].append(frame_data)

    # Save to file
    with open(output_path, 'w') as f:
        json.dump(face_data, f, indent=2)

    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"  ✓ Saved successfully!")
    print(f"  File size: {file_size_mb:.2f} MB")
    print()


def create_visualization_video(landmarks, output_path, fps=30, width=1920, height=1080,
                                original_landmarks=None):
    """
    Create visualization video showing predicted landmarks.

    Args:
        landmarks: Predicted landmarks (num_frames, 468, 3)
        output_path: Output video file path
        fps: Video frames per second
        width: Video width
        height: Video height
        original_landmarks: Optional original landmarks for comparison
    """
    print(f"Creating visualization video: {output_path}...")

    num_frames = landmarks.shape[0]

    # Determine video layout and get original frame count
    # Add 500px extra offset to ensure predicted face is visible
    PREDICTED_OFFSET = 500  # Extra horizontal offset for debugging

    if original_landmarks is not None:
        # Side-by-side comparison with extra offset for predicted
        video_width = width * 2 + PREDICTED_OFFSET
        canvas_width = width
        num_original_frames = original_landmarks.shape[0]
        print("  Mode: Side-by-side comparison (Original | Predicted)")
        print(f"  Predicted frames: {num_frames}")
        print(f"  Original frames: {num_original_frames}")
        print(f"  🔍 DEBUG: Predicted face offset by {PREDICTED_OFFSET}px for visibility")

        if num_frames > num_original_frames:
            print(f"  ⚠ Warning: Predicted has {num_frames - num_original_frames} more frames than original")
            print(f"  → Original will freeze at last frame after frame {num_original_frames}")
        elif num_frames < num_original_frames:
            print(f"  ⚠ Warning: Original has {num_original_frames - num_frames} more frames than predicted")
            print(f"  → Video will end at frame {num_frames}")
    else:
        # Single view
        video_width = width + PREDICTED_OFFSET
        canvas_width = width
        num_original_frames = 0  # No original frames
        print("  Mode: Predicted landmarks only")
        print(f"  🔍 DEBUG: Predicted face offset by {PREDICTED_OFFSET}px for visibility")

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (video_width, height))

    # MediaPipe face mesh connections for drawing
    # Key connections for face outline, eyes, eyebrows, nose, lips
    FACE_CONNECTIONS = [
        # Face oval
        (10, 338), (338, 297), (297, 332), (332, 284), (284, 251), (251, 389), (389, 356),
        (356, 454), (454, 323), (323, 361), (361, 288), (288, 397), (397, 365), (365, 379),
        (379, 378), (378, 400), (400, 377), (377, 152), (152, 148), (148, 176), (176, 149),
        (149, 150), (150, 136), (136, 172), (172, 58), (58, 132), (132, 93), (93, 234),
        (234, 127), (127, 162), (162, 21), (21, 54), (54, 103), (103, 67), (67, 109),
        # Lips outer
        (61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314), (314, 405),
        (405, 321), (321, 375), (375, 291), (291, 409), (409, 270), (270, 269), (269, 267),
        (267, 0), (0, 37), (37, 39), (39, 40), (40, 185), (185, 61),
        # Lips inner
        (78, 191), (191, 80), (80, 81), (81, 82), (82, 13), (13, 312), (312, 311),
        (311, 310), (310, 415), (415, 308), (308, 324), (324, 318), (318, 402), (402, 317),
        (317, 14), (14, 87), (87, 178), (178, 88), (88, 95), (95, 78),
        # Left eye
        (33, 246), (246, 161), (161, 160), (160, 159), (159, 158), (158, 157), (157, 173),
        (173, 133), (133, 155), (155, 154), (154, 153), (153, 145), (145, 144), (144, 163),
        (163, 7), (7, 33),
        # Right eye
        (263, 466), (466, 388), (388, 387), (387, 386), (386, 385), (385, 384), (384, 398),
        (398, 362), (362, 382), (382, 381), (381, 380), (380, 374), (374, 373), (373, 390),
        (390, 249), (249, 263),
        # Left eyebrow
        (46, 53), (53, 52), (52, 65), (65, 55), (70, 63), (63, 105), (105, 66), (66, 107),
        # Right eyebrow
        (276, 283), (283, 282), (282, 295), (295, 285), (300, 293), (293, 334), (334, 296), (296, 336),
        # Nose
        (168, 6), (6, 197), (197, 195), (195, 5), (5, 4), (4, 1), (1, 19), (19, 94),
    ]

    print(f"  Processing {num_frames} frames...")

    for frame_idx in range(num_frames):
        # Create canvas
        canvas = np.zeros((height, video_width, 3), dtype=np.uint8)

        # Draw predicted landmarks with extra offset for visibility
        pred_landmarks = landmarks[frame_idx]
        pred_offset_x = (canvas_width + PREDICTED_OFFSET) if original_landmarks is not None else PREDICTED_OFFSET
        draw_landmarks_on_canvas(canvas, pred_landmarks, width, height,
                                 offset_x=pred_offset_x,
                                 color=(0, 255, 0), connections=FACE_CONNECTIONS,
                                 label="PREDICTED (500px offset)" if original_landmarks is not None else "PREDICTED")

        # Draw original landmarks if provided (side-by-side)
        if original_landmarks is not None:
            # Check if frame_idx is within bounds of original landmarks
            if frame_idx < num_original_frames:
                # Use the actual frame from original data
                orig_landmarks = original_landmarks[frame_idx]
            else:
                # Use the last available frame when beyond original data
                orig_landmarks = original_landmarks[num_original_frames - 1]

            draw_landmarks_on_canvas(canvas, orig_landmarks, width, height,
                                     offset_x=0, color=(0, 165, 255),
                                     connections=FACE_CONNECTIONS, label="ORIGINAL")

        # Add frame counter
        cv2.putText(canvas, f"Frame: {frame_idx + 1}/{num_frames}",
                    (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)

        # Write frame
        video_writer.write(canvas)

        if (frame_idx + 1) % 50 == 0:
            print(f"  Rendered {frame_idx + 1}/{num_frames} frames...")

    video_writer.release()

    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"  ✓ Video created successfully!")
    print(f"  File size: {file_size_mb:.2f} MB")
    print()


def draw_landmarks_on_canvas(canvas, landmarks, canvas_width, canvas_height,
                              offset_x=0, color=(0, 255, 0), connections=None, label=None):
    """
    Draw landmarks and connections on canvas.

    Args:
        canvas: Image canvas to draw on
        landmarks: Landmarks array (468, 3)
        canvas_width: Width of drawing area
        canvas_height: Height of drawing area
        offset_x: Horizontal offset for drawing
        color: Color for landmarks and connections (BGR)
        connections: List of landmark index pairs to connect
        label: Optional label text to display
    """
    # Get actual number of landmarks (handle cases with fewer than 468)
    num_landmarks = min(landmarks.shape[0], 468)

    # Convert normalized coordinates to pixel coordinates
    for i in range(num_landmarks):
        x = int(landmarks[i, 0] * canvas_width) + offset_x
        y = int(landmarks[i, 1] * canvas_height)

        # Clamp to canvas bounds
        x = max(0, min(canvas.shape[1] - 1, x))
        y = max(0, min(canvas.shape[0] - 1, y))

        # Draw landmark point
        cv2.circle(canvas, (x, y), 1, color, -1)

    # Draw connections if provided
    if connections:
        for idx1, idx2 in connections:
            # Check if both indices are within bounds
            if idx1 < num_landmarks and idx2 < num_landmarks:
                x1 = int(landmarks[idx1, 0] * canvas_width) + offset_x
                y1 = int(landmarks[idx1, 1] * canvas_height)
                x2 = int(landmarks[idx2, 0] * canvas_width) + offset_x
                y2 = int(landmarks[idx2, 1] * canvas_height)

                # Clamp coordinates
                x1 = max(0, min(canvas.shape[1] - 1, x1))
                y1 = max(0, min(canvas.shape[0] - 1, y1))
                x2 = max(0, min(canvas.shape[1] - 1, x2))
                y2 = max(0, min(canvas.shape[0] - 1, y2))

                # Draw line
                cv2.line(canvas, (x1, y1), (x2, y2), color, 1)

    # Add label if provided
    if label:
        text_x = offset_x + 20
        cv2.putText(canvas, label, (text_x, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)


def generate_animation(
    model_path: str,
    audio_path: str,
    output_json: str,
    output_video: str = None,
    reference_json: str = None,
    fps: int = 30,
    target_frames: int = None
):
    """
    Generate facial animation from audio using trained model.

    Args:
        model_path: Path to trained model (.pth)
        audio_path: Path to input audio file
        output_json: Path to save generated landmarks JSON
        output_video: Path to save visualization video (optional)
        reference_json: Path to original landmarks for comparison (optional)
        fps: Target frames per second
        target_frames: Target number of frames (if None, calculated from audio)
    """
    print("=" * 70)
    print("FACIAL ANIMATION GENERATION")
    print("=" * 70)
    print()

    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Load model and preprocessing parameters
    model, preprocessing_params = load_model(model_path, device)
    print()

    # Extract MFCC features from audio
    mfcc_features, audio_metadata = extract_mfcc_from_audio(
        audio_path, target_frames=target_frames, fps=fps
    )

    # Predict landmarks (with preprocessing and denormalization)
    predicted_landmarks = predict_landmarks(model, mfcc_features, preprocessing_params, device)

    # Save to JSON
    save_landmarks_json(predicted_landmarks, audio_metadata, output_json)

    # Create visualization video if requested
    if output_video:
        original_landmarks = None

        # Load original landmarks for comparison if provided
        if reference_json:
            print(f"Loading reference landmarks from {reference_json}...")
            with open(reference_json, 'r') as f:
                ref_data = json.load(f)

            num_ref_frames = len(ref_data['frames'])
            original_landmarks = np.zeros((num_ref_frames, 468, 3))

            for i, frame in enumerate(ref_data['frames']):
                if frame['face_detected']:
                    for j, lm in enumerate(frame['landmarks']):
                        # Only process up to 468 landmarks to avoid index errors
                        if j < 468:
                            original_landmarks[i, j, 0] = lm['x']
                            original_landmarks[i, j, 1] = lm['y']
                            original_landmarks[i, j, 2] = lm['z']

            print(f"  ✓ Loaded {num_ref_frames} reference frames")
            print()

        create_visualization_video(
            predicted_landmarks,
            output_video,
            fps=audio_metadata['fps'],
            original_landmarks=original_landmarks
        )

    # Summary
    print("=" * 70)
    print("GENERATION COMPLETE!")
    print("=" * 70)
    print()
    print(f"📊 Generated Animation:")
    print(f"   - Total frames: {predicted_landmarks.shape[0]}")
    print(f"   - FPS: {audio_metadata['fps']}")
    print(f"   - Duration: {audio_metadata['duration']:.2f} seconds")
    print(f"   - Landmarks per frame: 468")
    print()
    print(f"💾 Output Files:")
    print(f"   - Landmarks JSON: {output_json}")
    if output_video:
        print(f"   - Visualization: {output_video}")
    print()
    print("=" * 70)
    print("🎉 Your AI-generated facial animation is ready!")
    print("=" * 70)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate facial animation from audio using trained model"
    )
    parser.add_argument(
        '--model',
        type=str,
        default='sarah_lipsync.pth',
        help='Path to trained model file (default: sarah_lipsync.pth)'
    )
    parser.add_argument(
        '--audio',
        type=str,
        default='sarah_audio.wav',
        help='Path to input audio file (default: sarah_audio.wav)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='generated_face.json',
        help='Path to output landmarks JSON (default: generated_face.json)'
    )
    parser.add_argument(
        '--video',
        type=str,
        default='preview.mp4',
        help='Path to output visualization video (default: preview.mp4, use "none" to skip)'
    )
    parser.add_argument(
        '--reference',
        type=str,
        default=None,
        help='Path to reference landmarks JSON for comparison (default: None)'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='Target frames per second (default: 30)'
    )
    parser.add_argument(
        '--frames',
        type=int,
        default=None,
        help='Target number of frames (default: calculated from audio duration)'
    )

    args = parser.parse_args()

    # Handle "none" for video output
    output_video = None if args.video.lower() == 'none' else args.video

    try:
        generate_animation(
            model_path=args.model,
            audio_path=args.audio,
            output_json=args.output,
            output_video=output_video,
            reference_json=args.reference,
            fps=args.fps,
            target_frames=args.frames
        )
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
