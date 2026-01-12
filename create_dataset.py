#!/usr/bin/env python3
"""
Dataset Creation Script
Combines facial landmarks and audio MFCC features into a synchronized training dataset.
"""

import json
import numpy as np
import librosa
from scipy import interpolate
from pathlib import Path


def load_landmarks(json_path: str):
    """
    Load facial landmarks from JSON file.

    Args:
        json_path: Path to face_data.json

    Returns:
        numpy array of shape (num_frames, 468, 3) containing x, y, z coordinates
    """
    print(f"Loading landmarks from {json_path}...")

    with open(json_path, 'r') as f:
        data = json.load(f)

    frames = data['frames']
    num_frames = len(frames)

    # Initialize array for landmarks (num_frames, 468 landmarks, 3 coordinates)
    landmarks_array = np.zeros((num_frames, 468, 3))

    frames_with_face = 0

    for i, frame in enumerate(frames):
        if frame['face_detected'] and len(frame['landmarks']) == 468:
            # Extract x, y, z coordinates for all 468 landmarks
            for j, landmark in enumerate(frame['landmarks']):
                landmarks_array[i, j, 0] = landmark['x']
                landmarks_array[i, j, 1] = landmark['y']
                landmarks_array[i, j, 2] = landmark['z']
            frames_with_face += 1
        else:
            # If no face detected, use zeros (could also interpolate from neighbors)
            landmarks_array[i] = np.zeros((468, 3))

    print(f"  Loaded {num_frames} frames")
    print(f"  Frames with face detected: {frames_with_face}/{num_frames} ({frames_with_face/num_frames*100:.1f}%)")
    print(f"  Landmarks shape: {landmarks_array.shape}")

    return landmarks_array, data['video_info']


def extract_mfcc_features(audio_path: str, n_mfcc=13):
    """
    Extract MFCC features from audio file.

    Args:
        audio_path: Path to audio WAV file
        n_mfcc: Number of MFCC coefficients to extract (default: 13)

    Returns:
        numpy array of shape (num_audio_frames, n_mfcc) and sample rate
    """
    print(f"Loading audio from {audio_path}...")

    # Load audio file
    audio, sample_rate = librosa.load(audio_path, sr=None)
    duration = len(audio) / sample_rate

    print(f"  Audio duration: {duration:.2f} seconds")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Audio samples: {len(audio)}")

    print(f"Extracting MFCC features (n_mfcc={n_mfcc})...")

    # Extract MFCC features
    # hop_length controls the number of samples between successive frames
    # Default hop_length=512 gives ~86 frames per second at 22050 Hz
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)

    # Transpose to (time, features) format
    mfcc = mfcc.T

    print(f"  MFCC shape: {mfcc.shape}")
    print(f"  MFCC frames: {mfcc.shape[0]}")

    return mfcc, sample_rate, duration


def interpolate_features(features, target_length):
    """
    Interpolate features to match target length using linear interpolation.

    Args:
        features: numpy array of shape (original_length, num_features)
        target_length: desired number of frames

    Returns:
        numpy array of shape (target_length, num_features)
    """
    original_length = features.shape[0]
    num_features = features.shape[1]

    print(f"Interpolating features from {original_length} to {target_length} frames...")

    # Create interpolation function for each feature dimension
    original_indices = np.linspace(0, 1, original_length)
    target_indices = np.linspace(0, 1, target_length)

    interpolated = np.zeros((target_length, num_features))

    for i in range(num_features):
        # Linear interpolation for each MFCC coefficient
        f = interpolate.interp1d(original_indices, features[:, i], kind='linear')
        interpolated[:, i] = f(target_indices)

    print(f"  Interpolated shape: {interpolated.shape}")

    return interpolated


def create_training_dataset(
    landmarks_path: str,
    audio_path: str,
    output_path: str,
    n_mfcc: int = 13
):
    """
    Create synchronized training dataset from landmarks and audio.

    Args:
        landmarks_path: Path to face_data.json
        audio_path: Path to sarah_audio.wav
        output_path: Path to save training_data.npz
        n_mfcc: Number of MFCC coefficients (default: 13)
    """
    print("=" * 60)
    print("Creating Training Dataset")
    print("=" * 60)

    # Load facial landmarks
    landmarks, video_info = load_landmarks(landmarks_path)
    num_video_frames = landmarks.shape[0]
    fps = video_info.get('fps', 30)

    print()

    # Extract MFCC features from audio
    mfcc_features, sample_rate, audio_duration = extract_mfcc_features(audio_path, n_mfcc)

    print()

    # Interpolate MFCC features to match video frame count
    mfcc_synchronized = interpolate_features(mfcc_features, num_video_frames)

    print()
    print("-" * 60)
    print("Dataset Statistics:")
    print("-" * 60)
    print(f"  Number of frames: {num_video_frames}")
    print(f"  Video FPS: {fps}")
    print(f"  Video duration: {num_video_frames / fps:.2f} seconds")
    print(f"  Audio duration: {audio_duration:.2f} seconds")
    print()
    print(f"  Landmarks shape: {landmarks.shape}")
    print(f"  Landmarks per frame: 468")
    print(f"  Coordinates per landmark: 3 (x, y, z)")
    print()
    print(f"  MFCC features shape: {mfcc_synchronized.shape}")
    print(f"  MFCC coefficients: {n_mfcc}")
    print()

    # Flatten landmarks for easier processing
    # Shape: (num_frames, 468 * 3) = (num_frames, 1404)
    landmarks_flat = landmarks.reshape(num_video_frames, -1)

    print(f"  Flattened landmarks shape: {landmarks_flat.shape}")
    print()

    # Save as compressed NumPy archive
    print(f"Saving training dataset to {output_path}...")

    np.savez_compressed(
        output_path,
        landmarks=landmarks,  # Original 3D array (frames, 468, 3)
        landmarks_flat=landmarks_flat,  # Flattened (frames, 1404)
        mfcc=mfcc_synchronized,  # MFCC features (frames, n_mfcc)
        video_info=np.array([fps, video_info['width'], video_info['height'], num_video_frames]),
        n_mfcc=n_mfcc,
        audio_sample_rate=sample_rate
    )

    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)

    print(f"  ✓ Saved successfully!")
    print(f"  File size: {file_size_mb:.2f} MB")
    print()

    # Verify saved data
    print("Verifying saved data...")
    loaded = np.load(output_path)
    print(f"  ✓ Contains {len(loaded.files)} arrays:")
    for key in loaded.files:
        if key in ['landmarks', 'landmarks_flat', 'mfcc']:
            print(f"    - {key}: {loaded[key].shape}")
        else:
            print(f"    - {key}: {loaded[key]}")

    print()
    print("=" * 60)
    print("Dataset creation complete! 🎉")
    print("=" * 60)
    print()
    print("Your training data is ready:")
    print(f"  📁 {output_path}")
    print()
    print("Next steps:")
    print("  1. Use 'landmarks' for 3D facial data (frames, 468, 3)")
    print("  2. Use 'landmarks_flat' for flattened data (frames, 1404)")
    print("  3. Use 'mfcc' for audio features (frames, 13)")
    print("  4. Train your model to predict landmarks from MFCC!")
    print()


if __name__ == "__main__":
    # Configuration
    LANDMARKS_FILE = "face_data.json"
    AUDIO_FILE = "sarah_audio.wav"
    OUTPUT_FILE = "training_data.npz"
    N_MFCC = 13  # Number of MFCC coefficients

    try:
        create_training_dataset(
            landmarks_path=LANDMARKS_FILE,
            audio_path=AUDIO_FILE,
            output_path=OUTPUT_FILE,
            n_mfcc=N_MFCC
        )
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
