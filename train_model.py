#!/usr/bin/env python3
"""
Neural Network Training Script
Trains a Multi-Layer Perceptron to predict facial landmarks from audio MFCC features.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import time
from pathlib import Path


class AudioToLandmarksDataset(Dataset):
    """PyTorch Dataset for audio-to-landmarks mapping."""

    def __init__(self, mfcc_features, landmarks):
        """
        Args:
            mfcc_features: numpy array of shape (num_frames, n_mfcc)
            landmarks: numpy array of shape (num_frames, num_landmarks * 3)
        """
        self.mfcc = torch.FloatTensor(mfcc_features)
        self.landmarks = torch.FloatTensor(landmarks)

    def __len__(self):
        return len(self.mfcc)

    def __getitem__(self, idx):
        return self.mfcc[idx], self.landmarks[idx]


class LipSyncMLP(nn.Module):
    """
    Multi-Layer Perceptron for audio-driven facial animation.

    Architecture:
        Input: MFCC features (13 dimensions)
        Hidden layers: 3 layers with ReLU activation and dropout
        Output: Flattened facial landmarks (1404 dimensions = 468 landmarks * 3 coordinates)
    """

    def __init__(self, input_size=13, hidden_sizes=[256, 512, 512], output_size=1404, dropout=0.2):
        super(LipSyncMLP, self).__init__()

        layers = []

        # Input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for mfcc_batch, landmarks_batch in dataloader:
        # Move to GPU
        mfcc_batch = mfcc_batch.to(device)
        landmarks_batch = landmarks_batch.to(device)

        # Forward pass
        predictions = model(mfcc_batch)
        loss = criterion(predictions, landmarks_batch)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for mfcc_batch, landmarks_batch in dataloader:
            # Move to GPU
            mfcc_batch = mfcc_batch.to(device)
            landmarks_batch = landmarks_batch.to(device)

            # Forward pass
            predictions = model(mfcc_batch)
            loss = criterion(predictions, landmarks_batch)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def train_model(
    data_path: str,
    output_path: str,
    epochs: int = 1000,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    train_split: float = 0.8
):
    """
    Train the lip-sync neural network.

    Args:
        data_path: Path to training_data.npz
        output_path: Path to save trained model (sarah_lipsync.pth)
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for Adam optimizer
        train_split: Fraction of data to use for training (rest for validation)
    """
    print("=" * 70)
    print("LIP-SYNC NEURAL NETWORK TRAINING")
    print("=" * 70)
    print()

    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print()

    # Load data
    print(f"Loading data from {data_path}...")
    data = np.load(data_path)

    mfcc_features = data['mfcc']  # Shape: (num_frames, 13)
    landmarks_flat = data['landmarks_flat']  # Shape: (num_frames, 1404)

    print(f"  MFCC features shape: {mfcc_features.shape}")
    print(f"  Landmarks shape: {landmarks_flat.shape}")
    print(f"  Total samples: {len(mfcc_features)}")
    print()

    # Create dataset
    full_dataset = AudioToLandmarksDataset(mfcc_features, landmarks_flat)

    # Split into train and validation
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"Train samples: {train_size}")
    print(f"Validation samples: {val_size}")
    print(f"Batch size: {batch_size}")
    print()

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    input_size = mfcc_features.shape[1]  # 13 MFCC coefficients
    output_size = landmarks_flat.shape[1]  # 1404 (468 landmarks * 3 coordinates)

    model = LipSyncMLP(
        input_size=input_size,
        hidden_sizes=[256, 512, 512],  # 3 hidden layers
        output_size=output_size,
        dropout=0.2
    )

    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model Architecture:")
    print(model)
    print()
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()

    # Loss function and optimizer
    criterion = nn.MSELoss()  # Mean Squared Error for regression
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Loss function: MSE (Mean Squared Error)")
    print(f"Optimizer: Adam")
    print(f"Learning rate: {learning_rate}")
    print()

    # Training loop
    print("=" * 70)
    print("TRAINING")
    print("=" * 70)
    print()

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)

        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        epoch_time = time.time() - epoch_start

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, output_path)
            best_marker = " ⭐ NEW BEST"
        else:
            best_marker = ""

        # Print progress every 50 epochs
        if (epoch + 1) % 50 == 0 or epoch == 0:
            elapsed = time.time() - start_time
            eta = (elapsed / (epoch + 1)) * (epochs - epoch - 1)

            print(f"Epoch [{epoch + 1:4d}/{epochs}] | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"Time: {epoch_time:.2f}s | "
                  f"ETA: {eta/60:.1f}m{best_marker}")

    total_time = time.time() - start_time

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print()
    print(f"Total training time: {total_time/60:.2f} minutes")
    print(f"Average epoch time: {total_time/epochs:.2f} seconds")
    print()
    print(f"Final train loss: {train_losses[-1]:.6f}")
    print(f"Final validation loss: {val_losses[-1]:.6f}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print()

    # Save final model
    print(f"Saving final model to {output_path}...")
    torch.save({
        'epoch': epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_losses[-1],
        'val_loss': val_losses[-1],
        'best_val_loss': best_val_loss,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'model_config': {
            'input_size': input_size,
            'hidden_sizes': [256, 512, 512],
            'output_size': output_size,
            'dropout': 0.2
        }
    }, output_path)

    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"  ✓ Model saved successfully!")
    print(f"  File size: {file_size_mb:.2f} MB")
    print()

    # Summary
    print("=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print()
    print(f"📊 Dataset:")
    print(f"   - Total frames: {len(mfcc_features)}")
    print(f"   - Training samples: {train_size}")
    print(f"   - Validation samples: {val_size}")
    print()
    print(f"🧠 Model:")
    print(f"   - Input: {input_size} MFCC features")
    print(f"   - Hidden: [256, 512, 512] neurons")
    print(f"   - Output: {output_size} landmark coordinates")
    print(f"   - Parameters: {trainable_params:,}")
    print()
    print(f"📈 Training:")
    print(f"   - Epochs: {epochs}")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Learning rate: {learning_rate}")
    print(f"   - Device: {device}")
    print()
    print(f"🎯 Results:")
    print(f"   - Final train loss: {train_losses[-1]:.6f}")
    print(f"   - Final val loss: {val_losses[-1]:.6f}")
    print(f"   - Best val loss: {best_val_loss:.6f}")
    print(f"   - Training time: {total_time/60:.2f} minutes")
    print()
    print(f"💾 Output:")
    print(f"   - Model: {output_path}")
    print(f"   - Size: {file_size_mb:.2f} MB")
    print()
    print("=" * 70)
    print("🎉 Training complete! Your lip-sync model is ready!")
    print("=" * 70)
    print()
    print("Next step: Use this model to generate new facial animations from audio!")
    print()


if __name__ == "__main__":
    # Configuration
    DATA_FILE = "training_data.npz"
    MODEL_OUTPUT = "sarah_lipsync.pth"
    EPOCHS = 1000
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001

    try:
        train_model(
            data_path=DATA_FILE,
            output_path=MODEL_OUTPUT,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE
        )
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
