import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

# -----------------------------
# Global settings
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

grid_size = 16  # Matches the "image_size / mask_size" from TF code (128/8=16)

# -----------------------------
# Data generation (same logic as TF version)
# -----------------------------
def make_numbers(X, y, X_num, y_num):
    """
    Generates a new data sample by placing random MNIST digits (0 or 1)
    around a 128x128 frame. Exactly mirrors the TF version.
    """
    for _ in range(3):  # place three digits around the image
        idx = np.random.randint(len(X_num))
        number = X_num[idx]
        kls = y_num[idx]

        px, py = np.random.randint(0, 100), np.random.randint(0, 100)
        mx, my = (px + 14) // grid_size, (py + 14) // grid_size
        
        channels = y[my, mx]
        
        # prevent duplicates (if an object is already there, skip)
        if channels[0] > 0:
            continue

        # Fill in label map
        channels[0] = 1.0
        channels[1] = px - (mx * grid_size)  # x1
        channels[2] = py - (my * grid_size)  # y1
        channels[3] = 28.0                   # x2 (width)
        channels[4] = 28.0                   # y2 (height)
        channels[5 + kls] = 1.0  # classification: channel 5 or 6

        # Place the digit into the 128x128 image
        X[py:py+28, px:px+28] += number

def make_data(size, X_num, y_num):
    """
    Generates a new dataset for YOLO training:
      - X: shape (size, 128, 128, 1)
      - y: shape (size, 8, 8, 15), but effectively we use only the first 7 channels
    """
    X = np.zeros((size, 128, 128, 1), dtype=np.float32)
    y = np.zeros((size, 8, 8, 15), dtype=np.float32)

    for i in range(size):
        make_numbers(X[i], y[i], X_num, y_num)

    X = np.clip(X, 0.0, 1.0)
    return X, y

def show_predict(X, y, threshold=0.1):
    """
    Displays a single prediction result:
      - X is grayscale image shape (128,128)
      - y is shape (8,8,7) or (8,8,15) but we only read channels 0..6
    """
    X = X.copy()  # avoid modifying the original
    for mx in range(8):
        for my in range(8):
            channels = y[my, mx]
            prob, x1, y1, x2, y2 = channels[:5]

            if prob < threshold:
                continue

            px = int((mx * grid_size) + x1)
            py = int((my * grid_size) + y1)
            w = int(x2)
            h = int(y2)

            # Draw bounding box
            cv2.rectangle(X, (px, py), (px + w, py + h), (1,), 1)

            # Draw classification above the box
            cv2.rectangle(X, (px, py - 10), (px + 12, py), (1,), -1)
            kls = np.argmax(channels[5:])
            cv2.putText(X, f'{kls}', (px + 2, py - 2),
                        cv2.FONT_HERSHEY_PLAIN, 0.7, (0.0,), thickness=1)

            print(f"digit: {kls}, prob: {prob:.2f}, x1: {px}, y1: {py}, x2: {px + w}, y2: {py + h}")

    plt.imshow(X[..., 0], cmap="gray")
    plt.axis('off')
    plt.show()

# -----------------------------
# Neural Network Components
# -----------------------------
class ConvBlock(nn.Module):
    """
    Reusable convolutional block:
    Conv2D -> ReLU -> MaxPool -> BatchNorm
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        return self.block(x)

class GateLayer(nn.Module):
    """
    Custom gating layer that:
      - Thresholds x_prob at 0.5
      - Multiplies x_boxes and x_cls by that gate
    Mirrors the behavior:
        gate = tf.where(x_prob > 0.5, 1, 0)
    """
    def __init__(self):
        super(GateLayer, self).__init__()

    def forward(self, x_prob, x_boxes, x_cls):
        # x_prob is shape (B, 1, H, W)
        # apply gate
        gate = (x_prob > 0.5).float()
        x_boxes_gated = x_boxes * gate
        x_cls_gated   = x_cls * gate
        return x_prob, x_boxes_gated, x_cls_gated

class TinyYOLO(nn.Module):
    """
    Tiny YOLO implementation with:
    - Backbone: 4x ConvBlocks
    - Heads: probability, bounding boxes, classification
    - Gating mechanism
    """
    def __init__(self):
        super(TinyYOLO, self).__init__()
        
        # Backbone: stack of 4 identical conv blocks
        self.backbone = nn.ModuleList([
            ConvBlock(1, 16),  # First block: 1->16 channels
            *[ConvBlock(16, 16) for _ in range(3)]  # 3 more blocks: 16->16 channels
        ])
        
        # Output heads
        self.x_prob = nn.Conv2d(16, 1, kernel_size=3, padding=1)   # Probability
        self.x_boxes = nn.Conv2d(16, 4, kernel_size=3, padding=1)  # Bounding boxes
        self.x_cls = nn.Conv2d(16, 2, kernel_size=3, padding=1)    # Classification
        
        self.gate = GateLayer()

    def forward(self, x):
        """
        Forward pass:
          - x shape: (B, 1, 128, 128)
          - output shape: (B, 7, 8, 8) 
            (channels=prob + 4 box coords + 2 classes)
        """
        # Pass through backbone
        for block in self.backbone:
            x = block(x)
        
        # Generate predictions
        prob = torch.sigmoid(self.x_prob(x))   # shape (B, 1, 8, 8)
        boxes = self.x_boxes(x)                # shape (B, 4, 8, 8)
        cls_ = torch.sigmoid(self.x_cls(x))    # shape (B, 2, 8, 8)
        
        # Apply gating
        prob, boxes_gated, cls_gated = self.gate(prob, boxes, cls_)
        
        # Combine outputs
        return torch.cat([prob, boxes_gated, cls_gated], dim=1)

class TinyYOLOLoss(nn.Module):
    """
    Combined loss function for Tiny YOLO:
    - Bounding box regression (MSE)
    - Object probability (BCE)
    - Class prediction (BCE)
    """
    def __init__(self):
        super(TinyYOLOLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
    
    def forward(self, y_pred, y_true):
        # Bounding box loss (channels 1-4)
        bb_loss = self.mse_loss(y_pred[:, 1:5, :, :], y_true[:, 1:5, :, :])
        
        # Probability loss (channel 0)
        prob_loss = self.bce_loss(y_pred[:, 0:1, :, :], y_true[:, 0:1, :, :])
        
        # Classification loss (channels 5-6)
        cls_loss = self.bce_loss(y_pred[:, 5:7, :, :], y_true[:, 5:7, :, :])
        
        return bb_loss + prob_loss + cls_loss

# -----------------------------
# Main script
# -----------------------------
def main():
    # -----------------------------
    # Prepare MNIST (digits 0 and 1 only)
    # -----------------------------
    mnist_dataset = MNIST(root='.', download=True, train=True, transform=None)
    X_all = mnist_dataset.data.numpy()    # shape (60000, 28, 28)
    y_all = mnist_dataset.targets.numpy() # shape (60000,)
    
    # Filter for digits 0 or 1
    mask = (y_all == 0) | (y_all == 1)
    X_num = X_all[mask].astype(np.float32)
    y_num = y_all[mask]

    # Reshape to (N, 28, 28, 1) and normalize
    X_num = np.expand_dims(X_num, axis=-1) / 255.0

    # -----------------------------
    # Create model and loss
    # -----------------------------
    model = TinyYOLO().to(DEVICE)
    criterion = TinyYOLOLoss().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    # -----------------------------
    # Generate training data
    # -----------------------------
    batch_size = 32
    # e.g. 3200 training samples
    X_train, y_train = make_data(batch_size * 100, X_num, y_num)

    # Convert to torch tensors, and reorder to (B, C, H, W)
    X_train_t = torch.from_numpy(X_train).permute(0, 3, 1, 2).to(DEVICE)  # (B,1,128,128)
    y_train_t = torch.from_numpy(y_train).permute(0, 3, 1, 2).to(DEVICE)  # (B,15,8,8)

    # We only need the first 7 channels for training, to match model output
    # (the dataset is 15 channels but model only predicts 7)
    y_train_t = y_train_t[:, :7, :, :]

    # -----------------------------
    # Train model
    # -----------------------------
    model.train()
    num_epochs = 30
    
    # We have 3200 samples (by default) in memory; we can do a typical mini-batch loop
    num_samples = X_train_t.shape[0]
    for epoch in range(num_epochs):
        perm = torch.randperm(num_samples)
        epoch_loss = 0.0

        for i in range(0, num_samples, batch_size):
            idx = perm[i : i + batch_size]
            X_batch = X_train_t[idx]
            y_batch = y_train_t[idx]

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    # -----------------------------
    # Save model weights
    # -----------------------------
    torch.save(model.state_dict(), 'tiny_yolo_weights.pth')
    print("Model weights saved to tiny_yolo_weights.pth")

    # -----------------------------
    # Test prediction (1 sample)
    # -----------------------------
    X_test, y_test = make_data(1, X_num, y_num)
    X_test_t = torch.from_numpy(X_test).permute(0, 3, 1, 2).to(DEVICE)  # shape (1,1,128,128)
    model.eval()
    with torch.no_grad():
        y_pred_t = model(X_test_t)  # shape (1,7,8,8)

    # Convert back to numpy for visualization
    X_test_np = X_test_t[0].permute(1, 2, 0).cpu().numpy()  # (128,128,1)
    y_pred_np = y_pred_t[0].permute(1, 2, 0).cpu().numpy()  # (8,8,7)

    # Show predictions
    show_predict(X_test_np, y_pred_np, threshold=0.7)

if __name__ == "__main__":
    main()
