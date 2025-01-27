import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from LymphoMNIST.LymphoMNIST import LymphoMNIST
from src.models.TinyYolo import TinyYOLO, YOLOLoss

# -----------------------------
# Global settings
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# For a 128×256 image (H=128, W=256) and an 8×8 grid:
#   each grid cell is 16 px tall, 32 px wide
grid_size_x = 32
grid_size_y = 16

# -----------------------------
# Data generation
# -----------------------------
def make_cells(X, y, X_num, y_num):
    """
    Places random 28×28 patches into a 128×256 frame (H=128, W=256).
    - X has shape (128, 256, 1).
    - y has shape (8,8,15), but we only need channels 0..6 for training.
    - We randomly pick top-left px in [0..228], py in [0..100].
    """
    for _ in range(3):  # place three cells
        idx = np.random.randint(len(X_num))
        cell_img = X_num[idx]  # shape (28,28,1)
        cls_idx  = y_num[idx]  # 0 or 1

        # X.shape => (128,256): py in [0..100], px in [0..228]
        py = np.random.randint(0, 101)   # 0..100
        px = np.random.randint(0, 229)   # 0..228

        # Convert to cell index in an 8×8 grid
        #   Each cell: 16 px tall, 32 px wide
        #   +14 is a slight offset for center-based logic
        mx = (px + 14) // grid_size_x  # column index [0..7]
        my = (py + 14) // grid_size_y  # row index [0..7]

        channels = y[my, mx]

        # If this cell is already occupied, skip
        if channels[0] > 0:
            continue

        # Fill in the label map
        channels[0] = 1.0                          # probability
        channels[1] = px - (mx * grid_size_x)      # x offset
        channels[2] = py - (my * grid_size_y)      # y offset
        channels[3] = 28.0                         # width
        channels[4] = 28.0                         # height
        channels[5 + cls_idx] = 1.0                # 2-class one-hot

        # Place the 28×28 patch in X
        X[py:py+28, px:px+28] += cell_img

def make_data(size, X_num, y_num):
    """
    Creates a dataset of shape:
      - X: (size, 128, 256, 1)
      - y: (size, 8, 8, 15)
    We only *train* on the first 7 channels of y (prob + box + 2-class).
    """
    X = np.zeros((size, 128, 256, 1), dtype=np.float32)
    y = np.zeros((size, 8, 8, 15), dtype=np.float32)

    for i in range(size):
        make_cells(X[i], y[i], X_num, y_num)
    X = np.clip(X, 0.0, 1.0)
    return X, y

# -----------------------------
# Visualization
# -----------------------------
def show_predict(X, y, threshold=0.1, save_path=None, title=None):
    """
    Visualizes predictions on a single sample:
    - X has shape (128, 256, 1).
    - y has shape (8, 8, 7) or (8, 8, 15).
      We read prob, x1, y1, w, h from channels [0..4], then 2-class from [5..6].
    """
    X_disp = X.copy()
    plt.figure(figsize=(10, 5))  # (width, height) in inches
    if title:
        plt.title(title)

    for my in range(8):  # row index
        for mx in range(8):  # col index
            channels = y[my, mx]
            prob, x1, y1, w, h = channels[:5]

            if prob < threshold:
                continue

            px = int(mx * grid_size_x + x1)
            py = int(my * grid_size_y + y1)
            w  = int(w)
            h  = int(h)

            cv2.rectangle(X_disp, (px, py), (px + w, py + h), (1,), 1)
            cls_ = np.argmax(channels[5:7])  # 0 or 1
            cv2.rectangle(X_disp, (px, py - 10), (px + 12, py), (1,), -1)
            cv2.putText(X_disp, str(cls_), (px+2, py-2),
                        cv2.FONT_HERSHEY_PLAIN, 0.7, (0.0,), thickness=1)
            print(f"Class={cls_}, Prob={prob:.2f}, (px,py)=({px},{py}), (w,h)=({w},{h})")

    plt.imshow(X_disp[..., 0], cmap="gray")
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.show()

# -----------------------------
# Visualization Utility
# -----------------------------
def save_dataset_visualizations(X, y, num_samples, save_dir, prefix):
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    for i in range(num_samples):
        file_path = os.path.join(save_dir, f"{prefix}_sample_{i+1}.png")
        show_predict(X[i], y[i], threshold=0.1, save_path=file_path, title=f"{prefix} {i+1}")
        print(f"Saved visualization to {file_path}")

# -----------------------------
# Main script
# -----------------------------
def main():
    # 1) Load LymphoMNIST, filtering for 2 classes: B=0, T4=1
    lympho_dataset = LymphoMNIST(root='.', download=True, train=True, transform=None, num_classes=3)
    X_all = lympho_dataset.data.numpy()       # shape (N,28,28)
    y_all = lympho_dataset.targets.numpy()    # shape (N,)

    # Filter out T8 (label=2), keep B=0 or T4=1
    mask = (y_all == 0) | (y_all == 1)
    X_all = X_all[mask]
    y_all = y_all[mask]

    # Convert to float, downsample from 64x64 to 28x28, reshape to (N,28,28,1), normalize to [0..1]
    X_num = X_all.astype(np.float32)
    X_num = np.array([cv2.resize(img, (28, 28)) for img in X_num])
    X_num = np.expand_dims(X_num, axis=-1) / 255.0
    y_num = y_all  # 0 or 1

    # 2) Create the model and loss function
    model = TinyYOLO().to(DEVICE)
    criterion = YOLOLoss().to(DEVICE)
    from torchsummary import summary
    summary(model, (1, 128, 256))
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    # 3) Generate training data (e.g. 320 samples)
    batch_size = 32
    X_train, y_train = make_data(batch_size * 10, X_num, y_num)

    # Convert to PyTorch
    X_train_t = torch.from_numpy(X_train).permute(0,3,1,2).to(DEVICE)  # (B,1,128,256)
    y_train_t = torch.from_numpy(y_train).permute(0,3,1,2).to(DEVICE)  # (B,15,8,8)
    
    # Save a few visualizations
    save_dataset_visualizations(X_train[:5], y_train[:5], 5, "visualizations", "train")

    # Only need the first 7 channels of y for training
    y_train_t = y_train_t[:, :7, :, :]

    # 4) Train
    model.train()
    num_epochs = 10000
    n_samples = X_train_t.size(0)
    
    # Initialize learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)
    
    # Track best model
    best_loss = float('inf')
    
    # Generate test data
    X_test, y_test = make_data(5, X_num, y_num)  # Generate 5 test samples
    X_test_t = torch.from_numpy(X_test).permute(0,3,1,2).to(DEVICE)
    y_test_t = torch.from_numpy(y_test).permute(0,3,1,2).to(DEVICE)[:, :7, :, :]  # Only first 7 channels
    
    # Save test data visualizations
    save_dataset_visualizations(X_test[:5], y_test[:5], 5, "visualizations", "test")

    for epoch in range(num_epochs):
        model.train()
        perm = torch.randperm(n_samples)
        epoch_loss = 0.0
        epoch_obj_loss = 0.0
        epoch_noobj_loss = 0.0
        epoch_coord_loss = 0.0
        epoch_class_loss = 0.0

        for i in range(0, n_samples, batch_size):
            idx = perm[i : i+batch_size]
            xb = X_train_t[idx]
            yb = y_train_t[idx]

            optimizer.zero_grad()
            y_pred = model(xb)
            loss, loss_components = criterion(y_pred, yb, return_components=True)
            loss.backward()
            optimizer.step()
            
            # Accumulate losses
            epoch_loss += loss.item()
            epoch_obj_loss += loss_components['obj_loss']
            epoch_noobj_loss += loss_components['noobj_loss']
            epoch_coord_loss += loss_components['coord_loss']
            epoch_class_loss += loss_components['class_loss']

        # Calculate average losses
        num_batches = (n_samples + batch_size - 1) // batch_size
        epoch_loss /= num_batches
        epoch_obj_loss /= num_batches
        epoch_noobj_loss /= num_batches
        epoch_coord_loss /= num_batches
        epoch_class_loss /= num_batches

        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            test_pred = model(X_test_t)
            test_loss = criterion(test_pred, y_test_t).item()
            
            # Save test predictions periodically
            if (epoch + 1) % 50 == 0:
                for i in range(min(5, len(X_test))):
                    test_img = X_test_t[i].permute(1,2,0).cpu().numpy()
                    test_pred_np = test_pred[i].permute(1,2,0).cpu().numpy()
                    show_predict(test_img, test_pred_np, threshold=0.7, 
                               save_path=f"visualizations/test_pred_epoch_{epoch+1}_sample_{i+1}.png",
                               title=f"Test Prediction - Epoch {epoch+1}")

        # Update learning rate scheduler
        scheduler.step(test_loss)
        
        # Save best model
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), "tiny_yolo_2class_128x256_best.pth")
            print(f"New best model saved with test loss: {test_loss:.4f}")

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {epoch_loss:.4f} (obj: {epoch_obj_loss:.4f}, noobj: {epoch_noobj_loss:.4f}, "
              f"coord: {epoch_coord_loss:.4f}, class: {epoch_class_loss:.4f}) | "
              f"Test Loss: {test_loss:.4f} | "
              f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

    # Save final model weights
    torch.save(model.state_dict(), "tiny_yolo_2class_128x256_final.pth")
    print("Final model weights saved to tiny_yolo_2class_128x256_final.pth")

    # Final test predictions
    model.eval()
    with torch.no_grad():
        final_test_pred = model(X_test_t)
        
    for i in range(min(5, len(X_test))):
        test_img = X_test_t[i].permute(1,2,0).cpu().numpy()
        test_pred_np = final_test_pred[i].permute(1,2,0).cpu().numpy()
        show_predict(test_img, test_pred_np, threshold=0.7,
                    save_path=f"visualizations/final_test_pred_sample_{i+1}.png",
                    title="Final Test Prediction")

if __name__ == "__main__":
    main()
