import numpy as np
import cv2
import torch
from LymphoMNIST.LymphoMNIST import LymphoMNIST
from src.models.TinyYolo import TinyYOLO
import matplotlib.pyplot as plt
import os

# -----------------------------
# Global settings
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
grid_size_x = 32
grid_size_y = 16

# -----------------------------
# Data generation (reused from training)
# -----------------------------
def make_cells(X, y, X_num, y_num):
    for _ in range(3):
        idx = np.random.randint(len(X_num))
        cell_img = X_num[idx]
        cls_idx = y_num[idx]

        py = np.random.randint(0, 101)
        px = np.random.randint(0, 229)

        mx = (px + 14) // grid_size_x
        my = (py + 14) // grid_size_y

        channels = y[my, mx]

        if channels[0] > 0:
            continue

        channels[0] = 1.0
        channels[1] = px - (mx * grid_size_x)
        channels[2] = py - (my * grid_size_y)
        channels[3] = 28.0
        channels[4] = 28.0
        channels[5 + cls_idx] = 1.0

        X[py:py+28, px:px+28] += cell_img

def make_data(size, X_num, y_num):
    X = np.zeros((size, 128, 256, 1), dtype=np.float32)
    y = np.zeros((size, 8, 8, 15), dtype=np.float32)

    for i in range(size):
        make_cells(X[i], y[i], X_num, y_num)
    X = np.clip(X, 0.0, 1.0)
    return X, y

def show_predict(X, y, threshold=0.1, save_path=None, title=None):
    X_disp = X.copy()
    plt.figure(figsize=(10, 5))
    if title:
        plt.title(title)

    for my in range(8):
        for mx in range(8):
            channels = y[my, mx]
            prob, x1, y1, w, h = channels[:5]

            if prob < threshold:
                continue

            px = int(mx * grid_size_x + x1)
            py = int(my * grid_size_y + y1)
            w = int(w)
            h = int(h)

            cv2.rectangle(X_disp, (px, py), (px + w, py + h), (1,), 1)
            cls_ = np.argmax(channels[5:7])
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

def main():
    # Create output directory
    os.makedirs("inference_output", exist_ok=True)

    # Load LymphoMNIST dataset
    print("Loading LymphoMNIST dataset...")
    lympho_dataset = LymphoMNIST(root='.', download=True, train=True, transform=None, num_classes=3)
    X_all = lympho_dataset.data.numpy()
    y_all = lympho_dataset.targets.numpy()

    # Filter for B=0 and T4=1 classes
    mask = (y_all == 0) | (y_all == 1)
    X_all = X_all[mask]
    y_all = y_all[mask]

    # Preprocess images
    X_num = X_all.astype(np.float32)
    X_num = np.array([cv2.resize(img, (28, 28)) for img in X_num])
    X_num = np.expand_dims(X_num, axis=-1) / 255.0
    y_num = y_all

    # Load model and weights
    print("Loading model and weights...")
    model = TinyYOLO().to(DEVICE)
    model.load_state_dict(torch.load('tiny_yolo_2class_128x256_best.pth'))
    model.eval()

    # Generate test samples
    print("Generating test samples...")
    num_test_samples = 10
    X_test, y_test = make_data(num_test_samples, X_num, y_num)
    X_test_t = torch.from_numpy(X_test).permute(0, 3, 1, 2).to(DEVICE)

    # Run inference
    print(f"\nRunning inference on {num_test_samples} test samples...")
    with torch.no_grad():
        for i in range(num_test_samples):
            # Get prediction
            y_pred = model(X_test_t[i:i+1])
            
            # Convert to numpy for visualization
            X_test_np = X_test_t[i].permute(1, 2, 0).cpu().numpy()
            y_pred_np = y_pred[0].permute(1, 2, 0).cpu().numpy()
            
            # Save ground truth
            save_path = f"inference_output/sample_{i+1}_ground_truth.png"
            show_predict(X_test_np, y_test[i], threshold=0.1, 
                        save_path=save_path,
                        title=f"Ground Truth - Sample {i+1}")
            print(f"\nGround Truth - Sample {i+1}:")
            show_predict(X_test_np, y_test[i], threshold=0.1)
            
            # Save prediction
            save_path = f"inference_output/sample_{i+1}_prediction.png"
            show_predict(X_test_np, y_pred_np, threshold=0.5,
                        save_path=save_path,
                        title=f"Prediction - Sample {i+1}")
            print(f"\nPrediction - Sample {i+1}:")
            show_predict(X_test_np, y_pred_np, threshold=0.5)

    print("\nInference completed. Results saved in 'inference_output' directory.")

if __name__ == "__main__":
    main()
