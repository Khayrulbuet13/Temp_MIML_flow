import numpy as np
import cv2
import torch
from LymphoMNIST.LymphoMNIST import LymphoMNIST
from src.models.TinyYolo import TinyYOLO
import os
import time

# -----------------------------
# Global settings
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
grid_size_x = 32
grid_size_y = 16

trace_start = 470
trace_width = 256
trace_end = trace_start + trace_width  # = 470 + 256 = 726

# -----------------------------
# Helper to draw bounding boxes on color frames
# -----------------------------
def draw_predictions_on_color_frame(
    color_frame: np.ndarray,
    pred: np.ndarray,
    threshold: float = 0.5
):
    """
    color_frame: (H, W, 3), the original color frame
    pred: (8, 8, 15), YOLO prediction from TinyYOLO
    threshold: Probability threshold to draw bounding boxes
    """
    H, W, _ = color_frame.shape
    
    # pred shape is (grid_y=8, grid_x=8, channels=15)
    for my in range(pred.shape[0]):
        for mx in range(pred.shape[1]):
            channels = pred[my, mx]
            prob = channels[0]
            if prob < threshold:
                continue

            # YOLO channels: [prob, x, y, w, h, cls0_prob, cls1_prob, ...]
            # x, y are offsets from the top-left corner of cell
            px_offset = channels[1]
            py_offset = channels[2]
            w_box = channels[3]
            h_box = channels[4]

            # Convert cell coords -> absolute coords within the 128x256 crop
            abs_x = int(mx * grid_size_x + px_offset)
            abs_y = int(my * grid_size_y + py_offset)
            w_box = int(w_box)
            h_box = int(h_box)

            # We must shift x by `trace_start` to place on the full (original) frame
            draw_x = abs_x + trace_start
            draw_y = abs_y  # same y since no vertical cropping

            # Draw bounding box on color frame
            # Ensure bounding box is within frame boundaries
            x1 = max(0, draw_x)
            y1 = max(0, draw_y)
            x2 = min(W - 1, draw_x + w_box)
            y2 = min(H - 1, draw_y + h_box)

            cv2.rectangle(
                color_frame, (x1, y1), (x2, y2),
                (0, 255, 0), 2
            )

            # For class text, just pick whichever has the highest probability
            # For a 2-class scenario, channels[5] vs channels[6]
            cls_idx = np.argmax(channels[5:7])
            text = f"C{cls_idx}"

            cv2.rectangle(color_frame, (x1, y1 - 15), (x1 + 30, y1), (0, 255, 0), -1)
            cv2.putText(
                color_frame, text, (x1 + 2, y1 - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
            )

# -----------------------------
# Main
# -----------------------------
def main():
    # ---------------------
    # 1. Load your model
    # ---------------------
    
    print("Loading TinyYOLO model and weights...")
    model = TinyYOLO().to(DEVICE)
    model.load_state_dict(torch.load('./checkpoints/tiny_yolo_2class_128x256_best.pth', map_location=DEVICE))
    model.eval()

    # ---------------------
    # 2. Open video stream
    # ---------------------
    input_video_path = os.path.abspath("Input/output_video.mp4")
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {input_video_path}")
        return

    # Get video properties
    input_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps          = cap.get(cv2.CAP_PROP_FPS)

    # Prepare output video
    output_path = "annotated_output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(
        output_path, fourcc, fps,
        (input_width, input_height)  # same size as original frames
    )

    print(f"Reading video from: {input_video_path}")
    print(f"Saving annotated video to: {output_path}")

    # ---------------------
    # 3. Process frames
    # ---------------------
    frame_count = 0
    total_time = 0  # To accumulate time for all frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # `frame` is (H, W, 3) in BGR format
        # Crop horizontally from 470 to 726
        cropped = frame[:, trace_start:trace_end]  # shape: (128, 256, 3) if your video is 128 in height
        # Convert to grayscale
        gray_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)  # shape: (128, 256)

        # Convert to float32 and normalize (0-1)
        inp = gray_cropped.astype(np.float32) / 255.0
        # Add batch and channel dimension: (1,1,128,256)
        inp_torch = torch.from_numpy(inp).unsqueeze(0).unsqueeze(0).to(DEVICE)  # shape: [1, 1, 128, 256]

        # ---------------------
        # 4. Inference
        # ---------------------
        start_time = time.time()
        with torch.no_grad():
            pred = model(inp_torch)  # shape: [1, 15, 8, 8]
        end_time = time.time()
        pred_np = pred[0].permute(1, 2, 0).cpu().numpy()  # -> (8, 8, 15)
        
        frame_time = end_time - start_time
        total_time += frame_time

        # ---------------------
        # 5. Overlay bboxes
        # ---------------------
        # Convert the BGR frame to RGB if you prefer or just keep BGR.
        # We'll draw in BGR space for OpenCV convenience.
        draw_predictions_on_color_frame(frame, pred_np, threshold=0.5)

        # ---------------------
        # 6. Write out the annotated frame
        # ---------------------
        # `frame` is still BGR, which is typically fine for .mp4 output
        out_writer.write(frame)

        if frame_count % 50 == 0:
            print(f"Processed frame {frame_count}...")

    # Cleanup
    cap.release()
    out_writer.release()
    print("Done! Video saved:", output_path)
    avg_fps = frame_count / total_time if total_time > 0 else 0
    print(f"\nProcessed {frame_count} frames in {total_time:.2f} seconds.")
    print(f"Average FPS: {avg_fps:.2f}")

if __name__ == "__main__":
    main()
