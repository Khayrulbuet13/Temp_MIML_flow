import cv2
import time
import numpy as np
import torch
import os
import warnings

# Local module imports
from src.detection.yolo_detect import YOLOv5CellDetector
from src.trackers.minimal_tracker import Tracker
from src.analysis.cell_analyze import CellAnalyze
from src.analysis.cell_patch_extractor import CellPatchExtractor
from src.models.MIML import CombinedModel, MLP

import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T


warnings.filterwarnings("ignore", category=UserWarning, module="torch")


val_transform = T.Compose([
    T.Resize((64, 64)),
    T.ToTensor(),
    T.Normalize((0.3622, 0.3622, 0.3622), (0.1403, 0.1403, 0.1403)) 
])


class RealTimeMIMLInference:
    """
    Loads the MIML (CombinedModel) for real-time inference on the cropped
    64x64 patch + numeric cell properties (e.g., from CellAnalyze).
    """

    def __init__(
        self,
        model_path: str,
        device='cuda',
        debug=False
    ):
        """
        :param model_path: Path to the .pt or .pth file containing trained model state_dict
        :param device: 'cuda' or 'cpu'
        :param debug: Print debug messages if True
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.debug = debug

        # Build/Load MIML Model
        mlp = MLP(input_size=3, hidden_size=32, output_size=16)  # adjust input_size to your needs
        self.model = CombinedModel(mlp=mlp, n_classes=2, train_resnet=False)
        
        # Load saved weights
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()

        # Example class mapping
        self.idx_to_class = {0: 'T4', 1: 'T8'}

        if self.debug:
            print(f"[DEBUG RealTimeMIMLInference] Loaded MIML model from {model_path}")

    def _log_debug(self, msg: str):
        if self.debug:
            print(f"[DEBUG RealTimeMIMLInference] {msg}")

    def _make_property_tensor(self, cell_props: dict) -> torch.Tensor:
        """
        Convert the cell properties into a numeric tensor for the MLP.
        Uses 'max_DI', 'max_velocity', 'transition_time' by default.
        """
        di = cell_props.get("max_DI")
        vel = cell_props.get("max_velocity")
        transition_time = cell_props.get("transition_time")

        if any(v is None for v in [di, vel, transition_time]):
            raise ValueError("One or more required properties are None for inference")

        return torch.tensor([[di, vel, transition_time]], dtype=torch.float)

    def _preprocess_patch(self, patch: np.ndarray) -> torch.Tensor:
        """
        Convert the 64x64 BGR patch (NumPy) to a [1, 3, 64, 64] tensor,
        applying the same val_transform used in training.
        """
        patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(patch_rgb)
        patch_tensor = val_transform(pil_img).unsqueeze(0).to(self.device)  # [1, 3, 64, 64]
        return patch_tensor

    def infer(self, patch: np.ndarray, cell_props: dict) -> dict:
        """
        Run forward pass of the CombinedModel. Returns a dict with predicted class, 
        confidence, and raw logits/probs.
        """
        if patch is None:
            self._log_debug("No patch provided. Skipping inference.")
            return {}

        try:
            with torch.no_grad():
                # 1) Preprocess patch
                patch_tensor = self._preprocess_patch(patch)

                # 2) Create property tensor
                prop_tensor = self._make_property_tensor(cell_props).to(self.device)

                # 3) Forward pass
                logits = self.model(patch_tensor, prop_tensor)
                probs = F.softmax(logits, dim=1)
                pred_idx = probs.argmax(dim=1).item()
                pred_class = self.idx_to_class.get(pred_idx, f"Class{pred_idx}")
                confidence = probs[0, pred_idx].item()

            result = {
                "pred_class": pred_class,
                "confidence": confidence,
                "logits": logits.cpu().numpy().tolist(),
                "probs": probs.cpu().numpy().tolist(),
            }
            self._log_debug(f"Inference => {result}")
            return result

        except ValueError as e:
            self._log_debug(f"Skipping inference: {str(e)}")
            return {}


def main():
    DEBUG = True
    input_video_path = os.path.abspath("Input/output_video.mp4")

    # Horizontal positions where the ROI starts/ends
    trace_start = 470
    trace_end = trace_start + 256
    GAP = 20

    miml_model_path = os.path.abspath("checkpoints/31_July_10_34-model.pt")

    cap = cv2.VideoCapture(input_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 20

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cell_detector = YOLOv5CellDetector(
        model_path='checkpoints/yolov5.pt',
        device=device,
        debug=False
    )
    tracker = Tracker(dist_thresh=50, max_frames_to_skip=100, max_trace_length=10)
    analyzer = CellAnalyze(traceStart=trace_start, traceEnd=trace_end, fps=fps, debug=DEBUG)
    patch_extractor = CellPatchExtractor(
        trace_start=trace_start,
        trace_end=trace_end,
        gap=GAP,
        patch_size=64,
        debug=DEBUG
    )
    miml_infer = RealTimeMIMLInference(
        model_path=miml_model_path,
        device=device,
        debug=DEBUG
    )

    RESIZE_WIDTH = 64 * 4
    RESIZE_HEIGHT = 32 * 4

    total_frames = 0
    elapsed = 0.0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()
        centers, boxes, weights = cell_detector.detect(
            frame,
            crop_x_start=trace_start,
            crop_x_end=trace_end,
            resize_width=RESIZE_WIDTH,
            resize_height=RESIZE_HEIGHT
        )

        tracked_results = tracker.update(centers, frame)
        end_time = time.time()
        elapsed += (end_time - start_time)
        total_frames += 1

        # Draw region boundaries
        cv2.line(frame, (trace_start, 0), (trace_start, height), (0, 255, 0), 2)
        cv2.line(frame, (trace_end, 0), (trace_end, height), (0, 0, 255), 2)

        for i, (track_id, (tx, ty)) in enumerate(tracked_results):
            matching_box = boxes[i] if i < len(boxes) else None

            # Update cell analysis
            analyzer.update_properties(track_id, (tx, ty), matching_box, frame_count)

            # Attempt to extract patch if crossing
            patch_extractor.update_patch(track_id, frame, (tx, ty))

            # Get track data to check if crossing is completed
            track_data = analyzer.track_data.get(track_id, {})
            start_frame = track_data.get("start_frame_index")
            end_frame = track_data.get("end_frame_index")

            # Get the actual track object
            track_obj = next((t for t in tracker.tracks if t.track_id == track_id), None)
            if track_obj is None:
                continue

            # If the cell has completed crossing AND not marked done
            if start_frame is not None and end_frame is not None and not track_obj.done:
                if "transition_time" not in track_data:
                    continue

                # Ensure max_DI, max_velocity, etc. are not None
                required_keys = ["max_DI", "max_velocity", "transition_time"]
                if any(track_data.get(k) is None for k in required_keys):
                    if DEBUG:
                        print(f"Track {track_id} missing required properties. Skipping inference.")
                    continue

                finalize_result = patch_extractor.finalize(track_id, track_data)
                if finalize_result is not None:
                    patch, cell_props = finalize_result
                    # MIML Inference
                    inference_out = miml_infer.infer(patch, cell_props)
                    if inference_out:
                        track_obj.done = True  # mark the track as processed
                        if DEBUG:
                            print(f"Track {track_id} => MIML inference: {inference_out} with input: {cell_props}")

            # Optionally draw track lines
            if len(track_obj.trace) > 1:
                for idx in range(1, len(track_obj.trace)):
                    pt1 = (int(track_obj.trace[idx - 1][0]), int(track_obj.trace[idx - 1][1]))
                    pt2 = (int(track_obj.trace[idx][0]), int(track_obj.trace[idx][1]))
                    cv2.line(frame, pt1, pt2, (0, 255, 255), 2)

        frame_count += 1

        if DEBUG:
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1)
            if key == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

    print(f"Processed {total_frames} frames in {elapsed:.2f}s ({total_frames / elapsed:.2f} FPS)")

    # Final results
    results = analyzer.get_results()
    for item in results:
        print(item)


if __name__ == "__main__":
    main()
