import logging
import os
import cv2
import time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
from typing import Dict, Any


# Local module imports
from src.detection.yolo_detect import YOLOv5CellDetector
from src.trackers.minimal_tracker import Tracker
from src.analysis.cell_analyze import CellAnalyze
from src.analysis.cell_patch_extractor import CellPatchExtractor
from src.models.MIML import CombinedModel, MLP


logger = logging.getLogger(__name__)

# Example transform (adapt to your model training)
val_transform = T.Compose([
    T.Resize((64, 64)),
    T.ToTensor(),
    T.Normalize((0.3622, 0.3622, 0.3622), (0.1403, 0.1403, 0.1403))
])

class RealTimeMIMLInference:
    """
    Loads the MIML (CombinedModel) for real-time inference using a 64x64 patch
    plus numeric cell properties (e.g., from CellAnalyze).
    """

    def __init__(
        self,
        model_path: str,
        device: str = 'cuda',
        debug: bool = False
    ):
        """
        :param model_path: Path to the trained model .pt or .pth file.
        :param device: 'cuda' or 'cpu'.
        :param debug: Whether to log debug messages.
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.debug = debug

        # Build MLP for numeric properties
        mlp = MLP(input_size=3, hidden_size=32, output_size=16)
        self.model = CombinedModel(mlp=mlp, n_classes=2, train_resnet=False)

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()

        self.idx_to_class = {0: 'T4', 1: 'T8'}

        if self.debug:
            logger.debug(f"Loaded MIML model from {model_path}")

    def _make_property_tensor(self, cell_props: Dict[str, Any]) -> torch.Tensor:
        """
        Convert the cell properties dict into a numeric tensor for the MLP.

        :param cell_props: A dict with keys like 'max_DI', 'max_velocity', 'transition_time'.
        :return: torch.Tensor of shape (1, 3).
        """
        di = cell_props.get("max_DI")
        vel = cell_props.get("max_velocity")
        transition_time = cell_props.get("transition_time")

        if any(v is None for v in [di, vel, transition_time]):
            raise ValueError("One or more required cell properties are None.")

        return torch.tensor([[di, vel, transition_time]], dtype=torch.float)

    def _preprocess_patch(self, patch: np.ndarray) -> torch.Tensor:
        """
        Preprocess the 64x64 BGR patch for the ResNet.

        :param patch: NumPy array (H, W, 3) in BGR format.
        :return: Torch tensor of shape (1, 3, 64, 64).
        """
        patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(patch_rgb)
        patch_tensor = val_transform(pil_img).unsqueeze(0).to(self.device)
        return patch_tensor

    def infer(self, patch: np.ndarray, cell_props: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run inference on a single cell patch + cell properties.

        :param patch: 64x64 NumPy BGR patch of the cell.
        :param cell_props: Dict with cell properties from CellAnalyze.
        :return: A dict with predicted class, confidence, logits, and probabilities.
        """
        if patch is None:
            if self.debug:
                logger.debug("No patch provided for inference.")
            return {}

        try:
            with torch.no_grad():
                patch_tensor = self._preprocess_patch(patch)
                prop_tensor = self._make_property_tensor(cell_props).to(self.device)
                logits = self.model(patch_tensor, prop_tensor)
                probs = F.softmax(logits, dim=1)
                pred_idx = probs.argmax(dim=1).item()
                pred_class = self.idx_to_class.get(pred_idx, f"Class{pred_idx}")
                confidence = probs[0, pred_idx].item()

            result = {
                "pred_class": pred_class,
                "confidence": confidence,
                "logits": logits.cpu().numpy().tolist(),
                "probs": probs.cpu().numpy().tolist()
            }
            if self.debug:
                logger.debug(f"Inference => {result}")
            return result
        except ValueError as e:
            if self.debug:
                logger.debug(f"Skipping inference: {str(e)}")
            return {}

def main():
    """
    Main function to demonstrate YOLO detection, minimal tracking,
    cell analysis, patch extraction, and MIML inference.
    """

    # Set to True for more debugging statements throughout
    DEBUG = False
    
    # Set logging level based on DEBUG flag
    logging_level = logging.DEBUG if DEBUG else logging.INFO
    
    # Add stream=sys.stdout to write to standard output
    import sys
    logging.basicConfig(
        level=logging_level,
        format='[%(levelname)s] %(name)s - %(message)s',
        stream=sys.stdout  # Add this line
    )

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

    # Initialize detection, tracking, analysis, patch extraction, and inference
    cell_detector = YOLOv5CellDetector(
        model_path='checkpoints/yolov5.pt',
        device=device,
        debug=DEBUG
    )
    tracker = Tracker(dist_thresh=50, max_frames_to_skip=100, max_trace_length=10, debug=DEBUG)
    analyzer = CellAnalyze(traceStart=trace_start, traceEnd=trace_end, fps=fps, debug=DEBUG)
    patch_extractor = CellPatchExtractor(trace_start, trace_end, GAP, 64, debug=DEBUG)
    miml_infer = RealTimeMIMLInference(model_path=miml_model_path, device=device, debug=DEBUG)

    RESIZE_WIDTH  = int(256)  # 64 in production
    RESIZE_HEIGHT = int(128)  # 32 in production

    total_frames = 0
    elapsed = 0.0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.info("Video processing complete - reached end of video.")
            break

        start_time = time.time()

        # Detect
        centers, boxes, weights = cell_detector.detect(
            frame, 
            crop_x_start=trace_start, 
            crop_x_end=trace_end,
            resize_width=RESIZE_WIDTH, 
            resize_height=RESIZE_HEIGHT
        )

        # Track
        tracked_results = tracker.update(centers, frame)

        end_time = time.time()
        elapsed += (end_time - start_time)
        total_frames += 1

        # Draw region lines
        cv2.line(frame, (trace_start, 0), (trace_start, height), (0, 255, 0), 2)
        cv2.line(frame, (trace_end, 0), (trace_end, height), (0, 0, 255), 2)

        for i, (track_id, (tx, ty)) in enumerate(tracked_results):
            matching_box = boxes[i] if i < len(boxes) else None

            # Update cell analysis
            analyzer.update_properties(track_id, (tx, ty), matching_box, frame_count)

            # Attempt to extract patch if crossing
            patch_extractor.update_patch(track_id, frame, (tx, ty))

            # Retrieve track data
            track_data = analyzer.track_data.get(track_id, {})
            start_frame = track_data.get("start_frame_index")
            end_frame = track_data.get("end_frame_index")

            # Mark track done if crossing is complete
            track_obj = next((t for t in tracker.tracks if t.track_id == track_id), None)
            if track_obj and (start_frame is not None) and (end_frame is not None) and not track_obj.done:
                # We also need transition_time for MIML
                if "transition_time" not in track_data:
                    continue
                # Check if DI, velocity, etc. are non-None
                required_keys = ["max_DI", "max_velocity", "transition_time"]
                if any(track_data.get(k) is None for k in required_keys):
                    logger.debug(f"Track {track_id} missing required properties. Skipping inference.")
                    continue

                finalize_result = patch_extractor.finalize(track_id, track_data)
                if finalize_result is not None:
                    patch, cell_props = finalize_result
                    inference_out = miml_infer.infer(patch, cell_props)
                    if inference_out:
                        track_obj.done = True
                        logger.info(f"Track {track_id} => MIML inference: {inference_out} with cell properties {cell_props}")

        frame_count += 1

        if DEBUG:
            cv2.imshow("Main Frame", frame)
            key = cv2.waitKey(1)
            if key == 27:
                logger.info("Esc pressed. Terminating.")
                break

    cap.release()
    cv2.destroyAllWindows()

    logger.info(f"Processed {total_frames} frames in {elapsed:.2f}s "
                f"({total_frames / elapsed:.2f} FPS)")

    # # Print final analysis
    # results = analyzer.get_results()
    # for item in results:
    #     logger.info(item)


if __name__ == "__main__":
    main()
