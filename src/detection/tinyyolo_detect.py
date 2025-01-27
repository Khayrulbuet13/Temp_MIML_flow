import logging
import cv2
import torch
import numpy as np
from typing import List, Tuple
from src.models.TinyYolo import TinyYOLO

# To suppress warnings
import warnings
import os
os.environ['PYTHONWARNINGS'] = 'ignore::FutureWarning'
warnings.filterwarnings('ignore', category=FutureWarning)

logger = logging.getLogger(__name__)

class TinyYOLOCellDetector:
    """
    Wrapper around our custom TinyYOLO for cell detection.
    Optimized for grayscale images with 256x128 input size.
    Maintains same interface as YOLOv5CellDetector for seamless integration.
    """

    def __init__(
        self,
        model_path: str,
        device: str = 'cuda',
        debug: bool = True,
        conf_threshold: float = 0.5,  # Default threshold matches training
        **kwargs  # Accept but ignore other YOLOv5 specific params
    ):
        """
        :param model_path: Path to the TinyYOLO weights file (.pth)
        :param device: 'cuda' or 'cpu'
        :param debug: Whether to log debug info
        :param conf_threshold: Confidence threshold for detections
        """
        self.debug = debug
        self.device = device
        self.conf_threshold = conf_threshold

        # Initialize our custom TinyYOLO model
        self.model = TinyYOLO()
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        self.model.eval()

        if self.debug:
            logger.debug(f"Loaded TinyYOLO model from {model_path} on {self.device}")

    def detect(
        self,
        frame: np.ndarray,
        crop_x_start: int,
        crop_x_end: int,
        resize_width: int,
        resize_height: int
    ) -> Tuple[List[Tuple[int, int]], List[List[int]], List[float]]:
        """
        Process frame and run TinyYOLO inference.
        Maintains exact same interface as YOLOv5CellDetector.detect().
        
        :param frame: Original grayscale image/frame
        :param crop_x_start: Left boundary of crop region
        :param crop_x_end: Right boundary of crop region
        :param resize_width: Width to resize to
        :param resize_height: Height to resize to
        :return: Tuple of (centers, boxes, weights) in same format as YOLOv5
        """
        # Crop frame
        cropped_frame = frame[:, crop_x_start:crop_x_end]
        
        # Resize to 128x256 (model's expected input size)
        resized_frame = cv2.resize(cropped_frame, (256, 128))
        
        # Normalize to [0,1] and prepare for PyTorch
        x = resized_frame.astype(np.float32) / 255.0
        x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        x = x.to(self.device)

        # Run inference
        with torch.no_grad():
            pred = self.model(x)  # Shape: (1, 7, 8, 8)
            pred = pred[0].cpu().numpy()  # Remove batch dim
        
        # Convert predictions to original image coordinates
        centers: List[Tuple[int, int]] = []
        boxes: List[List[int]] = []
        weights: List[float] = []
        
        grid_size_x = 32  # 256/8 = 32 pixels per grid cell (width)
        grid_size_y = 16  # 128/8 = 16 pixels per grid cell (height)
        scale_x = (crop_x_end - crop_x_start) / 256.0
        scale_y = cropped_frame.shape[0] / 128.0

        # Process each grid cell
        for y in range(8):  # height grid
            for x in range(8):  # width grid
                prob = pred[0, y, x]  # Channel 0: probability
                if prob < self.conf_threshold:
                    continue
                    
                # Get box coordinates (relative to grid cell)
                bx = pred[1, y, x]  # x offset within cell
                by = pred[2, y, x]  # y offset within cell
                bw = pred[3, y, x]  # width
                bh = pred[4, y, x]  # height
                
                # Convert to pixel coordinates in 256x128 space
                px = int((x * grid_size_x) + bx)
                py = int((y * grid_size_y) + by)
                pw = int(bw)
                ph = int(bh)
                
                # Scale back to original image coordinates
                x_orig = int(px * scale_x) + crop_x_start
                y_orig = int(py * scale_y)
                w_orig = int(pw * scale_x)
                h_orig = int(ph * scale_y)
                
                # Calculate center
                cx = x_orig + w_orig // 2
                cy = y_orig + h_orig // 2
                
                centers.append((cx, cy))
                boxes.append([x_orig, y_orig, w_orig, h_orig])
                weights.append(float(prob))

        if self.debug:
            logger.debug(
                f"Detected {len(centers)} object(s) with TinyYOLO"
            )

        return centers, boxes, weights

    def update(
        self,
        centers: List[Tuple[int, int]],
        boxes: List[List[int]],
        weights: List[float]
    ) -> Tuple[Tuple[int, int], List[int]]:
        """
        Optional fusion method, kept for interface compatibility.
        Same implementation as YOLOv5CellDetector.
        """
        if not centers:
            if self.debug:
                logger.debug("No detections to fuse.")
            return (0, 0), []

        cx = int(sum(c[0] for c in centers) / len(centers))
        cy = int(sum(c[1] for c in centers) / len(centers))
        fused_center = (cx, cy)

        max_idx = np.argmax(weights)
        fused_box = boxes[max_idx] if boxes else []

        if self.debug:
            logger.debug(f"Fused center: {fused_center}, Box: {fused_box}")
        return fused_center, fused_box
