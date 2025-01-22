import logging
import cv2
import torch
import numpy as np
from typing import List, Tuple
import warnings

logger = logging.getLogger(__name__)

class YOLOv5CellDetector:
    """
    Wrapper around YOLOv5 for cell detection. 
    Crops/resizes frames, runs YOLO inference, and returns bounding boxes and centers.
    """

    def __init__(
        self,
        model_path: str,
        device: str = 'cuda',
        debug: bool = True,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_det: int = 1000
    ):
        """
        :param model_path: Path to the YOLOv5 .pt weights file.
        :param device: 'cuda' or 'cpu'.
        :param debug: Whether to log debug info.
        :param conf_threshold: Confidence threshold for YOLO.
        :param iou_threshold: Intersection-over-Union threshold for YOLO.
        :param max_det: Maximum number of detections.
        """
        self.debug = debug
        self.device = device
        # Add this line to suppress the specific warning
        warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda.amp.autocast")


        # Load YOLOv5 from torch.hub
        self.model = torch.hub.load(
            'ultralytics/yolov5',
            'custom',
            path=model_path,
            force_reload=False
        )
        self.model.to(self.device)
        self.model.conf = conf_threshold
        self.model.iou = iou_threshold
        self.model.max_det = max_det

        if self.debug:
            logger.debug(f"Loaded YOLOv5 model from {model_path} on {self.device}")

    def detect(
        self,
        frame: np.ndarray,
        crop_x_start: int,
        crop_x_end: int,
        resize_width: int,
        resize_height: int
    ) -> Tuple[List[Tuple[int, int]], List[List[int]], List[float]]:
        """
        Crop the input frame horizontally from crop_x_start to crop_x_end,
        resize to (resize_width, resize_height), and run YOLOv5 inference.
        
        :param frame: Original image/frame as a NumPy array.
        :param crop_x_start: Left boundary of the crop region.
        :param crop_x_end: Right boundary of the crop region.
        :param resize_width: Resized image width for YOLO inference.
        :param resize_height: Resized image height for YOLO inference.
        :return: A tuple of:
                 1. centers: List of (cx, cy) for each detection
                 2. boxes: List of bounding boxes [x, y, w, h]
                 3. weights: List of confidence scores
        """
        original_h, original_w = frame.shape[:2]

        cropped_frame = frame[:, crop_x_start:crop_x_end]
        resized_frame = cv2.resize(cropped_frame, (resize_width, resize_height))
        resized_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

        # YOLO inference
        results = self.model(resized_rgb, size=resize_width)
        if len(results.xyxy) == 0 or len(results.xyxy[0]) == 0:
            if self.debug:
                logger.debug("No YOLO detections found")
            return [], [], []

        detections = results.xyxy[0].cpu().numpy()

        scale_x = (crop_x_end - crop_x_start) / float(resize_width)
        scale_y = cropped_frame.shape[0] / float(resize_height)

        centers: List[Tuple[int, int]] = []
        boxes: List[List[int]] = []
        weights: List[float] = []

        for (x1, y1, x2, y2, conf, cls) in detections:
            w = x2 - x1
            h = y2 - y1

            # Map back to original coords
            x_original = int(x1 * scale_x) + crop_x_start
            y_original = int(y1 * scale_y)
            w_original = int(w * scale_x)
            h_original = int(h * scale_y)

            cx = x_original + w_original // 2
            cy = y_original + h_original // 2

            centers.append((cx, cy))
            boxes.append([x_original, y_original, w_original, h_original])
            weights.append(float(conf))

        if self.debug:
            logger.debug(
                f"Detected {len(centers)} object(s) with YOLOv5 in the cropped/resized image."
            )

        return centers, boxes, weights

    def update(
        self,
        centers: List[Tuple[int, int]],
        boxes: List[List[int]],
        weights: List[float]
    ) -> Tuple[Tuple[int, int], List[int]]:
        """
        Optional example method to 'fuse' multiple detections into a single bounding box/center.
        Not used in main code but can be adapted as needed.

        :param centers: List of detection centers.
        :param boxes: List of bounding boxes.
        :param weights: List of confidence scores.
        :return: A fused (cx, cy) and a single bounding box.
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
