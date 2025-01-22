import cv2
import torch
import numpy as np


class YOLOv5CellDetector:
    """
    A wrapper around YOLOv5 for cell detection. 
    It crops, resizes, and runs inference with a YOLOv5 model.
    """

    def __init__(
        self,
        model_path,
        device='cuda',
        debug=True,
        conf_threshold=0.25,
        iou_threshold=0.45,
        max_det=1000
    ):
        """
        :param model_path: Path to the YOLOv5 .pt model
        :param device: 'cuda' or 'cpu'
        :param debug: Whether to show debug messages / windows
        :param conf_threshold: Confidence threshold for YOLO
        :param iou_threshold: IOU threshold for YOLO
        :param max_det: Max detections
        """
        self.debug = debug
        self.device = device

        # Load YOLOv5 from torch.hub
        self.model = torch.hub.load(
            'ultralytics/yolov5',
            'custom',
            path=model_path,
            force_reload=False
        )
        self.model.to(device)
        self.model.conf = conf_threshold
        self.model.iou = iou_threshold
        self.model.max_det = max_det

        self._log_debug(f"Loaded YOLOv5 model from {model_path} on {device}")

    def _log_debug(self, message: str, frame=None, window_name="Debug Frame", pause_time=1):
        """Print or visualize debug info if enabled."""
        if self.debug and message != "":
            print(f"[DEBUG yolo_detect.py] {message}")

        if self.debug and frame is not None:
            if len(frame.shape) == 2:
                display_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                display_frame = frame
            cv2.imshow(window_name, display_frame)
            key = cv2.waitKey(pause_time)
            if key == 27:  # Esc key
                print("[DEBUG yolo_detect.py] Esc pressed. Exiting debug visualization.")
                cv2.destroyAllWindows()
                exit()

    def detect(
        self,
        frame: np.ndarray,
        crop_x_start: int,
        crop_x_end: int,
        resize_width: int,
        resize_height: int
    ):
        """
        Crops the input frame horizontally from crop_x_start to crop_x_end,
        resizes to (resize_width, resize_height), 
        and runs YOLOv5 inference. 
        Returns centers, boxes, and weights for each detection.
        """
        original_h, original_w = frame.shape[:2]

        cropped_frame = frame[:, crop_x_start:crop_x_end]
        resized_frame = cv2.resize(cropped_frame, (resize_width, resize_height))
        resized_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

        # YOLOv5 inference
        results = self.model(resized_rgb, size=resize_width)
        if len(results.xyxy) == 0 or len(results.xyxy[0]) == 0:
            self._log_debug("No YOLO detections found")
            return [], [], []

        detections = results.xyxy[0].cpu().numpy()
        scale_x = (crop_x_end - crop_x_start) / float(resize_width)
        scale_y = cropped_frame.shape[0] / float(resize_height)

        centers, boxes, weights = [], [], []

        # For debug overlay
        debug_overlay = resized_frame.copy() if self.debug else None

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
                # Draw bounding box
                cv2.rectangle(debug_overlay, (int(x1), int(y1)), 
                              (int(x2), int(y2)), (0, 255, 0), 2)
                # Draw center
                cv2.circle(debug_overlay, 
                           (int(x1 + w/2), int(y1 + h/2)), 
                           3, (0, 0, 255), -1)
                # Draw confidence
                cv2.putText(
                    debug_overlay,
                    f"{conf:.2f}",
                    (int(x1), int(y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, 
                    (255, 0, 0), 
                    1
                )

        if self.debug:
            self._log_debug(
                f"Detected {len(centers)} object(s) with YOLOv5",
                debug_overlay,
                "YOLOv5 Detections"
            )

        return centers, boxes, weights

    def update(self, centers, boxes, weights):
        """
        Example method that could 'fuse' multiple detections into a single bounding box/center.
        Not used in main code but might be extended for advanced usage.
        """
        if not centers:
            self._log_debug("No detections to fuse")
            return None, None

        cx = sum([c[0] for c in centers]) / len(centers)
        cy = sum([c[1] for c in centers]) / len(centers)
        fused_center = (int(cx), int(cy))

        max_idx = np.argmax(weights)
        fused_box = boxes[max_idx]

        if self.debug and boxes:
            debug_frame_h = max([b[1] + b[3] for b in boxes]) + 50
            debug_frame_w = max([b[0] + b[2] for b in boxes]) + 50
            debug_frame = np.zeros((debug_frame_h, debug_frame_w, 3), dtype=np.uint8)

            # Draw all boxes
            for box in boxes:
                cv2.rectangle(
                    debug_frame, 
                    (box[0], box[1]),
                    (box[0]+box[2], box[1]+box[3]),
                    (128, 128, 128), 
                    1
                )

            # Highlight fused box
            cv2.rectangle(
                debug_frame, 
                (fused_box[0], fused_box[1]),
                (fused_box[0]+fused_box[2], fused_box[1]+fused_box[3]),
                (0, 255, 0), 
                2
            )
            # Fused center
            cv2.circle(debug_frame, fused_center, 5, (0,0,255), -1)

            self._log_debug("Fused YOLO detection", debug_frame, "Fused YOLOv5")

        return fused_center, fused_box
