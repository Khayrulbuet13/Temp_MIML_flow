import logging
import numpy as np
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class CellAnalyze:
    """
    Handles per-track analysis of cells, including:
    - Max DI (Deformation Index)
    - Start/end frames for crossing region of interest
    - Maximum velocity
    - Transition time
    """

    def __init__(
        self, 
        traceStart: float, 
        traceEnd: float, 
        fps: float = 20.0, 
        debug: bool = True
    ):
        """
        :param traceStart: X coordinate to start analysis.
        :param traceEnd: X coordinate to end analysis.
        :param fps: Frame rate (FPS) used for time-based metrics.
        :param debug: Whether to log debug messages.
        """
        self.traceStart = traceStart * 1.2
        self.traceEnd = traceEnd * 0.8
        self.fps = fps
        self.debug = debug

        # Per-track storage:
        # track_data[track_id] = {
        #   "max_DI": float or None,
        #   "start_frame_index": int or None,
        #   "end_frame_index": int or None,
        #   "prev_center": (x, y) or None,
        #   "max_velocity": float or None,
        #   "transition_time": float or None,
        # }
        self.track_data: Dict[int, Dict[str, Any]] = {}

    def _ensure_track(self, track_id: int) -> None:
        """Initialize a dict entry for the given track if it doesn't exist."""
        if track_id not in self.track_data:
            self.track_data[track_id] = {
                "max_DI": None,
                "start_frame_index": None,
                "end_frame_index": None,
                "prev_center": None,
                "max_velocity": None
            }

    def update_properties(
        self,
        track_id: int,
        detection: tuple,
        box: list,
        frame_ctr: int
    ) -> None:
        """
        Update analysis metrics (DI, velocity, start/end frame, transition time) for a given track.

        :param track_id: The track ID.
        :param detection: (x, y) center coordinates of the detection.
        :param box: [x, y, w, h] bounding box or None.
        :param frame_ctr: Current frame index in the video/processing.
        """
        self._ensure_track(track_id)
        data = self.track_data[track_id]

        # Update Deformation Index
        if box is not None:
            x, y, w, h = box
            max_radius = max(w, h) / 2.0
            min_radius = min(w, h) / 2.0
            denominator = (max_radius + min_radius)
            if denominator > 0:
                DI = (max_radius - min_radius) / denominator
                if data["max_DI"] is None or DI > data["max_DI"]:
                    data["max_DI"] = DI
                    if self.debug:
                        logger.debug(f"Track {track_id} max_DI updated to {DI:.2f}")

        # Mark start frame
        if data["start_frame_index"] is None and detection[0] >= self.traceStart:
            data["start_frame_index"] = frame_ctr
            if self.debug:
                logger.debug(f"Track {track_id} started at frame {frame_ctr}")

        # Mark end frame
        if (data["start_frame_index"] is not None and 
            data["end_frame_index"] is None and 
            detection[0] >= self.traceEnd):
            data["end_frame_index"] = frame_ctr
            if self.debug:
                logger.debug(f"Track {track_id} ended at frame {frame_ctr}")

        # Compute transition_time if we have both start/end
        if (data["start_frame_index"] is not None and
            data["end_frame_index"] is not None and
            "transition_time" not in data):
            transition_frames = data["end_frame_index"] - data["start_frame_index"]
            if transition_frames > 0:
                data["transition_time"] = transition_frames / self.fps
                if self.debug:
                    logger.debug(
                        f"Track {track_id} transition_time={data['transition_time']:.2f}s"
                    )

        # Update velocity
        prev_center = data["prev_center"]
        if prev_center is not None:
            dist = np.linalg.norm(np.array(detection) - np.array(prev_center))
            if data["max_velocity"] is None or dist > data["max_velocity"]:
                data["max_velocity"] = dist
                if self.debug:
                    logger.debug(f"Track {track_id} max_velocity updated to {dist:.2f}")

        data["prev_center"] = detection

    def get_results(self) -> List[dict]:
        """
        Return a list of dicts with final analysis for each track.

        :return: List of dictionaries with:
            "track_id", "max_DI", "transition_frames", 
            "transition_time", "max_velocity".
        """
        results = []
        for track_id, data in self.track_data.items():
            if self.debug:
                logger.debug(f"Finalizing track {track_id} -> {data}")

            transition_frames = None
            transition_time = None
            if (data["start_frame_index"] is not None and 
                data["end_frame_index"] is not None):
                transition_frames = data["end_frame_index"] - data["start_frame_index"]
                if "transition_time" in data:
                    transition_time = data["transition_time"]
                elif transition_frames > 0:
                    transition_time = transition_frames / self.fps

            results.append({
                "track_id": track_id,
                "max_DI": data["max_DI"],
                "transition_frames": transition_frames,
                "transition_time": transition_time,
                "max_velocity": data["max_velocity"]
            })
        return results
