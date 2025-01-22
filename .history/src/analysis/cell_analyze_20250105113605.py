import numpy as np


class CellAnalyze:
    """
    Handles per-track analysis of cells: 
    - Max DI (Deformation Index)
    - Start/end frames for crossing region of interest
    - Maximum velocity
    - Computation of transition_time
    """

    def __init__(self, traceStart, traceEnd, fps=20.0, debug=True):
        """
        :param traceStart: X coordinate to start analysis
        :param traceEnd: X coordinate to end analysis
        :param fps: Frame rate for computing time-based metrics
        :param debug: Whether to log debug messages
        """
        self.traceStart = traceStart * 1.2
        self.traceEnd = traceEnd * 0.8
        self.fps = fps
        self.debug = debug

        # Dictionary for track data
        # track_data[track_id] = {
        #   "max_DI": float or None
        #   "start_frame_index": int or None
        #   "end_frame_index": int or None
        #   "prev_center": (x, y) or None
        #   "max_velocity": float or None
        #   "transition_time": float or None
        # }
        self.track_data = {}

    def _ensure_track(self, track_id):
        """Initialize track_data entry if not present."""
        if track_id not in self.track_data:
            self.track_data[track_id] = {
                "max_DI": None,
                "start_frame_index": None,
                "end_frame_index": None,
                "prev_center": None,
                "max_velocity": None
                # "transition_time" will be added after we have both start/end frames
            }

    def _log_debug(self, msg):
        if self.debug:
            print(f"[DEBUG cell_analyze.py] {msg}")

    def update_properties(self, track_id, detection, box, frame_ctr):
        """
        Update analysis metrics (DI, velocity, start/end frame, etc.) for a given track.

        :param track_id: integer ID of the track
        :param detection: (x, y) center position
        :param box: bounding box [x, y, w, h] or None
        :param frame_ctr: current frame index
        """
        self._ensure_track(track_id)
        data = self.track_data[track_id]

        # Update Deformation Index
        if box is not None:
            x, y, w, h = box
            max_radius = max(w, h) / 2.0
            min_radius = min(w, h) / 2.0
            if max_radius + min_radius > 0:
                DI = (max_radius - min_radius) / (max_radius + min_radius)
                if data["max_DI"] is None or DI > data["max_DI"]:
                    data["max_DI"] = DI
                    self._log_debug(f"Track {track_id} max_DI updated to {DI:.2f}")

        # Mark start frame index once it crosses traceStart
        if data["start_frame_index"] is None and detection[0] >= self.traceStart:
            data["start_frame_index"] = frame_ctr
            self._log_debug(f"Track {track_id} started at frame {frame_ctr}")

        # Mark end frame index once it crosses traceEnd
        if (data["start_frame_index"] is not None
                and data["end_frame_index"] is None
                and detection[0] >= self.traceEnd):
            data["end_frame_index"] = frame_ctr
            self._log_debug(f"Track {track_id} ended at frame {frame_ctr}")

        # Compute transition_time if we have both start/end frames
        if (data["start_frame_index"] is not None and 
                data["end_frame_index"] is not None and 
                "transition_time" not in data):
            transition_frames = data["end_frame_index"] - data["start_frame_index"]
            if transition_frames > 0:
                data["transition_time"] = transition_frames / self.fps
                self._log_debug(
                    f"Track {track_id} transition_time={data['transition_time']:.2f}s"
                )

        # Update max velocity
        prev_center = data["prev_center"]
        if prev_center is not None:
            dist = np.linalg.norm(np.array(detection) - np.array(prev_center))
            if data["max_velocity"] is None or dist > data["max_velocity"]:
                data["max_velocity"] = dist
                self._log_debug(f"Track {track_id} max_velocity updated to {dist:.2f}")

        data["prev_center"] = detection

    def get_results(self):
        """
        Return a list of dicts with the final analysis results for each track:
        [
            {
                "track_id": int,
                "max_DI": float or None,
                "transition_frames": int or None,
                "transition_time": float or None,
                "max_velocity": float or None
            },
            ...
        ]
        """
        results = []
        for track_id, data in self.track_data.items():
            self._log_debug(f"Finalizing track {track_id} -> {data}")

            transition_frames = None
            transition_time = None

            if (data["start_frame_index"] is not None and 
                    data["end_frame_index"] is not None):
                # If transition_time wasn't computed, compute here
                if "transition_time" not in data:
                    transition_frames = data["end_frame_index"] - data["start_frame_index"]
                    if transition_frames > 0:
                        transition_time = transition_frames / self.fps
                else:
                    # Use the already-computed transition_time
                    transition_frames = data["end_frame_index"] - data["start_frame_index"]
                    transition_time = data["transition_time"]

                if transition_frames is not None and transition_time is not None:
                    self._log_debug(
                        f"Track {track_id} transition_frames={transition_frames}, time={transition_time:.2f}s"
                    )

            results.append({
                "track_id": track_id,
                "max_DI": data["max_DI"],
                "transition_frames": transition_frames,
                "transition_time": transition_time or data.get("transition_time"),
                "max_velocity": data["max_velocity"],
            })
        return results
