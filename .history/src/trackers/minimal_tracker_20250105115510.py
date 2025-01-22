import logging
import numpy as np
import cv2
from typing import List, Tuple

logger = logging.getLogger(__name__)

class Track:
    """
    Represents a single track of a detected cell/object.
    """

    def __init__(self, initial_pos: Tuple[float, float], track_id: int):
        """
        Initialize the Track object.

        :param initial_pos: (x, y) initial center coordinates of the detection.
        :param track_id: Unique integer ID for this track.
        """
        self.track_id = track_id
        self.prediction = np.array(initial_pos, dtype=float)
        self.skipped_frames = 0
        self.trace = []
        self.done = False


class Tracker:
    """
    A minimal multiple-object tracker associating new detections with existing tracks
    based on a simple distance threshold. Tracks are removed if they skip too many frames.
    """

    def __init__(
        self,
        dist_thresh: float = 50.0,
        max_frames_to_skip: int = 10,
        max_trace_length: int = 10,
        debug: bool = True
    ):
        """
        Initialize the Tracker.

        :param dist_thresh: Distance threshold for matching a track with a detection.
        :param max_frames_to_skip: Max consecutive frames a track can be unmatched before removal.
        :param max_trace_length: Maximum length of the track's coordinate trace.
        :param debug: Whether to log debug statements.
        """
        self.dist_thresh = dist_thresh
        self.max_frames_to_skip = max_frames_to_skip
        self.max_trace_length = max_trace_length
        self.debug = debug

        self.tracks: List[Track] = []
        self.next_id = 0

        # For color-based visualization
        self.colors = np.random.randint(0, 255, (1000, 3))

    def update(
        self,
        detections: List[Tuple[float, float]],
        frame: np.ndarray = None
    ) -> List[Tuple[int, Tuple[float, float]]]:
        """
        Match new detections to existing tracks and update them.

        :param detections: List of (x, y) detection center coordinates.
        :param frame: Optional frame for visualization (if in debug mode).
        :return: List of (track_id, (x, y)) for each active track.
        """
        if self.debug:
            logger.debug(f"Updating with {len(detections)} detection(s).")

        # Sort by X for naive matching
        detections = sorted(detections, key=lambda d: d[0])
        self.tracks.sort(key=lambda t: t.prediction[0])

        N = len(self.tracks)
        M = len(detections)
        assignment = [-1] * N

        # Naive 1-to-1 assignment by index
        for i in range(min(N, M)):
            tx, ty = self.tracks[i].prediction
            dx, dy = detections[i]
            dist = np.sqrt((tx - dx)**2 + (ty - dy)**2)
            if dist < self.dist_thresh:
                assignment[i] = i
                if self.debug:
                    logger.debug(
                        f"Matched track {self.tracks[i].track_id} with detection {i}, dist={dist:.2f}"
                    )

        # Increment skipped_frames for unmatched tracks
        for i, a in enumerate(assignment):
            if a == -1:
                self.tracks[i].skipped_frames += 1
                if self.debug:
                    logger.debug(
                        f"Track {self.tracks[i].track_id} missed (skipped_frames={self.tracks[i].skipped_frames})."
                    )

        # Remove dead tracks
        before_remove = len(self.tracks)
        self.tracks = [t for t in self.tracks if t.skipped_frames <= self.max_frames_to_skip]
        after_remove = len(self.tracks)
        removed_count = before_remove - after_remove
        if self.debug and removed_count > 0:
            logger.debug(f"Removed {removed_count} dead track(s).")

        # Update matched tracks
        for i, a in enumerate(assignment):
            if a != -1:
                track = self.tracks[i]
                track.prediction = np.array(detections[a])
                track.skipped_frames = 0
                track.trace.append(track.prediction.copy())
                if len(track.trace) > self.max_trace_length:
                    track.trace.pop(0)

        # Handle unmatched detections
        matched_dets = [a for a in assignment if a != -1]
        unmatched_dets = [detections[i] for i in range(M) if i not in matched_dets]
        for d in unmatched_dets:
            new_track = Track(d, self.next_id)
            self.next_id += 1
            new_track.trace.append(np.array(d))
            self.tracks.append(new_track)
            if self.debug:
                logger.debug(f"Created new track {new_track.track_id} at {d}")

        # Debug visualization (optional)
        if self.debug and frame is not None:
            debug_frame = frame.copy()
            for track in self.tracks:
                color = self.colors[track.track_id % len(self.colors)]
                color = (int(color[0]), int(color[1]), int(color[2]))

                # Draw trace
                if len(track.trace) > 1:
                    for t_idx in range(len(track.trace) - 1):
                        pt1 = (int(track.trace[t_idx][0]), int(track.trace[t_idx][1]))
                        pt2 = (int(track.trace[t_idx+1][0]), int(track.trace[t_idx+1][1]))
                        cv2.line(debug_frame, pt1, pt2, color, 2)

                # Draw current position and ID
                curr_pos = (int(track.prediction[0]), int(track.prediction[1]))
                cv2.circle(debug_frame, curr_pos, 5, color, -1)
                cv2.putText(debug_frame, f"ID:{track.track_id}",
                            (curr_pos[0] + 10, curr_pos[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv2.imshow("Tracker Debug", debug_frame)
            key = cv2.waitKey(2)
            if key == 27:
                logger.info("Esc pressed. Closing visualization.")
                cv2.destroyAllWindows()
                exit()

        # Return updated (track_id, (x, y)) for alive tracks
        return [(t.track_id, (t.prediction[0], t.prediction[1])) for t in self.tracks]
