import numpy as np
import cv2


class Track:
    """
    Represents a single track of a detected cell/object.
    """

    def __init__(self, initial_pos, track_id):
        """
        :param initial_pos: (x, y) tuple for the initial detection center
        :param track_id: integer track ID
        """
        self.track_id = track_id
        self.prediction = np.array(initial_pos, dtype=float)
        self.skipped_frames = 0
        self.trace = []
        self.done = False


class Tracker:
    """
    A minimal multiple-object tracker that associates new detections with existing tracks
    based on a simple distance threshold and keeps track of cell/object positions over time.
    """

    def __init__(
        self, 
        dist_thresh=50, 
        max_frames_to_skip=10, 
        max_trace_length=10, 
        debug=True
    ):
        """
        :param dist_thresh: Distance threshold for matching track to detection
        :param max_frames_to_skip: Maximum number of frames a track can go unmatched before removal
        :param max_trace_length: Maximum number of trace points to store for each track
        :param debug: Whether to enable debugging/visualization
        """
        self.dist_thresh = dist_thresh
        self.max_frames_to_skip = max_frames_to_skip
        self.max_trace_length = max_trace_length
        self.debug = debug

        self.tracks = []
        self.next_id = 0

        # For debug visualization
        self.colors = np.random.randint(0, 255, (1000, 3))

    def _log_debug(self, message: str, frame=None, window_name="Tracker Frame", pause_time=2):
        """Prints and/or shows debug info if self.debug is True."""
        if self.debug and message != "":
            print(f"[DEBUG minimal_tracker.py] {message}")

        if self.debug and frame is not None:
            if len(frame.shape) == 2:
                display_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                display_frame = frame
            cv2.imshow(window_name, display_frame)
            key = cv2.waitKey(pause_time)
            if key == 27:  # Esc to exit
                print("[DEBUG minimal_tracker.py] Esc pressed. Exiting debug visualization.")
                cv2.destroyAllWindows()
                exit()

    def update(self, detections, frame=None):
        """
        Matches new detections to existing tracks and updates them accordingly.
        
        :param detections: List of (x, y) center coordinates for each detection
        :param frame: Optional frame for debug visualization
        :return: List of tuples [(track_id, (x, y)), ...] for alive (not skipped too many times) tracks
        """
        # Sort detections by X, sort existing tracks by X
        detections = sorted(detections, key=lambda d: d[0])
        self.tracks.sort(key=lambda t: t.prediction[0])

        if self.debug:
            self._log_debug(f"Updating with {len(detections)} detection(s).")

        N = len(self.tracks)
        M = len(detections)
        assignment = [-1] * N

        # Match by index (naive matching)
        for i in range(min(N, M)):
            tx, ty = self.tracks[i].prediction
            dx, dy = detections[i]
            dist = np.sqrt((tx - dx)**2 + (ty - dy)**2)
            if dist < self.dist_thresh:
                assignment[i] = i
                if self.debug:
                    self._log_debug(
                        f"Matched track {self.tracks[i].track_id} with detection {i}, dist={dist:.2f}"
                    )

        # Increment skipped frames for unmatched tracks
        for i, a in enumerate(assignment):
            if a == -1:
                self.tracks[i].skipped_frames += 1
                if self.debug:
                    self._log_debug(
                        f"Track {self.tracks[i].track_id} missed (skipped_frames={self.tracks[i].skipped_frames})."
                    )

        # Remove tracks that have exceeded max_frames_to_skip
        before_remove = len(self.tracks)
        self.tracks = [t for t in self.tracks if t.skipped_frames <= self.max_frames_to_skip]
        after_remove = len(self.tracks)
        if self.debug and before_remove != after_remove:
            self._log_debug(f"Removed {before_remove - after_remove} dead track(s).")

        # Update matched tracks
        for i, a in enumerate(assignment):
            if a != -1:
                track = self.tracks[i]
                track.prediction = np.array(detections[a])
                track.skipped_frames = 0
                track.trace.append(track.prediction.copy())
                if len(track.trace) > self.max_trace_length:
                    track.trace.pop(0)

        # Handle unmatched detections by creating new tracks
        matched_dets = [a for a in assignment if a != -1]
        unmatched_dets = [detections[i] for i in range(M) if i not in matched_dets]
        for d in unmatched_dets:
            new_track = Track(d, self.next_id)
            self.next_id += 1
            new_track.trace.append(np.array(d))
            self.tracks.append(new_track)
            if self.debug:
                self._log_debug(f"Created new track {new_track.track_id} at {d}")

        # Draw debug info
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

            self._log_debug("", debug_frame, "Tracker Debug")

        # Return updated list of track IDs + positions
        return [(t.track_id, (t.prediction[0], t.prediction[1])) for t in self.tracks]

