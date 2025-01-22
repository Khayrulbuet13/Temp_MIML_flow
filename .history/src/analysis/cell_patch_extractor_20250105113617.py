import cv2
import numpy as np


class CellPatchExtractor:
    """
    Crops and stores 64x64 cell patches once the cell crosses (trace_start + gap).
    After the cell crosses trace_end, the stored patch (if any) is retrieved for MIML.
    """

    def __init__(
        self, 
        trace_start, 
        trace_end, 
        gap=20, 
        patch_size=64, 
        debug=False
    ):
        """
        :param trace_start: X coordinate of the region start
        :param trace_end: X coordinate of the region end
        :param gap: Additional gap after crossing trace_start before we crop
        :param patch_size: Size of the patch (square) to crop
        :param debug: Whether to print debug info
        """
        self.trace_start = trace_start
        self.trace_end = trace_end
        self.gap = gap
        self.patch_size = patch_size
        self.debug = debug

        # Dictionary storing patches for each track_id => single patch or multiple
        self.track_patches = {}

    def _log_debug(self, msg):
        if self.debug:
            print(f"[DEBUG cell_patch_extractor.py] {msg}")

    def update_patch(self, track_id, frame, center_xy):
        """
        Attempt to crop a patch around the cell center_xy if 
        the cell has crossed (trace_start + gap).

        :param track_id: Track ID of the cell
        :param frame: Current video frame
        :param center_xy: (x, y) center of the cell
        """
        x_center, y_center = center_xy
        x_center = int(round(x_center))
        y_center = int(round(y_center))

        # If we've already stored a patch for this track, skip
        if track_id in self.track_patches:
            return

        if x_center < (self.trace_start + self.gap):
            return

        half = self.patch_size // 2

        x1 = max(0, x_center - half)
        y1 = max(0, y_center - half)
        x2 = x1 + self.patch_size
        y2 = y1 + self.patch_size

        h, w = frame.shape[:2]
        if x2 > w:
            diff = x2 - w
            x1 -= diff
            x2 -= diff
        if y2 > h:
            diff = y2 - h
            y1 -= diff
            y2 -= diff

        patch = frame[y1:y2, x1:x2].copy()
        self.track_patches[track_id] = patch
        self._log_debug(f"Track {track_id} patch cropped at x=[{x1}:{x2}], y=[{y1}:{y2}]")

    def finalize(self, track_id, cell_properties):
        """
        Once the cell crosses trace_end, retrieve the patch (if any) plus the cell properties.
        Return them so they can be fed to the MIML model or saved for further analysis.

        :param track_id: Track ID
        :param cell_properties: Dict of cell properties (from CellAnalyze)
        :return: (patch, selected_cell_properties) or None if no patch found
        """
        patch = self.track_patches.get(track_id, None)
        if patch is None:
            self._log_debug(f"Track {track_id} has no stored patch.")
            return None

        # Example: reduce cell_properties to only those needed by MIML
        MIML_cell_properties = {
            'max_DI': cell_properties.get('max_DI', None),
            'max_velocity': cell_properties.get('max_velocity', None),
            'transition_time': cell_properties.get('transition_time', None)
        }

        print(f"[DEBUG cell_patch_extractor.py] cell_properties {cell_properties} patch finalized.")
        return patch, MIML_cell_properties
