import logging
import cv2
import numpy as np
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

class CellPatchExtractor:
    """
    Crops and stores 64x64 cell patches once the cell crosses (trace_start + gap).
    After the cell crosses trace_end, the stored patch (if any) can be retrieved
    for passing to a MIML model or further processing.
    """

    def __init__(
        self,
        trace_start: float,
        trace_end: float,
        gap: int = 20,
        patch_size: int = 64,
        debug: bool = False
    ):
        """
        :param trace_start: X coordinate of the region start.
        :param trace_end: X coordinate of the region end.
        :param gap: Additional gap after crossing trace_start before we crop.
        :param patch_size: Square size of the patch to crop.
        :param debug: Whether to log debug information.
        """
        self.trace_start = trace_start
        self.trace_end = trace_end
        self.gap = gap
        self.patch_size = patch_size
        self.debug = debug

        # Dictionary to store patches for each track_id
        self.track_patches: Dict[int, np.ndarray] = {}

    def update_patch(
        self,
        track_id: int,
        frame: np.ndarray,
        center_xy: Tuple[float, float]
    ) -> None:
        """
        Attempt to crop and store a patch around the cell center_xy once
        it crosses (trace_start + gap).

        :param track_id: ID of the track.
        :param frame: Current frame from which to crop.
        :param center_xy: (x, y) center of the cell in the frame.
        """
        x_center, y_center = center_xy
        x_center = int(round(x_center))
        y_center = int(round(y_center))

        # If we already have a patch for this track, skip
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
        if self.debug:
            logger.debug(
                f"Track {track_id} patch cropped at x=[{x1}:{x2}], y=[{y1}:{y2}]"
            )

    def finalize(
        self,
        track_id: int,
        cell_properties: dict
    ) -> Optional[Tuple[np.ndarray, dict]]:
        """
        Once the cell crosses trace_end, retrieve the patch (if any)
        plus the cell properties for MIML or other usage.

        :param track_id: Track ID being finalized.
        :param cell_properties: Dictionary of cell properties (e.g., from CellAnalyze).
        :return: (patch, selected_cell_properties) or None if no patch exists.
        """
        patch = self.track_patches.get(track_id, None)
        if patch is None:
            if self.debug:
                logger.debug(f"Track {track_id} has no stored patch.")
            return None

        logger.debug(f"Track {track_id} finalized with properties {cell_properties}.")

        # Example: reduce to the keys we need for MIML
        miml_cell_properties = {
            'max_DI': cell_properties.get('max_DI'),
            'max_velocity': cell_properties.get('max_velocity'),
            'transition_time': cell_properties.get('transition_time')
        }

        return patch, miml_cell_properties
