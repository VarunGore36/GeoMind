"""Vision Engine storage utilities - reads frames from camera pipeline storage."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..cameras.storage import FrameStorage


class VisionStorage:
    """Reads frames from the camera pipeline's storage."""

    def __init__(self, camera_storage: FrameStorage | None = None):
        self.camera_storage = camera_storage or FrameStorage()

    def get_latest_frame_path(self, camera_id: str) -> Optional[Path]:
        """Get path to the latest frame for a camera."""
        return self.camera_storage.get_latest_frame_path(camera_id)

    def read_frame(self, camera_id: str, frame_path: Path | None = None) -> Optional[bytes]:
        """Read a frame from disk."""
        if frame_path is None:
            frame_path = self.get_latest_frame_path(camera_id)
        if not frame_path or not frame_path.exists():
            return None
        try:
            with open(frame_path, "rb") as f:
                return f.read()
        except Exception:
            return None


vision_storage = VisionStorage()