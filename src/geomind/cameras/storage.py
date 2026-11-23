"""Frame storage with deduplication."""

from __future__ import annotations

import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .config import get_camera_config


class FrameStorage:
    def __init__(self, base_path: str | None = None):
        config = get_camera_config()
        self.base_path = Path(base_path or config.storage.path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.max_frames = config.storage.max_frames_per_camera
        self.deduplicate = config.storage.deduplicate
        self.similarity_threshold = config.storage.similarity_threshold

    def _camera_dir(self, camera_id: str) -> Path:
        cam_dir = self.base_path / camera_id
        cam_dir.mkdir(parents=True, exist_ok=True)
        return cam_dir

    def _generate_filename(self, camera_id: str, timestamp: datetime, frame_number: int) -> str:
        ts = timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]
        return f"{camera_id}_{ts}_{frame_number:06d}.jpg"

    def _compute_hash(self, image_data: bytes) -> str:
        return hashlib.sha256(image_data).hexdigest()

    def _compute_phash(self, image_data: bytes) -> str:
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return ""
        img = cv2.resize(img, (32, 32))
        dct = cv2.dct(np.float32(img))
        dct_low = dct[:8, :8]
        median = np.median(dct_low)
        return "".join("1" if x > median else "0" for x in dct_low.flatten())

    def _hamming_distance(self, hash1: str, hash2: str) -> int:
        if len(hash1) != len(hash2):
            return len(hash1)
        return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))

    def is_duplicate(self, camera_id: str, image_data: bytes, phash: str) -> bool:
        if not self.deduplicate or not phash:
            return False
        cam_dir = self._camera_dir(camera_id)
        frames = sorted(cam_dir.glob("*.jpg"))
        if not frames:
            return False
        last_frame = frames[-1]
        try:
            with open(last_frame, "rb") as f:
                last_data = f.read()
            last_phash = self._compute_phash(last_data)
            if not last_phash:
                return False
            distance = self._hamming_distance(phash, last_phash)
            similarity = 1 - (distance / len(phash))
            return similarity >= self.similarity_threshold
        except Exception:
            return False

    def save_frame(
        self,
        camera_id: str,
        image_data: bytes,
        timestamp: datetime,
        frame_number: int,
    ) -> tuple[str, int, int, int, str]:
        cam_dir = self._camera_dir(camera_id)
        filename = self._generate_filename(camera_id, timestamp, frame_number)
        file_path = cam_dir / filename

        with open(file_path, "wb") as f:
            f.write(image_data)

        file_size = len(image_data)
        content_hash = self._compute_hash(image_data)

        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        width = height = 0
        if img is not None:
            height, width = img.shape[:2]

        self._cleanup_old_frames(camera_id)

        return str(file_path), file_size, width, height, content_hash

    def _cleanup_old_frames(self, camera_id: str) -> int:
        cam_dir = self._camera_dir(camera_id)
        frames = sorted(cam_dir.glob("*.jpg"))
        if len(frames) <= self.max_frames:
            return 0
        to_delete = frames[:-self.max_frames]
        for frame in to_delete:
            frame.unlink(missing_ok=True)
        return len(to_delete)

    def get_latest_frame_path(self, camera_id: str) -> Optional[Path]:
        cam_dir = self._camera_dir(camera_id)
        frames = sorted(cam_dir.glob("*.jpg"))
        return frames[-1] if frames else None


storage = FrameStorage()