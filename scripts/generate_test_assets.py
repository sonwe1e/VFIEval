from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TEST_STYLE = ROOT / "videos" / "test_style"
EDGE_CASES = ROOT / "videos" / "test_edge_cases"
TEST_4K = ROOT / "videos" / "test_4k"


def main() -> None:
    TEST_STYLE.mkdir(parents=True, exist_ok=True)
    EDGE_CASES.mkdir(parents=True, exist_ok=True)
    TEST_4K.mkdir(parents=True, exist_ok=True)
    write_video(TEST_STYLE / "gradient_motion.avi", gradient_motion, frames=12, size=(96, 64), fourcc="MJPG")
    write_video(TEST_STYLE / "blocks_motion.avi", blocks_motion, frames=12, size=(96, 64), fourcc="MJPG")
    write_video(TEST_STYLE / "thin_lines_motion.mp4", thin_lines_motion, frames=12, size=(96, 64), fourcc="mp4v")
    write_video(TEST_STYLE / "occlusion_motion.avi", occlusion_motion, frames=12, size=(96, 64), fourcc="MJPG")
    write_video(TEST_STYLE / "large_displacement.mp4", large_displacement, frames=12, size=(96, 64), fourcc="mp4v")
    write_video(TEST_STYLE / "repeated_texture.avi", repeated_texture, frames=12, size=(96, 64), fourcc="MJPG")
    write_video(TEST_STYLE / "odd_resolution_中文 空格.avi", gradient_motion, frames=12, size=(95, 63), fourcc="MJPG")
    write_video(EDGE_CASES / "short_less_than_3_frames.avi", blocks_motion, frames=2, size=(64, 48), fourcc="MJPG")
    write_video(EDGE_CASES / "alternate_encoding.mp4", repeated_texture, frames=8, size=(80, 56), fourcc="mp4v")
    write_video(TEST_4K / "uhd_gradient_motion.mp4", gradient_motion, frames=6, size=(3840, 2160), fourcc="mp4v")


def write_video(path: Path, renderer, frames: int, size: tuple[int, int], fourcc: str) -> None:
    width, height = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), 8.0, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to create {path}")
    try:
        for index in range(frames):
            frame = renderer(index, frames, width, height)
            writer.write(frame)
    finally:
        writer.release()


def gradient_motion(index: int, frames: int, width: int, height: int) -> np.ndarray:
    x = np.linspace(0, 255, width, dtype=np.uint8)
    y = np.linspace(0, 120, height, dtype=np.uint8)
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :, 0] = np.roll(x, index * 3)
    frame[:, :, 1] = y[:, None]
    frame[:, :, 2] = 160
    return frame


def blocks_motion(index: int, frames: int, width: int, height: int) -> np.ndarray:
    frame = np.full((height, width, 3), (20, 35, 45), dtype=np.uint8)
    x = 4 + index * max(1, (width - 28) // max(1, frames - 1))
    y = height // 3
    cv2.rectangle(frame, (x, y), (min(width - 1, x + 20), min(height - 1, y + 18)), (210, 80, 40), -1)
    cv2.rectangle(frame, (width - x - 24, height - y - 20), (width - x - 4, height - y), (40, 170, 230), -1)
    return frame


def thin_lines_motion(index: int, frames: int, width: int, height: int) -> np.ndarray:
    frame = np.full((height, width, 3), (8, 8, 10), dtype=np.uint8)
    for offset in range(0, width, 10):
        x = (offset + index * 2) % width
        cv2.line(frame, (x, 0), (x, height - 1), (230, 230, 230), 1)
    cv2.line(frame, (0, (index * 3) % height), (width - 1, (index * 3) % height), (50, 190, 80), 1)
    return frame


def occlusion_motion(index: int, frames: int, width: int, height: int) -> np.ndarray:
    frame = repeated_texture(index, frames, width, height)
    x = width // 4 + index * 3
    cv2.circle(frame, (min(width - 1, x), height // 2), 14, (230, 230, 70), -1)
    cv2.rectangle(frame, (width // 2 - 8, 0), (width // 2 + 12, height), (30, 30, 35), -1)
    return frame


def large_displacement(index: int, frames: int, width: int, height: int) -> np.ndarray:
    frame = np.full((height, width, 3), (12, 18, 32), dtype=np.uint8)
    x = int((width - 18) * index / max(1, frames - 1))
    y = int((height - 18) * (frames - 1 - index) / max(1, frames - 1))
    cv2.rectangle(frame, (x, y), (min(width - 1, x + 18), min(height - 1, y + 18)), (70, 220, 130), -1)
    return frame


def repeated_texture(index: int, frames: int, width: int, height: int) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    tile = 8
    for y in range(0, height, tile):
        for x in range(0, width, tile):
            value = 80 if ((x // tile + y // tile + index) % 2) else 180
            frame[y : y + tile, x : x + tile] = (value, value // 2, 220 - value // 2)
    return frame


if __name__ == "__main__":
    main()
