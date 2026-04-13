"""Video export utilities for video generation models."""

import hashlib
import os
import time
from pathlib import Path


def save_and_emit_video_artifact(
    frames,
    fps: int,
    output_dir: str,
    emitter,
    *,
    index: int = 0,
    count: int = 1,
    metadata: dict | None = None,
    elapsed: float | None = None,
) -> str | None:
    """Export video frames as MP4 and emit artifact event.

    Args:
        frames: Video frames — either a list of PIL Images or a torch tensor
                of shape (num_frames, H, W, 3) with values in [0, 1].
        fps: Frame rate for the output video.
        output_dir: Directory to write the file into.
        emitter: EventEmitter for artifact/progress events.
        index: Video index within a batch (0-based).
        count: Total videos in the batch.
        metadata: Dict of metadata (stored in sidecar, not embedded in MP4).
        elapsed: Time taken for this video (used for ETA calculation).

    Returns:
        The output filepath, or None on failure.
    """
    os.makedirs(output_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"{timestamp}_{index:03d}.mp4" if count > 1 else f"{timestamp}.mp4"
    filepath = os.path.join(output_dir, filename)

    try:
        _export_frames_to_mp4(frames, filepath, fps)
    except Exception as exc:
        emitter.error(
            "VIDEO_EXPORT_FAILED",
            f"Failed to export video: {exc}",
            recoverable=False,
        )
        return None

    # Hash the output
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)

    size_bytes = os.path.getsize(filepath)

    emitter.artifact(path=filepath, sha256=sha256.hexdigest(), size_bytes=size_bytes)
    emitter.progress(
        stage="generate",
        step=index + 1,
        total_steps=count,
        eta_seconds=elapsed * (count - index - 1) if elapsed and count > 1 else None,
    )
    emitter.info(
        f"Video {index + 1}/{count}: {filepath}"
        + (f" ({elapsed:.1f}s)" if elapsed else "")
    )

    return filepath


def _export_frames_to_mp4(frames, output_path: str, fps: int) -> None:
    """Write video frames to an MP4 file using diffusers' export utility.

    Falls back to imageio if diffusers export is not available.
    """
    try:
        # diffusers provides export_to_video which handles torch tensors
        from diffusers.utils import export_to_video

        export_to_video(frames, output_path, fps=fps)
        return
    except ImportError:
        pass

    # Fallback: use imageio
    try:
        import imageio
        import numpy as np

        writer = imageio.get_writer(output_path, fps=fps, codec="libx264")
        for frame in frames:
            if hasattr(frame, "numpy"):
                # torch tensor
                arr = frame.cpu().numpy()
                if arr.max() <= 1.0:
                    arr = (arr * 255).clip(0, 255).astype(np.uint8)
            elif hasattr(frame, "convert"):
                # PIL Image
                import numpy as np
                arr = np.array(frame)
            else:
                arr = frame
            writer.append_data(arr)
        writer.close()
        return
    except ImportError:
        pass

    raise RuntimeError(
        "Cannot export video: neither diffusers.utils.export_to_video "
        "nor imageio is available. Install diffusers>=0.37 or imageio."
    )
