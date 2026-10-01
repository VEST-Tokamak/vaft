"""The private PyAV video encoder behind ``Animation.save`` (issue #1050).

It knows how to encode frames and nothing about what they mean: frames
arrive already rendered, on the scientific renderer's colour scale, with the
builder's own annotations.  ``av`` is imported only when a ``require_av``
call asks for it, so ``import vaft.plot`` never needs the ``video`` extra.
"""

from __future__ import annotations

from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

__all__ = ["VIDEO_CODECS", "encode_video", "require_av"]

#: The codecs tried for each container, in order: the first the local FFmpeg
#: build can encode is used.  Every one of them takes ``yuv420p``.
VIDEO_CODECS = {
    ".mp4": ("libx264", "h264", "mpeg4"),
    ".webm": ("libvpx-vp9", "libvpx"),
}


def require_av() -> Any:
    """``av``, or an ImportError naming ``vaft[video]``."""
    try:
        import av
    except ImportError as error:
        raise ImportError(
            "Writing .mp4/.webm needs the PyAV package, which is optional; "
            "install it with `pip install vaft[video]` (or `pip install av`). "
            "A .gif needs no extra."
        ) from error
    return av


def _codec(av: Any, suffix: str, codecs: tuple[str, ...] | None = None) -> str:
    candidates = VIDEO_CODECS[suffix] if codecs is None else codecs
    for name in candidates:
        try:
            av.codec.Codec(name, "w")
        except Exception:  # UnknownCodecError, or a build without the encoder
            continue
        return name
    raise RuntimeError(
        f"this FFmpeg build (PyAV {av.__version__}) has none of the {suffix} encoders "
        f"{', '.join(candidates)}; write a .gif, or install a PyAV wheel that has them"
    )


def even_frame(frame: np.ndarray) -> np.ndarray:
    """``frame`` padded to even width and height by repeating its last row/column.

    ``yuv420p`` subsamples chroma by two, so an odd dimension cannot be
    encoded.  The pad is one edge pixel, deterministic, and never a resize.
    """
    rows, columns = frame.shape[:2]
    pad = ((0, rows % 2), (0, columns % 2), (0, 0))
    return np.pad(frame, pad, mode="edge") if any(p[1] for p in pad[:2]) else frame


@contextmanager
def _encoding(path: Path, codec: str) -> Iterator[None]:
    try:
        yield
    except Exception as error:
        raise RuntimeError(f"encoding {path.name} with {codec} failed: {error}") from error


def encode_video(
    frames: Iterable[np.ndarray], path: Path, *, fps: float, codecs: tuple[str, ...] | None = None,
    name: str | None = None,
) -> dict[str, Any]:
    """Encode ``uint8`` RGB ``frames`` to ``path`` at ``fps``; return the encoder facts.

    Frames are consumed one at a time.  Frame ``i`` is presented at
    ``i / fps`` seconds; nothing is dropped, repeated or interpolated.
    ``codecs`` narrows the candidates of :data:`VIDEO_CODECS` (the notebook
    preview accepts only H.264, the one every browser plays in an .mp4).
    ``name`` is the file the caller asked for, when ``path`` is a temporary
    file beside it; error messages name it.
    """
    av = require_av()
    suffix = path.suffix.lower()
    codec = _codec(av, suffix, codecs)
    label = Path(name) if name else path
    rate = Fraction(fps).limit_denominator(1000) or Fraction(fps).limit_denominator()
    frames = iter(frames)
    count = 0
    # Only the encoder's own failures are rewrapped: an error raised while a
    # frame is drawn reaches the caller as it was raised.
    with _encoding(label, codec):
        container = av.open(str(path), mode="w")
    try:
        stream = None
        for frame in frames:
            frame = even_frame(frame)
            with _encoding(label, codec):
                if stream is None:
                    stream = container.add_stream(codec, rate=rate)
                    stream.height, stream.width = frame.shape[:2]
                    stream.pix_fmt = "yuv420p"
                    stream.codec_context.time_base = 1 / rate
                picture = av.VideoFrame.from_ndarray(np.ascontiguousarray(frame), format="rgb24")
                picture.pts = count
                picture.time_base = 1 / rate
                for packet in stream.encode(picture):
                    container.mux(packet)
            count += 1
        with _encoding(label, codec):
            if stream is not None:
                for packet in stream.encode():
                    container.mux(packet)
    except BaseException:
        try:
            container.close()
        except Exception:  # the error already in flight is the one to report
            pass
        raise
    with _encoding(label, codec):
        container.close()
    return {"container": suffix.lstrip("."), "codec": codec, "pix_fmt": "yuv420p",
            "av_version": av.__version__, "frames": count}
