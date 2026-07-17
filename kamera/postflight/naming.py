"""Parsing helpers for KAMERA file and directory naming conventions.

Image basenames look like::

    <effort...>_<flight>_<channel>_<date>_<time>_<modality>.<ext>
    e.g. test_seattle_2020_fl09_R_20200830_020748.058907_rgb.jpg

The effort prefix may itself contain underscores, so fields must be
parsed from the *end* of the name, never by absolute index.

Camera directories (the per-camera image folders used to organize a
colmap workspace, e.g. ``images0/<camera_dir>/<image>``) look like::

    <prefix...>_<channel>_<modality>
    e.g. 85mm_25_5deg_center_rgb

Never swap modalities with ``str.replace`` on a filename or path: the
modality string ("ir", "rgb", ...) can appear as a substring elsewhere in
the name. Parse, replace the field, and rebuild instead.
"""

import dataclasses
import os
import os.path as osp
from dataclasses import dataclass

__all__ = [
    "KameraImageName",
    "KameraCameraName",
    "swap_image_name_modality",
    "VIEW_BY_CHANNEL",
]

# Maps a single-letter machine/channel code to its view name.
VIEW_BY_CHANNEL = {"C": "center_view", "L": "left_view", "R": "right_view"}


@dataclass(frozen=True)
class KameraImageName:
    """Fields of a KAMERA image (or meta json) basename."""

    prefix: str  # effort/project prefix, may contain underscores
    flight: str  # e.g. "fl09"
    channel: str  # machine/channel code, e.g. "C", "L", "R"
    date: str  # e.g. "20200830"
    time: str  # e.g. "020748.058907"
    modality: str  # e.g. "rgb", "uv", "ir", "meta"
    ext: str  # extension without leading dot, e.g. "jpg", "json"

    @classmethod
    def parse(cls, fname: str | os.PathLike) -> "KameraImageName":
        base = osp.basename(str(fname))
        parts = base.split("_")
        if len(parts) < 5:
            raise ValueError(
                f"'{base}' does not look like a KAMERA image name "
                "(<effort>_<flight>_<channel>_<date>_<time>_<modality>.<ext>)"
            )
        modality, _, ext = parts[-1].partition(".")
        time = parts[-2]
        date = parts[-3]
        try:
            float(time)
            int(date)
        except ValueError:
            raise ValueError(
                f"'{base}' does not look like a KAMERA image name: expected "
                f"numeric <date>_<time> fields, got '{date}_{time}'"
            )
        return cls(
            prefix="_".join(parts[:-5]),
            flight=parts[-5],
            channel=parts[-4],
            date=date,
            time=time,
            modality=modality,
            ext=ext,
        )

    @property
    def base_name(self) -> str:
        """The name with modality and extension stripped.

        This is the key shared between an image and its `_meta.json`
        (matches the historical ``get_base_name`` behavior).
        """
        fields = [self.flight, self.channel, self.date, self.time]
        if self.prefix:
            fields.insert(0, self.prefix)
        return "_".join(fields)

    @property
    def name(self) -> str:
        """The full basename."""
        last = f"{self.modality}.{self.ext}" if self.ext else self.modality
        return f"{self.base_name}_{last}"

    @property
    def view(self) -> str:
        return VIEW_BY_CHANNEL.get(self.channel, "null")

    def with_modality(self, modality: str, ext: str | None = None) -> "KameraImageName":
        return dataclasses.replace(
            self, modality=modality, ext=self.ext if ext is None else ext
        )


@dataclass(frozen=True)
class KameraCameraName:
    """Fields of a per-camera image directory name."""

    prefix: str  # rig/effort prefix, may contain underscores
    channel: str  # e.g. "center", "left", "right"
    modality: str  # e.g. "rgb", "uv", "ir"

    @classmethod
    def parse(cls, name: str | os.PathLike) -> "KameraCameraName":
        base = osp.basename(str(name))
        parts = base.split("_")
        if len(parts) < 2:
            raise ValueError(
                f"'{base}' does not look like a KAMERA camera directory "
                "(<prefix>_<channel>_<modality>)"
            )
        return cls(
            prefix="_".join(parts[:-2]), channel=parts[-2], modality=parts[-1]
        )

    @property
    def name(self) -> str:
        fields = [self.channel, self.modality]
        if self.prefix:
            fields.insert(0, self.prefix)
        return "_".join(fields)

    def with_modality(self, modality: str) -> "KameraCameraName":
        return dataclasses.replace(self, modality=modality)


def swap_image_name_modality(image_name: str, modality: str) -> str:
    """Swap the modality of a colmap image name (``<camera_dir>/<basename>``).

    Both the camera directory and the basename carry the modality; both are
    swapped. Colmap image names always use "/" separators.
    """
    dirname, _, base = image_name.rpartition("/")
    new_base = KameraImageName.parse(base).with_modality(modality).name
    if not dirname:
        return new_base
    new_dir = KameraCameraName.parse(dirname).with_modality(modality).name
    return f"{new_dir}/{new_base}"
