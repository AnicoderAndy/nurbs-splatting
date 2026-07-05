"""
We use this script to generate our emoji dataset for Layer-Wise Image Vectorization.
Use the default settings to generate an identical dataset to the one we used in our paper.

Instructions:
1. Change to this directory.
2. Download Iconify JSON data by running:
    `npm install @iconify-json/noto-v1 --save-dev`
3. Run this script.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import cairosvg
from typing import Any, Dict, Optional, Tuple


def parse_size(value: str) -> Tuple[int, int]:
    raw = value.strip().lower().replace(" ", "")
    if "x" in raw:
        parts = raw.split("x", 1)
    elif "," in raw:
        parts = raw.split(",", 1)
    else:
        parts = [raw, raw]
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError("size must be like 512 or 512x512")
    return int(parts[0]), int(parts[1])


def resolve_icon(
    name: str, icons: Dict[str, Dict[str, Any]], stack: Optional[list] = None
) -> Dict[str, Any]:
    if stack is None:
        stack = []
    if name in stack:
        raise ValueError(f"parent cycle detected: {' -> '.join(stack + [name])}")
    icon = icons.get(name)
    if not icon:
        raise KeyError(f"icon not found: {name}")
    parent_name = icon.get("parent")
    if parent_name:
        stack.append(name)
        parent = resolve_icon(parent_name, icons, stack)
        stack.pop()
        merged = dict(parent)
        merged.update(icon)
        return merged
    return dict(icon)


def build_svg(
    body: str,
    width: int,
    height: int,
    left: int = 0,
    top: int = 0,
    output_size: Optional[Tuple[int, int]] = None,
    padding: int = 0,
    background: Optional[str] = None,
) -> str:
    out_w, out_h = output_size if output_size else (width, height)
    view_box = f"0 0 {out_w} {out_h}"
    background = (background or "").strip().lower()
    rect = ""
    if background and background not in {"none", "transparent"}:
        rect = f'<rect width="100%" height="100%" fill="{background}"/>'

    padding = max(0, int(padding))
    avail_w = max(0.0, float(out_w - padding * 2))
    avail_h = max(0.0, float(out_h - padding * 2))
    if width <= 0 or height <= 0 or avail_w == 0 or avail_h == 0:
        scale = 1.0
        offset_x = float(padding)
        offset_y = float(padding)
    else:
        scale = min(avail_w / float(width), avail_h / float(height))
        content_w = float(width) * scale
        content_h = float(height) * scale
        offset_x = float(padding) + (avail_w - content_w) / 2.0
        offset_y = float(padding) + (avail_h - content_h) / 2.0

    transform = f"translate({offset_x:.6f} {offset_y:.6f}) scale({scale:.6f}) translate({-left} {-top})"
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="{view_box}">{rect}<g transform="{transform}">{body}</g></svg>'
    )


def rasterize_all(
    input_path: str,
    output_dir: str,
    output_size: Tuple[int, int],
    force: bool = False,
    padding: int = 0,
    background: Optional[str] = None,
    names: Optional[list] = None,
) -> None:
    with open(input_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    icons = data.get("icons", {})
    default_width = int(data.get("width", 16))
    default_height = int(data.get("height", 16))

    if not icons:
        raise ValueError("no icons found in JSON")

    # Filter icons if specific names are provided
    if names:
        icons_to_process = {name: icons[name] for name in names if name in icons}
    else:
        icons_to_process = icons

    os.makedirs(output_dir, exist_ok=True)

    total = len(icons_to_process)
    success = 0
    skipped = 0
    failed = 0

    for name in icons_to_process.keys():
        try:
            icon = resolve_icon(name, icons)
            body = icon.get("body")
            if not body:
                skipped += 1
                continue
            width = int(icon.get("width", default_width))
            height = int(icon.get("height", default_height))
            left = int(icon.get("left", 0))
            top = int(icon.get("top", 0))

            svg = build_svg(
                body,
                width,
                height,
                left,
                top,
                output_size=output_size,
                padding=padding,
                background=background,
            )
            output_path = os.path.join(output_dir, f"{name}.png")
            if os.path.exists(output_path) and not force:
                skipped += 1
                continue

            cairosvg.svg2png(
                bytestring=svg.encode("utf-8"),
                write_to=output_path,
                output_width=output_size[0],
                output_height=output_size[1],
            )
            success += 1
        except Exception as exc:
            failed += 1
            print(f"failed: {name}: {exc}", file=sys.stderr)

    print(
        f"done: {success} rendered, {skipped} skipped, {failed} failed, {total} total",
        file=sys.stderr,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Rasterize Iconify JSON icons to PNG.")
    parser.add_argument(
        "input",
        nargs="?",
        default=os.path.join("node_modules", "@iconify-json", "noto-v1", "icons.json"),
        help="Path to icons.json",
    )
    parser.add_argument(
        "--out",
        default="emoji-480",
        help="Output directory for PNGs",
    )
    parser.add_argument(
        "--size",
        default="480",
        help="Output size, e.g. 480 or 480x480",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing PNGs",
    )
    parser.add_argument(
        "--bg",
        default="#fff",
        help="Background color (e.g. #fff). Use 'transparent' for no fill.",
    )
    parser.add_argument(
        "--pad",
        type=int,
        default=10,
        help="Padding in output pixels.",
    )
    parser.add_argument(
        "--names",
        help="Comma-separated list of icon names to rasterize (e.g., 'anchor,ant'). If not provided, all icons are rasterized.",
    )

    args = parser.parse_args()
    try:
        size = parse_size(args.size)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    # Parse names if provided
    names = [
        "anger-symbol",
        "angry-face",
        "angry-face-with-horns",
        "anguished-face",
        "antenna-bars",
        "anxious-face-with-sweat",
        "aries",
        "astonished-face",
        "backhand-index-pointing-down",
        "backhand-index-pointing-left",
        "barber-pole",
        "beaming-face-with-smiling-eyes",
        "cat-face",
        "clown-face",
        "confounded-face",
        "confused-face",
        "cow-face",
        "cowboy-hat-face",
        "crying-face",
        "disappointed-face",
        "dizzy-face",
        "dog-face",
        "downcast-face-with-sweat",
        "drooling-face",
        "expressionless-face",
        "extract_filenames",
        "face-blowing-a-kiss",
        "face-savoring-food",
        "face-screaming-in-fear",
        "face-with-head-bandage",
        "face-with-medical-mask",
        "face-with-open-mouth",
        "face-with-rolling-eyes",
        "face-with-steam-from-nose",
        "face-with-tears-of-joy",
        "face-with-thermometer",
        "face-with-tongue",
        "face-without-mouth",
        "fearful-face",
        "first-quarter-moon-face",
        "flushed-face",
        "frowning-face",
        "frowning-face-with-open-mouth",
        "full-moon",
        "full-moon-face",
        "grimacing-face",
        "grinning-face",
        "grinning-face-with-big-eyes",
        "grinning-face-with-smiling-eyes",
        "grinning-face-with-sweat",
        "grinning-squinting-face",
        "heart-suit",
        "high-voltage",
        "hole",
        "hugging-face",
        "hushed-face",
        "kissing-cat",
        "kissing-face",
        "kissing-face-with-closed-eyes",
        "kissing-face-with-smiling-eyes",
        "last-quarter-moon-face",
        "light-bulb",
        "loudly-crying-face",
        "lying-face",
        "money-mouth-face",
        "monkey-face",
        "mouse-face",
        "nauseated-face",
        "nerd-face",
        "neutral-face",
        "new-moon-face",
        "pensive-face",
        "persevering-face",
        "pig-face",
        "pouting-face",
        "rabbit-face",
        "relieved-face",
        "sad-but-relieved-face",
        "sleeping-face",
        "sleepy-face",
        "slightly-frowning-face",
        "slightly-smiling-face",
        "smiling-face",
        "smiling-face-with-halo",
        "smiling-face-with-heart-eyes",
        "smiling-face-with-horns",
        "smiling-face-with-smiling-eyes",
        "smiling-face-with-sunglasses",
        "smirking-face",
        "sneezing-face",
        "squinting-face-with-tongue",
        "sun-with-face",
        "thinking-face",
        "tired-face",
        "unamused-face",
        "upside-down-face",
        "weary-face",
        "winking-face",
        "winking-face-with-tongue",
        "worried-face",
        "zipper-mouth-face",
    ]
    if args.names:
        names = [n.strip() for n in args.names.split(",")]

    rasterize_all(args.input, args.out, size, args.force, args.pad, args.bg, names)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
