"""
We use this script to generate our calligraphy reconstruction dataset.
Use default settings to generate identical images to those in the paper.
Open with UTF-8 encoding to see the characters.
Download the Chinese font from: https://www.foundertype.com/index.php/FontInfo/index/id/6583
Download the Japanese font from: https://karu-k.booth.pm/items/2985842
"""

import argparse
import os
import re
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import cairosvg
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

"""
Common Chinese characters (Please make sure you open the file with UTF-8 encoding):
一乙二十丁厂七卜八人入儿匕几九刁了刀力乃又三干于亏工土士才下寸大丈与万上小口山巾千乞川亿个夕久么勺凡丸及广亡门丫义之尸己已巳弓子卫也女刃飞习叉马乡丰王开井天夫元无云专丐扎艺木五支厅不犬太区历歹友尤匹车
All Japanese Kana:
あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをんアイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a font dataset as per-character images."
    )
    parser.add_argument(
        "--font",
        default="pmzd.ttf",
        help="Font name or path to a .ttf/.otf file.",
    )
    parser.add_argument(
        "--chars",
        default="一乙二十丁厂",
        help="Character list to render, e.g. 'ABC123'.",
    )
    parser.add_argument(
        "--out",
        default="data",
        help="Output directory for generated images.",
    )
    parser.add_argument(
        "--file-prefix",
        default="sp_",
        help="Prefix for output filenames, e.g. 'zh_', 'jp_'.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Image resolution (square). Default: 512.",
    )
    return parser.parse_args()


def split_graphemes(text: str) -> List[str]:
    """Split text into grapheme clusters, handling emoji sequences properly.

    Handles variation selectors, ZWJ sequences, skin tone modifiers,
    combining enclosing keycap, and tag sequences.
    """
    clusters: List[str] = []
    i = 0
    while i < len(text):
        start = i
        i += 1
        while i < len(text):
            code = ord(text[i])
            if code in (0xFE0E, 0xFE0F):  # variation selectors
                i += 1
            elif code == 0x200D:  # ZWJ
                i += 1
                if i < len(text):
                    i += 1
            elif 0x1F3FB <= code <= 0x1F3FF:  # skin tone modifiers
                i += 1
            elif code == 0x20E3:  # combining enclosing keycap
                i += 1
            elif 0xE0020 <= code <= 0xE007F:  # tag characters
                i += 1
            else:
                break
        clusters.append(text[start:i])
    return clusters


def strip_svg_table(font_path: str) -> str:
    """Remove SVG table from font if present.

    Pillow cannot render SVG-based color emoji (FreeType SVG hooks not set).
    Stripping the SVG table forces FreeType to fall back to COLR/CPAL tables
    which Pillow supports via ``embedded_color=True``.

    Returns path to a modified temp file, or the original path if no SVG table.
    """
    import tempfile

    tt = TTFont(font_path)
    if "SVG " in tt:
        del tt["SVG "]
        fd, tmp = tempfile.mkstemp(suffix=os.path.splitext(font_path)[1] or ".ttf")
        os.close(fd)
        tt.save(tmp)
        tt.close()
        return tmp
    tt.close()
    return font_path


class SVGFontRenderer:
    """Render individual glyphs from an SVG-based color emoji font.

    Parses the font's SVG table once, then extracts and rasterises glyphs
    on demand via *cairosvg*.
    """

    def __init__(self, font_path: str) -> None:
        tt = TTFont(font_path)
        self.cmap: Dict[int, str] = tt.getBestCmap()
        self.glyph_order = tt.getGlyphOrder()
        self.upem: int = tt["head"].unitsPerEm
        self.ascent: int = tt["hhea"].ascent
        self.descent: int = tt["hhea"].descent  # negative value
        self._hmtx = dict(tt["hmtx"].metrics)  # glyph_name -> (advanceWidth, lsb)

        # Build glyph-name → glyph-ID mapping
        self._name_to_id: Dict[str, int] = {
            name: idx for idx, name in enumerate(self.glyph_order)
        }
        self._id_to_name: Dict[int, str] = {
            idx: name for idx, name in enumerate(self.glyph_order)
        }

        # Pre-parse SVG docs: list of (doc_str, start_gid, end_gid)
        svg_table = tt["SVG "]
        self._svg_docs: List[Tuple[str, int, int]] = [
            (doc, s, e) for doc, s, e in svg_table.docList
        ]

        # Cache extracted <defs> per doc index
        self._defs_cache: Dict[int, str] = {}
        tt.close()

    def _get_glyph_id(self, character: str) -> Optional[int]:
        """Map a (possibly multi-codepoint) character to a glyph ID.

        For simple emoji or emoji with variation selectors, the base codepoint
        is looked up in the cmap.
        """
        # Strip variation selectors for cmap lookup
        base = "".join(c for c in character if ord(c) not in (0xFE0E, 0xFE0F))
        if not base:
            return None

        # Single codepoint lookup (covers the vast majority of cases)
        cp = ord(base[0])
        glyph_name = self.cmap.get(cp)
        if glyph_name is None:
            return None
        return self._name_to_id.get(glyph_name)

    def _find_svg_doc(self, glyph_id: int) -> Optional[Tuple[int, str]]:
        """Return (doc_index, doc_string) for the SVG doc covering *glyph_id*."""
        for idx, (doc, start, end) in enumerate(self._svg_docs):
            if start <= glyph_id <= end:
                return idx, doc
        return None

    def _extract_defs(self, doc_idx: int, doc: str) -> str:
        if doc_idx not in self._defs_cache:
            m = re.search(r"(<defs>.*?</defs>)", doc, re.DOTALL)
            self._defs_cache[doc_idx] = m.group(1) if m else ""
        return self._defs_cache[doc_idx]

    @staticmethod
    def _extract_glyph_element(doc: str, glyph_id: int) -> Optional[str]:
        """Extract ``<g id="glyph{N}">...</g>`` handling nested ``<g>`` tags."""
        pattern = rf'<g\s+id="glyph{glyph_id}">'
        match = re.search(pattern, doc)
        if not match:
            return None

        start_pos = match.start()
        depth = 0
        i = start_pos
        while i < len(doc):
            if doc[i : i + 2] == "<g":
                depth += 1
                # Skip to end of opening tag
                i = doc.index(">", i) + 1
            elif doc[i : i + 4] == "</g>":
                depth -= 1
                if depth == 0:
                    return doc[start_pos : i + 4]
                i += 4
            else:
                i += 1
        return None

    def render(self, character: str, resolution: int) -> Optional[Image.Image]:
        """Render *character* to a ``resolution x resolution`` RGB PIL Image."""
        glyph_id = self._get_glyph_id(character)
        if glyph_id is None:
            return None

        result = self._find_svg_doc(glyph_id)
        if result is None:
            return None
        doc_idx, doc = result

        glyph_svg = self._extract_glyph_element(doc, glyph_id)
        if glyph_svg is None:
            return None

        defs = self._extract_defs(doc_idx, doc)

        # Determine viewBox from font metrics
        glyph_name = self._id_to_name.get(glyph_id, "")
        advance_width, _ = self._hmtx.get(glyph_name, (self.upem, 0))
        vb_x = 0
        vb_y = -self.ascent  # top of em square (negative in font coords)
        vb_w = advance_width
        vb_h = self.ascent - self.descent  # full height from ascender to descender

        # Add padding (percentage of the larger dimension)
        pad = int(max(vb_w, vb_h) * 0.05)
        vb_x -= pad
        vb_y -= pad
        vb_w += pad * 2
        vb_h += pad * 2

        standalone = (
            f'<?xml version="1.0" encoding="UTF-8"?>'
            f'<svg xmlns="http://www.w3.org/2000/svg"'
            f' xmlns:xlink="http://www.w3.org/1999/xlink"'
            f' viewBox="{vb_x} {vb_y} {vb_w} {vb_h}"'
            f' width="{resolution}" height="{resolution}">'
            f"{defs}{glyph_svg}</svg>"
        )

        png_data = cairosvg.svg2png(
            bytestring=standalone.encode("utf-8"),
            output_width=resolution,
            output_height=resolution,
        )

        rgba = Image.open(BytesIO(png_data)).convert("RGBA")
        bg = Image.new("RGB", (resolution, resolution), (255, 255, 255))
        bg.paste(rgba, mask=rgba.split()[3])
        return bg


def _font_has_svg(font_path: str) -> bool:
    """Return True if the font contains an SVG table."""
    tt = TTFont(font_path, lazy=True)
    has_svg = "SVG " in tt
    tt.close()
    return has_svg


def load_font(font_name: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(font_name, size)
    except OSError as exc:
        raise SystemExit(
            f"Failed to load font '{font_name}'. Provide a valid font name or file path."
        ) from exc


def fit_font_size(
    font_name: str, character: str, resolution: int
) -> ImageFont.FreeTypeFont:
    padding = int(resolution * 0.1)
    max_size = resolution
    min_size = 1

    test_img = Image.new("L", (resolution, resolution), 0)
    draw = ImageDraw.Draw(test_img)

    size = max_size
    while size >= min_size:
        font = load_font(font_name, size)
        bbox = draw.textbbox((0, 0), character, font=font)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        if width <= (resolution - padding * 2) and height <= (resolution - padding * 2):
            return font
        size -= 1

    return load_font(font_name, min_size)


def render_character(character: str, font_name: str, resolution: int) -> Image.Image:
    """Render a character using Pillow (for non-SVG fonts)."""
    font = fit_font_size(font_name, character, resolution)
    image = Image.new("RGB", (resolution, resolution), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    bbox = draw.textbbox((0, 0), character, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    x = (resolution - width) / 2 - bbox[0]
    y = (resolution - height) / 2 - bbox[1]

    draw.text((x, y), character, font=font, fill=(0, 0, 0))
    return image


def generate_dataset(
    font_name: str,
    characters: List[str],
    resolution: int,
    out_dir: str,
    file_prefix: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    digits = max(1, len(str(len(characters))))

    # Choose renderer based on font capabilities
    use_svg = _font_has_svg(font_name)
    svg_renderer: Optional[SVGFontRenderer] = None
    pillow_font_path: Optional[str] = None

    if use_svg:
        svg_renderer = SVGFontRenderer(font_name)
        # Also prepare a Pillow-compatible font (SVG stripped) as fallback
        pillow_font_path = strip_svg_table(font_name)
    else:
        pillow_font_path = font_name

    for index, character in enumerate(characters, start=1):
        image = None

        # Try SVG renderer first
        if svg_renderer is not None:
            image = svg_renderer.render(character, resolution)

        # Fall back to Pillow for non-SVG glyphs or non-SVG fonts
        if image is None and pillow_font_path is not None:
            image = render_character(character, pillow_font_path, resolution)

        if image is not None:
            filename = f"{file_prefix}{index:0{digits}d}.png"
            image.save(os.path.join(out_dir, filename))


def main() -> None:
    args = parse_args()
    if args.resolution <= 0:
        raise SystemExit("Resolution must be a positive integer.")
    if not args.chars:
        raise SystemExit("Character list cannot be empty.")

    os.makedirs(args.out, exist_ok=True)
    characters = split_graphemes(args.chars)
    generate_dataset(args.font, characters, args.resolution, args.out, args.file_prefix)


if __name__ == "__main__":
    main()
