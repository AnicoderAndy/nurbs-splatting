from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
from matplotlib.patches import PathPatch
from matplotlib.path import Path
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from PIL import Image
import matplotlib.patches as patches
from . import geom


def set_axis_limits(box, pad=0, invert=True, ax=None, y_limits_only=False):
    # UNUSED
    if ax is None:
        ax = plt.gca()

    xlim = [box[0][0] - pad, box[1][0] + pad]
    ylim = [box[0][1] - pad, box[1][1] + pad]

    ax.set_ylim(ylim)
    ax.set_ybound(ylim)
    if not y_limits_only:
        ax.set_xlim(xlim)
        ax.set_xbound(xlim)

    # Hack to get matplotlib to actually respect limits?
    stroke_rect(
        [geom.vec(xlim[0], ylim[0]), geom.vec(xlim[1], ylim[1])],
        "r",
        plot=False,
        alpha=0,
    )
    # ax.set_clip_on(True)
    if invert:
        ax.invert_yaxis()


def stroke_rect(rect, clr="k", plot=True, **kwargs):
    x, y = rect[0]
    w, h = rect[1] - rect[0]

    plt.gca().add_patch(
        patches.Rectangle((x, y), w, h, fill=False, edgecolor=clr, **kwargs)
    )


def setup(invert_y=True, axis=False, box=None, debug_box=False):
    ax = plt.gca()
    ax.axis("scaled")
    if not axis:
        ax.axis("off")
    else:
        ax.axis("on")
    if invert_y:
        ax.invert_yaxis()
    if debug_box and box is not None:
        stroke_rect(box, "r", plot=False)
    if box is not None:
        set_axis_limits(box, invert=invert_y, ax=ax, y_limits_only=False)


def font_to_image(
    text,
    image_size,
    padding=0,
    font_path=None,
    grayscale=True,
    return_outline=False,
    separate_glyphs=False,
    vspace=1.0,
    subd=20,
):
    # Load font
    font_size = 60
    if font_path:
        font_prop = FontProperties(fname=font_path, size=font_size)
    else:
        font_prop = FontProperties(size=font_size)

    lines = text.splitlines()

    # Get vector path
    vertices = []
    codes = []

    for i, text in enumerate(lines):
        text_path = TextPath((0, 0), text, prop=font_prop)
        tvertices = text_path.vertices.copy()
        tvertices[:, 1] *= -1
        tcodes = text_path.codes

        box = geom.bounding_box(tvertices)
        center = geom.rect_center(box)
        tvertices = geom.tsm(geom.trans_2d(-center + [0, font_size * i]), tvertices)
        vertices.append(tvertices)
        codes.append(tcodes)
    vertices = np.vstack(vertices)
    codes = np.concatenate(codes)
    # for text in lines:
    #     text_path = TextPath((0, 0), text, prop=font_prop)
    #     vertices = text_path.vertices.copy()
    #     codes = text_path.codes

    # Transform to image space
    box = geom.make_rect(0, 0, image_size[0], image_size[1])
    src_box = geom.bounding_box(vertices)
    mat = geom.rect_in_rect_transform(src_box, box, padding=padding)
    vertices = geom.tsm(mat, vertices)
    text_path = Path(vertices, codes)

    # Draw to image
    fig, ax = plt.subplots(figsize=(image_size[0] / 100, image_size[1] / 100), dpi=100)
    # ax = fig.add_axes([0, 0, 1, 1])
    patch = PathPatch(text_path, facecolor="k", edgecolor="none")
    ax.add_patch(patch)

    setup(box=box)
    # Key for sizing!
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)

    image = Image.open(buf)
    if grayscale:
        image = image.convert("L")

    if not return_outline:
        return image

    # Optional: Return outline samples too
    curves = []
    i = 0
    while i < len(codes):
        code = codes[i]
        if code == Path.MOVETO:
            start = vertices[i]
            i += 1
            curves.append([start])
        elif code == Path.LINETO:
            end = vertices[i]
            line = np.linspace(start, end, subd)
            curves[-1].append(line[1:])
            start = end
            i += 1
        elif code == Path.CURVE3:  # Quadratic Bézier
            ctrl = vertices[i]
            end = vertices[i + 1]
            t = np.linspace(0, 1, subd).reshape(-1, 1)
            pts = (1 - t) ** 2 * start + 2 * (1 - t) * t * ctrl + t**2 * end
            curves[-1].append(pts[1:])
            start = end
            i += 2
        elif code == Path.CURVE4:  # Cubic Bézier
            ctrl1 = vertices[i]
            ctrl2 = vertices[i + 1]
            end = vertices[i + 2]
            t = np.linspace(0, 1, subd).reshape(-1, 1)
            pts = (
                (1 - t) ** 3 * start
                + 3 * (1 - t) ** 2 * t * ctrl1
                + 3 * (1 - t) * t**2 * ctrl2
                + t**3 * end
            )
            curves[-1].append(pts[1:])
            start = end
            i += 3
        elif code == Path.CLOSEPOLY:
            i += 1
        else:
            i += 1

    return image, [np.vstack(X) for X in curves]
