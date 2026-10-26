"""Printable, true-scale board template, tiled onto US Letter pages.

Markers come from the layout (layout.py); string lines, fret lines and the
strum zone come from guitar.py, so what you print is what the code plays.

Usage:  python gen_board.py                      -> board.pdf + board_layout.json
        python gen_board.py --length 50 --width 11 --marker 2.5 --per-edge 4

Print at "Actual size" / 100% (not "fit to page") and check the 10 cm bar.
"""

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

import layout as layout_mod
from guitar import FRET_ZONE, NUM_FRETS, STRUM_START, NUM_STRINGS, fret_u, string_v

DPI = 300
PAGE_CM = (27.94, 21.59)  # US Letter, landscape
MARGIN_CM = 1.0
OVERLAP_CM = 1.0
FONT = cv2.FONT_HERSHEY_SIMPLEX


def px(cm):
    return int(round(cm * DPI / 2.54))


def render_board(layout):
    """The whole board as one grayscale image at DPI."""
    W, H = px(layout["length_cm"]), px(layout["width_cm"])
    img = np.full((H, W), 255, dtype=np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(layout_mod.ARUCO_DICT)
    x0, y0, x1, y1 = layout["play"]

    def at(u, v):  # board space -> image pixel
        return px(x0 + u * (x1 - x0)), px(y0 + v * (y1 - y0))

    # strum zone shading, then frets and strings over it
    cv2.rectangle(img, at(FRET_ZONE, 0), at(1, 1), 235, -1)
    x0d, _ = at(FRET_ZONE, 0)
    x1d, _ = at(STRUM_START, 0)
    _, yt = at(0, 0)
    _, yb = at(0, 1)
    for x in range(x0d - (yb - yt), x1d, 40):  # hatch the dead band
        cv2.line(img, (max(x, x0d), yt + max(0, x0d - x)), (min(x + (yb - yt), x1d), yb - max(0, x + (yb - yt) - x1d)),
                 200, 2)
    cv2.line(img, at(STRUM_START, 0), at(STRUM_START, 1), 120, 2)
    for f in range(1, NUM_FRETS + 1):
        cv2.line(img, at(fret_u(f), 0), at(fret_u(f), 1), 90, 4 if f == 1 else 2)
        label_x, _ = at((fret_u(f) + fret_u(f + 1)) / 2, 0)
        cv2.putText(img, str(f), (label_x - 10, px(y0) - 8), FONT, 0.9, 120, 2, cv2.LINE_AA)
    cv2.line(img, at(FRET_ZONE, 0), at(FRET_ZONE, 1), 90, 3)
    for s in range(NUM_STRINGS):
        cv2.line(img, at(0, string_v(s)), at(1, string_v(s)), 150, 2)
    cv2.rectangle(img, at(0, 0), at(1, 1), 60, 2)
    nx, ny = at(0, 1)
    cv2.putText(img, "NUT", (nx + 8, ny + 30), FONT, 0.8, 120, 2, cv2.LINE_AA)
    sx, _ = at((STRUM_START + 1) / 2, 1)
    cv2.putText(img, "STRUM", (sx - 45, ny + 30), FONT, 0.8, 120, 2, cv2.LINE_AA)

    for mid, corners in layout["markers"].items():
        (mx, my), side = corners[0], px(layout["marker_cm"])
        img[px(my):px(my) + side, px(mx):px(mx) + side] = cv2.aruco.generateImageMarker(dictionary, int(mid), side)

    cv2.rectangle(img, (0, 0), (W - 1, H - 1), 0, 2)  # cut line
    return img


def paginate(board):
    """Split the board into Letter pages with an overlap strip and alignment marks."""
    pw, ph = px(PAGE_CM[0]), px(PAGE_CM[1])
    m, ov = px(MARGIN_CM), px(OVERLAP_CM)
    cw, ch = pw - 2 * m, ph - 2 * m          # printable area
    bh, bw = board.shape
    cols = max(1, math.ceil((bw - ov) / (cw - ov)))
    rows = max(1, math.ceil((bh - ov) / (ch - ov)))
    pages = []
    for r in range(rows):
        for c in range(cols):
            bx, by = c * (cw - ov), r * (ch - ov)
            tile = board[by:by + ch, bx:bx + cw]
            page = np.full((ph, pw), 255, dtype=np.uint8)
            page[m:m + tile.shape[0], m:m + tile.shape[1]] = tile
            th, tw = tile.shape
            if c > 0:  # trim here, then lay this edge over the previous page's matching ticks
                for y in range(m, m + th, 30):
                    cv2.line(page, (m + ov, y), (m + ov, min(y + 15, m + th)), 0, 2)
                cv2.putText(page, "trim, overlap onto previous page", (m + ov + 6, m + th + 30),
                            FONT, 0.6, 0, 1, cv2.LINE_AA)
            if c < cols - 1 and tw == cw:  # where the next page's trim line lands
                x = m + cw - ov
                for y in (m - 25, m + th + 5):
                    cv2.line(page, (x, y), (x, y + 20), 0, 2)
            label_y = min(m + th + 80, ph - m)
            cv2.putText(page, f"page {len(pages) + 1}/{rows * cols}  (row {r + 1}, col {c + 1})",
                        (m, label_y), FONT, 0.8, 0, 2, cv2.LINE_AA)
            bar_y = min(label_y + 50, ph - m // 2)
            cv2.rectangle(page, (m, bar_y), (m + px(10), bar_y + 20), 0, -1)
            cv2.putText(page, "10 cm - check this measures 10 cm", (m + px(10) + 15, bar_y + 18),
                        FONT, 0.7, 0, 2, cv2.LINE_AA)
            pages.append(page)
    return pages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=float, default=60.0, help="board length, cm")
    ap.add_argument("--width", type=float, default=12.0, help="board width, cm")
    ap.add_argument("--marker", type=float, default=2.5, help="marker side, cm")
    ap.add_argument("--per-edge", type=int, default=5, help="markers along each long edge")
    ap.add_argument("--out", default="board.pdf")
    args = ap.parse_args()

    lay = layout_mod.make_layout(args.length, args.width, args.marker, args.per_edge)
    step = PAGE_CM[0] - 2 * MARGIN_CM - OVERLAP_CM
    seams = [c * step + OVERLAP_CM for c in range(1, math.ceil(args.length / step) + 1)]
    for mid, corners in lay["markers"].items():
        x0, x1 = corners[0][0], corners[1][0]
        if any(x0 - 0.2 < s < x1 + 0.2 for s in seams):
            print(f"warning: marker {mid} straddles a page seam; try a different --length or --per-edge")
    pages = paginate(render_board(lay))
    from PIL import Image
    imgs = [Image.fromarray(p) for p in pages]
    imgs[0].save(args.out, save_all=True, append_images=imgs[1:], resolution=DPI)
    layout_path = Path(args.out).with_name("board_layout.json")
    layout_mod.save(lay, layout_path)
    print(f"wrote {args.out} ({len(pages)} Letter pages) and {layout_path} "
          f"({len(lay['markers'])} markers, {args.length:g} x {args.width:g} cm)")


if __name__ == "__main__":
    main()
