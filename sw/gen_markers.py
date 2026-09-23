"""Generate a printable sheet of ArUco markers for the fretboard corners.

Prints markers 0-3. Tape them to the corners of your board like this,
holding it as a guitar neck (nut on the left, strings horizontal):

    [0]--------------------[1]
     |   nut          bridge |
    [3]--------------------[2]

Usage:  python gen_markers.py            -> writes markers.png
        python gen_markers.py --size 300 -> bigger markers
"""

import argparse

import cv2
import numpy as np

from board import ARUCO_DICT, MARKER_IDS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=250, help="marker side in pixels")
    ap.add_argument("--out", default="markers.png")
    args = ap.parse_args()

    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    pad = args.size // 4
    tile = args.size + 2 * pad
    sheet = np.full((tile, tile * len(MARKER_IDS)), 255, dtype=np.uint8)

    for i, mid in enumerate(MARKER_IDS):
        marker = cv2.aruco.generateImageMarker(dictionary, mid, args.size)
        x = i * tile + pad
        sheet[pad:pad + args.size, x:x + args.size] = marker
        cv2.putText(sheet, f"id {mid}", (x, pad // 2 + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 1, cv2.LINE_AA)

    cv2.imwrite(args.out, sheet)
    print(f"wrote {args.out} ({sheet.shape[1]}x{sheet.shape[0]}) - print it, cut out the four squares")


if __name__ == "__main__":
    main()
