"""Physical board layout, in centimetres. Shared by gen_board.py (printing) and
board.py (tracking) so the print and the code can't disagree.

    x: 0 .. length_cm along the neck (nut end at 0)
    y: 0 .. width_cm across it (top edge at 0)

    +------------------------------------------------------+
    | [0]        [1]        [2]        [3]        [4]      |  marker strip
    |------------------------------------------------------|
    | nut   frets ...            |  strum zone             |  playing area
    |------------------------------------------------------|  = board space u, v in [0, 1]
    | [5]        [6]        [7]        [8]        [9]      |  marker strip
    +------------------------------------------------------+

Each marker stores its 4 corners in ArUco's order (top-left, top-right,
bottom-right, bottom-left as printed), so detected corners pair up with known
positions one to one. The legacy layout stores one point (the centre) per marker.
"""

import json
from pathlib import Path

import cv2

ARUCO_DICT = cv2.aruco.DICT_4X4_50
LEGACY_IDS = [0, 1, 2, 3]  # top-left, top-right, bottom-right, bottom-left
DEFAULT_PATH = Path(__file__).parent / "board_layout.json"


def make_layout(length_cm=60.0, width_cm=12.0, marker_cm=2.5, per_edge=5, gap_cm=0.4, end_cm=1.0):
    """Markers in a strip along the top and bottom edges; playing area between them.

    gap_cm: white quiet zone around each marker (ArUco needs one).
    end_cm: playing area starts/ends this far in from the short edges.
    """
    strip = marker_cm + 2 * gap_cm
    if width_cm <= 2 * strip + 2:
        raise ValueError(f"board too narrow: {width_cm} cm leaves no playing area")
    span = length_cm - 2 * gap_cm - marker_cm  # travel of a marker's left edge along the board
    markers = {}
    for row, y in enumerate([gap_cm, width_cm - gap_cm - marker_cm]):
        for k in range(per_edge):
            x = gap_cm + span * k / max(per_edge - 1, 1)
            markers[str(row * per_edge + k)] = [[x, y], [x + marker_cm, y],
                                                [x + marker_cm, y + marker_cm], [x, y + marker_cm]]
    return {
        "length_cm": length_cm, "width_cm": width_cm, "marker_cm": marker_cm,
        "play": [end_cm, strip, length_cm - end_cm, width_cm - strip],  # x0, y0, x1, y1
        "markers": markers,
        "min_markers": 2,
    }


def legacy_layout():
    """The original setup: markers 0-3 whose centres are the corners of board space."""
    corners = [[0, 0], [1, 0], [1, 1], [0, 1]]
    return {
        "length_cm": 1.0, "width_cm": 1.0, "marker_cm": None,
        "play": [0, 0, 1, 1],
        "markers": {str(i): [c] for i, c in zip(LEGACY_IDS, corners)},
        "min_markers": 4,
    }


def save(layout, path=DEFAULT_PATH):
    Path(path).write_text(json.dumps(layout, indent=1) + "\n")


def load(path=DEFAULT_PATH):
    return json.loads(Path(path).read_text())


def play_corners_cm(layout):
    """The 4 corners of the playing area (= board-space unit square), TL TR BR BL."""
    x0, y0, x1, y1 = layout["play"]
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
