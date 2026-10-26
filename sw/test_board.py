"""Board tracking tests on synthetic frames (no camera needed).

The printed template (gen_board.render_board) is warped into a fake camera
frame with a known perspective, so the true board coords of any pixel are known.

    .venv/bin/python test_board.py
"""

import cv2
import numpy as np

import gen_board
import layout as layout_mod
from board import Board

FRAME = (720, 1280)
GRID = [(u, v) for u in np.linspace(0.02, 0.98, 9) for v in np.linspace(0.05, 0.95, 5)]


def scene(lay, cover=(), mirror=False):
    """Warp the board into a frame. Returns (bgr frame, ground-truth pixel->board fn)."""
    board_img = gen_board.render_board(lay)
    bh, bw = board_img.shape
    src = np.float32([[0, 0], [bw, 0], [bw, bh], [0, bh]])
    dst = np.float32([[140, 250], [1170, 190], [1140, 470], [160, 430]])  # tilted, foreshortened
    H = cv2.getPerspectiveTransform(src, dst)
    frame = cv2.warpPerspective(board_img, H, FRAME[::-1], borderValue=110)
    cm_to_img = 300 / 2.54

    for mid in cover:  # grey out a marker as a hand would
        c = np.float32(lay["markers"][str(mid)])
        c = c.mean(0) + 1.3 * (c - c.mean(0))
        poly = cv2.perspectiveTransform((c * cm_to_img).reshape(-1, 1, 2), H).astype(int)
        cv2.fillPoly(frame, [poly], 128)

    x0, y0, x1, y1 = lay["play"]
    def truth(u, v):
        cm = np.float32([[x0 + u * (x1 - x0), y0 + v * (y1 - y0)]]) * cm_to_img
        p = cv2.perspectiveTransform(cm.reshape(-1, 1, 2), H).reshape(2)
        if mirror:
            p[0] = FRAME[1] - 1 - p[0]
        return p

    if mirror:
        frame = cv2.flip(frame, 1)
    return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR), truth


def max_err(board, truth):
    pts = np.array([truth(u, v) for u, v in GRID])
    return float(np.abs(board.to_board(pts) - np.array(GRID)).max())


def test_all_visible():
    lay = layout_mod.load()
    frame, truth = scene(lay)
    b = Board(lay)
    assert b.update(frame) and b.mode == "tracking", b.mode
    assert len(b.seen_ids) == 10, b.seen_ids
    assert max_err(b, truth) < 0.01, max_err(b, truth)


def test_two_corners_covered():
    lay = layout_mod.load()
    frame, truth = scene(lay, cover=[0, 9])
    b = Board(lay)
    assert b.update(frame) and b.mode == "tracking", b.mode
    assert 0 not in b.seen_ids and 9 not in b.seen_ids
    assert max_err(b, truth) < 0.01, max_err(b, truth)


def test_mirrored_frame():
    lay = layout_mod.load()
    frame, truth = scene(lay, mirror=True)
    naive = Board(lay)
    naive.update(frame)  # mirrored codes mostly fail, and some misread as *other* IDs
    assert len(naive.seen_ids) < 10, "expected mirrored markers to mis-decode without mirrored=True"
    b = Board(lay)
    assert b.update(frame, mirrored=True), b.mode
    assert max_err(b, truth) < 0.01, max_err(b, truth)


def test_legacy_four_corners():
    frame = np.full((*FRAME, 3), 255, np.uint8)
    d = cv2.aruco.getPredefinedDictionary(layout_mod.ARUCO_DICT)
    centres = np.float32([[200, 200], [1080, 200], [1080, 520], [200, 520]])
    for mid, (cx, cy) in zip(layout_mod.LEGACY_IDS, centres.astype(int)):
        frame[cy - 40:cy + 40, cx - 40:cx + 40] = cv2.aruco.generateImageMarker(d, mid, 80)[..., None]
    b = Board()
    assert b.update(frame)
    got = b.to_board(centres - 0.5)  # marker spans cx-40..cx+39, so its centre is half a pixel left/up
    assert np.allclose(got, [[0, 0], [1, 0], [1, 1], [0, 1]], atol=1e-3), got


def test_events_from_board_serialize():
    """Board output is float32; events built from it must still go through json (--log)."""
    import json
    from guitar import GuitarState
    lay = layout_mod.load()
    frame, truth = scene(lay)
    b = Board(lay)
    b.update(frame)
    g = GuitarState()
    ev = []
    for t, v in [(0.0, 0.02), (0.03, 0.98)]:  # strum across all strings in the strum zone
        ev += g.update_pick(tuple(b.to_board([truth(0.8, v)])[0]), t)
    assert len(ev) == 6, ev
    for e in ev:
        json.dumps(e.__dict__)


def test_only_nut_markers_not_trusted():
    """TODO(Carlos): passes once board.well_spread() is written."""
    lay = layout_mod.load()
    frame, _ = scene(lay, cover=[1, 2, 3, 4, 6, 7, 8, 9])  # only 0 and 5 (nut end) visible
    b = Board(lay)
    b.update(frame)
    assert b.seen_ids == [0, 5] or sorted(b.seen_ids) == [0, 5], b.seen_ids
    assert b.mode != "tracking", "fit from the nut end alone was trusted - implement well_spread()"


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
