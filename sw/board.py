"""Fretboard reference tracking.

Detects ArUco markers on a hand-held board and builds a homography from camera
pixels to *board space*:

    u: 0 (nut) .......... 1 (bridge end)     along the neck
    v: 0 (top string) ... 1 (bottom string)  across the neck

Everything downstream (fret cells, string lines, strum zone) is defined in
board space, so the board can be moved/tilted freely.

Where the markers sit comes from a layout (layout.py). Every visible marker
contributes its known positions in cm; cv2.findHomography with RANSAC fits
pixels -> cm over whichever markers are visible, so a hand covering some of
them doesn't lose the board. The fitted map places the playing area's corners
in pixels, and those corners define board space as before.

If no markers are seen yet, a fixed rectangle in the middle of the frame is
used instead so you can try things out before printing markers.
"""

import cv2
import numpy as np

import layout as layout_mod
from layout import ARUCO_DICT

UNIT_SQUARE = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)


def well_spread(points_cm, layout):
    """Are the visible marker points spread out enough to trust a fit over the whole board?

    points_cm: (N, 2) known positions (cm) of the detected marker corners.

    A homography fitted to points bunched near the nut is accurate there and
    increasingly wrong towards the far end of the neck. Decide here when that
    risk is too high, e.g. from how much of the board's length the points span,
    or whether both the top and bottom marker strips are represented.

    TODO(Carlos): implement. test_board.py has a case that fails until you do.
    """
    return True


class Board:
    def __init__(self, layout=None, smoothing=0.5, hold_frames=15, ransac_cm=0.5):
        dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
        self.detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
        self.layout = layout or layout_mod.legacy_layout()
        self.known = {int(k): np.array(v, dtype=np.float32) for k, v in self.layout["markers"].items()}
        self.play_cm = np.array(layout_mod.play_corners_cm(self.layout), dtype=np.float32)
        self.min_markers = self.layout.get("min_markers", 2)
        self.ransac_cm = ransac_cm      # RANSAC inlier threshold, a real distance on the board
        self.smoothing = smoothing      # EMA factor for corner positions (0 = no smoothing)
        self.hold_frames = hold_frames  # keep last homography this many frames after losing markers
        self.corners = None             # (4, 2) pixel corners of the playing area, smoothed
        self.H = None                   # pixel -> board
        self.H_inv = None               # board -> pixel
        self.missing = 0
        self.tracking = False           # True when driven by real markers
        # diagnostics, refreshed every update()
        self.mode = "fallback"          # tracking | hold | fallback
        self.detected = ([], None)      # (corners, ids) as returned by the detector, frame coords
        self.seen_ids = []
        self.inliers = 0
        self.reproj_err_cm = None
        self.marker_px = None           # median marker side in pixels

    # -- update ------------------------------------------------------------

    def update(self, frame_bgr, mirrored=False):
        """Detect markers and refresh the homography. Returns True if tracking.

        mirrored: the frame was cv2.flip'ed for display. A mirrored marker reads
        as a different (usually invalid) code, so detection runs on the
        un-mirrored image and the corners are mapped back into frame coords.
        """
        h, w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if mirrored:
            gray = cv2.flip(gray, 1)
        corners, ids, _ = self.detector.detectMarkers(gray)
        if mirrored:
            corners = tuple(np.column_stack([w - 1 - c[0, :, 0], c[0, :, 1]])[None].astype(np.float32)
                            for c in corners)
        self.detected = (corners, ids)

        px, cm, sides = [], [], []
        self.seen_ids = []
        if ids is not None:
            for c, i in zip(corners, ids.flatten()):
                known = self.known.get(int(i))
                if known is None:
                    continue
                c = c.reshape(4, 2)
                self.seen_ids.append(int(i))
                sides.append(np.linalg.norm(c - np.roll(c, 1, axis=0), axis=1).mean())
                if len(known) == 4:
                    px.extend(c)
                else:  # legacy layout: one point per marker, its centre
                    px.append(c.mean(axis=0))
                cm.extend(known)
        self.marker_px = float(np.median(sides)) if sides else None

        quad = self._fit(np.array(px, dtype=np.float32), np.array(cm, dtype=np.float32))
        if quad is not None:
            if self.corners is None or not self.tracking:
                self.corners = quad
            else:
                a = self.smoothing
                self.corners = a * self.corners + (1 - a) * quad
            self._set_homography(self.corners)
            self.missing = 0
            self.tracking = True
            self.mode = "tracking"
        else:
            self.missing += 1
            self.mode = "hold"
            if self.H is None or self.missing > self.hold_frames:
                self.tracking = False
                self.mode = "fallback"
                self._use_fallback(frame_bgr.shape)

        return self.tracking

    def _fit(self, px, cm):
        """Fit pixels -> cm over all visible marker points; return the playing area's
        corners in pixels, or None if there isn't enough to trust."""
        self.inliers, self.reproj_err_cm = 0, None
        if len(self.seen_ids) < self.min_markers or len(px) < 4 or not well_spread(cm, self.layout):
            return None
        # Fit in cm, not u/v: the board is much longer than it is wide, so an
        # error threshold in u/v units would mean different distances per axis.
        H_cm, mask = cv2.findHomography(px, cm, cv2.RANSAC, self.ransac_cm)
        if H_cm is None:
            return None
        inl = mask.ravel().astype(bool)
        if inl.sum() < max(4, len(px) // 2):
            return None
        err = np.linalg.norm(cv2.perspectiveTransform(px[inl].reshape(-1, 1, 2), H_cm).reshape(-1, 2)
                             - cm[inl], axis=1)
        self.inliers, self.reproj_err_cm = int(inl.sum()), float(err.mean())
        quad = cv2.perspectiveTransform(self.play_cm.reshape(-1, 1, 2), np.linalg.inv(H_cm))
        return quad.reshape(4, 2).astype(np.float32)

    def _use_fallback(self, shape):
        h, w = shape[:2]
        x0, x1 = int(w * 0.15), int(w * 0.85)
        y0, y1 = int(h * 0.35), int(h * 0.65)
        self.corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
        self._set_homography(self.corners)

    def _set_homography(self, corners):
        self.H = cv2.getPerspectiveTransform(corners, UNIT_SQUARE)
        self.H_inv = cv2.getPerspectiveTransform(UNIT_SQUARE, corners)

    # -- mapping -----------------------------------------------------------

    def to_board(self, pts):
        """(N, 2) pixel coords -> (N, 2) board coords (u, v)."""
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)

    def to_pixel(self, pts):
        """(N, 2) board coords -> (N, 2) pixel coords."""
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H_inv).reshape(-1, 2)

    # -- drawing -----------------------------------------------------------

    def draw_outline(self, frame):
        color = (0, 255, 120) if self.tracking else (0, 140, 255)
        cv2.polylines(frame, [self.corners.astype(int).reshape(-1, 1, 2)], True, color, 2, cv2.LINE_AA)
        if not self.tracking:
            x, y = self.corners[3].astype(int)
            cv2.putText(frame, "no markers - using fixed region", (x, y + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    def draw_markers(self, frame):
        """Outline and label every detected marker (diagnostics)."""
        corners, ids = self.detected
        if ids is not None and len(corners):
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

    def draw_line(self, frame, p0, p1, color, thickness=1):
        """Draw a line given two board-space points."""
        a, b = self.to_pixel([p0, p1]).astype(int)
        cv2.line(frame, tuple(a), tuple(b), color, thickness, cv2.LINE_AA)

    def draw_quad(self, frame, u0, v0, u1, v1, color, alpha=0.35):
        """Fill a board-space rectangle with translucent colour."""
        quad = self.to_pixel([[u0, v0], [u1, v0], [u1, v1], [u0, v1]]).astype(int)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [quad.reshape(-1, 1, 2)], color)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
