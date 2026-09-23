"""Fretboard reference tracking.

Detects four ArUco markers at the corners of a hand-held board and builds a
homography from camera pixels to *board space*:

    u: 0 (nut) .......... 1 (bridge end)     along the neck
    v: 0 (top string) ... 1 (bottom string)  across the neck

Everything downstream (fret cells, string lines, strum zone) is defined in
board space, so the board can be moved/tilted freely.

If no markers are seen yet, a fixed rectangle in the middle of the frame is
used instead so you can try things out before printing markers.
"""

import cv2
import numpy as np

ARUCO_DICT = cv2.aruco.DICT_4X4_50
MARKER_IDS = [0, 1, 2, 3]  # top-left, top-right, bottom-right, bottom-left

UNIT_SQUARE = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)


class Board:
    def __init__(self, smoothing=0.5, hold_frames=15):
        dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
        self.detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
        self.smoothing = smoothing      # EMA factor for corner positions (0 = no smoothing)
        self.hold_frames = hold_frames  # keep last homography this many frames after losing markers
        self.corners = None             # (4, 2) pixel corners, smoothed
        self.H = None                   # pixel -> board
        self.H_inv = None               # board -> pixel
        self.missing = 0
        self.tracking = False           # True when driven by real markers

    # -- update ------------------------------------------------------------

    def update(self, frame_bgr):
        """Detect markers and refresh the homography. Returns True if tracking."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)

        found = {}
        if ids is not None:
            for c, i in zip(corners, ids.flatten()):
                if i in MARKER_IDS:
                    found[int(i)] = c.reshape(4, 2).mean(axis=0)  # marker centre

        if len(found) == 4:
            pts = np.array([found[i] for i in MARKER_IDS], dtype=np.float32)
            if self.corners is None or not self.tracking:
                self.corners = pts
            else:
                a = self.smoothing
                self.corners = a * self.corners + (1 - a) * pts
            self._set_homography(self.corners)
            self.missing = 0
            self.tracking = True
        else:
            self.missing += 1
            if self.H is None or self.missing > self.hold_frames:
                self.tracking = False
                self._use_fallback(frame_bgr.shape)

        return self.tracking

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
