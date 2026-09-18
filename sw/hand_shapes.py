"""Webcam hand tracker: fingertips become points, points get connected into shapes.

Controls
  m  cycle draw mode (polygon -> hull -> web -> skeleton)
  h  toggle the hand skeleton overlay
  q  quit
"""

import time
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

MODEL_PATH = Path(__file__).parent / "models" / "hand_landmarker.task"

# Landmark indices (see https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]
FINGER_TIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
FINGER_PIPS = [THUMB_IP, INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]

# Bone connections for the optional skeleton overlay.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

DRAW_MODES = ["polygon", "hull", "web", "skeleton"]

# One colour per finger so you can tell which point is which.
FINGER_COLORS = [
    (80, 200, 255),   # thumb  - orange-ish (BGR)
    (80, 255, 120),   # index  - green
    (255, 200, 80),   # middle - light blue
    (255, 120, 220),  # ring   - pink
    (120, 120, 255),  # pinky  - red
]

SHAPE_NAMES = {0: "", 1: "point", 2: "line", 3: "triangle", 4: "quad", 5: "pentagon"}


# ---------------------------------------------------------------------------
# Geometry / gesture helpers
# ---------------------------------------------------------------------------

def to_pixels(landmarks, w, h):
    """Normalized landmarks -> (21, 2) int array of pixel coords."""
    return np.array([(lm.x * w, lm.y * h) for lm in landmarks], dtype=np.float32)


def extended_fingers(pts):
    """Return a list of 5 bools: is each finger extended?

    Rotation-tolerant heuristic: a finger is extended if its tip is farther
    from the wrist than its middle joint. The thumb is measured against the
    pinky MCP instead so it works whether the palm faces in or out.
    """
    wrist = pts[WRIST]
    out = []
    for i, (tip, pip) in enumerate(zip(FINGER_TIPS, FINGER_PIPS)):
        if i == 0:  # thumb
            ref = pts[PINKY_MCP]
            out.append(bool(np.linalg.norm(pts[tip] - ref) > np.linalg.norm(pts[pip] - ref) * 1.15))
        else:
            out.append(bool(np.linalg.norm(pts[tip] - wrist) > np.linalg.norm(pts[pip] - wrist)))
    return out


def classify_gesture(ext, pts):
    """Map a finger-extension pattern to a friendly gesture name."""
    thumb, index, middle, ring, pinky = ext
    n = sum(ext)

    # Thumb + index tips touching -> OK sign
    pinch = np.linalg.norm(pts[THUMB_TIP] - pts[INDEX_TIP])
    palm = np.linalg.norm(pts[WRIST] - pts[MIDDLE_MCP])
    if pinch < palm * 0.35 and middle and ring and pinky:
        return "ok"

    if n == 0:
        return "fist"
    if n == 5:
        return "open palm"
    if ext == [False, True, False, False, False]:
        return "point"
    if ext == [False, True, True, False, False]:
        return "peace"
    if ext == [True, False, False, False, False]:
        return "thumbs up"
    if ext == [True, False, False, False, True]:
        return "call me"
    if ext == [False, True, False, False, True]:
        return "rock"
    if ext == [True, True, False, False, True]:
        return "love"
    if ext == [False, True, True, True, False]:
        return "three"
    if ext == [False, True, True, True, True]:
        return "four"
    return f"{n} up"


def order_around_centroid(points):
    """Sort points by angle around their centroid so the polygon doesn't self-cross."""
    if len(points) < 3:
        return points
    c = points.mean(axis=0)
    ang = np.arctan2(points[:, 1] - c[1], points[:, 0] - c[0])
    return points[np.argsort(ang)]


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_skeleton(frame, pts):
    ip = pts.astype(int)
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, tuple(ip[a]), tuple(ip[b]), (90, 90, 90), 1, cv2.LINE_AA)
    for p in ip:
        cv2.circle(frame, tuple(p), 2, (160, 160, 160), -1, cv2.LINE_AA)


def draw_shape(frame, points, mode, color):
    """Connect a set of points according to the draw mode."""
    n = len(points)
    if n < 2:
        return
    ip = points.astype(int)

    if mode == "polygon":
        ordered = order_around_centroid(points).astype(int)
        if n >= 3:
            overlay = frame.copy()
            cv2.fillPoly(overlay, [ordered.reshape(-1, 1, 2)], color)
            cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
        cv2.polylines(frame, [ordered.reshape(-1, 1, 2)], n >= 3, color, 2, cv2.LINE_AA)

    elif mode == "hull":
        if n >= 3:
            hull = cv2.convexHull(ip.reshape(-1, 1, 2))
            overlay = frame.copy()
            cv2.fillPoly(overlay, [hull], color)
            cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
            cv2.polylines(frame, [hull], True, color, 2, cv2.LINE_AA)
        else:
            cv2.line(frame, tuple(ip[0]), tuple(ip[1]), color, 2, cv2.LINE_AA)

    elif mode == "web":
        for i in range(n):
            for j in range(i + 1, n):
                cv2.line(frame, tuple(ip[i]), tuple(ip[j]), color, 1, cv2.LINE_AA)


def draw_label(frame, text, origin, scale=0.7, color=(255, 255, 255)):
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    if not MODEL_PATH.exists():
        raise SystemExit(
            f"Missing model at {MODEL_PATH}.\nDownload it with:\n"
            "  curl -L -o models/hand_landmarker.task "
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
        )

    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.5,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam (index 0).")

    mode_idx = 0
    show_skeleton = True
    t0 = time.time()
    fps = 0.0
    last = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # mirror so it feels natural
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_img, int((time.time() - t0) * 1000))

        mode = DRAW_MODES[mode_idx]
        all_tips = []  # extended fingertips across both hands
        y_text = 60

        for hand_lms, handed in zip(result.hand_landmarks, result.handedness):
            pts = to_pixels(hand_lms, w, h)
            ext = extended_fingers(pts)
            gesture = classify_gesture(ext, pts)
            side = handed[0].category_name  # "Left" / "Right" (already mirror-corrected by flip)

            if show_skeleton or mode == "skeleton":
                draw_skeleton(frame, pts)

            tips = np.array([pts[t] for t, e in zip(FINGER_TIPS, ext) if e], dtype=np.float32)
            all_tips.extend(tips.tolist())

            # One shape per hand (unless in skeleton-only mode)
            hand_color = (0, 220, 255) if side == "Right" else (255, 180, 0)
            if mode != "skeleton":
                draw_shape(frame, tips, mode, hand_color)

            # Fingertip points, colour-coded per finger
            for i, tip in enumerate(FINGER_TIPS):
                p = tuple(pts[tip].astype(int))
                if ext[i]:
                    cv2.circle(frame, p, 9, FINGER_COLORS[i], -1, cv2.LINE_AA)
                    cv2.circle(frame, p, 9, (255, 255, 255), 1, cv2.LINE_AA)
                else:
                    cv2.circle(frame, p, 4, FINGER_COLORS[i], 1, cv2.LINE_AA)

            shape = SHAPE_NAMES.get(len(tips), f"{len(tips)}-gon")
            draw_label(frame, f"{side}: {gesture}  [{shape}]", (10, y_text), 0.7, hand_color)
            y_text += 28

        # Two hands -> also connect all extended tips into one big shape
        if len(result.hand_landmarks) == 2 and mode != "skeleton" and len(all_tips) >= 3:
            draw_shape(frame, np.array(all_tips, dtype=np.float32), mode, (200, 200, 200))

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
        last = now

        draw_label(frame, f"mode: {mode}   fps: {fps:.0f}", (10, 28), 0.6)
        draw_label(frame, "m: mode  h: skeleton  q: quit", (10, h - 12), 0.5, (200, 200, 200))

        cv2.imshow("hand shapes", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
        elif key == ord("m"):
            mode_idx = (mode_idx + 1) % len(DRAW_MODES)
        elif key == ord("h"):
            show_skeleton = not show_skeleton

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
