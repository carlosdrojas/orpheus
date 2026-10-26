"""Air guitar: webcam -> hand landmarks -> board space -> guitar events -> sound.

    camera --> MediaPipe hands --> Board (ArUco homography) --> GuitarState --> events
                                                                                 |-> Synth / MIDI
                                                                                 |-> console / JSONL log

Hold the board (with ArUco markers 0-3 at its corners) like a guitar neck.
Without markers, a fixed rectangle in the middle of the frame is the board.

  fretting hand: index/middle/ring/pinky tips inside the fret zone hold frets
  picking hand:  index tip crossing a string line in the strum zone plucks it

Controls
  s  toggle skeleton overlay
  d  toggle board diagnostics (marker IDs, fit quality, tracking %)
  l  swap which hand frets / picks (lefty mode)
  q  quit

Options
  --midi         also send to a virtual MIDI port named "orpheus"
  --no-audio     don't run the built-in synth
  --log FILE     append events as JSON lines
  --lefty        start with fretting = left hand
  --diag         start with diagnostics on
  --layout FILE  board layout JSON from gen_board.py (default: board_layout.json if present,
                 else the original 4 corner markers)
"""

import argparse
import json
import time
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

import layout as layout_mod
from board import Board
from guitar import (FRET_ZONE, NUM_FRETS, STRUM_START, NUM_STRINGS, OPEN_NOTES, GuitarState,
                    fret_u, note_name, string_v)
from hand_shapes import (INDEX_TIP, MIDDLE_TIP, MODEL_PATH, PINKY_TIP, RING_TIP,
                         draw_label, draw_skeleton, to_pixels)

FRET_TIPS = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
PICK_TIP = INDEX_TIP

COL_STRING = (200, 200, 200)
COL_FRET = (120, 120, 120)
COL_HELD = (0, 200, 255)
COL_PLUCK = (80, 255, 120)
COL_FRET_HAND = (255, 180, 0)
COL_PICK_HAND = (0, 220, 255)


def make_landmarker():
    return vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.5,
    ))


def draw_board(frame, board, state, flash):
    """Strings, frets, strum zone, held cells and recently plucked strings."""
    board.draw_quad(frame, STRUM_START, 0, 1, 1, (60, 60, 60), alpha=0.25)  # strum zone
    board.draw_quad(frame, FRET_ZONE, 0, STRUM_START, 1, (0, 0, 0), alpha=0.45)  # dead band
    for f in range(1, NUM_FRETS + 1):
        u = fret_u(f)
        board.draw_line(frame, (u, 0), (u, 1), COL_FRET)
    board.draw_line(frame, (FRET_ZONE, 0), (FRET_ZONE, 1), COL_FRET, 2)
    for s in range(NUM_STRINGS):
        v = string_v(s)
        if state.held[s]:
            u0, u1 = fret_u(state.held[s]), fret_u(state.held[s] + 1)
            board.draw_quad(frame, u0, v - 0.5 / NUM_STRINGS, u1, v + 0.5 / NUM_STRINGS, COL_HELD, 0.35)
        age = time.time() - flash[s]
        if age < 0.25:
            board.draw_line(frame, (0, v), (1, v), COL_PLUCK, 3)
        else:
            board.draw_line(frame, (0, v), (1, v), COL_STRING, 1)
        px = board.to_pixel([(1.02, v)])[0].astype(int)
        cv2.putText(frame, note_name(OPEN_NOTES[s] + state.held[s]), tuple(px),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_HELD if state.held[s] else COL_STRING, 1, cv2.LINE_AA)
    board.draw_outline(frame)


def draw_diag(frame, board, modes):
    """Marker outlines/IDs plus fit stats, top right."""
    board.draw_markers(frame)
    total = len(board.known)
    err = f"{board.reproj_err_cm:.2f}cm" if board.reproj_err_cm is not None else "-"
    mpx = f"{board.marker_px:.0f}px" if board.marker_px else "-"
    track = 100 * sum(m == "tracking" for m in modes) / max(len(modes), 1)
    lines = [f"board: {board.mode}  ({track:.0f}% tracking, last {len(modes)} frames)",
             f"markers {len(board.seen_ids)}/{total}: {sorted(board.seen_ids)}",
             f"inliers {board.inliers}  reproj err {err}  marker size {mpx}"]
    w = frame.shape[1]
    for i, line in enumerate(lines):
        draw_label(frame, line, (w - 470, 28 + 22 * i), 0.5, (0, 255, 120) if board.tracking else (0, 140, 255))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--midi", action="store_true")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--log")
    ap.add_argument("--lefty", action="store_true")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--diag", action="store_true")
    ap.add_argument("--layout")
    args = ap.parse_args()

    outputs = []
    if not args.no_audio:
        from synth import Synth
        outputs.append(Synth())
    if args.midi:
        from synth import MidiOut
        outputs.append(MidiOut())
    log = open(args.log, "a") if args.log else None

    landmarker = make_landmarker()
    layout_path = args.layout or (layout_mod.DEFAULT_PATH if layout_mod.DEFAULT_PATH.exists() else None)
    board = Board(layout_mod.load(layout_path) if layout_path else None)
    print(f"board layout: {layout_path or 'legacy 4 corner markers'}")
    state = GuitarState()
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam.")

    fret_label = "Left" if args.lefty else "Right"
    show_skeleton = False
    show_diag = args.diag
    modes = deque(maxlen=150)               # recent board modes, for the rolling tracking %
    mode_counts = {"tracking": 0, "hold": 0, "fallback": 0}
    marker_px_seen = []
    flash = [0.0] * NUM_STRINGS
    recent = []  # last few events for on-screen log
    t0 = time.time()
    fps, last = 0.0, time.time()

    failed_reads = 0
    while True:
        ok, frame = cap.read()
        if not ok:  # cameras often drop the first reads while starting up
            failed_reads += 1
            if failed_reads > 60:
                print(f"Camera {args.camera} stopped giving frames. Check camera permission for this "
                      "terminal, or try another --camera index (virtual cameras like OBS give none).")
                break
            time.sleep(0.03)
            continue
        failed_reads = 0
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        now = time.time()

        board.update(frame, mirrored=True)
        modes.append(board.mode)
        mode_counts[board.mode] += 1
        if board.marker_px:
            marker_px_seen.append(board.marker_px)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                                             int((now - t0) * 1000))

        fret_tips_b, pick_tip_b = [], None
        for lms, handed in zip(result.hand_landmarks, result.handedness):
            pts = to_pixels(lms, w, h)
            is_fret = handed[0].category_name == fret_label
            if show_skeleton:
                draw_skeleton(frame, pts)
            if is_fret:
                px = pts[FRET_TIPS]
                fret_tips_b = board.to_board(px)
                for p in px.astype(int):
                    cv2.circle(frame, tuple(p), 7, COL_FRET_HAND, -1, cv2.LINE_AA)
            else:
                px = pts[PICK_TIP]
                pick_tip_b = tuple(board.to_board([px])[0])
                cv2.circle(frame, tuple(px.astype(int)), 9, COL_PICK_HAND, -1, cv2.LINE_AA)

        events = state.update_frets(fret_tips_b, now) + state.update_pick(pick_tip_b, now)
        for ev in events:
            if ev.kind == "pluck":
                flash[ev.string] = now
                for out in outputs:
                    out.pluck(ev.string, ev.note, ev.velocity)
            elif ev.kind == "unfret":  # lifting the finger damps the string
                for out in outputs:
                    out.mute(ev.string)
            print(ev)
            if log:
                log.write(json.dumps(ev.__dict__) + "\n")
            recent.append(str(ev))
        recent = recent[-8:]

        draw_board(frame, board, state, flash)

        now2 = time.time()
        fps = 0.9 * fps + 0.1 / max(now2 - last, 1e-6)
        last = now2
        draw_label(frame, f"fret hand: {fret_label}   fps: {fps:.0f}", (10, 28), 0.6)
        if show_diag:
            draw_diag(frame, board, modes)
        for i, line in enumerate(recent):
            draw_label(frame, line, (10, h - 12 - 20 * (len(recent) - 1 - i)), 0.45, (200, 200, 200))

        cv2.imshow("air guitar", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("s"):
            show_skeleton = not show_skeleton
        elif key == ord("d"):
            show_diag = not show_diag
        elif key == ord("l"):
            fret_label = "Left" if fret_label == "Right" else "Right"

    frames = sum(mode_counts.values())
    if frames:
        pct = "  ".join(f"{k} {100 * n / frames:.0f}%" for k, n in mode_counts.items())
        mpx = f"{np.median(marker_px_seen):.0f}px" if marker_px_seen else "none seen"
        print(f"board: {pct} | median marker {mpx} | {frames / (time.time() - t0):.1f} fps over {frames} frames")

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    for out in outputs:
        out.close()
    if log:
        log.close()


if __name__ == "__main__":
    main()
