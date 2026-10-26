"""
test_logic.py - checks the gesture + sound logic with fake hand data (no camera needed).

  python test_logic.py
"""
import math

import numpy as np

from air_guitar import (CHORDS, FingerChordSelector, SignChordSelector, StrumDetector,
                        StrumRenderer, ZoneChordSelector, auto_voicing, chord_notes,
                        count_raised_fingers, karplus_strong, midi_to_hz,
                        read_hand_number, scale_chords)

FPS = 30.0
HAND = 0.10  # hand length as a fraction of the frame


def run(detector, ys, fps=FPS):
    events = []
    for i, y in enumerate(ys):
        ev = detector.update(y, HAND, i / fps)
        if ev:
            events.append(ev)
    return events


def test_strumming_alternates():
    # hand swings 0.35..0.75 at 2 Hz (4 strums per second) for 3 seconds
    t = np.arange(0, 3, 1 / FPS)
    ys = 0.55 + 0.20 * np.sin(2 * math.pi * 2 * t)
    ev = run(StrumDetector(), ys)
    dirs = [e.direction for e in ev]
    assert 10 <= len(ev) <= 13, len(ev)
    assert all(a != b for a, b in zip(dirs, dirs[1:])), dirs  # down/up alternate
    print(f"ok  steady strumming: {len(ev)} strums in 3 s, alternating {dirs[:4]}...")


def test_slow_drift_is_silent():
    ys = np.linspace(0.30, 0.80, int(4 * FPS))  # crosses the line slowly over 4 s
    assert run(StrumDetector(), ys) == []
    print("ok  slow drift across the line makes no sound")


def test_jitter_near_line_is_silent():
    rng = np.random.default_rng(0)
    ys = 0.55 + rng.normal(0, 0.008, int(3 * FPS))  # tracking noise sitting on the line
    assert run(StrumDetector(), ys) == []
    print("ok  tracking jitter on the line does not retrigger")


def test_fast_jump_between_frames_is_caught():
    ys = [0.40] * 5 + [0.70] * 5  # hand crosses the whole band in one frame
    ev = run(StrumDetector(), ys)
    assert len(ev) == 1 and ev[0].direction == "down" and ev[0].strength > 0.5
    print(f"ok  very fast strum caught, strength {ev[0].strength:.2f}")


def test_strength_scales_with_speed():
    def strength(amplitude_per_frame):
        ys = [min(0.40 + amplitude_per_frame * i, 0.95) for i in range(30)]
        ev = run(StrumDetector(), ys)
        return ev[0].strength
    soft, hard = strength(0.012), strength(0.05)
    assert hard > soft, (soft, hard)
    print(f"ok  harder strum is louder ({soft:.2f} -> {hard:.2f})")


def test_low_frame_rate_still_works():
    t = np.arange(0, 3, 1 / 15)
    ys = 0.55 + 0.20 * np.sin(2 * math.pi * 2 * t)
    ev = run(StrumDetector(), ys, fps=15)
    assert len(ev) >= 10, len(ev)
    print(f"ok  still works at 15 fps ({len(ev)} strums)")


def test_zones_with_hysteresis():
    z = ZoneChordSelector(4, 0.0, 0.4, hysteresis=0.02)
    assert [z.update(x) for x in (0.05, 0.15, 0.25, 0.35)] == [0, 1, 2, 3]
    z.update(0.15)
    assert z.update(0.21) == 1  # just past the edge at 0.20: stays (hysteresis)
    assert z.update(0.23) == 2
    print("ok  neck zones pick the right chord and don't flicker at edges")


def fake_hand(raised):
    """21 landmarks: wrist at origin, fingers point up; raised fingers extended."""
    pts = [(0.0, 0.0)] * 21
    pts = list(pts)
    for f, (mcp, pip, dip, tip) in enumerate([(5, 6, 7, 8), (9, 10, 11, 12),
                                               (13, 14, 15, 16), (17, 18, 19, 20)]):
        x = -0.03 + 0.02 * f
        pts[mcp] = (x, -0.10)
        pts[pip] = (x, -0.14)
        if f < raised:
            pts[dip], pts[tip] = (x, -0.17), (x, -0.20)
        else:  # curled back toward the palm
            pts[dip], pts[tip] = (x, -0.12), (x, -0.09)
    return pts


def test_finger_counting():
    for n in range(5):
        assert count_raised_fingers(fake_hand(n)) == n
    sel = FingerChordSelector(4, stable_frames=3)
    for _ in range(3):
        sel.update(3)
    assert sel.zone == 2
    sel.update(1)  # one noisy frame should not switch the chord
    assert sel.zone == 2
    print("ok  finger counting 0-4 and debouncing")


def sign_hand(up=(), thumb="folded", ok=False, flip=False, rotate=0.0):
    """
    Build 21 landmarks for a hand facing the camera, fingers pointing up.
    up      : which of "imrp" (index, middle, ring, pinky) are raised
    thumb   : "out" (sticking out sideways) or "folded" (across the palm)
    ok      : index curls round to touch the thumb tip ("OK" sign)
    flip    : mirror left/right (back of the hand toward the camera)
    rotate  : degrees, clockwise on screen
    """
    p = [(0.0, 0.0)] * 21
    p[1], p[2] = (-0.03, -0.03), (-0.05, -0.05)
    if thumb == "out":
        p[3], p[4] = (-0.08, -0.07), (-0.11, -0.09)
    elif thumb == "loose":   # relaxed: only a little away from the index, pointing mostly up
        p[3], p[4] = (-0.065, -0.09), (-0.07, -0.13)
    elif thumb == "half":    # folded partway, tip resting near the index knuckle
        p[3], p[4] = (-0.045, -0.085), (-0.025, -0.10)
    else:
        p[3], p[4] = (-0.03, -0.08), (0.00, -0.09)
    for name, (mcp, pip, dip, tip), x in zip("imrp", [(5, 6, 7, 8), (9, 10, 11, 12),
                                                     (13, 14, 15, 16), (17, 18, 19, 20)],
                                             (-0.03, -0.01, 0.01, 0.03)):
        p[mcp], p[pip] = (x, -0.10 if name != "p" else -0.09), (x, -0.14)
        if name in up:
            p[dip], p[tip] = (x, -0.17), (x, -0.20)
        else:
            p[dip], p[tip] = (x, -0.12), (x, -0.09)
    if ok:  # index bends forward so its tip meets the thumb tip
        p[3], p[4] = (-0.07, -0.09), (-0.065, -0.128)
        p[6], p[7], p[8] = (-0.05, -0.14), (-0.065, -0.145), (-0.06, -0.13)
    if flip:
        p = [(-x, y) for x, y in p]
    a = math.radians(rotate)
    p = [(x * math.cos(a) - y * math.sin(a), x * math.sin(a) + y * math.cos(a)) for x, y in p]
    return [(0.5 + x, 0.5 + y) for x, y in p]


# Wrist, thumb and index positions read off a real screenshot of the 7
# (pixels / 436): wrist at the top, thumb running sideways, index hanging
# down. Folded fingers drawn as tight curls, matching the [TI---] readout.
_px = {0: (135, 62), 1: (183, 64), 2: (237, 88), 3: (275, 100), 4: (310, 97),
       5: (230, 128), 6: (234, 198), 7: (236, 235), 8: (236, 265),
       9: (198, 135), 10: (200, 175), 11: (195, 160), 12: (192, 145),
       13: (162, 140), 14: (164, 178), 15: (160, 162), 16: (157, 148),
       17: (128, 145), 18: (130, 175), 19: (127, 160), 20: (125, 150)}
PHOTO_SEVEN = [(_px[i][0] / 436, _px[i][1] / 436) for i in range(21)]


def test_hand_signs_one_to_seven():
    cases = {
        "1 (index)":                  (sign_hand("i"), 1),
        "1 (index, thumb sticking out)": (sign_hand("i", thumb="out"), 1),
        "2 (index + middle)":         (sign_hand("im"), 2),
        "2 (index + middle, thumb out)": (sign_hand("im", thumb="out"), 2),
        "3 (index, middle, ring)":    (sign_hand("imr"), 3),
        "3 (index, middle, ring, thumb out)": (sign_hand("imr", thumb="out"), 3),
        "4 (four fingers)":           (sign_hand("imrp"), 4),
        "5 (open hand)":              (sign_hand("imrp", thumb="out"), 5),
        "6 (OK sign)":                (sign_hand("mrp", ok=True), 6),
        "6 (OK sign, tilted)":        (sign_hand("mrp", ok=True, rotate=30), 6),
        "7 (upside down, thumb right)": (sign_hand("i", thumb="out", rotate=180), 7),
        "7 (upside down, thumb left)":  (sign_hand("i", thumb="out", flip=True, rotate=180), 7),
        "7 (index angled down)":      (sign_hand("i", thumb="out", rotate=145), 7),
        "7 (no clear thumb)":         (sign_hand("i", rotate=180), 7),
        "7 (from your photo)":        (PHOTO_SEVEN, 7),
        "1 (index sideways, not a 7)": (sign_hand("i", thumb="out", rotate=90), 1),
        "5 (thumb only loosely out)": (sign_hand("imrp", thumb="loose"), 5),
        "1 (thumb half tucked)":      (sign_hand("i", thumb="half"), 1),
        "4 (thumb half tucked)":      (sign_hand("imrp", thumb="half"), 4),
        "fist":                       (sign_hand(""), None),
        "5 (open hand, tilted)":      (sign_hand("imrp", thumb="out", rotate=-25), 5),
    }
    for label, (pts, want) in cases.items():
        got = read_hand_number(pts)
        assert got == want, f"{label}: read {got}, expected {want}"
    print(f"ok  hand signs read correctly ({len(cases)} shapes, 1-7 and fist)")


def test_sign_selector_debounce():
    sel = SignChordSelector(7, stable_frames=4)
    for _ in range(4):
        sel.update(7)
    assert sel.zone == 6
    sel.update(2)                 # one stray frame while changing shape
    sel.update(None)              # fist
    assert sel.zone == 6
    for _ in range(4):
        sel.update(3)
    assert sel.zone == 2
    print("ok  chord changes only after a sign holds for 4 frames; fist keeps the chord")


def test_scale_chords():
    assert [n for n, _ in scale_chords("G")] == ["G", "Am", "Bm", "C", "D", "Em", "F#dim"]
    assert [r for _, r in scale_chords("G")] == ["I", "ii", "iii", "IV", "V", "vi", "vii°"]
    assert [n for n, _ in scale_chords("D")] == ["D", "Em", "F#m", "G", "A", "Bm", "C#dim"]
    assert [n for n, _ in scale_chords("F")] == ["F", "Gm", "Am", "Bb", "C", "Dm", "Edim"]
    assert [n for n, _ in scale_chords("Am")] == ["Am", "Bdim", "C", "Dm", "Em", "F", "G"]
    # the automatic shape finder reproduces the standard open chords
    assert auto_voicing({0, 4, 7}, 0) == [None, 3, 2, 0, 1, 0]          # C
    assert auto_voicing({7, 11, 2}, 7) == [3, 2, 0, 0, 0, 3]            # G
    assert auto_voicing({2, 6, 9}, 2) == [None, None, 0, 2, 3, 2]       # D
    for key in ("G", "C", "D", "E", "F", "Bb", "Em", "Am", "F#m"):
        for name, _ in scale_chords(key):
            notes = chord_notes(CHORDS[name])
            assert len(notes) >= 4, (key, name, CHORDS[name])
    print("ok  keys give the right 7 chords (G: G Am Bm C D Em F#dim), each with a shape")


def test_pluck_pitch():
    for midi in (40, 64):
        s = karplus_strong(midi_to_hz(midi), duration=1.0, seed=1)
        seg = s[2000:2000 + 8192].astype(np.float64)
        ac = np.correlate(seg, seg, mode="full")[len(seg) - 1:]
        lo, hi = int(44100 / 1400), int(44100 / 60)   # search 60..1400 Hz
        lag = lo + int(np.argmax(ac[lo:hi]))
        f0, expected = 44100 / lag, midi_to_hz(midi)
        assert abs(f0 / expected - 1) < 0.02, (f0, expected)  # within ~1/3 semitone
    print("ok  synthesized strings are in tune")


def test_chords_and_render():
    assert chord_notes(CHORDS["Em"]) == [40, 47, 52, 55, 59, 64]
    r = StrumRenderer()
    down = r.render("G", "down", 0.8)
    up = r.render("G", "up", 0.3)
    assert np.max(np.abs(down)) <= 1.0 and np.max(np.abs(down)) > 0.1
    assert np.max(np.abs(up)) < np.max(np.abs(down))
    print("ok  chord strums render, soft up-strum is quieter")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("\nall tests passed")
