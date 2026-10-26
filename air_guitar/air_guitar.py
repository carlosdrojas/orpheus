"""
air_guitar.py - Air guitar -> sound (and optionally -> servos) proof of concept.

What it does
  * Webcam + MediaPipe Hand Landmarker tracks both hands (21 points each).
  * STRUM HAND: a strum fires when the hand crosses an on-screen "string line".
      - direction  = which way it crossed (down / up)
      - loudness   = how fast it crossed
  * FRET HAND picks the chord, two modes (press M to switch):
      - scale : a hand sign 1-7 picks that chord of the key (default)
                  1-4 = index, +middle, +ring, +pinky (thumb ignored)
                  5   = open hand (all four + thumb)
                  6   = "OK" sign (thumb and index tips touching)
                  7   = hand upside down: thumb sideways along the top,
                        index pointing down, other fingers folded
                  fist = keep the current chord
                In G major: 1 G, 2 Am, 3 Bm, 4 C, 5 D, 6 Em, 7 F#dim
      - zones : where the hand sits along the imaginary neck (--chords list)
  * Plays a synthesized guitar strum (Karplus-Strong) through your speakers.
  * Optional: sends one-line text commands over USB serial to an MSPM0
    LaunchPad that drives the strum / fret servos (see firmware_mspm0/).

Run
  pip install -r requirements.txt
  python air_guitar.py                      # sound only, key of G major
  python air_guitar.py --key D              # any major key
  python air_guitar.py --key Am             # natural minor keys end in "m"
  python air_guitar.py --port COM5          # sound + servos (Windows COM port)
  python air_guitar.py --lefty              # left-handed players
  python air_guitar.py --no-sound --port COM5

Keys (in the video window)
  Q / Esc  quit
  M        switch chord mode (scale <-> zones)
  C        move the string line to where your strum hand is now
  [ / ]    strum sensitivity down / up
  S        toggle sound on/off

Serial protocol (115200 baud, ASCII, one command per line, '\n' terminated)
  C<i>        select chord i, 0-based (scale mode: 0 = chord 1 ... 6 = chord 7;
              zones mode: index into the --chords list)
  D<v>        strum down, v = 0..127 strength
  U<v>        strum up,   v = 0..127 strength
  S<ch>,<us>  set servo channel <ch> to a raw pulse width in microseconds
  R           all servos to rest
  P           ping (firmware answers "PONG")
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------
# Music data
# --------------------------------------------------------------------------
SAMPLE_RATE = 44100
OPEN_STRINGS = [40, 45, 50, 55, 59, 64]  # E2 A2 D3 G3 B3 E4 as MIDI note numbers

# Fret pressed on each string, low E -> high E. None = string not played.
# These same shapes are what a fretting mechanism has to press.
CHORDS = {
    "G":  [3, 2, 0, 0, 0, 3],
    "C":  [None, 3, 2, 0, 1, 0],
    "D":  [None, None, 0, 2, 3, 2],
    "Em": [0, 2, 2, 0, 0, 0],
    "Am": [None, 0, 2, 2, 1, 0],
    "E":  [0, 2, 2, 1, 0, 0],
    "A":  [None, 0, 2, 2, 2, 0],
    "F":  [1, 3, 3, 2, 1, 1],
}
DEFAULT_PROGRESSION = ["G", "C", "D", "Em"]

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")


def midi_to_hz(m: float) -> float:
    return 440.0 * 2.0 ** ((m - 69) / 12.0)


def chord_notes(frets: Sequence[Optional[int]]) -> List[int]:
    """MIDI notes for a chord shape, low string first."""
    return [OPEN_STRINGS[i] + f for i, f in enumerate(frets) if f is not None]


# --------------------------------------------------------------------------
# Keys and scale-degree chords (1-7)
# --------------------------------------------------------------------------
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
MAJOR_STEPS = [0, 2, 4, 5, 7, 9, 11]
MINOR_STEPS = [0, 2, 3, 5, 7, 8, 10]   # natural minor
ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII"]


def parse_key(key: str) -> Tuple[int, bool, bool]:
    """'G' -> (7, major, sharps), 'Am' -> (9, minor, sharps), 'Bb' -> (10, major, flats)."""
    k = key.strip()
    minor = k.endswith("m") and len(k) > 1
    tonic = k[:-1] if minor else k
    tonic = tonic[0].upper() + tonic[1:]
    if tonic in NOTE_NAMES:
        pc = NOTE_NAMES.index(tonic)
    elif tonic in FLAT_NAMES:
        pc = FLAT_NAMES.index(tonic)
    else:
        raise ValueError(f"Unknown key {key!r}. Examples: G, D, Bb, F#, Em, Am")
    use_flats = "b" in tonic or tonic == "F" or (minor and tonic in ("D", "G", "C", "F"))
    return pc, minor, use_flats


def auto_voicing(chord_pcs: Sequence[int], root_pc: int, max_fret: int = 4) -> List[Optional[int]]:
    """
    Simple open-position guitar shape: the lowest string that can play the root
    (within frets 0-4) plays it, strings below are muted, and every higher
    string plays its lowest chord tone. Reproduces the standard open G, C, D,
    Am and Em shapes, and gives playable shapes for the other chords.
    """
    frets: List[Optional[int]] = [None] * 6
    started = False
    for s, open_midi in enumerate(OPEN_STRINGS):
        options = [f for f in range(max_fret + 1) if (open_midi + f) % 12 in chord_pcs]
        if not started:
            roots = [f for f in options if (open_midi + f) % 12 == root_pc]
            if roots:
                frets[s] = roots[0]
                started = True
        elif options:
            frets[s] = options[0]
    return frets


def scale_chords(key: str) -> List[Tuple[str, str]]:
    """
    The 7 triads of a key as (chord name, roman numeral), e.g. for G:
    G (I), Am (ii), Bm (iii), C (IV), D (V), Em (vi), F#dim (vii°).
    Every chord gets a shape in CHORDS so the synth and the fretter can use it.
    """
    tonic, minor, flats = parse_key(key)
    steps = MINOR_STEPS if minor else MAJOR_STEPS
    names = FLAT_NAMES if flats else NOTE_NAMES
    out = []
    for d in range(7):
        root = (tonic + steps[d]) % 12
        third = (tonic + steps[(d + 2) % 7]) % 12
        fifth = (tonic + steps[(d + 4) % 7]) % 12
        i3, i5 = (third - root) % 12, (fifth - root) % 12
        if i3 == 4 and i5 == 7:
            suffix, roman = "", ROMAN[d]
        elif i3 == 3 and i5 == 7:
            suffix, roman = "m", ROMAN[d].lower()
        elif i3 == 3 and i5 == 6:
            suffix, roman = "dim", ROMAN[d].lower() + "°"
        else:
            suffix, roman = "aug", ROMAN[d] + "+"
        name = names[root] + suffix
        if name not in CHORDS:
            CHORDS[name] = auto_voicing({root, third, fifth}, root)
        out.append((name, roman))
    return out


# --------------------------------------------------------------------------
# Sound: Karplus-Strong plucked string (vectorised, no extra libraries)
# --------------------------------------------------------------------------
def karplus_strong(freq: float, duration: float = 2.5, sr: int = SAMPLE_RATE,
                   decay: float = 0.996, brightness: float = 0.6,
                   seed: Optional[int] = None) -> np.ndarray:
    """
    Plucked-string tone. y[n] = decay * (y[n-N] + y[n-N-1]) / 2.
    Computed one period (N samples) at a time, which is exact because every
    sample in a block depends only on samples at least N-1 steps earlier.
    """
    rng = np.random.default_rng(seed)
    n_total = int(sr * duration)
    period = max(2, int(round(sr / freq)))
    out = np.zeros(n_total + period + 1, dtype=np.float64)

    burst = rng.uniform(-1.0, 1.0, period)
    # one-pole low-pass on the burst: lower brightness = softer, rounder tone
    a = float(np.clip(brightness, 0.05, 1.0))
    for i in range(1, period):
        burst[i] = a * burst[i] + (1 - a) * burst[i - 1]
    out[:period] = burst - burst.mean()

    k = period
    length = len(out)
    while k < length:
        end = min(k + period, length)
        span = end - k
        prev = out[k - period:k - period + span]
        idx = np.arange(k - period - 1, k - period - 1 + span)
        prev1 = np.where(idx >= 0, out[np.maximum(idx, 0)], 0.0)
        out[k:end] = decay * 0.5 * (prev + prev1)
        k = end

    out = out[:n_total]
    fade = min(len(out), int(0.05 * sr))
    out[-fade:] *= np.linspace(1.0, 0.0, fade)
    peak = np.max(np.abs(out)) or 1.0
    return (out / peak).astype(np.float32)


class StrumRenderer:
    """Caches one sample per note and mixes them into strums on demand."""

    def __init__(self, sr: int = SAMPLE_RATE):
        self.sr = sr
        self._cache = {}

    def note(self, midi: int) -> np.ndarray:
        if midi not in self._cache:
            self._cache[midi] = karplus_strong(midi_to_hz(midi), sr=self.sr, seed=midi)
        return self._cache[midi]

    def warm(self, chord_names: Sequence[str]) -> None:
        for name in chord_names:
            for m in chord_notes(CHORDS[name]):
                self.note(m)

    def render(self, chord_name: str, direction: str, strength: float) -> np.ndarray:
        """
        direction 'down' = low string first (pick moves toward the floor).
        strength 0..1: louder and tighter (less gap between strings) when higher.
        """
        notes = chord_notes(CHORDS[chord_name])
        if direction == "up":
            notes = list(reversed(notes))
            notes = notes[:4]          # up-strums usually catch only the top strings
        strength = float(np.clip(strength, 0.0, 1.0))
        gap = int(self.sr * (0.030 - 0.022 * strength))  # 30 ms .. 8 ms per string
        gain = 0.25 + 0.55 * strength
        length = max(len(self.note(m)) for m in notes) + gap * len(notes)
        mix = np.zeros(length, dtype=np.float32)
        for i, m in enumerate(notes):
            s = self.note(m)
            start = i * gap
            # strings hit later in the strum are a touch quieter
            mix[start:start + len(s)] += s * (1.0 - 0.06 * i)
        mix *= gain / max(1.0, len(notes) ** 0.5)
        return np.clip(mix, -1.0, 1.0)


class AudioEngine:
    """Low-latency mixer on top of sounddevice. Silently disabled if unavailable."""

    def __init__(self, sr: int = SAMPLE_RATE, enabled: bool = True):
        self.sr = sr
        self.enabled = False
        self._voices: List[dict] = []
        self._lock = threading.Lock()
        self._stream = None
        if not enabled:
            return
        try:
            import sounddevice as sd  # type: ignore
            self._stream = sd.OutputStream(samplerate=sr, channels=1, dtype="float32",
                                           blocksize=256, latency="low",
                                           callback=self._callback)
            self._stream.start()
            self.enabled = True
        except Exception as exc:  # pragma: no cover - depends on the machine
            print(f"[audio] disabled ({exc}). Install with: pip install sounddevice")

    def _callback(self, outdata, frames, _time, _status):
        buf = np.zeros(frames, dtype=np.float32)
        with self._lock:
            alive = []
            for v in self._voices:
                chunk = v["buf"][v["pos"]:v["pos"] + frames]
                n = len(chunk)
                if v["fade"] is not None:
                    env = np.clip((v["fade"] - np.arange(n)) / v["fade_len"], 0, 1)
                    chunk = chunk * env
                    v["fade"] -= n
                buf[:n] += chunk
                v["pos"] += frames
                if v["pos"] < len(v["buf"]) and (v["fade"] is None or v["fade"] > 0):
                    alive.append(v)
            self._voices = alive
        outdata[:, 0] = np.tanh(buf)  # soft limiter

    def play(self, samples: np.ndarray, damp_previous: bool = True) -> None:
        if not self.enabled:
            return
        with self._lock:
            if damp_previous:  # a new strum mutes the old one, like a real guitar
                fade_len = int(0.04 * self.sr)
                for v in self._voices:
                    if v["fade"] is None:
                        v["fade"] = fade_len
                        v["fade_len"] = fade_len
            self._voices.append({"buf": samples, "pos": 0, "fade": None, "fade_len": 1})

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()


# --------------------------------------------------------------------------
# Gesture logic (pure Python, unit-tested in test_logic.py)
# --------------------------------------------------------------------------
@dataclass
class StrumEvent:
    direction: str      # "down" or "up"
    strength: float     # 0..1
    speed: float        # hand-lengths per second at the crossing
    t: float            # timestamp (s)


class StrumDetector:
    """
    Fires when the strum hand crosses a horizontal "string line".

    * Positions are normalised screen coords (0 top .. 1 bottom).
    * Speed is measured in hand-lengths per second, so it works at any
      distance from the camera.
    * A dead band around the line stops jitter from re-triggering.
    * Checking the side (not the exact line) means a very fast hand that
      jumps across the band between two frames is still caught.
    """

    def __init__(self, line_y: float = 0.55, band: float = 0.03,
                 min_speed: float = 3.0, max_speed: float = 30.0,
                 refractory_s: float = 0.06, alpha: float = 0.7):
        self.line_y = line_y
        self.band = band
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.refractory_s = refractory_s
        self.alpha = alpha
        self.reset()

    def reset(self) -> None:
        self._y = None
        self._t = None
        self._side = None
        self._speeds: List[float] = []
        self._last_fire = -1e9

    def update(self, y: float, hand_size: float, t: float) -> Optional[StrumEvent]:
        hand_size = max(hand_size, 1e-3)
        if self._y is None:
            self._y, self._t = y, t
            self._side = self._which_side(y)
            return None
        dt = max(t - self._t, 1e-3)
        y_s = self.alpha * y + (1 - self.alpha) * self._y        # light smoothing
        speed = abs(y_s - self._y) / dt / hand_size
        self._speeds = (self._speeds + [speed])[-3:]
        self._y, self._t = y_s, t

        side = self._which_side(y_s)
        event = None
        if side is not None and self._side is not None and side != self._side:
            peak = max(self._speeds)
            if peak >= self.min_speed and (t - self._last_fire) >= self.refractory_s:
                strength = (peak - self.min_speed) / (self.max_speed - self.min_speed)
                event = StrumEvent("down" if side == "below" else "up",
                                   float(np.clip(strength, 0.0, 1.0)), peak, t)
                self._last_fire = t
        if side is not None:
            self._side = side
        return event

    def _which_side(self, y: float) -> Optional[str]:
        if y < self.line_y - self.band:
            return "above"
        if y > self.line_y + self.band:
            return "below"
        return None  # inside the dead band: keep previous side


class ZoneChordSelector:
    """Splits the imaginary neck (an x-range on screen) into equal zones."""

    def __init__(self, n_zones: int, x_start: float = 0.04, x_end: float = 0.50,
                 hysteresis: float = 0.015):
        self.n = n_zones
        self.x_start = x_start
        self.x_end = x_end
        self.hysteresis = hysteresis
        self.zone = 0

    def edges(self) -> List[float]:
        w = (self.x_end - self.x_start) / self.n
        return [self.x_start + i * w for i in range(self.n + 1)]

    def update(self, x: float) -> int:
        """x = neck coordinate: 0 near the headstock, larger toward the body."""
        e = self.edges()
        lo, hi = e[self.zone] - self.hysteresis, e[self.zone + 1] + self.hysteresis
        if lo <= x <= hi:
            return self.zone
        w = (self.x_end - self.x_start) / self.n
        self.zone = int(np.clip((x - self.x_start) // w, 0, self.n - 1))
        return self.zone


# Tuning knobs for hand-sign reading. If a sign is misread, watch the
# on-screen "sees:" readout and nudge these.
FINGER_UP_RATIO = 1.2    # finger is "up" if its tip is this much farther from the wrist than its middle joint
THUMB_OUT_COS = 0.1      # thumb is "out" when the on-screen "thumb" score is below this.
                         #   Folded across the palm: about +0.3 to +1. Loosely out: about -0.4.
                         #   Straight out sideways: about -0.9. Raise it if 5 still fails;
                         #   lower it if 4 gets read as 5.
THUMB_OUT_RATIO = 1.15   # ...or if its tip is this much farther from the pinky knuckle than its base
TOUCH_RATIO = 0.35       # two fingertips "touch" if closer than this x hand length
INDEX_DOWN_RATIO = 0.3   # for the 7: index tip at least this x hand length below its knuckle


def _dist(pts, a: int, b: int) -> float:
    return math.hypot(pts[a][0] - pts[b][0], pts[a][1] - pts[b][1])


def thumb_score(pts: Sequence[Tuple[float, float]]) -> float:
    """
    How much the thumb points across the palm toward the pinky: about +1 when
    folded over the palm, about 0 when sticking up beside the index, about -1
    when sticking straight out sideways. Works at any hand angle and whether
    the palm or the back of the hand faces the camera.
    """
    tx, ty = pts[4][0] - pts[2][0], pts[4][1] - pts[2][1]      # thumb base -> tip
    px, py = pts[17][0] - pts[5][0], pts[17][1] - pts[5][1]    # index knuckle -> pinky knuckle
    norm = math.hypot(tx, ty) * math.hypot(px, py)
    return (tx * px + ty * py) / norm if norm > 1e-9 else 1.0


def finger_states(pts: Sequence[Tuple[float, float]]) -> Tuple[bool, List[bool]]:
    """
    (thumb out?, [index, middle, ring, pinky] up?). Works with the hand at
    any angle, including upside down.
    """
    fingers = [_dist(pts, tip, 0) > FINGER_UP_RATIO * _dist(pts, pip, 0)
               for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18))]
    thumb = (thumb_score(pts) < THUMB_OUT_COS or
             _dist(pts, 4, 17) > THUMB_OUT_RATIO * max(_dist(pts, 2, 17), 1e-6))
    return thumb, fingers


def aspect_correct(pts: Sequence[Tuple[float, float]], width: int, height: int):
    """MediaPipe x and y are fractions of width and height; make them the same scale."""
    k = width / float(height)
    return [(x * k, y) for x, y in pts]


def count_raised_fingers(pts: Sequence[Tuple[float, float]]) -> int:
    """Raised index/middle/ring/pinky fingers (thumb ignored)."""
    return sum(finger_states(pts)[1])


def read_hand_number(pts: Sequence[Tuple[float, float]]) -> Optional[int]:
    """
    Reads a hand sign as a number 1-7, or None (fist / nothing clear).
      1    index
      2    index + middle
      3    index + middle + ring
      4    index + middle + ring + pinky
      5    all four + thumb (open hand)
      6    "OK" sign: thumb tip touching index tip, other fingers up
      7    hand upside down: thumb sideways along the top, index pointing
           DOWN, middle/ring/pinky folded (looks like a 7)
    The thumb only matters for 5, so a stray thumb can't change 1-4.
    1-4 count raised fingers, so it doesn't matter exactly which ones.
    """
    size = max(_dist(pts, 0, 9), 1e-6)
    thumb, (index, middle, ring, pinky) = finger_states(pts)

    if _dist(pts, 4, 8) < TOUCH_RATIO * size and (middle + ring + pinky) >= 2:
        return 6
    if index and not (middle or ring or pinky):
        # Index pointing down: its tip well below its knuckle on screen.
        # An upright "1" points up, so it can't be mistaken for this.
        index_down = (pts[8][1] - pts[5][1]) > INDEX_DOWN_RATIO * size  # screen y grows downward
        if index_down:
            return 7
    n = index + middle + ring + pinky
    if n == 4 and thumb:
        return 5
    return n if n >= 1 else None


class SignChordSelector:
    """
    Hand sign 1..n -> chord index 0..n-1. The sign must hold steady for a few
    frames before the chord changes, so passing through other shapes while
    moving your fingers doesn't trigger chords. A fist keeps the current chord.
    """

    def __init__(self, n_chords: int = 7, stable_frames: int = 4):
        self.n = n_chords
        self.stable_frames = stable_frames
        self.zone = 0
        self.seen: Optional[int] = None   # what the camera reads right now (for the HUD)
        self._candidate: Optional[int] = None
        self._count = 0

    def update(self, number: Optional[int]) -> int:
        self.seen = number
        if number == self._candidate:
            self._count += 1
        else:
            self._candidate, self._count = number, 1
        if self._count >= self.stable_frames and number is not None and 1 <= number <= self.n:
            self.zone = number - 1
        return self.zone


# kept so older scripts that import it still work
FingerChordSelector = SignChordSelector


def hand_center_and_size(pts: Sequence[Tuple[float, float]]) -> Tuple[float, float, float]:
    """Centre = between wrist (0) and middle-finger knuckle (9); size = their distance."""
    (x0, y0), (x9, y9) = pts[0], pts[9]
    return (x0 + x9) / 2, (y0 + y9) / 2, math.hypot(x9 - x0, y9 - y0)


# --------------------------------------------------------------------------
# Serial link to the MSPM0
# --------------------------------------------------------------------------
class ServoLink:
    def __init__(self, port: Optional[str], baud: int = 115200):
        self.ser = None
        if not port:
            return
        try:
            import serial  # type: ignore
            self.ser = serial.Serial(port, baud, timeout=0, write_timeout=0)
            time.sleep(0.1)
            threading.Thread(target=self._reader, daemon=True).start()
            print(f"[serial] connected to {port}")
            self.send("P")
        except Exception as exc:
            print(f"[serial] could not open {port}: {exc}")
            self.ser = None

    @property
    def connected(self) -> bool:
        return self.ser is not None

    def send(self, line: str) -> None:
        if self.ser is None:
            return
        try:
            self.ser.write((line + "\n").encode("ascii"))
        except Exception as exc:
            print(f"[serial] write failed: {exc}")

    def _reader(self) -> None:
        buf = b""
        while self.ser is not None:
            try:
                buf += self.ser.read(64)
            except Exception:
                return
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                print("[mspm0]", line.decode(errors="replace").strip())
            time.sleep(0.005)

    def close(self) -> None:
        if self.ser is not None:
            self.send("R")
            self.ser.close()
            self.ser = None


# --------------------------------------------------------------------------
# Drawing helpers (plain OpenCV, no MediaPipe drawing utils needed)
# --------------------------------------------------------------------------
HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
                    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


def draw_hand(cv2, img, pts, color):
    h, w = img.shape[:2]
    p = [(int(x * w), int(y * h)) for x, y in pts]
    for a, b in HAND_CONNECTIONS:
        cv2.line(img, p[a], p[b], color, 2)
    for q in p:
        cv2.circle(img, q, 3, (255, 255, 255), -1)


def ensure_model(path: str) -> str:
    if not os.path.exists(path):
        print(f"[model] downloading hand_landmarker.task (~8 MB) to {path}")
        urllib.request.urlretrieve(MODEL_URL, path)
    return path


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Air guitar proof of concept")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--port", help="MSPM0 serial port, e.g. COM5 or /dev/ttyACM0")
    ap.add_argument("--key", default="G",
                    help="key for scale mode: G, D, Bb, F#... add 'm' for minor (Em, Am)")
    ap.add_argument("--chords", default=",".join(DEFAULT_PROGRESSION),
                    help=f"zones mode chord list, comma separated, from {list(CHORDS)}")
    ap.add_argument("--mode", choices=["scale", "zones"], default="scale")
    ap.add_argument("--lefty", action="store_true", help="left-handed player")
    ap.add_argument("--no-sound", action="store_true")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=60, help="requested camera fps")
    ap.add_argument("--model", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                    "hand_landmarker.task"))
    args = ap.parse_args()

    import cv2  # imported here so the logic above can be tested without OpenCV
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision

    progression = [c.strip() for c in args.chords.split(",") if c.strip()]
    for c in progression:
        if c not in CHORDS:
            sys.exit(f"Unknown chord {c!r}. Choose from {list(CHORDS)}")
    try:
        scale = scale_chords(args.key)        # [(name, roman), ...] for signs 1-7
    except ValueError as exc:
        sys.exit(str(exc))
    scale_names = [name for name, _ in scale]
    print(f"[key] {args.key}: " + "  ".join(f"{i + 1}={n}" for i, n in enumerate(scale_names)))

    renderer = StrumRenderer()
    print("[audio] pre-rendering notes...")
    renderer.warm(progression + scale_names)
    audio = AudioEngine(enabled=not args.no_sound)
    link = ServoLink(args.port)

    options = vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=ensure_model(args.model)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    cap = cv2.VideoCapture(args.camera, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # always use the newest frame
    if not cap.isOpened():
        sys.exit("Could not open the camera. Try --camera 1")

    strum = StrumDetector()
    zones = ZoneChordSelector(len(progression))
    signs = SignChordSelector(len(scale_names))
    mode = args.mode
    sound_on = audio.enabled
    chord_idx = {"scale": 0, "zones": 0}     # remembered separately per mode
    sent_chord = None
    fret_state = None                        # (thumb, [i, m, r, p]) for the HUD
    last_strum: Optional[StrumEvent] = None
    last_pos = {"strum": None, "fret": None}
    fps_t, fps_n, fps = time.perf_counter(), 0, 0.0
    t0 = time.perf_counter()
    last_ts = -1

    # neck coordinate: 0 at the headstock end. Right-handers hold the neck to
    # their left, which is screen-left in the mirrored image.
    def neck_x(x: float) -> float:
        return (1.0 - x) if args.lefty else x

    print("Ready. Strum across the yellow line. Keys: Q quit, M mode, C set line, [ ] sensitivity")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("camera read failed"); break
            frame = cv2.flip(frame, 1)  # mirror, so it feels like a mirror
            now = time.perf_counter() - t0
            ts = int(now * 1000)
            if ts <= last_ts:
                ts = last_ts + 1
            last_ts = ts

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
            hands = [[(lm.x, lm.y) for lm in hand] for hand in (result.hand_landmarks or [])]

            # --- decide which hand strums and which frets (by screen position)
            strum_pts = fret_pts = None
            if len(hands) >= 2:
                hands.sort(key=lambda p: hand_center_and_size(p)[0])
                fret_pts, strum_pts = (hands[0], hands[-1]) if not args.lefty else (hands[-1], hands[0])
            elif len(hands) == 1:
                cx, cy, _ = hand_center_and_size(hands[0])

                def dist(role):
                    lp = last_pos[role]
                    return math.inf if lp is None else math.hypot(cx - lp[0], cy - lp[1])
                if dist("strum") == math.inf and dist("fret") == math.inf:
                    is_strum = (cx > 0.5) != args.lefty
                else:
                    is_strum = dist("strum") <= dist("fret")
                if is_strum:
                    strum_pts = hands[0]
                else:
                    fret_pts = hands[0]

            # --- fret hand -> chord
            if fret_pts is not None:
                fx, fy, _ = hand_center_and_size(fret_pts)
                last_pos["fret"] = (fx, fy)
                if mode == "zones":
                    chord_idx["zones"] = zones.update(neck_x(fx))
                else:
                    fh, fw = frame.shape[:2]
                    sign_pts = aspect_correct(fret_pts, fw, fh)
                    fret_state = (*finger_states(sign_pts), thumb_score(sign_pts))
                    chord_idx["scale"] = signs.update(read_hand_number(sign_pts))
                draw_hand(cv2, frame, fret_pts, (255, 160, 0))
            else:
                fret_state = None
                signs.seen = None

            chords_now = scale_names if mode == "scale" else progression
            idx = chord_idx[mode]
            if (mode, idx) != sent_chord:        # chord first, so the fretter can move
                link.send(f"C{idx}")             # before the next strum arrives
                sent_chord = (mode, idx)

            # --- strum hand -> strum events
            if strum_pts is not None:
                sx, sy, size = hand_center_and_size(strum_pts)
                last_pos["strum"] = (sx, sy)
                ev = strum.update(sy, size, now)
                draw_hand(cv2, frame, strum_pts, (0, 220, 255))
                if ev is not None:
                    last_strum = ev
                    chord = chords_now[idx]
                    vel = int(round(ev.strength * 127))
                    link.send(f"{'D' if ev.direction == 'down' else 'U'}{vel}")
                    if sound_on:
                        audio.play(renderer.render(chord, ev.direction, ev.strength))
            else:
                strum.reset()

            # --- overlay
            h, w = frame.shape[:2]
            ly = int(strum.line_y * h)
            cv2.line(frame, (w // 2, ly), (w, ly) if not args.lefty else (0, ly), (0, 255, 255), 2)
            cv2.putText(frame, "strings", (w - 90 if not args.lefty else 10, ly - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            if mode == "zones":
                for i, (a, b) in enumerate(zip(zones.edges()[:-1], zones.edges()[1:])):
                    xa, xb = (a, b) if not args.lefty else (1 - b, 1 - a)
                    pa, pb = int(xa * w), int(xb * w)
                    col = (0, 200, 0) if i == idx else (90, 90, 90)
                    cv2.rectangle(frame, (pa, 10), (pb, 50), col, 2)
                    cv2.putText(frame, progression[i], (pa + 6, 38),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
            else:  # scale mode: a strip showing sign number -> chord
                cell = (w - 20) // len(scale_names)
                for i, name in enumerate(scale_names):
                    pa = 10 + i * cell
                    col = (0, 200, 0) if i == idx else (90, 90, 90)
                    cv2.rectangle(frame, (pa, 10), (pa + cell - 4, 50), col, 2)
                    cv2.putText(frame, f"{i + 1} {name}", (pa + 5, 37),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
            fps_n += 1
            if time.perf_counter() - fps_t >= 1.0:
                fps = fps_n / (time.perf_counter() - fps_t)
                fps_t, fps_n = time.perf_counter(), 0
            if mode == "scale":
                roman = scale[idx][1].replace("°", "o")   # OpenCV fonts have no degree sign
                top = f"key {args.key}: {idx + 1} ({roman}) = {chords_now[idx]}"
                if fret_state is not None:
                    thumb, (fi, fm, fr, fp), tscore = fret_state
                    marks = "".join(c if up else "-" for c, up in
                                    zip("TIMRP", (thumb, fi, fm, fr, fp)))
                    seen = signs.seen if signs.seen is not None else "fist"
                    top += f"   sees: {seen} [{marks}] thumb {tscore:+.2f}"
            else:
                top = f"chord: {chords_now[idx]}   mode: zones"
            hud = [top,
                   f"fps {fps:4.1f}  min speed {strum.min_speed:.1f}  sound {'on' if sound_on else 'off'}"
                   f"  servos {'on' if link.connected else 'off'}"]
            if last_strum is not None and now - last_strum.t < 0.4:
                hud.append(f"STRUM {last_strum.direction.upper()}  strength {last_strum.strength:.2f}")
            for i, line in enumerate(hud):
                cv2.putText(frame, line, (10, h - 60 + i * 22), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (255, 255, 255), 2)
            cv2.imshow("Air Guitar PoC", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("m"):
                mode = "scale" if mode == "zones" else "zones"
            if key == ord("c") and last_pos["strum"] is not None:
                strum.line_y = last_pos["strum"][1]
                strum.reset()
            if key == ord("["):
                strum.min_speed = min(strum.min_speed + 0.5, 10.0)   # less sensitive
            if key == ord("]"):
                strum.min_speed = max(strum.min_speed - 0.5, 0.5)    # more sensitive
            if key == ord("s") and audio.enabled:
                sound_on = not sound_on
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()
        audio.close()
        link.close()


if __name__ == "__main__":
    main()
