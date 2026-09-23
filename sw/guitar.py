"""Guitar event logic. Pure functions of board-space fingertip positions.

Input each frame:  fretting-hand tips (u, v), picking-hand tip (u, v), time.
Output:            a list of Event objects - this is the stream the mechanical
                   system (or a synth) consumes.

Board layout (board space, see board.py):

    u:  [0 ............ FRET_ZONE) [FRET_ZONE ....... 1]
              fret cells                strum zone
    v:  six string lines at (i + 0.5) / 6, i = 0 (top) .. 5 (bottom)
"""

from dataclasses import dataclass, field
import time

NUM_STRINGS = 6
NUM_FRETS = 5          # frets represented on the board (keep small - it's a hand-held board)
FRET_ZONE = 0.65       # fraction of the board length that is frets; the rest is strum zone
STRING_SLOP = 0.6      # how far from a string line (in string spacings) a fingertip still counts

# Standard tuning, MIDI note numbers, top string first (low E .. high E)
OPEN_NOTES = [40, 45, 50, 55, 59, 64]
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def note_name(midi):
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def string_v(i):
    """Board-space v of string i's centre line."""
    return (i + 0.5) / NUM_STRINGS


def fret_u(f):
    """Board-space u of the *start* of fret f (f = 1 .. NUM_FRETS)."""
    return (f - 1) / NUM_FRETS * FRET_ZONE


@dataclass
class Event:
    kind: str            # "fret" | "unfret" | "pluck"
    string: int
    t: float
    fret: int = 0        # for fret / pluck (fret currently held)
    note: int = 0        # MIDI note, for pluck
    direction: int = 0   # +1 down, -1 up, for pluck
    velocity: float = 0  # 0..1, for pluck

    def __str__(self):
        if self.kind == "pluck":
            arrow = "v" if self.direction > 0 else "^"
            return f"pluck  s{self.string} f{self.fret} {note_name(self.note):4s} {arrow} vel={self.velocity:.2f}"
        return f"{self.kind:6s} s{self.string} f{self.fret}"


@dataclass
class GuitarState:
    held: list = field(default_factory=lambda: [0] * NUM_STRINGS)      # fret per string (0 = open)
    _pending: list = field(default_factory=lambda: [0] * NUM_STRINGS)  # candidate fret awaiting debounce
    _pending_n: list = field(default_factory=lambda: [0] * NUM_STRINGS)
    _pick_prev: tuple = None   # (v, t) of the picking tip last frame, if it was in the strum zone
    debounce: int = 2          # frames a fret change must persist

    # -- fretting ------------------------------------------------------------

    def update_frets(self, tips, t):
        """tips: iterable of (u, v) board coords for the fretting hand's fingertips."""
        events = []
        wanted = [0] * NUM_STRINGS
        for u, v in tips:
            cell = cell_of(u, v)
            if cell is None:
                continue
            s, f = cell
            wanted[s] = max(wanted[s], f)  # highest fret on a string wins, like a real guitar

        for s in range(NUM_STRINGS):
            if wanted[s] == self.held[s]:
                self._pending_n[s] = 0
                continue
            if wanted[s] == self._pending[s]:
                self._pending_n[s] += 1
            else:
                self._pending[s] = wanted[s]
                self._pending_n[s] = 1
            if self._pending_n[s] >= self.debounce:
                self.held[s] = wanted[s]
                self._pending_n[s] = 0
                kind = "fret" if wanted[s] else "unfret"
                events.append(Event(kind, s, t, fret=wanted[s]))
        return events

    # -- picking -------------------------------------------------------------

    def update_pick(self, tip, t):
        """tip: (u, v) board coords of the picking tip, or None if not visible.

        Emits a pluck for every string line the tip crossed since last frame.
        """
        events = []
        if tip is None or not (FRET_ZONE <= tip[0] <= 1.0) or not (-0.1 <= tip[1] <= 1.1):
            self._pick_prev = None
            return events

        v = tip[1]
        if self._pick_prev is not None:
            v0, t0 = self._pick_prev
            dt = max(t - t0, 1e-3)
            speed = abs(v - v0) / dt              # board-heights per second
            velocity = min(1.0, speed / 4.0)      # ~4 board-heights/s = max velocity
            lo, hi = min(v0, v), max(v0, v)
            direction = 1 if v > v0 else -1
            strings = range(NUM_STRINGS) if direction > 0 else range(NUM_STRINGS - 1, -1, -1)
            for s in strings:
                if lo < string_v(s) <= hi:
                    note = OPEN_NOTES[s] + self.held[s]
                    events.append(Event("pluck", s, t, fret=self.held[s], note=note,
                                        direction=direction, velocity=max(0.15, velocity)))
        self._pick_prev = (v, t)
        return events


def cell_of(u, v):
    """Board coords -> (string, fret) if inside the fret zone, else None."""
    if not (0.0 <= u < FRET_ZONE):
        return None
    s = int(v * NUM_STRINGS)
    if not (0 <= s < NUM_STRINGS):
        return None
    # must be reasonably close to the string's centre line
    if abs(v - string_v(s)) > STRING_SLOP / NUM_STRINGS:
        return None
    f = int(u / FRET_ZONE * NUM_FRETS) + 1
    return s, min(f, NUM_FRETS)
