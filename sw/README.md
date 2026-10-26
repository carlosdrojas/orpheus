# orpheus / sw

Vision side of the air-guitar project: webcam → hand landmarks → board space →
guitar events (fret / unfret / pluck). The event stream is the interface to the
mechanical guitar; a built-in synth and optional MIDI let you hear it now.

## Setup (run from `sw/`)

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
curl -L -o models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

## Air guitar

```bash
.venv/bin/python gen_board.py       # board.pdf (3 Letter pages) + board_layout.json
.venv/bin/python air_guitar.py      # run (no board? a fixed on-screen rectangle is used)
```

Hold the board like a neck, nut to your fretting side. Right hand fingertips in
the fret zone hold frets; left index finger crossing strings in the strum zone
(shaded part of the board) plucks them.

Keys: `s` skeleton, `d` board diagnostics, `l` swap fret/pick hands, `q` quit.
Flags: `--diag`, `--layout FILE`, `--midi` (virtual port "orpheus"), `--no-audio`,
`--log events.jsonl`, `--lefty`, `--camera N`.

### Building the board (60 x 12 cm foam board or cardboard)

1. `python gen_board.py` (sizes: `--length --width --marker --per-edge`; it writes the
   matching `board_layout.json`, which `air_guitar.py` loads automatically).
2. Print `board.pdf` at **Actual size / 100%**, not "fit to page", on matte paper.
   Check the 10 cm bar on each page with a ruler.
3. Trim pages 2 and 3 along the dashed line and lay each over the previous page,
   lining up the printed strings and the tick marks. Glue stick or spray mount onto
   the board. Don't put tape over the markers: glossy tape causes glare that breaks detection.
4. Cut the board along the outer black outline.
5. First run: `python air_guitar.py --diag --log session1.jsonl`. Check:
   - the outline is green and `markers` lists most IDs at playing distance;
   - `marker size` is above ~20 px (if not, regenerate with a bigger `--marker`);
   - covering one end with your hand keeps it green.
   On quit, a summary line prints % of frames tracking / hold / fallback.
6. Try the laptop camera, then a camera low and to the side of the fretting hand.
   Afterwards, tune `FRET_ZONE`, `NUM_FRETS`, `STRING_SLOP` and `debounce` in
   `guitar.py`, then **rerun gen_board.py and reprint**: the print comes from those constants.

Tests (synthetic frames, no camera): `.venv/bin/python test_board.py`.

Files:
- `layout.py`  – physical board layout in cm (marker corners, playing area), JSON load/save
- `gen_board.py` – printable tiled template from the layout + `guitar.py` constants
- `board.py`   – ArUco detection → RANSAC homography over visible markers → board coords (u along neck, v across strings)
- `guitar.py`  – pure event logic: held frets per string (debounced), string-crossing plucks
- `synth.py`   – Karplus-Strong plucked string + MIDI out
- `air_guitar.py` – main loop / overlay / diagnostics
- `gen_markers.py` – old 4-corner marker sheet (used when there's no `board_layout.json`)

Event format (also what `--log` writes, one JSON object per line):
```
{"kind": "pluck", "string": 1, "t": 1758400000.12, "fret": 3, "note": 48, "direction": 1, "velocity": 0.7}
{"kind": "fret",  "string": 1, "t": ..., "fret": 3, ...}
```

## Hand shapes (earlier demo)

```bash
.venv/bin/python hand_shapes.py
```
Fingertips → points → connected into shapes; labels basic gestures.
Keys: `m` cycle draw mode, `h` skeleton, `q` quit.
