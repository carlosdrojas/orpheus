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
.venv/bin/python gen_markers.py     # writes markers.png - print, cut out, tape to board corners
.venv/bin/python air_guitar.py      # run (no board? a fixed on-screen rectangle is used)
```

Hold the board like a neck: marker 0 top-left (nut), 1 top-right, 2 bottom-right, 3 bottom-left.
Left hand fingertips in the fret zone hold frets; right index finger crossing
strings in the strum zone (right third of the board) plucks them.

Keys: `s` skeleton, `l` swap fret/pick hands, `q` quit.
Flags: `--midi` (virtual port "orpheus"), `--no-audio`, `--log events.jsonl`, `--lefty`, `--camera N`.

Files:
- `board.py`   – ArUco corner detection → homography → board coords (u along neck, v across strings)
- `guitar.py`  – pure event logic: held frets per string (debounced), string-crossing plucks
- `synth.py`   – Karplus-Strong plucked string + MIDI out
- `air_guitar.py` – main loop / overlay

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
