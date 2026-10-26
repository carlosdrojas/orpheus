# Orpheus

Air guitar that plays a real guitar. Hold a printed, neck-shaped board and play it like a guitar. A webcam tracks your hands, software turns them into fret and pluck events, and those events drive a synth today and, next, a small servo rig on a real guitar: fret levers that press chord shapes plus a strum servo, on a TI MSPM0G3507 LaunchPad.

```
webcam → hand landmarks (MediaPipe) → board space (ArUco + homography) → fret / unfret / pluck events
                                                                           ├─ synth / MIDI / JSONL log   (working)
                                                                           └─ USB serial → MSPM0 → chord-pose + strum servos  (planned)
```

Senior design project, Fall 2026.

## Status

| Part | State |
| --- | --- |
| Vision prototype (`sw/`) | Working on a laptop webcam, 25–30 fps |
| Printed board, 10-marker tracking | Built, in live testing |
| Synth, MIDI, event log | Working |
| Host-to-hardware bridge (chord matching, strum grouping) | Designed (`docs/system-design.md` §8) |
| MSPM0 firmware + chord-pose servo rig | Designed (§9), based on Kamsi's proof of concept (branch `air-guitar-poc-kamsi`); lever mechanism open |

## Quick start

macOS, Python 3.13, a webcam.

```bash
cd sw
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
curl -L -o models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task

.venv/bin/python gen_board.py                               # board.pdf: print at 100%, build the board
.venv/bin/python air_guitar.py --diag --log session1.jsonl  # play; q to quit
.venv/bin/python test_board.py                              # tests on synthetic frames, no camera
```

There's no board yet? `air_guitar.py` falls back to a fixed rectangle on screen. Board building steps, keys and flags are in [`sw/README.md`](sw/README.md).

## Repo layout

```
docs/
  system-design.md    full system design: board, vision, event contract, servo hardware, protocol, milestones
  vision-status.md    vision pipeline details, limitations, ML plan
  vision-brief.md     notes for the CV review meeting
  software-design.md  earlier software design (superseded by system-design.md)
sw/                   Python: vision, board tracking, event logic, synth/MIDI, board template generator
hw/                   firmware and hardware (planned)
```

## The contract

Vision and hardware only share the event stream: one JSON object per event.

```json
{"kind": "pluck", "string": 1, "fret": 3, "note": 48, "direction": 1, "velocity": 0.7, "t": 1793034221.81}
```

The full field list and its semantics, plus the serial protocol to the servo controller, are in [`docs/system-design.md`](docs/system-design.md).
