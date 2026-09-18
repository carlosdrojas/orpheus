# orpheus / sw — hand shapes

Webcam hand tracking with MediaPipe. Extended fingertips become points and get
connected into shapes (triangle, quad, pentagon…). Also labels basic gestures
(fist, open palm, peace, point, thumbs up, ok, rock, …).

## Setup (run from `sw/`)

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
curl -L -o models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

## Run

```bash
.venv/bin/python hand_shapes.py
```

Keys: `m` cycle draw mode (polygon → hull → web → skeleton), `h` toggle
skeleton overlay, `q` quit.
