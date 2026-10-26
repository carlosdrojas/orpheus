# Air Guitar to Real Guitar — Proof of Concept

Webcam air guitar that plays sound on your laptop and (optionally) drives
strum/fret servos on a TI LP-MSPM0G3507 LaunchPad.

```
air_guitar_poc/
  air_guitar.py          main app: camera -> hand tracking -> strums + chords -> sound / serial
  test_logic.py          tests the gesture + sound logic with fake data (no camera needed)
  servo_console.py       type commands to the MSPM0 to calibrate servo angles
  strum_demo.wav         what the built-in guitar synth sounds like (G C D Em)
  requirements.txt
  firmware_mspm0/main.c  MSPM0G3507 firmware (SysConfig + DriverLib)
```

## 1. Sound only (start here)

Python 3.9–3.12 recommended (MediaPipe wheels lag new Python releases).

```
pip install -r requirements.txt
python test_logic.py        # should print "all tests passed"
python air_guitar.py        # downloads the hand model (~8 MB) on first run
```

How to play (right-handed; add `--lefty` otherwise):

1. Stand 1–2 m from the camera so both hands are in view.
2. **Strum hand** (screen right): swing it up and down through the yellow
   "strings" line. Faster = louder. Press **C** with your hand where your
   guitar strings would be to move the line there.
3. **Fret hand** (screen left): make a hand sign for the chord number in the key.
   The strip at the top shows which chord each number plays.

   | Sign | Hand shape | Key of G |
   |---|---|---|
   | 1 | index | G |
   | 2 | index + middle | Am |
   | 3 | index + middle + ring | Bm |
   | 4 | index + middle + ring + pinky | C |
   | 5 | open hand (all four + thumb) | D |
   | 6 | "OK" sign: thumb and index tips touching, other three fingers up | Em |
   | 7 | hand upside down: thumb sideways along the top, index pointing down, rest folded | F#dim |
   | fist | | keeps the current chord |

   The thumb is ignored for 1–4 and only counts for 5. Hold a sign for a
   moment; the chord changes after it stays steady for 4 frames.
4. **[ / ]** make strums less / more sensitive. **Q** quits.

Other keys: `python air_guitar.py --key D` (major) or `--key Em` (minor; add "m").

**If a sign is misread,** look at the `sees:` readout at the bottom of the
video. It shows the number it reads and which fingers it thinks are up,
e.g. `sees: 3 [TIM--] thumb +0.70` = thumb, index, middle. The `thumb`
number is how far the thumb points across the palm: below 0.10 counts as
"out". If 5 shows `[-IMRP]`, read the thumb number while holding 5 and set
`THUMB_OUT_COS` (near the middle of `air_guitar.py`) a little above it.
Other tuning values sit next to it (`TOUCH_RATIO` for the 6,
`INDEX_DOWN_RATIO` for the 7: lower it if a 7 with the index angled
down still reads as 1).

**Zones mode (the old way):** press **M** and slide your fret hand along the
boxes instead. Pick its chords with `--chords Am,C,G,E` (any of G C D Em Am E A F).

## 2. Add the servos (MSPM0G3507)

### Wiring
| Servo | Signal pin | Timer channel |
|---|---|---|
| 0 Strum | the pin SysConfig assigns to PWM_0 C0 | PWM_0 CC0 |
| 1 Fret lever A | PWM_0 C1 | PWM_0 CC1 |
| 2 Fret lever B (optional) | PWM_1 C0 | PWM_1 CC0 |
| 3 Pick lift (optional) | PWM_1 C1 | PWM_1 CC1 |

* Servo power (red) and ground (brown/black) go to a **separate 5–6 V supply**
  (≥ 1 A per servo that moves at once). Connect that supply's ground to a
  LaunchPad GND pin. Servo signal (orange/yellow) goes to the PWM pin.
* MSPM0 pins are 3.3 V; most hobby servos accept a 3.3 V signal. If yours
  jitters, add a 3.3→5 V level shifter.

### SysConfig (CCS Theia or CCS 12, MSPM0 SDK)
1. Start from the SDK's empty DriverLib example for LP-MSPM0G3507.
2. Add **PWM** → name `PWM_0`, any TIMG/TIMA instance with 2 channels.
   * Timer clock: BUSCLK, divider 8, prescaler 3 → about 1 MHz.
   * PWM period count: 20000 (= 50 Hz). Edge-aligned. Enable C0 and C1.
3. (Optional) Add a second PWM `PWM_1` the same way for 2 more servos.
4. Add **UART** → name `UART_0`, instance UART0, PA10 TX / PA11 RX,
   115200 baud, 8N1, enable the **RX interrupt**. These pins reach your PC
   through the LaunchPad's USB cable (XDS110 back-channel).
5. Replace the generated `main` file with `firmware_mspm0/main.c`, build, flash.
6. Open a terminal at 115200 baud: you should see `AIRGUITAR READY`.

The firmware reads the real timer period at runtime, so a different clock
setup still gives correct pulse widths.

**One thing to check with your first servo:** if it slams to an end and buzzes,
change `#define CC_SETS_LOW_TIME 1` to `0` in main.c (it depends on the PWM
polarity SysConfig picked).

### Calibrate
```
python servo_console.py COM5      # find the port in Device Manager -> Ports
> sweep 0                         # watch the strum servo, note good angles
> s 0 1250                        # pick resting just above the low E string
> s 0 1750                        # pick just past the high E string
```
Put the numbers into `STRUM_SIDE_A_US`, `STRUM_SIDE_B_US` and `CHORD_POSE`
at the top of main.c, rebuild, flash.

### Play
```
python air_guitar.py --port COM5
```

## Serial protocol (laptop -> MSPM0)

| Command | Meaning |
|---|---|
| `C<i>` | select chord *i*: 0–6 for signs 1–7 in scale mode, or index into `--chords` in zones mode |
| `D<v>` / `U<v>` | strum down / up, strength 0–127 |
| `S<ch>,<us>` | raw servo pulse in µs (clamped 500–2500) |
| `R` | all servos to rest |
| `P` | ping → `PONG` |

Chord commands are sent the moment the fret hand changes zone, which is
normally well before the next strum, so the fretting servo has time to land.

## Known limits

* Total delay motion → servo strike is roughly 70–120 ms (camera frame wait,
  tracking, servo travel). Sound-only mode is faster. A 60 fps webcam helps.
* Good, even lighting matters most for tracking. Avoid a bright window behind you.
* The strum detector always alternates down/up, like real strumming. If you
  return your hand slowly (no sound), the firmware either lifts the pick back
  (with a lift servo) or strums the other way so the beat is not lost.
* Strum strength is used for sound volume only; hobby servos always move at
  full speed.
