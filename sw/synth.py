"""Audio output for guitar events.

- Synth: a tiny Karplus-Strong plucked-string synth so you can hear results
  with zero setup. Each pluck is pre-rendered into a buffer and mixed by the
  audio callback. Re-plucking a string cuts off the previous note (like a real
  string).
- MidiOut: optional. Opens a virtual MIDI port named "orpheus" that any DAW /
  soft synth on the machine can listen to.
"""

import threading

import numpy as np

from guitar import NUM_STRINGS


class Synth:
    def __init__(self, sample_rate=44100, blocksize=512, duration=2.0):
        import sounddevice as sd
        self.sr = sample_rate
        self.duration = duration
        self.voices = [None] * NUM_STRINGS  # per string: [buffer, position]
        self.lock = threading.Lock()
        self.stream = sd.OutputStream(samplerate=sample_rate, channels=1, blocksize=blocksize,
                                      dtype="float32", callback=self._callback)
        karplus_strong(110, sample_rate, 0.1)  # warm up numpy paths so the first real pluck is fast
        self.stream.start()

    def pluck(self, string, midi_note, velocity=0.8):
        buf = karplus_strong(midi_to_hz(midi_note), self.sr, self.duration, velocity)
        with self.lock:
            self.voices[string] = [buf, 0]

    def mute(self, string, fade_s=0.02):
        """Damp a string with a short fade (a hard cut clicks)."""
        with self.lock:
            voice = self.voices[string]
            if voice is None:
                return
            buf, pos = voice
            tail = buf[pos:pos + int(self.sr * fade_s)]
            self.voices[string] = [tail * np.linspace(1, 0, len(tail), dtype=np.float32), 0]

    def _callback(self, out, frames, time_info, status):
        mix = np.zeros(frames, dtype=np.float32)
        with self.lock:
            for i, voice in enumerate(self.voices):
                if voice is None:
                    continue
                buf, pos = voice
                chunk = buf[pos:pos + frames]
                mix[:len(chunk)] += chunk
                voice[1] = pos + frames
                if voice[1] >= len(buf):
                    self.voices[i] = None
        out[:, 0] = np.tanh(mix)  # soft clip

    def close(self):
        self.stream.stop()
        self.stream.close()


def midi_to_hz(n):
    return 440.0 * 2 ** ((n - 69) / 12)


def karplus_strong(freq, sr, duration, velocity=0.8, decay=0.996):
    """Render a plucked string. Block-wise so it's fast in numpy."""
    n = max(2, int(sr / freq))
    total = int(sr * duration)
    rng = np.random.default_rng()
    block = rng.uniform(-1, 1, n).astype(np.float32) * velocity
    # a softer pluck: low-pass the initial noise burst a bit for higher velocity brightness
    block = 0.5 * (block + np.roll(block, 1)) if velocity < 0.5 else block
    out = np.empty(total, dtype=np.float32)
    pos = 0
    while pos < total:
        k = min(n, total - pos)
        out[pos:pos + k] = block[:k]
        block = decay * 0.5 * (block + np.roll(block, 1))
        pos += k
    # short fade-in/out to avoid clicks
    fade = min(64, total // 2)
    out[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
    out[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
    return out


class MidiOut:
    """Note-off is sent on re-pluck, on mute, or after `duration` seconds,
    so sustained patches don't hang (matches the synth's ring time)."""

    def __init__(self, port_name="orpheus", channel=0, duration=2.0):
        import mido
        self.mido = mido
        self.port = mido.open_output(port_name, virtual=True)
        self.channel = channel
        self.duration = duration
        self.sounding = [None] * NUM_STRINGS  # per string: (note, timer)
        self.lock = threading.Lock()

    def pluck(self, string, midi_note, velocity=0.8):
        with self.lock:
            self._off(string)
            self.port.send(self.mido.Message("note_on", note=midi_note,
                                             velocity=int(1 + 126 * velocity), channel=self.channel))
            timer = threading.Timer(self.duration, self._expire, (string,))
            timer.daemon = True
            self.sounding[string] = (midi_note, timer)
            timer.start()

    def mute(self, string):
        with self.lock:
            self._off(string)

    def _expire(self, string):
        with self.lock:  # only if this timer's note is still the one sounding
            if self.sounding[string] and self.sounding[string][1] is threading.current_thread():
                self._off(string)

    def _off(self, string):
        """Caller holds the lock."""
        if self.sounding[string] is not None:
            note, timer = self.sounding[string]
            timer.cancel()
            self.port.send(self.mido.Message("note_off", note=note, channel=self.channel))
            self.sounding[string] = None

    def close(self):
        for s in range(NUM_STRINGS):
            self.mute(s)
        self.port.close()
