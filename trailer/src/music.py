"""Original score for the Hogline 2 trailer, synthesised from scratch. 100 BPM, 30 bars, D minor."""
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, fftconvolve, lfilter

SR = 44100
BEAT = 0.6
BAR = 4 * BEAT
BARS = 30
N = int(SR * (BARS * BAR + 3.0))
rng = np.random.default_rng(7)
mix = np.zeros((N, 2))

def hz(note):  # midi -> Hz
    return 440.0 * 2 ** ((note - 69) / 12)

def lp(x, cutoff, order=2):
    b, a = butter(order, cutoff / (SR / 2), 'low'); return lfilter(b, a, x)

def hp(x, cutoff, order=2):
    b, a = butter(order, cutoff / (SR / 2), 'high'); return lfilter(b, a, x)

def add(sig, at, gain=1.0, pan=0.0):
    i = int(at * SR)
    if i >= N: return
    sig = sig[: N - i]
    mix[i:i + len(sig), 0] += sig * gain * np.sqrt((1 - pan) / 2)
    mix[i:i + len(sig), 1] += sig * gain * np.sqrt((1 + pan) / 2)

def t_(dur): return np.arange(int(dur * SR)) / SR

def saw(f, dur, detune=0.0):
    t = t_(dur); out = np.zeros_like(t)
    for d in (-detune, 0, detune):
        out += 2 * ((t * f * (1 + d) + rng.random()) % 1) - 1
    return out / 3

def pad(notes, dur, cutoff=900):
    t = t_(dur); out = np.zeros_like(t)
    for n in notes: out += saw(hz(n), dur, 0.004)
    env = np.minimum(1, t / 0.9) * np.minimum(1, (dur - t) / 0.8)
    return lp(out, cutoff, 3) * env / len(notes)

def pluck(n, dur=0.5, bright=2600):
    t = t_(dur)
    s = saw(hz(n), dur, 0.002) * np.exp(-t * 7)
    return lp(s, bright) * np.minimum(1, t / 0.004)

def bass(n, dur):
    t = t_(dur)
    s = np.sin(2 * np.pi * hz(n) * t) + 0.35 * lp(saw(hz(n), dur), 420)
    return s * np.minimum(1, t / 0.006) * np.exp(-t * 3.2) * np.minimum(1, (dur - t) / 0.02)

def kick(power=1.0):
    t = t_(0.42)
    f = 46 + 110 * np.exp(-t * 28)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 9) * power

def hat(dur=0.05):
    t = t_(dur); return hp(rng.standard_normal(len(t)), 7000) * np.exp(-t / dur * 5)

def clap():
    t = t_(0.22); n = hp(lp(rng.standard_normal(len(t)), 5200), 900)
    env = sum(np.exp(-np.maximum(0, t - d) * 60) * (t >= d) for d in (0, 0.011, 0.023)) / 3 + 0.4 * np.exp(-t * 16)
    return n * env

def boom():
    t = t_(3.2)
    sub = np.sin(2 * np.pi * np.cumsum(34 + 60 * np.exp(-t * 6)) / SR) * np.exp(-t * 1.5)
    crash = hp(rng.standard_normal(len(t)), 2500) * np.exp(-t * 2.4) * 0.35
    return np.tanh(2.2 * sub) * 0.9 + crash

def riser(dur):
    t = t_(dur); n = rng.standard_normal(len(t)); out = np.zeros_like(t)
    step = SR // 20
    for i in range(0, len(t), step):                      # sweep a filter up over the length
        k = i / len(t)
        out[i:i + step] = lp(n[max(0, i - step):i + step], 300 + 9000 * k ** 2)[-len(out[i:i + step]):]
    return out * (t / dur) ** 2

# D minor: Dm - Bb - F - C, voiced around middle C
CHORDS = [[50, 57, 62, 65, 69], [46, 58, 62, 65, 70], [53, 57, 60, 65, 69], [48, 55, 60, 64, 67]]
ROOTS = [38, 34, 41, 36]
ARP = [[62, 65, 69, 74, 69, 65, 69, 77], [58, 62, 65, 70, 65, 62, 65, 74], [60, 65, 69, 72, 69, 65, 69, 77], [60, 64, 67, 72, 67, 64, 67, 76]]

def groove(bar, level=1.0, drums=True, arp=True, clapon=True):
    at = bar * BAR; c = bar % 4
    add(pad(CHORDS[c], BAR + 0.5, 1100), at, 0.5 * level)
    for b in range(8):                                    # eighth notes
        add(bass(ROOTS[c] + (12 if b % 4 == 3 else 0), BEAT / 2), at + b * BEAT / 2, 0.5 * level)
        if arp:
            add(pluck(ARP[c][b]), at + b * BEAT / 2, 0.26 * level, pan=(-0.5 if b % 2 else 0.5))
            add(pluck(ARP[c][b] + 12, 0.35, 3400), at + b * BEAT / 2 + BEAT / 4, 0.09 * level, pan=(0.6 if b % 2 else -0.6))
    if drums:
        for b in range(4):
            add(kick(), at + b * BEAT, 0.85 * level)
            add(hat(), at + b * BEAT + BEAT / 2, 0.16 * level, 0.2)
            add(hat(0.03), at + b * BEAT + BEAT / 4, 0.06 * level, -0.3)
            add(hat(0.03), at + b * BEAT + 3 * BEAT / 4, 0.06 * level, -0.3)
        if clapon:
            add(clap(), at + BEAT, 0.4 * level); add(clap(), at + 3 * BEAT, 0.4 * level)

# --- bars 0-3: cold open. drone, heartbeat, lonely plucks
add(pad([38, 45, 50, 57], 4 * BAR + 0.4, 500), 0, 0.75)
for bar in range(4):
    add(kick(0.8), bar * BAR, 0.6); add(kick(0.5), bar * BAR + 0.28, 0.4)
for i, n in enumerate([74, 69, 77, 72, 74, 69, 65, 62]):
    add(pluck(n, 1.2, 1800), 1.2 + i * BEAT * 2, 0.34, pan=(-0.4 if i % 2 else 0.4))
add(riser(2.4)[::-1].copy()[::-1], 3 * BAR, 0.22)       # swell into the title

# --- bar 4-5: title hit, groove starts without claps
add(boom(), 4 * BAR, 0.95)
groove(4, 0.8, arp=False, clapon=False); groove(5, 0.9, clapon=False)
# --- bars 6-21: the four steps, full groove; drop the drums for one bar at each step change
for bar in range(6, 22):
    first = (bar - 6) % 4 == 0
    groove(bar, 1.0)
    if first: add(boom(), bar * BAR, 0.35)
    if (bar - 6) % 4 == 3:
        for k in range(4): add(clap(), bar * BAR + 3 * BEAT + k * BEAT / 4, 0.22 + 0.06 * k)   # fill into next step
# --- bars 22-24: questions. half-time, tension
for bar in range(22, 25):
    add(boom(), bar * BAR, 0.6)
    add(pad(CHORDS[0] if bar < 24 else [45, 57, 61, 64, 69], BAR + 0.4, 700 + 300 * (bar - 22)), bar * BAR, 0.6)
    for b in range(8): add(bass(38 if bar < 24 else 33, BEAT / 2), bar * BAR + b * BEAT / 2, 0.42)
    for b in range(4): add(hat(), bar * BAR + b * BEAT + BEAT / 2, 0.12)
add(riser(2 * BAR), 23 * BAR, 0.5)
for k in range(16): add(clap(), 24 * BAR + k * BEAT / 4, 0.12 + 0.03 * k)                       # roll
# --- bar 25: "Stop guessing." big hit, then air
add(boom(), 25 * BAR, 1.0)
add(pad(CHORDS[0], BAR + 0.4, 1500), 25 * BAR, 0.5)
# --- bars 26-29: end card. groove returns and resolves on D
add(boom(), 26 * BAR, 0.8)
for bar in range(26, 29): groove(bar + 2 if False else bar, 0.95)
add(boom(), 29 * BAR, 0.9)
add(pad([38, 50, 57, 62, 66, 69], 4.6, 1400), 29 * BAR, 0.85)       # D major: it lands somewhere brighter
for i, n in enumerate([62, 69, 74, 78, 81]): add(pluck(n, 1.6, 2400), 29 * BAR + i * 0.15, 0.3, pan=(i - 2) / 3)

# --- reverb, master
ir_t = t_(2.2)
ir = rng.standard_normal((len(ir_t), 2)) * np.exp(-ir_t * 3.0)[:, None]
ir = np.stack([lp(ir[:, 0], 6000), lp(ir[:, 1], 6000)], 1)
wet = np.stack([fftconvolve(hp(mix[:, c], 180), ir[:, c])[:N] for c in range(2)], 1)
wet /= np.abs(wet).max() + 1e-9
out = mix + 0.16 * wet * np.abs(mix).max()
out = np.tanh(out / np.abs(out).max() * 1.6)
end = int(SR * (BARS * BAR + 2.2))
out = out[:end]
fade = int(SR * 1.5); out[-fade:] *= np.linspace(1, 0, fade)[:, None] ** 2
out *= 0.89 / np.abs(out).max()
wavfile.write('score.wav', SR, (out * 32767).astype(np.int16))
print('score.wav', len(out) / SR, 's')
