"""Sample full-resolution mosaic frames (and a short-baseline motion mask) from a recording.

Usage: sample_frames.py SOURCE.mp4 OUTDIR START END STEP [--pair 0.5]
Writes OUTDIR/t{seconds:07.1f}.jpg (q95) and, with --pair, t{...}.diff.png holding the
max-channel absolute difference to the frame PAIR seconds later (stationarity evidence).
Existing outputs are skipped, so the command can be re-run to resume.
"""
import subprocess, sys, os
import numpy as np, cv2

W, H = 1920, 1080


def grab(src, t, n=1, every=1):
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", src, "-frames:v", str((n - 1) * every + 1),
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    k = len(raw) // (W * H * 3)
    if k == 0:
        return []
    a = np.frombuffer(raw[: k * W * H * 3], np.uint8).reshape(k, H, W, 3)
    return [a[i] for i in range(0, k, every)]


def main():
    src, out, start, end, step = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
    pair = float(sys.argv[7]) if len(sys.argv) > 7 and sys.argv[6] == "--pair" else 0
    os.makedirs(out, exist_ok=True)
    t = start
    while t < end:
        name = os.path.join(out, f"t{t:07.1f}")
        if not os.path.exists(name + ".jpg"):
            frames = grab(src, t, 2 if pair else 1, int(round(pair * 60)) if pair else 1)
            if frames:
                if pair and len(frames) == 2:
                    d = np.abs(frames[0].astype(np.int16) - frames[1].astype(np.int16)).max(axis=2).astype(np.uint8)
                    cv2.imwrite(name + ".diff.png", d)
                cv2.imwrite(name + ".jpg", frames[0], [cv2.IMWRITE_JPEG_QUALITY, 95])
        t += step
    print("done", start, end)


if __name__ == "__main__":
    main()
