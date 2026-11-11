"""Per-pixel temporal median of sampled mosaic frames: people and stones vanish, paint stays."""
import sys, glob
import numpy as np, cv2

files = sorted(glob.glob(sys.argv[1] + "/t*.jpg"))
stack = np.stack([cv2.imread(f) for f in files])  # N,H,W,3 uint8
med = np.empty(stack.shape[1:], np.uint8)
mad = np.empty(stack.shape[1:3], np.uint8)
for y0 in range(0, stack.shape[1], 120):  # strips keep peak memory low
    s = stack[:, y0 : y0 + 120]
    m = np.median(s, axis=0)
    med[y0 : y0 + 120] = m.astype(np.uint8)
    # robust spread: how stable each pixel was (median absolute deviation)
    mad[y0 : y0 + 120] = np.median(np.abs(s.astype(np.int16) - m.astype(np.int16)).max(axis=3), axis=0)
cv2.imwrite(sys.argv[2], med)
cv2.imwrite(sys.argv[2].replace(".png", ".mad.png"), mad)
print(len(files), "frames ->", sys.argv[2])
