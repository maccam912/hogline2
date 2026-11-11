"""Synthetic checks of the geometry and the tracker's identity logic (no video needed)."""
import numpy as np

from stonetrack.geometry import HOG, TEE_DISTANCE, Camera, apply_h, pinhole_pose, solve_h, tee
from stonetrack.track import Stone, Tracker


def synthetic_camera(position, look_at, f=5000.0, pp=(360.0, 495.0)):
    """Ice->image homography of an ideal pinhole camera."""
    c = np.asarray(position, float)
    fwd = np.asarray(look_at, float) - c
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.vstack([right, down, fwd])
    K = np.array([[f, 0, pp[0]], [0, f, pp[1]], [0, 0, 1.0]])
    P = K @ np.c_[R, -R @ c]
    return P[:, [0, 1, 3]], P


def test_homography_round_trip():
    src = [[0, 0], [1, 0], [1, 2], [0, 2]]
    dst = [[10, 20], [200, 30], [190, 400], [5, 380]]
    H = solve_h(src, dst)
    assert np.allclose(apply_h(H, src), dst, atol=1e-9)


def test_pinhole_pose_recovers_camera():
    G, _ = synthetic_camera([-0.3, 41.7, 4.4], [0, 0, 0])
    f, centre = pinhole_pose(G, (360.0, 495.0))
    assert abs(f - 5000) < 1
    assert np.allclose(centre, [-0.3, 41.7, 4.4], atol=1e-3)


def test_elevated_point_matches_true_projection():
    G, P = synthetic_camera([0.2, 41.0, 4.2], [0, 0, 0])
    cam = Camera("p", "perspective", "A", (0, 0, 720, 990), G, [0.2, 41.0], 4.2)
    stone, z = np.array([0.7, 15.0]), 0.135
    true = P @ [stone[0], stone[1], z, 1.0]
    assert np.allclose(cam.to_image([stone], z)[0], true[:2] / true[2], atol=1e-6)
    assert np.allclose(cam.to_ice([true[:2] / true[2]], z)[0], stone, atol=1e-6)


def _frame(t, y, x, colour, views):
    cov = {"overhead_b": [1e-4, 0, 1e-4], "perspective_b": [2e-3, 0, 2e-2], "perspective_a": [2e-3, 0, 2e-2]}
    return [dict(view=v, colour=colour, ice=[x, y], cov=cov[v], q=1.0, px=[0, 0]) for v in views]


def test_deliveries_get_alternating_stable_ids():
    eye = np.eye(3)
    cams = {v: Camera(v, "overhead" if v.startswith("o") else "perspective", v[-1].upper(), (0, 0, 10, 10), eye)
            for v in ("overhead_b", "perspective_b", "perspective_a")}
    Stone._next = 0
    tr = Tracker(cams, "B")
    tr._overhead_sees = lambda p: abs(p[1] - tee("B")[1]) < 1.9
    tr._visible_empty = lambda s, fg: False
    t = 0.0
    for k, colour in enumerate(["red", "yellow", "red"]):
        x = -0.5 + 0.5 * k
        for i in range(400):  # 20 s at 20 fps: slide from behind tee B past the far hog line
            tau = i / 20
            travelled = -1.5 + 2.4 * tau - 0.5 * 0.085 * tau * tau
            y = tee("B")[1] - travelled
            views = ["overhead_b", "perspective_b"] if travelled < 1.9 else ["perspective_b"] if travelled < 17 else ["perspective_a"]
            if 8 < travelled < 9.5:
                views = []  # hidden behind a sweeper
            tr.step(t, _frame(t, y, x, colour, views), {})
            t += 0.05
        for _ in range(400):  # 20 s of quiet before the next stone
            tr.step(t, _frame(t, y, x, colour, ["perspective_a"]), {})
            t += 0.05
    res = tr.result()
    assert [s["id"] for s in res["stones"]] == ["R1", "Y1", "R2"]
    assert all(s["hog_to_hog_s"] and 8 < s["hog_to_hog_s"] < 16 for s in res["stones"])
    assert TEE_DISTANCE - HOG > 0
