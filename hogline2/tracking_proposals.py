"""Cheap color/motion proposals only; accepted objects receive SAM masks."""

import math

import cv2
import numpy as np

from .tracking_geometry import TEE_DISTANCE, project, projection_variance


def candidates(rgb, view, background=None, require_granite=False):
    tee_distance = view.get("tee_distance_m", TEE_DISTANCE)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    delta = cv2.absdiff(rgb, background).mean(axis=2) if background is not None else None
    result = []
    scale = rgb.shape[1] / (724 if view["kind"] == "perspective" else 458)
    for color in ("red", "yellow"):
        if color == "red":
            mask = (
                ((hsv[:, :, 0] < 12) | (hsv[:, :, 0] > 170)) & (hsv[:, :, 1] > 100) & (hsv[:, :, 2] > 65)
            ).astype(np.uint8)
        else:
            mask = (
                (hsv[:, :, 0] >= 18) & (hsv[:, :, 0] <= 40) & (hsv[:, :, 1] > 95) & (hsv[:, :, 2] > 100)
            ).astype(np.uint8)
        count, components, stats, centers = cv2.connectedComponentsWithStats(mask)
        for i in range(1, count):
            x, y, w, h, area = stats[i]
            if (
                not max(2, 3 * scale**2) <= area <= 1300 * scale**2
                or w > 65 * scale
                or h > 50 * scale
                or w < max(1, 2 * scale)
            ):
                continue
            if delta is not None:
                region = components[y : y + h, x : x + w] == i
                change = float(delta[y : y + h, x : x + w][region].mean())
                if view["kind"] == "perspective":
                    below = delta[y + h : min(rgb.shape[0], y + h + max(8, w // 2)), x : x + w]
                    if below.size:
                        change = max(change, float(below.mean()))
                if change < 24:
                    continue  # Static painted ice/logo is not a stone proposal.
            px, py = centers[i]
            px += view["crop"][0]
            py += view["crop"][1]
            # Perspective handle height is not an ice point; a small body-height proposal
            # offset is replaced by the SAM mask's ground-contact estimate after initialization.
            foot = [px, py + (max(3, h * 0.7) if view["kind"] == "perspective" else 0)]
            if require_granite and view["kind"] == "perspective":
                from .tracking_stone import granite_evidence

                evidence = granite_evidence(rgb, components == i, color)
                if evidence:
                    bx0, _, bx1, by1 = evidence["bbox"]
                    foot = [(bx0 + bx1) / 2 + view["crop"][0], by1 - 1 + view["crop"][1]]
            try:
                xy = project(view["image_to_ice"], foot)
            except ValueError:
                continue
            local_y = xy[1] if view["end"] == "A" else tee_distance - xy[1]
            if abs(xy[0]) > 2.3 or not -4.4 <= local_y <= (
                TEE_DISTANCE + 4.4 if view["kind"] == "perspective" else 2.3
            ):
                continue
            result.append(
                dict(
                    color=color,
                    point=[float(px), float(py)],
                    xy=xy,
                    area=int(area),
                    variance=projection_variance(view, foot, pixel_sigma=4),
                )
            )
    return result


class DeliveryGate:
    """Require a colored candidate to originate at the hack and move toward the ice."""

    def __init__(self):
        self.pending = []
        self.last_birth = -1e9

    def update(self, seconds, view, proposals):
        hack = view["landmarks"].get("hack")
        if not hack or view["kind"] != "perspective":
            return []
        scale = (view.get("crop", [0, 0, 724, 990])[2] - view.get("crop", [0, 0, 724, 990])[0]) / 724
        button = view["landmarks"].get("button", [hack[0], hack[1] + 1])
        length = math.dist(button, hack) or 1
        direction = [(button[i] - hack[i]) / length for i in range(2)]
        output = []
        self.pending = [p for p in self.pending if seconds - p["last"] <= 1.5]
        for candidate in proposals:
            matches = [
                p
                for p in self.pending
                if p["color"] == candidate["color"] and math.dist(p["point"], candidate["point"]) < 35 * scale
            ]
            previous = (
                min(matches, key=lambda p: math.dist(p["point"], candidate["point"])) if matches else None
            )
            if previous:
                previous.update(point=candidate["point"], xy=candidate["xy"], last=seconds)
                previous["samples"] += 1
                dy = sum((candidate["point"][i] - previous["origin"][i]) * direction[i] for i in range(2))
                if dy < 8 * scale:
                    previous["started"] = seconds  # Waiting at the hack must not expire a future delivery.
                if (
                    dy > 12 * scale
                    and previous["samples"] >= 3
                    and seconds - self.last_birth > 15
                    and seconds - previous["started"] <= 15
                ):
                    output.append(candidate)
                    self.last_birth = seconds
                    self.pending.clear()
                    break
            elif math.dist(hack, candidate["point"]) < 32 * scale:
                self.pending.append(
                    {
                        **candidate,
                        "origin": candidate["point"],
                        "started": seconds,
                        "last": seconds,
                        "samples": 1,
                    }
                )
        return output


def mask_observation(mask, quality, rgb, view, color, background=None, require_granite=False):
    yy, xx = np.where(mask)
    if len(xx) < 8 or len(xx) > mask.size * 0.04:
        return None
    if yy.max() >= rgb.shape[0] - 1:
        return None  # A crop edge is not a visible ice-contact point.
    if view["kind"] == "overhead" and (xx.min() <= 0 or yy.min() <= 0 or xx.max() >= rgb.shape[1] - 1):
        return None  # A truncated stone has a biased centroid.
    width, height = xx.max() - xx.min() + 1, yy.max() - yy.min() + 1
    if height > width * 1.8 or width > rgb.shape[1] * 0.20:
        return None
    tee_distance = view.get("tee_distance_m", TEE_DISTANCE)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    colored = (hsv[:, :, 1] > 80) & (hsv[:, :, 2] > 60) & mask
    colored &= (
        ((hsv[:, :, 0] < 15) | (hsv[:, :, 0] > 165))
        if color == "red"
        else ((hsv[:, :, 0] >= 17) & (hsv[:, :, 0] <= 42))
    )
    if colored.sum() < 2:
        return None
    if background is not None:
        delta = cv2.absdiff(rgb, background).mean(axis=2)
        change = float(delta[colored].mean())
        if view["kind"] == "perspective":
            below = delta[
                int(yy.max()) : min(rgb.shape[0], int(yy.max() + width)),
                int(xx.min()) : int(xx.max() + 1),
            ]
            if below.size:
                change = max(change, float(below.mean()))
        if change < 24:
            return None
    ground_uncertain = False
    ground_inferred = False
    if view["kind"] == "overhead":
        point = [float(xx.mean()), float(yy.mean())]
    elif require_granite:
        from .tracking_stone import granite_anchor, granite_evidence

        point = granite_anchor(mask, granite_evidence(rgb, mask, color))
        if point is None:
            return None
    else:
        bottom = np.quantile(yy, 0.94)
        point = [float(np.median(xx[yy >= bottom])), float(yy.max())]
        # SAM sometimes segments the colored handle alone. Locate adjacent granite below it
        # for the ice-contact estimate; this does not expand or fabricate the saved SAM mask.
        raw_ground = project(
            view["image_to_ice"],
            [point[0] + view["crop"][0], point[1] + view["crop"][1]],
        )
        local_ground_y = raw_ground[1] if view["end"] == "A" else tee_distance - raw_ground[1]
        if height < width * 0.85 and abs(local_ground_y) < 4:
            ground_uncertain = True
            cy, cx = np.where(colored)
            center_x, center_y = float(cx.mean()), float(cy.mean())
            x0, x1 = (
                max(0, int(center_x - width * 1.5)),
                min(rgb.shape[1], int(center_x + width * 1.5 + 1)),
            )
            y0, y1 = (
                max(0, int(center_y)),
                min(rgb.shape[0], int(center_y + width * 2 + 1)),
            )
            patch = hsv[y0:y1, x0:x1]
            gray = ((patch[:, :, 1] < 70) & (patch[:, :, 2] > 45) & (patch[:, :, 2] < 210)).astype(np.uint8)
            if background is not None:
                gray &= (delta[y0:y1, x0:x1] > 20).astype(np.uint8)
            n, _, stats, _ = cv2.connectedComponentsWithStats(gray)
            plausible = []
            for j in range(1, n):
                gx, gy, gw, gh, ga = stats[j]
                if (
                    ga > 10
                    and gw > width * 0.45
                    and gh < width * 1.8
                    and gw < width * 3
                    and gw > gh
                    and gy < width * 0.8
                    and gx <= center_x - x0 <= gx + gw
                ):
                    plausible.append((ga, [float(x0 + gx + gw / 2), float(y0 + gy + gh)]))
            if plausible:
                point = max(plausible, key=lambda row: row[0])[1]
                ground_uncertain = False
                ground_inferred = True
    point = [point[0] + view["crop"][0], point[1] + view["crop"][1]]
    try:
        xy = project(view["image_to_ice"], point)
    except ValueError:
        return None
    local_y = xy[1] if view["end"] == "A" else tee_distance - xy[1]
    if abs(xy[0]) > 2.5 or not -4.5 < local_y < (
        TEE_DISTANCE + 4.5 if view["kind"] == "perspective" else 2.5
    ):
        return None
    variance = projection_variance(view, point)
    if ground_inferred:
        variance[1] = max(variance[1], 0.3**2)
    if ground_uncertain:
        variance = [max(variance[0], 0.15**2), max(variance[1], 0.8**2)]
    return dict(
        xy=xy,
        point=point,
        quality=quality,
        variance=variance,
        ground_uncertain=ground_uncertain,
        measurement_kind=("visible_granite_bottom" if require_granite else "legacy_ground_estimate")
        if view["kind"] == "perspective"
        else "overhead_centroid",
        kind=view["kind"],
        view_id=view["id"],
        bbox=[int(xx.min()), int(yy.min()), int(xx.max() + 1), int(yy.max() + 1)],
        area=int(mask.sum()),
    )
