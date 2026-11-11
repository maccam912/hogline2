// Sheet geometry, camera projection and the top-down drawing. No DOM access except a canvas context.
//
// Ice coordinates are metres: tee A is (0, 0), tee B is (0, TEE), X is lateral.
// Image coordinates are pixels in the full broadcast mosaic.

export const FT = 0.3048;
export const TEE = 114 * FT;
export const HOG = 21 * FT;
export const BACK = 6 * FT;
export const HACK = 12 * FT;
export const HALF_WIDTH = 4.75 / 2;
export const STONE_RADIUS = 0.1455;
const STONE_TOP = 0.115; // height of the coloured cap, which is what the eye finds in the picture
const RINGS = [[6 * FT, '#2f7fc4'], [4 * FT, '#eef4f6'], [2 * FT, '#58b368'], [0.5 * FT, '#eef4f6']];

function homography(h, [x, y]) {
  const z = h[6] * x + h[7] * y + h[8];
  return [(h[0] * x + h[1] * y + h[2]) / z, (h[3] * x + h[4] * y + h[5]) / z];
}

/** Mosaic pixel of a point `z` metres above the ice. */
export function toImage(camera, [x, y], z = 0) {
  if (z && camera.height) {
    const k = camera.height / (camera.height - z), [gx, gy] = camera.ground;
    return homography(camera.ice_to_image, [gx + (x - gx) * k, gy + (y - gy) * k]);
  }
  return homography(camera.ice_to_image, [x, y]);
}

/** Ice position of a mosaic pixel, taken to be on the ice surface. */
export const toIce = (camera, pixel) => homography(camera.image_to_ice, pixel);

export const cameraAt = (cameras, [x, y]) =>
  cameras.find(c => x >= c.crop[0] && x < c.crop[2] && y >= c.crop[1] && y < c.crop[3]) || null;

/** Pixels per lateral metre at an ice point: the size a stone is drawn at. */
export function pixelsPerMetre(camera, [x, y]) {
  const a = toImage(camera, [x - 0.5, y]), b = toImage(camera, [x + 0.5, y]);
  return Math.hypot(b[0] - a[0], b[1] - a[1]);
}

export const stoneInImage = (camera, position) => toImage(camera, position, STONE_TOP);

/** The part of a perspective panel that shows the far house, from the back line out to past the hog line. */
export function houseCrop(camera) {
  const tee = camera.end === 'A' ? 0 : TEE, inward = camera.end === 'A' ? 1 : -1;
  const points = [];
  for (const x of [-HALF_WIDTH - 0.4, HALF_WIDTH + 0.4]) {
    points.push(toImage(camera, [x, tee + inward * (HOG + 2.5)]));
    points.push(toImage(camera, [x, tee - inward * (BACK + 0.6)]));
    points.push(toImage(camera, [x, tee - inward * BACK], 2.0)); // headroom for people standing at the back
  }
  const [x0, y0, x1, y1] = camera.crop;
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const xs = points.map(p => p[0]), ys = points.map(p => p[1]);
  const box = [clamp(Math.min(...xs), x0, x1), clamp(Math.min(...ys), y0, y1),
    clamp(Math.max(...xs), x0, x1), clamp(Math.max(...ys), y0, y1)];
  return box[2] - box[0] > 50 && box[3] - box[1] > 50 ? box.map(Math.round) : camera.crop;
}

/**
 * Where a stone is at `seconds`, from its compact path of [seconds, x, y, moving] rows.
 * Returns null before it is thrown and after it has left play; `hidden` when the tracker had no
 * sight of it and the last known position is being held.
 */
export function positionAt(stone, seconds) {
  const path = stone.path;
  if (!path.length || seconds < path[0][0] - 0.05) return null;
  const last = path[path.length - 1];
  if (seconds >= last[0]) {
    return stone.out && seconds > last[0] + 0.3 ? null : {x: last[1], y: last[2], moving: false, hidden: false};
  }
  let lo = 0, hi = path.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (path[mid][0] <= seconds) lo = mid; else hi = mid;
  }
  const a = path[lo], b = path[hi], gap = b[0] - a[0];
  if (gap > 0.3 && (a[3] || b[3])) return {x: a[1], y: a[2], moving: false, hidden: true};
  const f = gap > 0 ? (seconds - a[0]) / gap : 0;
  return {x: a[1] + (b[1] - a[1]) * f, y: a[2] + (b[2] - a[2]) * f, moving: !!a[3], hidden: false};
}

/**
 * A top-down view as the thrower sees it. `along` is metres from the delivery tee toward the
 * far house and `right` is metres to the thrower's right of the centre line.
 * `sideways` lays the sheet out left to right (delivery on the left) instead of bottom to top.
 */
export function sheetView(rect, deliveryEnd, alongRange, sideways = false) {
  const d = deliveryEnd === 'A' ? 1 : -1, origin = deliveryEnd === 'A' ? 0 : TEE;
  const [lo, hi] = alongRange, length = hi - lo, width = 2 * HALF_WIDTH;
  const scale = sideways ? Math.min(rect.w / length, rect.h / width) : Math.min(rect.w / width, rect.h / length);
  const cx = rect.x + rect.w / 2, cy = rect.y + rect.h / 2;
  const place = (along, right) => sideways
    ? [cx + (along - (lo + hi) / 2) * scale, cy + right * scale]
    : [cx + right * scale, cy - (along - (lo + hi) / 2) * scale];
  return {
    scale, lo, hi,
    fromIce: ([x, y]) => place((y - origin) * d, -d * x),
    place,
  };
}

/** Ice, lines and rings. */
export function drawSheet(ctx, view) {
  const corners = [view.place(view.lo, -HALF_WIDTH), view.place(view.hi, HALF_WIDTH)];
  const x = Math.min(corners[0][0], corners[1][0]), y = Math.min(corners[0][1], corners[1][1]);
  const w = Math.abs(corners[1][0] - corners[0][0]), h = Math.abs(corners[1][1] - corners[0][1]);
  ctx.save();
  ctx.fillStyle = '#eef4f6';
  ctx.fillRect(x, y, w, h);
  ctx.beginPath();
  ctx.rect(x, y, w, h);
  ctx.clip();
  for (const tee of [0, TEE]) {
    if (tee + 6 * FT < view.lo || tee - 6 * FT > view.hi) continue;
    const [cx, cy] = view.place(tee, 0);
    for (const [radius, colour] of RINGS) {
      ctx.fillStyle = colour;
      ctx.beginPath();
      ctx.arc(cx, cy, radius * view.scale, 0, 2 * Math.PI);
      ctx.fill();
    }
  }
  const line = (a, b, colour, width) => {
    ctx.strokeStyle = colour; ctx.lineWidth = width;
    ctx.beginPath(); ctx.moveTo(...a); ctx.lineTo(...b); ctx.stroke();
  };
  line(view.place(view.lo, 0), view.place(view.hi, 0), '#9fb0b6', 1);
  for (const along of [0, TEE]) line(view.place(along, -HALF_WIDTH), view.place(along, HALF_WIDTH), '#9fb0b6', 1);
  for (const along of [-BACK, TEE + BACK]) line(view.place(along, -HALF_WIDTH), view.place(along, HALF_WIDTH), '#9fb0b6', 1);
  for (const along of [HOG, TEE - HOG]) line(view.place(along, -HALF_WIDTH), view.place(along, HALF_WIDTH), '#d2555a', Math.max(2, 0.1 * view.scale));
  ctx.restore();
}

export const STONE_COLOURS = {red: '#e5484d', yellow: '#f2c230'};

export function drawStone(ctx, view, position, colour, {highlight = false, hidden = false, label = ''} = {}) {
  const [x, y] = view.fromIce([position.x, position.y]);
  const radius = Math.max(3.5, STONE_RADIUS * view.scale);
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, 2 * Math.PI);
  ctx.fillStyle = hidden ? 'rgba(120,130,135,.35)' : '#6d777c';
  ctx.fill();
  ctx.beginPath();
  ctx.arc(x, y, radius * 0.68, 0, 2 * Math.PI);
  ctx.fillStyle = STONE_COLOURS[colour];
  ctx.globalAlpha = hidden ? 0.45 : 1;
  ctx.fill();
  ctx.globalAlpha = 1;
  if (highlight) {
    ctx.beginPath();
    ctx.arc(x, y, radius + 3, 0, 2 * Math.PI);
    ctx.strokeStyle = '#0b1113'; ctx.lineWidth = 2;
    ctx.stroke();
  }
  if (label && radius >= 7) {
    ctx.fillStyle = '#0b1113';
    ctx.font = `600 ${Math.round(radius * 0.95)}px system-ui, sans-serif`;
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(label, x, y + 0.5);
  }
}

/** The travelled path of one stone between two times, as a polyline. */
export function drawPath(ctx, view, stone, from, to, colour) {
  ctx.beginPath();
  let started = false;
  for (const [t, x, y] of stone.path) {
    if (t < from) continue;
    if (t > to) break;
    const [px, py] = view.fromIce([x, y]);
    if (started) ctx.lineTo(px, py); else { ctx.moveTo(px, py); started = true; }
  }
  ctx.strokeStyle = colour; ctx.lineWidth = 2; ctx.lineJoin = 'round';
  ctx.stroke();
}

export function drawBroom(ctx, [x, y], colour = '#00d0ff', size = 9) {
  ctx.lineCap = 'round';
  for (const [stroke, width] of [['#0b1113', size * 0.55], [colour, size * 0.25]]) {
    ctx.strokeStyle = stroke; ctx.lineWidth = width;
    ctx.beginPath();
    ctx.moveTo(x - size, y); ctx.lineTo(x - size * 0.3, y); ctx.moveTo(x + size * 0.3, y); ctx.lineTo(x + size, y);
    ctx.moveTo(x, y - size); ctx.lineTo(x, y - size * 0.3); ctx.moveTo(x, y + size * 0.3); ctx.lineTo(x, y + size);
    ctx.stroke();
    ctx.beginPath(); ctx.arc(x, y, size * 0.62, 0, 2 * Math.PI); ctx.stroke();
  }
  ctx.lineCap = 'butt';
}
