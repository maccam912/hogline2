// UI geometry: camera projection, stone positions over time and the top-down view.
import assert from 'node:assert/strict';
import test from 'node:test';
import {TEE, cameraAt, houseCrop, positionAt, sheetView, toIce, toImage} from '../hogline2/static/js/sheet.js';

// a camera straight above house A: 100 px per metre, tee A at pixel (500, 500)
const overhead = {
  id: 'overhead_a', kind: 'overhead', end: 'A', crop: [0, 0, 1000, 1000],
  ice_to_image: [100, 0, 500, 0, 100, 500, 0, 0, 1], image_to_ice: [0.01, 0, -5, 0, 0.01, -5, 0, 0, 1],
  ground: [0, 0], height: 4,
};

test('pixels and ice positions round-trip, and height shifts a point away from the camera', () => {
  assert.deepEqual(toImage(overhead, [1, 2]), [600, 700]);
  assert.deepEqual(toIce(overhead, [600, 700]).map(v => Math.round(v * 1e6) / 1e6), [1, 2]);
  const lifted = toImage(overhead, [1, 0], 0.115);
  assert.ok(lifted[0] > 600 && lifted[0] < 604 && lifted[1] === 500);
  assert.equal(cameraAt([overhead], [10, 10]), overhead);
  assert.equal(cameraAt([overhead], [1000, 10]), null);
});

test('a stone is absent before delivery, interpolated while seen, held while hidden, gone once out', () => {
  const stone = {out: true, path: [[10, 0, 30, 1], [10.1, 0, 29, 1], [12, 0, 20, 1], [12.1, 0, 20, 0]]};
  assert.equal(positionAt(stone, 9), null);
  assert.ok(Math.abs(positionAt(stone, 10.05).y - 29.5) < 1e-9);
  assert.deepEqual(positionAt(stone, 11), {x: 0, y: 29, moving: false, hidden: true});
  assert.equal(positionAt(stone, 12.2).y, 20);
  assert.equal(positionAt(stone, 13), null);
  assert.equal(positionAt({...stone, out: false}, 500).y, 20);
});

test('the top-down view is drawn as the thrower sees it', () => {
  const rect = {x: 0, y: 0, w: 100, h: 400};
  const fromB = sheetView(rect, 'B', [0, TEE]), fromA = sheetView(rect, 'A', [0, TEE]);
  // thrown from B, the far tee (A) is at the top, and +X on the ice is to the thrower's right
  assert.ok(fromB.fromIce([0, 0])[1] < fromB.fromIce([0, TEE])[1]);
  assert.ok(fromB.fromIce([1, 10])[0] > fromB.fromIce([0, 10])[0]);
  assert.ok(fromA.fromIce([1, 10])[0] < fromA.fromIce([0, 10])[0]);
  const strip = sheetView({x: 0, y: 0, w: 400, h: 60}, 'B', [0, TEE], true);
  assert.ok(strip.fromIce([0, TEE])[0] < strip.fromIce([0, 0])[0]);
});

test('the house crop stays inside its panel', () => {
  const [x0, y0, x1, y1] = houseCrop(overhead);
  assert.ok(x0 >= 0 && y0 >= 0 && x1 <= 1000 && y1 <= 1000 && x1 > x0 && y1 > y0);
});
