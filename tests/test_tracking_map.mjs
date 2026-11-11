import assert from 'node:assert/strict';
import {mapPoint,nearHouse,TEE_DISTANCE,reviewEvents} from '../hogline2/static/tracking_map.mjs';
// Perspective at B maps image-right to world negative x. Facing that camera,
// A is near us and the plotted stone must remain on our right.
assert.deepEqual(mapPoint([-1,30],nearHouse('auto','B')),[1,30]);
assert.deepEqual(mapPoint([1,4],nearHouse('auto','A')),[1,TEE_DISTANCE-4]);
assert.equal(nearHouse('B','B'),'B');
// View changes rotate both horizontal and longitudinal axes, not just mirror x.
const a=mapPoint([1,7],'A'),b=mapPoint([1,7],'B');
assert.equal(a[0],-b[0]);assert.equal(a[1]+b[1],TEE_DISTANCE);
console.log('Map orientation checks passed');

const events=[{type:'delivery',stone:'red_1',seconds:0},...Array.from({length:60},(_,i)=>({type:'measurement_rejected',stone:'red_1',seconds:10+i/3})),{type:'possible_contact',stones:['red_1','yellow_1'],seconds:40}];
const grouped=reviewEvents(events);
assert.equal(grouped.length,3);assert.equal(grouped[0].type,'delivery');assert.equal(grouped[1].count,60);assert.equal(grouped[2].type,'possible_contact');
assert.deepEqual(mapPoint([1,4],'B',36.1),[1,32.1]);

const {sheetLayout}=await import('../hogline2/static/tracking_map.mjs');
const layout=sheetLayout();
assert.ok(Math.abs(layout.max_y-layout.min_y-45.720)<1e-10);
assert.equal(layout.width_m,4.750);
assert.deepEqual(layout.markings.find(m=>m.name==='Hack').ys,[-3.658,34.747+3.658]);
assert.deepEqual(layout.markings.find(m=>m.name==='Hog').ys,[6.401,34.747-6.401]);
for(const m of layout.markings)for(const y of m.ys)assert.ok(y>layout.min_y&&y<layout.max_y);
assert.equal(sheetLayout({sheet_geometry:{width_m:4.6}}).width_m,4.6);
console.log('Sheet extent and marking checks passed');
