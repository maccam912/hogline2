import {test} from 'node:test';
import assert from 'node:assert/strict';
import {project, rayX, rayBand, compareTarget, pairAt} from '../hogline2/static/tracking_aim_math.mjs';
test('rotated camera projects ray and broom in the same plane',()=>{
  const v={image_to_ice:[-.01,0,5,0,-.01,39.747,0,0,1]};
  const ray={origin:[.2,6.401],slope_dx_dy:.02,direction:1,coefficient_covariance:[[.001,0],[0,.0001]]};
  const point=[400,500], ice=project(v.image_to_ice,point);
  assert.deepEqual(ice,[1,34.747]);
  assert.equal(compareTarget(ray,v,point).offsetRight,1-rayX(ray,34.747));
  assert.ok(rayBand(ray,34.747)>rayBand(ray,6.401));
  assert.equal(project([1,0,0,0,1,0,0,0,0],point),null);
});
test('pair playback withholds stale rays in gaps',()=>{
  const a={previous_seconds:1,seconds:1.333},b={previous_seconds:2,seconds:2.333};
  assert.equal(pairAt([a,b],1.2),a);
  assert.equal(pairAt([a,b],1.8),null);
  assert.equal(pairAt([a,b],2.333),b);
  assert.equal(pairAt([a,b],2.4),null);
});
