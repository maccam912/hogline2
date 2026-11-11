export function project(h, [x, y]) {
  const z = h[6] * x + h[7] * y + h[8];
  if (Math.abs(z) < 1e-8) return null;
  return [(h[0]*x+h[1]*y+h[2])/z, (h[3]*x+h[4]*y+h[5])/z];
}
export function rayX(ray, y) { return ray.origin[0] + ray.slope_dx_dy * (y-ray.origin[1]); }
export function rayBand(ray, y) {
  const d=y-ray.origin[1], c=ray.coefficient_covariance;
  return c ? 2*Math.sqrt(Math.max(0,c[0][0]+2*d*c[0][1]+d*d*c[1][1])) : 0;
}
export function compareTarget(ray, view, point) {
  const ice=project(view.image_to_ice,point);
  return {offsetRight: -ray.direction*(rayX(ray,ice[1])-ice[0]), band:rayBand(ray,ice[1])};
}
export function pairAt(pairs, seconds) {
  // A pair describes the interval ending at this observation. Never hold through a gap.
  return pairs.find(p=>seconds>=p.previous_seconds-1e-6&&seconds<=p.seconds+1e-6) || null;
}
