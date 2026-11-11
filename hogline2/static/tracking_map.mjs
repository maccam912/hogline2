export const TEE_DISTANCE = 34.747;
// World +x was defined as image-right in the perspective looking AT house A.
// With A nearest the viewer, world +x therefore appears on screen-left.
export function mapPoint(point, nearHouse = 'A', teeDistance = TEE_DISTANCE) {
  return nearHouse === 'B' ? [point[0], teeDistance - point[1]] : [-point[0], point[1]];
}
export function nearHouse(selection, deliveryHouse = 'B') {
  return selection === 'auto' ? (deliveryHouse === 'A' ? 'B' : 'A') : selection;
}

export function reviewEvents(events) {
  const allowed = new Set(['delivery','view_join','possible_contact','collision_identity_ambiguous','identity_uncertain','unexplained_displacement','measurement_rejected','view_disagreement']);
  const grouped = [], last = new Map();
  for (const event of events) {
    if (!allowed.has(event.type)) continue;
    const key = [event.type,event.stone,event.view,(event.stones||[]).join(',')].join('|');
    const prior = last.get(key);
    if (prior && !['delivery','view_join','possible_contact'].includes(event.type) && event.seconds-prior.last_seconds <= 2) {
      prior.count++;prior.last_seconds=event.seconds;
    } else {
      const item={...event,count:1,last_seconds:event.seconds};grouped.push(item);last.set(key,item);
    }
  }
  return grouped;
}


export function sheetLayout(calibration = {}) {
  const g = {length_m:45.720,width_m:4.750,hog_from_tee_m:6.401,
    hack_from_tee_m:3.658,back_from_tee_m:1.829,house_radius_m:1.829,
    ...calibration.sheet_geometry};
  const tee = calibration.tee_distance_m ?? TEE_DISTANCE;
  const margin = (g.length_m - tee) / 2;
  return {...g,tee_distance_m:tee,min_y:-margin,max_y:tee+margin,
    markings: [{name:'Tee',ys:[0,tee]},
      {name:'Hog',ys:[g.hog_from_tee_m,tee-g.hog_from_tee_m]},
      {name:'Back',ys:[-g.back_from_tee_m,tee+g.back_from_tee_m]},
      {name:'Hack',ys:[-g.hack_from_tee_m,tee+g.hack_from_tee_m]}]};
}
