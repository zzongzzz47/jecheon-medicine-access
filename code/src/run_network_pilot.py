"""Actual facility-point network audit. Population benefits are not estimated here."""
from pathlib import Path
from collections import Counter
import json, pickle, time
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from walking_network import WalkingNetwork, TO_XY

ROOT=Path(__file__).resolve().parents[1]
def save(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def coords(rows):return np.array(TO_XY.transform([r['lon'] for r in rows],[r['lat'] for r in rows])).T
def summary(v):
    v=np.asarray(v);v=v[np.isfinite(v)]
    return dict(n=len(v),mean=float(v.mean()),median=float(np.median(v)),p90=float(np.quantile(v,.9))) if len(v) else {'n':0}

def main():
    started=time.monotonic();out=ROOT/'results';out.mkdir(exist_ok=True)
    c=json.loads((ROOT/'data/processed/candidates.json').read_text());ex=json.loads((ROOT/'data/processed/existing_inventory.json').read_text())
    assert len(ex)==73 and all(r['lon'] is not None for r in ex),'Existing inventory is incomplete'
    meta=json.loads((ROOT/'data/raw/jecheon_roads_osm.metadata.json').read_text())
    print('Building road graph',flush=True)
    network=WalkingNetwork(ROOT/'data/raw/jecheon_roads_osm.json',meta['bbox']);network.attach(c+ex,80)
    print('Road graph',network.audit,flush=True)
    save(out/'road_network_audit.json',network.audit)
    save(out/'connector_audit.json',[dict(id=r['id'],name=r['name'],kind=r['kind'],**network.snaps[r['id']]) for r in c+ex])
    with (out/'facility_network.pkl').open('wb') as f:pickle.dump(network,f,pickle.HIGHEST_PROTOCOL)
    candidates_id=[r['id'] for r in c];existing_id=[r['id'] for r in ex]
    print('Calculating facility -> existing-site distances',flush=True)
    road=network.distances(candidates_id,existing_id);euclid=cdist(coords(c),coords(ex))
    strict_ex=np.array([not r['flags'] for r in ex]);geocoded_ex=np.array([network.snaps[r['id']]['matched'] for r in ex])
    np.savez_compressed(out/'facility_to_existing_distances.npz',walking=road,straight=euclid,origins=candidates_id,destinations=existing_id)
    rows=[]
    for i,r in enumerate(c):
        reachable=np.isfinite(road[i]).any();j=int(np.argmin(road[i])) if reachable else None
        ej=int(np.argmin(euclid[i]));matched_ej=int(np.argmin(np.where(geocoded_ex,euclid[i],np.inf)))
        d=float(road[i,j]) if j is not None else None;line=float(euclid[i,j]) if j is not None else None
        rows.append({**r,**network.snaps[r['id']], 'nearest_road_existing_id':ex[j]['id'] if j is not None else None,
            'nearest_road_existing_name':ex[j]['name'] if j is not None else None,'road_distance_m':d,
            'straight_to_same_destination_m':line,'detour_ratio_same_destination':d/line if line and line>10 else None,
            'nearest_straight_existing_name':ex[ej]['name'],'nearest_straight_distance_m':float(euclid[i,ej]),
            'nearest_straight_among_connected_existing_m':float(euclid[i,matched_ej]),
            'straight_choice_road_distance_m':float(road[i,ej]) if np.isfinite(road[i,ej]) else None,
            'straight_and_road_choice_differ':ex[ej]['id']!=ex[j]['id'] if j is not None else None,
            'address_verified_subset_distance_m':float(np.min(road[i,strict_ex])) if np.isfinite(road[i,strict_ex]).any() else None})
    save(out/'facility_accessibility.json',rows)
    pd.DataFrame(rows).drop(columns=['flags','source_ids']).to_csv(out/'facility_accessibility.csv',index=False,encoding='utf-8-sig')
    route_pool=[r for r in rows if r['road_distance_m'] is not None and 80<r['straight_to_same_destination_m']<1000 and r['road_distance_m']<3000 and r['offset_m']<40]
    route_pool.sort(key=lambda r:r['road_distance_m']-r['straight_to_same_destination_m'],reverse=True)
    routes=[]
    for r in route_pool[:12]:
        route=network.route(r['id'],r['nearest_road_existing_id'])
        route['properties'].update(name=r['name'],kind=r['kind'],destination_name=r['nearest_road_existing_name'],straight_m=r['straight_to_same_destination_m'])
        if route['properties']['boundary_margin_m']>1000:routes.append(route)
        if len(routes)==3:break
    save(out/'example_routes.geojson',{'type':'FeatureCollection','features':routes})
    finite=np.isfinite(road)
    validation=dict(network_not_shorter_than_straight=bool(np.all(road[finite]+1e-5>=euclid[finite])),
        all_existing_inventory_accounted_for=len(ex)==73,all_points_unique=len({r['id'] for r in c+ex})==len(c+ex),
        route_lengths_match=all(abs(f['properties']['length_m']-road[candidates_id.index(f['properties']['origin']),existing_id.index(f['properties']['destination'])])<1e-5 for f in routes))
    assert all(validation.values()),validation
    stats={}
    for kind in ['senior_centre','convenience']:
        group=[r for r in rows if r['kind']==kind];valid=[r for r in group if r['road_distance_m'] is not None]
        stats[kind]=dict(total_locations=len(group),connected_origins=sum(r['matched'] for r in group),reachable_existing=len(valid),
            walking=summary([r['road_distance_m'] for r in valid]),straight_same_dest=summary([r['straight_to_same_destination_m'] for r in valid]),
            straight_nearest_same_sample=summary([r['nearest_straight_among_connected_existing_m'] for r in valid]),
            median_detour_same_destination=float(np.median([r['detour_ratio_same_destination'] for r in valid if r['detour_ratio_same_destination'] is not None])),
            different_closest_site=sum(r['straight_and_road_choice_differ'] for r in valid),
            euclidean_within_350_but_walk_over_350=sum(r['nearest_straight_among_connected_existing_m']<=350<r['road_distance_m'] for r in valid))
    review=dict(analysis_unit='시설 위치. 주민·고령인구의 대표점으로 사용하지 않음',
        statistics=stats,validation=validation,existing_connected=int(sum(geocoded_ex)),existing_unconnected=[r['name'] for r,m in zip(ex,geocoded_ex) if not m],
        missing_inputs=['제천시 인구 격자와 격자 경계','제천시 집계구 연령별 인구·집계구 경계'],
        population_optimization_run=False,elapsed_seconds=time.monotonic()-started)
    save(out/'pilot_summary.json',review);print(json.dumps(review,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
