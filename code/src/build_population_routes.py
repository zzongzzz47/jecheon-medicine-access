"""Extract a real, aggregated resident-cell route example and boundary checks."""
import json
from pathlib import Path
import numpy as np
from walking_network import WalkingNetwork
from prepare_population import write_json

ROOT = Path(__file__).resolve().parents[1]
F = ROOT / 'results/residents_connector80'
def read(p): return json.loads(p.read_text())


def main():
    origins=read(F/'origins.json');ex=read(F/'existing.json');c=read(F/'candidates.json')
    result=read(F/'optimization_medoid_screened.json')
    plan=next(p for p in result['results'] if p['max_sites']==10 and p['candidate_group']=='mixed' and p['mode']=='efficiency')
    chosen=[next(i for i,r in enumerate(c) if r['id']==s['id']) for s in plan['selected']]
    matrix=np.load(F/'distances.npz')['walking'];n=len(ex)
    baseline=matrix[:,:n].min(axis=1);added=matrix[:,n+np.array(chosen)].min(axis=1)
    eligible=[i for i,r in enumerate(origins) if r['group']=='grid100m' and r['population']>=80 and np.isfinite(baseline[i]) and 700<baseline[i]<7000 and added[i]<baseline[i]-500]
    eligible.sort(key=lambda i:origins[i]['population']*(baseline[i]-added[i]),reverse=True)
    meta=read(ROOT/'data/raw/jecheon_roads_osm.metadata.json')
    net=WalkingNetwork(ROOT/'data/raw/jecheon_roads_osm.json',meta['bbox']);net.attach(origins+ex+c,80)
    routes=[];selected=None
    for i in eligible[:15]:
        j=int(np.argmin(matrix[i,:n]));k=chosen[int(np.argmin(matrix[i,n+np.array(chosen)]))]
        before=net.route(origins[i]['id'],ex[j]['id']);after=net.route(origins[i]['id'],c[k]['id'])
        if min(before['properties']['boundary_margin_m'],after['properties']['boundary_margin_m'])<500:continue
        for feature,name,stage,expected in [(before,ex[j]['name'],'before',baseline[i]),(after,c[k]['name'],'after',added[i])]:
            feature['properties'].update(destination_name=name,stage=stage)
            assert abs(feature['properties']['length_m']-expected)<1e-4
        selected={**origins[i], 'before_name':ex[j]['name'],'after_name':c[k]['name'],'before_m':float(baseline[i]),'after_m':float(added[i])}
        routes=[before,after];break
    if selected is None:raise ValueError('적절한 경로 사례 없음')
    # Check the most influential and longest baseline paths, where a clipped
    # study boundary would be most consequential. This is not an exhaustive audit.
    candidates=[i for i,r in enumerate(origins) if r['group']=='grid100m' and np.isfinite(baseline[i])]
    order_weight=sorted(candidates,key=lambda i:origins[i]['population']*baseline[i],reverse=True)[:12]
    order_length=sorted(candidates,key=lambda i:baseline[i],reverse=True)[:12]
    checks=[]
    for i in sorted(set(order_weight+order_length)):
        j=int(np.argmin(matrix[i,:n]));route=net.route(origins[i]['id'],ex[j]['id'])
        checks.append(dict(origin_id=origins[i]['id'],baseline_m=float(baseline[i]),population=origins[i]['population'],boundary_margin_m=route['properties']['boundary_margin_m']))
    write_json(F/'resident_example_routes.geojson',dict(type='FeatureCollection',features=routes))
    write_json(F/'route_audit.json',dict(example=selected,boundary_checked_paths=checks,
        min_boundary_margin_m=min(r['boundary_margin_m'] for r in checks),
        boundary_check_scope='총 이동부담 상위 12개·최장거리 상위 12개 출발점의 기존망 경로 표본',
        example_paths_match_distance_matrix=True))
    print(json.dumps(selected,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
