"""Small independently enumerable checks; no synthetic policy results are published."""
from pathlib import Path
import json,tempfile,itertools
import numpy as np
from walking_network import WalkingNetwork,TO_XY,TO_LL
from optimize_locations import solve_siting

def main():
    x,y=TO_XY.transform(128.21,37.14)
    def node(i,dx,dy):
        lon,lat=TO_LL.transform(x+dx,y+dy);return dict(type='node',id=i,lon=lon,lat=lat)
    def point(i,dx,dy):
        lon,lat=TO_LL.transform(x+dx,y+dy);return dict(id=i,lon=lon,lat=lat)
    elems=[node(1,0,0),node(2,100,0),node(3,50,-100),node(4,50,100),node(5,200,0),node(6,300,0),
      dict(type='way',id=11,nodes=[1,2],tags={'highway':'footway','oneway:foot':'yes'}),
      dict(type='way',id=12,nodes=[3,4],tags={'highway':'footway','bridge':'yes','layer':'1'}),
      dict(type='way',id=13,nodes=[5,6],tags={'highway':'service','access':'private'})]
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/'test.json';p.write_text(json.dumps({'elements':elems}));net=WalkingNetwork(p,[128.1,37.0,128.3,37.3])
        pts=[point('a',25,10),point('b',75,10),point('c',50,90),point('far',1000,1000)]
        net.attach(pts,80);dm=net.distances(['a','b','c'],['a','b','c'])
        assert abs(dm[0,1]-70)<1e-5,(dm[0,1],70)
        assert np.isinf(dm[1,0]),'Pedestrian one-way was lost'
        assert np.isinf(dm[0,2]),'Geometric overpass crossing became a junction'
        assert not net.snaps['far']['matched']
        assert not net.graph.has_edge(5,6)
    baseline=np.array([1000.,600.,800.,700.]);dist=np.array([[200,800,500],[500,100,300],[700,500,100],[200,400,500]],float)
    people=np.array([10,20,15,0]);elder=np.array([0,0,0,7])
    results=solve_siting(baseline,dist,people,1,elderly=elder,retain_efficiency=.9)
    choices=[]
    for k in range(2):
        for choice in itertools.combinations(range(3),k):
            after=np.minimum(baseline,dist[:,choice].min(axis=1)) if choice else baseline
            choices.append((float(people@(baseline-after)),float(elder@(baseline-after))))
    best=max(g for g,e in choices);fair_best=max(e for g,e in choices if g>=.9*best-1e-6)
    assert abs(results['efficiency']['population_distance_reduction_person_m']-best)<1e-5
    assert abs(results['elderly_priority']['elderly_distance_reduction_person_m']-fair_best)<1e-5
    report={'edge_projection_and_connector_distance':True,'pedestrian_oneway':True,'overpass_not_joined':True,
      'unconnected_point_not_forced':True,'private_road_excluded':True,'optimization_matches_enumeration':True,
      'elderly_constraint_matches_enumeration':True,'real_population_optimization_executed':False}
    (Path(__file__).resolve().parents[1]/'results/method_checks.json').write_text(json.dumps(report,indent=2));print(report)

if __name__=='__main__':main()
