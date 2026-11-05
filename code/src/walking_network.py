"""Directed pedestrian graph; edge splitting, finite connectors, auditable paths."""
from collections import Counter, defaultdict
import json
import numpy as np
import networkx as nx
from pyproj import Transformer
from shapely.geometry import LineString, Point, Polygon
from shapely.strtree import STRtree
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix

TO_XY=Transformer.from_crs(4326,5179,always_xy=True)
TO_LL=Transformer.from_crs(5179,4326,always_xy=True)
DISALLOWED={'motorway','motorway_link','trunk','trunk_link','construction','proposed','raceway','busway','platform','disused','abandoned','services','rest_area','emergency_bay','road'}
def allowed(tags, node=False):
    foot=tags.get('foot','')
    if foot in {'no','private','use_sidepath'}: return False
    if tags.get('access') in {'no','private','customers','permit'} and foot not in {'yes','designated','permissive'}: return False
    if node: return tags.get('barrier') not in {'wall','fence','retaining_wall'} or foot in {'yes','designated','permissive'}
    if tags.get('area')=='yes': return False
    if tags.get('highway') in DISALLOWED and foot not in {'yes','designated'}: return False
    return bool(tags.get('highway'))

class WalkingNetwork:
    def __init__(self,path,bbox):
        data=json.loads(path.read_text());nodes={r['id']:r for r in data['elements'] if r['type']=='node'}
        ways=[r for r in data['elements'] if r['type']=='way'];ids=list(nodes)
        xx,yy=TO_XY.transform([nodes[i]['lon'] for i in ids],[nodes[i]['lat'] for i in ids])
        self.positions={i:np.array([x,y]) for i,x,y in zip(ids,xx,yy)}
        self.graph=nx.DiGraph();accepted=Counter();omitted=Counter();blocked_nodes=0
        for w in ways:
            tags=w.get('tags',{});kind=tags.get('highway','none')
            if not allowed(tags): omitted[kind]+=1;continue
            accepted[kind]+=1;direction=tags.get('oneway:foot','no')
            for a,b in zip(w['nodes'][:-1],w['nodes'][1:]):
                if a not in nodes or b not in nodes or a==b: continue
                if not all(allowed(nodes[n].get('tags',{}),node=True) for n in (a,b)):
                    blocked_nodes+=1;continue
                length=float(np.linalg.norm(self.positions[a]-self.positions[b]))
                if length<=0:continue
                attr=dict(length=length,steps=kind=='steps',highway=kind,way_id=w['id'],layer=tags.get('layer','0'),bridge=tags.get('bridge','no'))
                for u,v,permit in [(a,b,direction!='-1'),(b,a,direction not in {'yes','true','1'})]:
                    if permit and (not self.graph.has_edge(u,v) or self.graph[u][v]['length']>length):self.graph.add_edge(u,v,**attr)
        self.segments=[];seen=set()
        for a,b,d in self.graph.edges(data=True):
            key=frozenset((a,b))
            if key in seen:continue
            seen.add(key);line=LineString([self.positions[a],self.positions[b]])
            self.segments.append((a,b,line,dict(d),self.graph.has_edge(a,b),self.graph.has_edge(b,a)))
        self.lines=[s[2] for s in self.segments];self.tree=STRtree(self.lines);self.snaps={}
        w,s,e,n=bbox
        self.bounds=Polygon([TO_XY.transform(lon,lat) for lon,lat in [(w,s),(e,s),(e,n),(w,n)]])
        self.audit=dict(raw_nodes=len(nodes),raw_ways=len(ways),accepted_way_types=dict(accepted),
            excluded_way_types=dict(omitted),blocked_node_segments=blocked_nodes,road_segments=len(self.segments),
            road_length_km=sum(s[2].length for s in self.segments)/1000,bbox=bbox,
            ordinary_vehicle_oneway_ignored=True,explicit_foot_oneway_preserved=True,
            crossing_rule='Shared OSM node IDs only; no joins at geometric crossings')

    def attach(self,records,max_offset=80):
        if self.snaps:raise ValueError('Attach all points once before calculating routes.')
        splits=defaultdict(list)
        for r in records:
            if r['id'] in self.snaps:raise ValueError('Duplicate ID: '+r['id'])
            p=Point(*TO_XY.transform(r['lon'],r['lat']));i=int(self.tree.nearest(p));line=self.lines[i]
            offset=float(line.distance(p));projection=float(line.project(p));rid=r['id']
            self.snaps[rid]=dict(offset_m=offset,matched=offset<=max_offset,segment=i,
                boundary_margin_m=float(p.distance(self.bounds.boundary)))
            if offset>max_offset:continue
            sid='SNAP_'+rid;q=line.interpolate(projection);self.positions[sid]=np.array([q.x,q.y]);self.positions[rid]=np.array([p.x,p.y])
            attr=dict(length=offset,highway='connector',steps=False,way_id=None,layer=None,bridge=None)
            self.graph.add_edge(rid,sid,**attr);self.graph.add_edge(sid,rid,**attr);splits[i].append((projection,sid))
        for i,entries in splits.items():
            a,b,line,attr,fwd,back=self.segments[i]
            if self.graph.has_edge(a,b):self.graph.remove_edge(a,b)
            if self.graph.has_edge(b,a):self.graph.remove_edge(b,a)
            chain=sorted([(0.,a),(line.length,b)]+entries,key=lambda p:p[0])
            for (s,n),(t,m) in zip(chain[:-1],chain[1:]):
                edge={**attr,'length':max(0.,t-s)}
                if fwd:self.graph.add_edge(n,m,**edge)
                if back:self.graph.add_edge(m,n,**edge)
        self.ids=list(self.graph);self.index={n:i for i,n in enumerate(self.ids)}
        self._matrix=None
        self.audit.update(graph_nodes=len(self.ids),graph_arcs=self.graph.number_of_edges(),
            connected_points=sum(r['matched'] for r in self.snaps.values()),
            unconnected_points=sum(not r['matched'] for r in self.snaps.values()),max_connector_m=max_offset,
            weak_components=nx.number_weakly_connected_components(self.graph))

    def matrix(self,no_steps=False):
        if self._matrix is None:
            rr=[];cc=[];vv=[];st=[]
            for a,b,d in self.graph.edges(data=True):rr.append(self.index[a]);cc.append(self.index[b]);vv.append(d['length']);st.append(d['steps'])
            self._matrix=(np.array(rr),np.array(cc),np.array(vv),np.array(st))
        rr,cc,vv,st=self._matrix;mask=~st if no_steps else np.ones(len(rr),dtype=bool)
        # Explicit zero edges are preserved: coincident projected connectors are connected.
        return csr_matrix((vv[mask],(rr[mask],cc[mask])),shape=(len(self.ids),len(self.ids)))

    def distances(self,origins,destinations,no_steps=False,batch_size=8):
        out=np.full((len(origins),len(destinations)),np.inf)
        valid_o=[i for i,k in enumerate(origins) if k in self.index]; valid_d=[i for i,k in enumerate(destinations) if k in self.index]
        if not valid_o or not valid_d:return out
        graph=self.matrix(no_steps);di=np.array([self.index[destinations[i]] for i in valid_d])
        # Reverse sources at destinations when they are fewer, preserving origin -> facility direction.
        reverse=len(valid_d)<len(valid_o)
        source=valid_d if reverse else valid_o
        for start in range(0,len(source),batch_size):
            chunk=source[start:start+batch_size]
            si=[self.index[(destinations if reverse else origins)[i]] for i in chunk]
            distance=dijkstra(graph.T.tocsr() if reverse else graph,directed=True,indices=si)
            if reverse:
                oi=[self.index[origins[i]] for i in valid_o]
                out[np.ix_(valid_o,chunk)]=distance[:,oi].T
            else:out[np.ix_(chunk,valid_d)]=distance[:,di]
        return out

    def route(self,origin,destination):
        path=nx.shortest_path(self.graph,origin,destination,weight='length')
        edges=[self.graph[a][b] for a,b in zip(path[:-1],path[1:])]
        coords=[TO_LL.transform(*self.positions[n]) for n in path]
        return dict(type='Feature',geometry=dict(type='LineString',coordinates=coords),properties=dict(
            origin=origin,destination=destination,length_m=sum(e['length'] for e in edges),
            connector_m=sum(e['length'] for e in edges if e['highway']=='connector'),
            steps_m=sum(e['length'] for e in edges if e['steps']),
            boundary_margin_m=min(Point(self.positions[n]).distance(self.bounds.boundary) for n in path)))
