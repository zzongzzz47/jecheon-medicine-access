"""Prepare Jecheon candidate locations and the official existing-site inventory."""
from collections import defaultdict
from pathlib import Path
import json, re, hashlib
import pandas as pd
from bs4 import BeautifulSoup
from shapely.geometry import shape, Point
from pyproj import Transformer
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT/'data/raw', ROOT/'data/processed'
NOTICE = 'https://www.jecheon.go.kr/health/selectBbsNttView.do?key=944&bbsNo=170&nttNo=386840'
def save(name, data):
    (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
def norm(name): return re.sub(r'\s+', '', name)
def health_name(name): return norm(name).replace('보건지소','지소').replace('보건진료소','진료소')
def address_key(address):
    m = re.search(r'([가-힣\d·]+(?:로|길)\d*(?:번길|길)?)[ ]+(\d+(?:-\d+)?)', address)
    return m.groups() if m else None

def main():
    OUT.mkdir(exist_ok=True,parents=True)
    boundary=shape(json.loads((RAW/'jecheon_osm_boundary_search.json').read_text())[0]['geojson'])
    save('boundary.geojson',dict(type='FeatureCollection',features=[dict(type='Feature',geometry=boundary.__geo_interface__,properties={'name':'제천시','source':'OpenStreetMap relation 2538334'})]))
    records=[]
    seniors=pd.read_csv(RAW/'jecheon_senior_centres_20251126.csv',encoding='cp949').fillna('')
    for r in seniors.to_dict('records'):
        records.append(dict(id=f"SC_{r['연번']:04d}",name=r['명칭'],kind='senior_centre',dong=r['읍면동'],
            address=r['도로명주소'],lon=float(r['경도']),lat=float(r['위도']),source_date=r['데이터기준일자'],
            source_url='https://www.data.go.kr/data/15013784/fileData.do',
            flags=['이용 대상·참여 조건 확인'],participation_verified=False))
    cvs=json.loads((RAW/'jecheon_convenience_20260630.json').read_text())
    for r in cvs:
        label=(r['상호명']+' '+r['지점명']).strip();flags=[]
        if re.search('노브랜드|더프레시|THE FRESH',label,re.I): flags.append('편의점 업종 재확인')
        if re.search('대학교|대학|병원|리조트|휴게소|구내|캠퍼스',label+' '+r['건물명']): flags.append('시설 출입 조건 확인')
        if r['층정보'] and r['층정보'] not in ('1','1층'): flags.append('층·출입구 위치 확인')
        records.append(dict(id='CV_'+r['상가업소번호'],name=label,kind='convenience',dong=r['행정동명'],
            address=r['도로명주소'],lon=float(r['경도']),lat=float(r['위도']),source_date='2026-06-30',
            source_url='https://www.data.go.kr/data/15083033/fileData.do',flags=flags,participation_verified=False))
    groups=defaultdict(list)
    for r in records: groups[(r['kind'],r['lon'],r['lat'])].append(r)
    candidates=[]
    for rows in groups.values():
        r=rows[0].copy();r['source_ids']=[x['id'] for x in rows];r['source_records']=len(rows)
        if len(rows)>1:
            r['name']=' / '.join(x['name'] for x in rows);r['flags'].append('동일 좌표 통합')
        r['inside_city_boundary']=boundary.covers(Point(r['lon'],r['lat']))
        if not r['inside_city_boundary']: r['flags'].append('시 경계 밖 좌표 확인')
        candidates.append(r)
    save('candidates.json',candidates)

    business=json.loads((RAW/'jecheon_all_businesses_20260630.json').read_text())
    hira_path=RAW/'hira_jecheon_institutions_20260930.json'
    hira=json.loads(hira_path.read_text()) if hira_path.exists() else []
    official=json.loads((RAW/'official_pharmacies.json').read_text())
    exclusions={'제천약국','제천튼튼온누리약국','나은약국','초정한약국'}
    inventory=[];excluded=[]
    for r in official:
        if norm(r['name']) in exclusions: excluded.append(r);continue
        inventory.append(dict(name=r['name'],address=r['address'],kind='existing_pharmacy',inventory_url=r['url']))
    soup=BeautifulSoup((RAW/'health_branches.html').read_bytes(),'html.parser')
    for tr in soup.select('#contents tbody tr'):
        t=tr.find_all(['td','th'])
        if len(t)>=2 and ('지소' in t[0].get_text() or '진료소' in t[0].get_text()):
            inventory.append(dict(name=t[0].get_text(' ',strip=True),address='충청북도 제천시 '+t[1].get_text(' ',strip=True),
                kind='existing_health',inventory_url='https://www.jecheon.go.kr/health/contents.do?key=872'))
    inventory.append(dict(name='제천시보건소',address='충청북도 제천시 의림대로 242',kind='existing_health',
        inventory_url='https://www.jecheon.go.kr/health/contents.do?key=874'))
    to_xy=Transformer.from_crs(4326,5179,always_xy=True)
    for k,r in enumerate(inventory):
        r.update(id=f'EX_{k+1:03d}',collection_evidence_url=NOTICE,collection_evidence_date='2025-08-28',
            collection_status='공식 안내 기준; 최신 운영 여부 미확인',lon=None,lat=None,coordinate_method=None)
        r['flags']=[]
        aliases={'우리메디칼약국':'우리메디컬약국'}  # Same road address, spelling verified against both inventories.
        match_name=aliases.get(r['name'],r['name'])
        hm=[h for h in hira if health_name(h['요양기관명'])==health_name(match_name) and h['좌표(X)'] and h['좌표(Y)']]
        if len(hm)==1:
            h=hm[0]
            r.update(lon=float(h['좌표(X)']),lat=float(h['좌표(Y)']),coordinate_method='심평원 기관명·제천시 일치',
                coordinate_source_url='https://www.data.go.kr/data/15051059/fileData.do',coordinate_date='2026-09-30',
                coordinate_source_address=h['주소'],coordinate_source_name=h['요양기관명'])
            if match_name!=r['name']:
                assert address_key(r['address'])==address_key(h['주소'])
                r['coordinate_name_note']='메디칼/메디컬 표기 차이; 도로명주소 일치 확인'
            if address_key(r['address'])!=address_key(h['주소']):r['flags'].append('공식 명부와 심평원 주소 불일치 확인')
            r['inside_city_boundary']=boundary.covers(Point(r['lon'],r['lat']))
            continue
        same_addr=[b for b in business if address_key(r['address']) and address_key(b['도로명주소'])==address_key(r['address'])]
        exact=[b for b in same_addr if norm(b['상호명'])==norm(r['name'])]
        match=exact or same_addr
        if match:
            ll=np.array([[float(b['경도']),float(b['위도'])] for b in match]);xx=np.array(to_xy.transform(ll[:,0],ll[:,1])).T
            span=float(np.linalg.norm(xx-xx.mean(axis=0),axis=1).max())
            if span<=50:
                r.update(lon=float(ll[:,0].mean()),lat=float(ll[:,1].mean()),
                    coordinate_method='상가정보 상호·주소 일치' if exact else '상가정보 동일 도로명주소 건물 좌표',
                    coordinate_source_ids=[b['상가업소번호'] for b in match],coordinate_spread_m=span)
        r['inside_city_boundary']=boundary.covers(Point(r['lon'],r['lat'])) if r['lon'] is not None else None
    save('existing_inventory.json',inventory);save('excluded_pharmacies.json',excluded)
    audit=dict(senior_raw=len(seniors),convenience_raw=len(cvs),candidate_locations=len(candidates),
        candidate_inside_city=sum(r['inside_city_boundary'] for r in candidates),
        existing_inventory_count=len(inventory),official_notice_total=73,
        excluded_pharmacies=len(excluded),existing_geocoded=sum(r['lon'] is not None for r in inventory),
        existing_missing=[{'id':r['id'],'name':r['name'],'address':r['address']} for r in inventory if r['lon'] is None])
    (ROOT/'results/facility_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    print(json.dumps(audit,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
