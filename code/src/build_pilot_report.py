"""Render the road-based pilot as figures and a local HTML research note."""
from pathlib import Path
from collections import Counter
import base64,html,json,pickle,hashlib,datetime,platform,importlib.metadata
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon as PolygonPatch
from matplotlib.font_manager import FontProperties
from shapely.geometry import shape
from shapely.ops import transform
from walking_network import TO_XY,TO_LL

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results';FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
FONT='/System/Library/Fonts/Supplemental/AppleGothic.ttf'
plt.rcParams.update({'font.family':FontProperties(fname=FONT).get_name(),'axes.unicode_minus':False,'font.size':11,
 'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','axes.titleweight':'bold'})
COL={'existing':'#0f4254','convenience':'#ed964b','senior_centre':'#279f94','route':'#d44b43'}
def read(p):return json.loads(p.read_text())
def xy(rows):return np.array(TO_XY.transform([r['lon'] for r in rows],[r['lat'] for r in rows])).T
def figure_save(fig,name):
    fig.savefig(FIG/(name+'.png'),dpi=170,bbox_inches='tight',facecolor='white');plt.close(fig)
def kmbar(ax,x,y,metres):
    ax.plot([x,x+metres],[y,y],color='#233b4a',lw=3);ax.text(x+metres/2,y+metres*.12,f'{metres/1000:g} km',ha='center',fontsize=9)
def baseroad(ax,segments,bounds,linewidth=.45):
    xmin,ymin,xmax,ymax=bounds
    sel=(segments[:,:,0].max(axis=1)>=xmin)&(segments[:,:,0].min(axis=1)<=xmax)&(segments[:,:,1].max(axis=1)>=ymin)&(segments[:,:,1].min(axis=1)<=ymax)
    ax.add_collection(LineCollection(segments[sel],colors='#d9e1e5',linewidths=linewidth,zorder=1))
    ax.set_xlim(xmin,xmax);ax.set_ylim(ymin,ymax);ax.set_aspect('equal');ax.axis('off')
def imgtag(name):
    data=base64.b64encode((FIG/(name+'.png')).read_bytes()).decode()
    return f'<img src="data:image/png;base64,{data}" alt="{html.escape(name)}">'
def table(headers,rows):return '<div class="table"><table><thead><tr>'+''.join(f'<th>{html.escape(str(h))}</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{html.escape(str(x))}</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'

def main():
    summary=read(OUT/'pilot_summary.json');audit=read(OUT/'facility_audit.json');road_audit=read(OUT/'road_network_audit.json')
    candidate=read(ROOT/'data/processed/candidates.json');existing=read(ROOT/'data/processed/existing_inventory.json')
    accesses=read(OUT/'facility_accessibility.json');routes=read(OUT/'example_routes.geojson')['features']
    boundary=transform(TO_XY.transform,shape(read(ROOT/'data/processed/boundary.geojson')['features'][0]['geometry']))
    with (OUT/'facility_network.pkl').open('rb') as f:net=pickle.load(f)
    segments=np.array([np.asarray(s[2].coords) for s in net.segments]);cxy=xy(candidate);exy=xy(existing)
    fig,axs=plt.subplots(1,2,figsize=(14,9),gridspec_kw={'width_ratios':[1.1,1]})
    citybounds=boundary.bounds
    centre=[*TO_XY.transform(128.175,37.105),*TO_XY.transform(128.25,37.18)]
    for ax,bounds,title in zip(axs,[citybounds,centre],['제천시 전역','도심부 확대']):
        geoms=[boundary] if boundary.geom_type=='Polygon' else list(boundary.geoms)
        for poly in geoms:ax.add_patch(PolygonPatch(np.asarray(poly.exterior.coords),facecolor='#f4f7f9',edgecolor='#b6c4cc',linewidth=.9,zorder=0))
        baseroad(ax,segments,bounds,.22 if ax is axs[0] else .55)
        for kind,label in [('senior_centre','경로당 후보'),('convenience','편의점 후보')]:
            mask=np.array([r['kind']==kind for r in candidate]);ax.scatter(cxy[mask,0],cxy[mask,1],s=10 if ax is axs[0] else 25,color=COL[kind],alpha=.8,label=label,zorder=4)
        ax.scatter(exy[:,0],exy[:,1],s=18 if ax is axs[0] else 40,marker='s',color=COL['existing'],label='기존 수거지 명부',zorder=5)
        ax.set_title(title,pad=14,fontsize=15);ax.text(.96,.94,'N ↑',transform=ax.transAxes,ha='right',fontsize=11)
        xmin,ymin,xmax,ymax=bounds;kmbar(ax,xmin+(xmax-xmin)*.08,ymin+(ymax-ymin)*.06,5000 if ax is axs[0] else 1000)
    axs[1].legend(loc='lower right',frameon=True,fontsize=10)
    fig.suptitle('기존 수거지 73곳과 편의점·경로당 후보 위치',fontsize=20,y=.98)
    fig.text(.5,.03,'후보: 동일 좌표 통합 후 편의점 138개 위치 · 경로당 341개 위치  |  지도: OpenStreetMap',ha='center',fontsize=10,color='#526675')
    fig.tight_layout(rect=[0,.05,1,.95]);figure_save(fig,'01_기존수거망과_후보위치')

    selected=next(f for f in routes if f['properties']['name']=='고평 경로당')
    ll=np.array(selected['geometry']['coordinates']);path=np.array(TO_XY.transform(ll[:,0],ll[:,1])).T
    xmin,ymin=path.min(axis=0)-220;xmax,ymax=path.max(axis=0)+220
    fig,ax=plt.subplots(figsize=(11,8));baseroad(ax,segments,[xmin,ymin,xmax,ymax],1.1)
    ax.plot(path[:,0],path[:,1],color=COL['route'],lw=4,label=f"보행도로 경로 {selected['properties']['length_m']:,.0f} m",zorder=6)
    ax.plot(path[[0,-1],0],path[[0,-1],1],color=COL['existing'],ls='--',lw=2,label=f"직선 {selected['properties']['straight_m']:,.0f} m",zorder=7)
    ax.scatter(*path[0],s=160,color=COL['senior_centre'],edgecolors='white',lw=2,zorder=8)
    ax.scatter(*path[-1],s=150,marker='s',color=COL['existing'],edgecolors='white',lw=2,zorder=8)
    ax.annotate('고평 경로당',path[0],xytext=(10,13),textcoords='offset points',bbox={'facecolor':'white','edgecolor':'none','alpha':.9},fontsize=13,zorder=9)
    ax.annotate('보성약국',path[-1],xytext=(10,-22),textcoords='offset points',bbox={'facecolor':'white','edgecolor':'none','alpha':.9},fontsize=13,zorder=9)
    ax.legend(loc='upper left',frameon=True,fontsize=12);kmbar(ax,xmin+50,ymin+55,200)
    ax.set_title('같은 출발지와 목적지도, 길을 따라가면 달라진다',fontsize=19,pad=22)
    fig.text(.5,.015,'OSM 보행도로 최단경로 추정 · 시설↔도로 연결거리 포함 · 현장 검증 전',ha='center',fontsize=10,color='#526675')
    figure_save(fig,'02_직선거리와_보행경로_비교')

    fig,axs=plt.subplots(1,2,figsize=(13,5.8))
    for ax,kind,label in zip(axs,['senior_centre','convenience'],['경로당 333개 위치','편의점 131개 위치']):
        valid=[r for r in accesses if r['kind']==kind and r['road_distance_m'] is not None]
        for key,color,lab in [('nearest_straight_among_connected_existing_m','#8ca9b7','직선거리 기준'),('road_distance_m',COL[kind],'보행도로 기준')]:
            d=np.sort([r[key]/1000 for r in valid]);ax.plot(d,np.arange(1,len(d)+1)/len(d)*100,color=color,lw=2.8,label=lab)
        ax.set_xlim(0,6);ax.set_ylim(0,101);ax.set_xticks(range(7));ax.grid(alpha=.17);ax.set_xlabel('가장 가까운 기존 수거지까지 거리 (km)');ax.set_ylabel('시설 위치의 누적 비율 (%)');ax.set_title(label);ax.legend(loc='lower right',fontsize=10)
    fig.suptitle('시설 위치에서 본 접근성: 직선거리와 보행도로의 차이',fontsize=18,y=1.03)
    fig.text(.5,-.02,'각 시설을 출발점으로 한 예비 진단. 주민 수·경로당 이용자 수로 가중하지 않았으며, 주민 커버율을 뜻하지 않음.',ha='center',fontsize=10,color='#526675')
    fig.tight_layout();figure_save(fig,'03_시설출발점_거리분포')

    stats=summary['statistics'];s=stats['senior_centre'];c=stats['convenience']
    conflict=[r for r in existing if r['flags']]
    disconnected=[r for r in accesses if not r['matched'] or r['road_distance_m'] is None]
    sources=[('경로당 현황','제천시','2025-11-26','https://www.data.go.kr/data/15013784/fileData.do','342건 → 341개 좌표'),
      ('상가(상권)정보','소상공인시장진흥공단','2026-06-30','https://www.data.go.kr/data/15083033/fileData.do','편의점 업종 147건 → 138개 좌표'),
      ('폐의약품 수거함 위치 및 배출방법 안내','제천시 보건소','2025-08-28','https://www.jecheon.go.kr/health/selectBbsNttView.do?key=944&bbsNo=170&nttNo=386840','수거망 73곳·제외 약국 4곳'),
      ('약국 명부','제천시 보건소','2026-10-22 조회','https://www.jecheon.go.kr/health/selectBbsNttList.do?bbsNo=409&key=889','57곳에서 제외 명단 4곳 적용'),
      ('보건지소 및 진료소','제천시 보건소','2026-10-22 조회','https://www.jecheon.go.kr/health/contents.do?key=872','19곳 및 보건소 1곳'),
      ('전국 병의원 및 약국 현황','건강보험심사평가원','2026-09-30','https://www.data.go.kr/data/15051059/fileData.do','73곳 좌표 연결·주소 불일치 3곳'),
      ('보행도로와 시 경계','OpenStreetMap','2026-10-22 추출','https://www.openstreetmap.org/copyright','ODbL / 주변 도로 포함'),
      ('인구·연령별 통계와 경계','SGIS','신청·확보 필요','https://sgis.mods.go.kr/view/pss/openDataIntrcn','이번 예비 결과에는 미사용')]
    source_rows=''.join(f'<tr><td><a href="{u}" target="_blank">{html.escape(n)}</a></td><td>{org}</td><td>{date}</td><td>{note}</td></tr>' for n,org,date,u,note in sources)
    methodchecks=read(OUT/'method_checks.json')
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>제천 폐의약품 수거망 · 보행도로 연결 1차 분석</title><style>
    :root{--ink:#173645;--teal:#158f82;--muted:#536975;--line:#dae4e8}*{box-sizing:border-box}body{margin:0;background:#f1f5f6;color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,'Apple SD Gothic Neo','Malgun Gothic',sans-serif;line-height:1.75}main{max-width:1120px;margin:auto;background:white;padding:52px 60px}h1{font-size:38px;line-height:1.28;letter-spacing:-1.4px;margin:12px 0 20px}h2{font-size:25px;margin:0 0 15px}h3{font-size:18px;margin:20px 0 8px}p{margin:10px 0 16px}a{color:#087d77}section{padding:36px 0;border-top:1px solid var(--line)}.eyebrow{color:var(--teal);font-weight:700;letter-spacing:1px;font-size:13px}.lead{font-size:19px;max-width:870px}.note{font-size:14px;color:var(--muted)}.callout{background:#eaf5f3;padding:20px 24px;border-left:4px solid var(--teal);margin:24px 0}.kpis{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:28px 0}.kpi{border:1px solid var(--line);padding:18px 20px;border-radius:10px}.kpi b{display:block;font-size:31px}.kpi span{font-size:14px;color:var(--muted)}img{width:100%;height:auto;display:block;margin:22px 0}.table{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:14px;margin:18px 0}th{background:#edf3f5;text-align:left}td,th{border-bottom:1px solid var(--line);padding:11px 13px;vertical-align:top}.flow{background:#173645;color:white;padding:22px 25px;border-radius:8px;font-size:18px}details{margin:16px 0}summary{cursor:pointer;font-weight:600}.pill{display:inline-block;font-size:12px;background:#f3e6c9;color:#705312;padding:3px 10px;border-radius:15px}footer{font-size:13px;color:var(--muted);margin-top:30px}@media(max-width:700px){main{padding:26px 20px}h1{font-size:29px}.kpis{grid-template-columns:1fr}.lead{font-size:17px}}@media print{body{background:white}main{padding:0}section{break-inside:avoid}details{display:block}}
    </style><main><header><div class="eyebrow">JECHEON · WALKING ACCESSIBILITY · 2026.10.22</div><h1>길을 따라 계산한<br>제천시 폐의약품 수거 접근성</h1><p class="lead">기존 수거지와 편의점·경로당을 보행도로에 연결했다. 주민 인구를 연결하기에 앞서, 직선거리와 도로거리의 차이와 지도 연결 상태를 점검한 1차 결과다.</p><span class="pill">시설 위치 기준 예비 분석 · 주민 최적 입지는 아직 미선정</span></header>'''
    page+='<div class="kpis"><div class="kpi"><b>73곳</b><span>공식 안내로 구성한 기존 수거망</span></div><div class="kpi"><b>138개 위치</b><span>편의점 후보 · 원자료 147건</span></div><div class="kpi"><b>341개 위치</b><span>경로당 후보 · 원자료 342건</span></div></div>'
    page+='<section><h2>1. 이번에는 무엇이 달라졌나</h2><p>이번 분석은 제천시의 기존 수거망을 유지하면서 편의점과 경로당을 보완 거점으로 검토한다. 시설을 중심으로 원을 그리는 방식에서 더 나아가, 주민이 이동할 수 있는 도로를 따라 거리를 계산하도록 구성했다.</p><div class="flow">주민 위치 → 도로까지 연결 → 보행 가능한 도로의 최단경로 → 수거지까지 연결</div><p>이번 단계에서는 확보한 시설 위치를 출발점으로 사용해 도로 연결과 거리 계산을 검증했다. SGIS 인구를 확보하면 출발점을 주민의 인구 격자·집계구 대표점으로 바꿔 주민 수에 따른 개선 효과를 계산한다.</p>'+imgtag('01_기존수거망과_후보위치')+'</section>'
    page+='<section><h2>2. 직선 360m가 도로에서는 약 1.38km</h2><p>고평 경로당에서 보성약국까지 같은 두 지점을 비교했다. 직선거리는 <b>약 360m</b>, 확보한 보행도로의 최단경로는 <b>약 1,381m</b>다. 시설에서 도로로 연결되는 구간 약 25m도 도로거리 합계에 포함했다.</p>'+imgtag('02_직선거리와_보행경로_비교')+'<p class="note">이 경로는 OpenStreetMap에서 계산한 추정 경로다. 현장 통행 가능 여부·미등록 지름길·신호 대기·경사는 추가 확인 대상이다. 그림은 경로 차이를 보여주는 사례이며, 경로당 이용자의 이동을 관측한 결과가 아니다.</p></section>'
    page+='<section><h2>3. 가까운 수거지도 달라진다</h2><p>연결 가능한 같은 출발점과 같은 기존 수거망을 놓고 비교해도, 직선거리로 가장 가까운 수거지와 도로로 가장 가까운 수거지가 달라지는 사례가 나타났다.</p>'
    page+=table(['시설 출발점','거리 비교 가능','직선 기준 중앙값','도로 기준 중앙값','가장 가까운 수거지 변경'],[
      ['경로당',f"{s['reachable_existing']} / 341개 위치",f"{s['straight_nearest_same_sample']['median']:,.0f} m",f"{s['walking']['median']:,.0f} m",f"{s['different_closest_site']}개 위치"],
      ['편의점',f"{c['reachable_existing']} / 138개 위치",f"{c['straight_nearest_same_sample']['median']:,.0f} m",f"{c['walking']['median']:,.0f} m",f"{c['different_closest_site']}개 위치"]])
    page+='<p>직선거리에서는 기존 수거지 350m 안에 있지만 도로거리에서는 350m를 넘는 위치는 경로당 31곳, 편의점 16곳이었다. 기존 350m 기준을 그대로 쓰더라도 거리 계산 방식이 결과에 영향을 준다.</p>'+imgtag('03_시설출발점_거리분포')+'<div class="callout">위 수치는 <b>시설 위치를 출발점으로 계산한 거리</b>다. 경로당 1곳을 고령인구 1명처럼 취급하지 않았으며, 주민 커버율·고령층 평균 이동거리·편의점 설치 효과는 아직 계산하지 않았다.</div></section>'
    page+='<section><h2>4. 도로망은 이렇게 만들었다</h2><ul><li>제천시 경계 밖 도로도 포함해 시 경계에서 경로가 잘리는 문제를 줄였다.</li><li>보행 금지·사유 접근 제한·공사 중 도로 등을 제외했다. 보행 방향 제한은 반영하고 자동차의 일방통행은 보행에 그대로 적용하지 않았다.</li><li>도로가 지도에서 교차해도 OSM의 교차점이 공유될 때만 연결했다. 고가도로와 아래 도로를 임의로 이어 붙이지 않았다.</li><li>각 지점을 가장 가까운 도로 구간에 투영해 연결하고 연결거리도 더했다. 80m를 넘는 연결은 보류했다.</li><li>계단은 기본 도로망에 포함하고, 계단을 제외한 거리 계산도 가능하도록 구성했다.</li></ul>'
    page+=table(['연결 점검','결과'],[['기존 수거지 도로 연결','73 / 73곳'],['편의점 도로 연결','134 / 138개 위치'],['경로당 도로 연결','333 / 341개 위치'],['도로 연결 후 기존 수거망까지 경로를 찾지 못한 편의점','3개 위치'],['계산 검증','직선거리 하한·경로 합계·보행 방향·입체교차·최적화 전수 비교 통과']])
    page+='<p class="note">80m는 시설 좌표를 도로에 붙이기 위한 잠정 허용치다. 주민의 이용 가능 반경 350m와는 다른 값이다. 긴 연결이나 끊긴 경로는 도로 누락·좌표 오류·출입 제한 가능성을 함께 확인한다.</p>'
    page+='<details><summary>연결 확인이 필요한 15개 후보 위치</summary>'+table(['시설','구분','도로까지 연결거리','상태'],[[r['name'],'경로당' if r['kind']=='senior_centre' else '편의점',f"{r['offset_m']:,.0f} m",'80m 초과' if not r['matched'] else '기존 수거망까지 경로 없음'] for r in disconnected])+'</details></section>'
    page+='<section><h2>5. 다음 최적화에서 결정할 것</h2><p><b>정해진 수의 거점을 추가할 때 주민들의 보행거리를 가장 많이 줄이는 편의점·경로당 조합</b>을 찾는다. 인구수는 이동거리 감소의 가중치로 사용하고, 고령층은 별도의 수요점과 인구로 평가한다.</p>'
    page+=table(['비교안','선택 가능한 시설','비교할 효과'],[['현재 수거망','기존 수거지 유지','현재의 주민·고령층 이동부담'],['편의점 보완안','편의점 후보','편의점 도입의 추가 효과'],['혼합 보완안','편의점 + 경로당 후보','경로당을 후보에 추가한 편익'],['고령층 우선 혼합안','같은 후보 + 전체 개선 효과 유지 조건','전체 효과와 고령층 개선의 절충']])
    page+='<p>설치 규모는 5·10·15·20곳을 비교하는 설정부터 시작한다. 고령층 우선안은 전체 개선 효과를 90·95·100% 유지하는 조건을 비교한다. 이 숫자들은 분석 시나리오이며, 실제 설치 예산이나 현장 수용 능력으로 확정한 값은 아니다.</p><p>거점 선정 계산 모듈은 작성했고 작은 검증 문제에서 가능한 배치를 전부 나열한 답과 일치함을 확인했다. 실제 제천시 주민을 넣은 최적화는 인구 자료를 확보한 뒤 실행한다. 지역 유형을 구분하는 AI 분석도 인구·접근성 변수를 연결한 뒤 검토한다.</p></section>'
    page+='<section><h2>6. 남은 자료와 확인 사항</h2><p><b>SGIS 인구 자료 확보가 다음 단계다.</b> 100m 총인구 격자와 경계, 집계구 연령별 인구와 경계, 읍면동 경계가 필요하다. 65세 이상 인구를 100m 격자에 임의 배분하지 않고, 제공된 집계구 단위를 유지한다.</p><p>기존 수거망은 2025년 안내를 바탕으로 구성했으며, 73곳의 위치는 2026년 심평원 기관 자료와 연결했다. 아래 세 곳은 두 자료의 주소가 달라 확인 대상으로 남겼다. 우리메디칼약국은 심평원 명부의 우리메디컬약국과 도로명주소가 일치함을 확인해 연결했다.</p>'
    page+=table(['시설','시청 명부 주소','심평원 좌표의 주소'],[[r['name'],r['address'],r['coordinate_source_address']] for r in conflict])
    page+='<p>편의점·경로당의 참여 의사와 이용 대상은 확인 전이다. 특히 경로당은 모든 주민이 자유롭게 드나드는 시설로 가정하지 않고, 참여 가능한 시설 범위를 바꿔 결과를 비교할 예정이다.</p></section>'
    page+='<section><h2>자료 출처</h2><div class="table"><table><thead><tr><th>자료</th><th>제공처</th><th>기준 시점</th><th>사용 내용</th></tr></thead><tbody>'+source_rows+'</tbody></table></div><p class="note">전국폐의약품수거함표준데이터도 조회했으나 제천시 등록분은 0건이었다. 수거지 부재로 해석하지 않고 제천시 보건소의 별도 안내와 기관 명부로 기존망을 구성했다.</p></section><footer>제천시 폐의약품 수거망 보행도로 연결 1차 분석 · 2026.10.22<br>© OpenStreetMap contributors · 지도자료 ODbL 1.0 · 인구를 사용한 정책 효과 분석은 후속 단계</footer></main></html>'
    target=OUT/'제천_보행도로연결_1차분석.html';target.write_text(page,encoding='utf-8')
    manifest={'created_on':'2026-10-22','datasets':[dict(name=n,provider=org,reference_date=date,url=u,usage=note) for n,org,date,u,note in sources],
      'raw_files':[dict(file=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in (ROOT/'data/raw').glob('*') if p.is_file() and p.suffix in ['.csv','.json','.zip']],
      'environment':{k:importlib.metadata.version(k) for k in ['numpy','scipy','pandas','networkx','pyproj','shapely','matplotlib']},
      'population_optimization_executed':False}
    (ROOT/'data/source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print(target.resolve())

if __name__=='__main__':main()
