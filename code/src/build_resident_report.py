"""Standalone figures and report for the completed Jecheon resident analysis."""
import base64
import csv
import html
import json
import pickle
from pathlib import Path

import geopandas as gpd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.font_manager import FontProperties

from candidate_policy import is_screened_candidate
from walking_network import TO_XY
from prepare_population import write_json
from run_population_optimization import metrics

ROOT=Path(__file__).resolve().parents[1]
F=ROOT/'results/residents_connector80'
OUT=ROOT/'results/resident_analysis'; FIG=OUT/'figures'
FIG.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':FontProperties(fname='/System/Library/Fonts/Supplemental/AppleGothic.ttf').get_name(),
    'axes.unicode_minus':False,'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
TEAL='#138f83'; ORANGE='#df8b36'; BLUE='#254d66'; PURPLE='#8267aa'; RED='#d85d55'
LABELS={'convenience_efficiency':'편의점안','mixed_efficiency':'혼합 효율안','mixed_elderly_priority':'고령층 우선 혼합안'}
COLORS=[ORANGE,TEAL,PURPLE]

def read(p): return json.loads(p.read_text())
def esc(x): return html.escape(str(x))
def key(p):return p['candidate_group']+'_'+p['mode']
def coords(rows): return np.array(TO_XY.transform([r['lon'] for r in rows],[r['lat'] for r in rows])).T
def savefig(fig,name):
    fig.savefig(FIG/(name+'.png'),dpi=175,bbox_inches='tight',facecolor='white');plt.close(fig)
def imgtag(name):
    b=base64.b64encode((FIG/(name+'.png')).read_bytes()).decode()
    return f'<figure><img src="data:image/png;base64,{b}" alt="{esc(name)}"></figure>'
def table(headers,rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+esc(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def fmt(x):return f'{x:,.0f}'


def main():
    r=read(F/'optimization_medoid_screened.json'); ext=read(F/'extension_results.json')
    audit=read(ROOT/'results/population_input_audit.json'); conn=read(F/'connection_audit.json')
    route_audit=read(F/'route_audit.json'); examples=read(F/'resident_example_routes.geojson')['features']
    origins=read(F/'origins.json');candidates=read(F/'candidates.json');existing=read(F/'existing.json')
    distance=np.load(F/'distances.npz')['walking'];nex=len(existing)
    candidate_lookup={x['id']:i for i,x in enumerate(candidates)}
    plans=[p for p in r['results'] if p['max_sites']==10]
    cv,mix,fair=plans
    baseline=distance[:,:nex].min(axis=1)
    population=np.array([x.get('population',0) or 0 if x['group']=='grid100m' else 0 for x in origins])
    valid=(population>0)&np.isfinite(baseline)
    points=coords(origins);exxy=coords(existing)
    admin=gpd.read_file(ROOT/'data/processed/population/population.gpkg',layer='admin')
    xmin,ymin,xmax,ymax=admin.total_bounds
    bounds=[xmin-800,ymin-800,xmax+800,ymax+800]
    def chosen(plan):return np.array([candidate_lookup[x['id']] for x in plan['selected']])
    def after(plan):return np.minimum(baseline,distance[:,nex+chosen(plan)].min(axis=1))
    def mapbase(ax,limit=bounds):
        admin.plot(ax=ax,facecolor='#f5f8fa',edgecolor='#c5d2d8',linewidth=.65,zorder=0)
        ax.set_xlim(limit[0],limit[2]);ax.set_ylim(limit[1],limit[3]);ax.set_aspect('equal');ax.axis('off')
        ax.text(.96,.96,'N ↑',transform=ax.transAxes,ha='right',fontsize=10)
        width=limit[2]-limit[0];length=5000 if width>15000 else 1000
        xx=limit[0]+width*.07;yy=limit[1]+(limit[3]-limit[1])*.045
        ax.plot([xx,xx+length],[yy,yy],color=BLUE,lw=3);ax.text(xx+length/2,yy+length*.08,f'{length//1000} km',ha='center',fontsize=8)

    # Paired maps keep the same colour scale and the same resident sample.
    fig,axs=plt.subplots(1,2,figsize=(12,8.6),layout='constrained')
    for ax,values,title in zip(axs,[baseline,after(mix)],['현재 수거망','10곳 추가: 편의점 5곳 + 경로당 5곳']):
        mapbase(ax)
        dots=ax.scatter(points[valid,0],points[valid,1],c=values[valid],cmap='YlOrRd',norm=Normalize(0,5000),
            s=np.clip(np.sqrt(population[valid])*1.3,4,22),edgecolors='none',alpha=.85,zorder=2)
        ax.scatter(exxy[:,0],exxy[:,1],s=12,marker='s',color=BLUE,zorder=3,label='기존 수거지')
        if ax is axs[1]:
            sxy=coords(mix['selected']);ax.scatter(sxy[:,0],sxy[:,1],marker='*',s=110,c=TEAL,edgecolors='white',linewidths=.6,zorder=5,label='추가 거점')
        ax.set_title(title,fontsize=15,pad=12)
    cb=fig.colorbar(dots,ax=axs,shrink=.72,pad=.025);cb.set_label('수거지까지 보행거리 (m)');cb.set_ticks([0,1000,2000,3000,4000,5000]);cb.set_ticklabels(['0','1,000','2,000','3,000','4,000','5,000 이상'])
    axs[1].legend(loc='lower right',fontsize=9)
    fig.suptitle('같은 주민 위치에서 비교한 수거 접근성',fontsize=20)
    savefig(fig,'01_현재와_10곳추가_거리지도')

    # Do not visually equate older residents with the 100m-grid population sample.
    labels=['현재 수거망']+[LABELS[key(p)] for p in plans]
    fig,axs=plt.subplots(1,3,figsize=(15,5.5),layout='constrained')
    measures=[('전체 평균 보행거리 (m)','residents','mean_m'),('고령층 평균 보행거리 (m)','elderly_lower','mean_m'),('350m 이내 커버율 (%)','residents','coverage')]
    for ax,(title,group,measure) in zip(axs,measures):
        rows=[r['baseline'][group]]+[p[group] for p in plans]
        values=[x['within_distance_percent']['350'] if measure=='coverage' else x[measure] for x in rows]
        ax.barh(np.arange(4),values,color=[BLUE]+COLORS,height=.58)
        ax.invert_yaxis();ax.set_yticks(np.arange(4),labels if ax is axs[0] else ['']*4);ax.set_title(title,pad=15,fontsize=14)
        ax.set_xlim(0,max(values)*1.23);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
        for i,value in enumerate(values):ax.text(value+max(values)*.02,i,f'{value:,.1f}' if measure=='coverage' else f'{value:,.0f}',va='center',fontsize=12)
    fig.suptitle('추가 거점 10곳: 무엇을 우선하느냐에 따라 달라지는 결과',fontsize=19)
    savefig(fig,'02_10곳_대안별_효과비교')

    fig,axs=plt.subplots(1,2,figsize=(13,5.5),layout='constrained')
    for ax,metric,title in zip(axs,['residents','elderly_lower'],['전체 평균 보행거리','고령층 평균 보행거리']):
        for label,color in zip(LABELS,COLORS):
            group=sorted([p for p in r['results'] if key(p)==label],key=lambda p:p['max_sites'])
            x=[0]+[p['max_sites'] for p in group];y=[r['baseline'][metric]['mean_m']]+[p[metric]['mean_m'] for p in group]
            ax.plot(x,y,'o-',color=color,label=LABELS[label],lw=2.5,markersize=6)
        ax.set_xticks([0,5,10,15,20]);ax.set_xlabel('추가 거점 수 (곳)');ax.set_ylabel('인구 가중 평균 보행거리 (m)');ax.grid(alpha=.2);ax.set_title(title,fontsize=15)
    axs[1].legend(fontsize=9);fig.suptitle('설치 규모별 효과 — 각 규모에서 별도로 선정한 결과',fontsize=19)
    savefig(fig,'03_추가규모별_효과')

    fig,axs=plt.subplots(1,3,figsize=(15,8),layout='constrained')
    for ax,plan in zip(axs,plans):
        mapbase(ax);ax.scatter(points[valid,0],points[valid,1],s=3,color='#bfcbd0',alpha=.45,zorder=1,label='인구 격자')
        ax.scatter(exxy[:,0],exxy[:,1],s=10,marker='s',color=BLUE,alpha=.5,zorder=2,label='기존 수거지')
        for kind,color,marker,label in [('convenience',ORANGE,'o','편의점'),('senior_centre',TEAL,'^','경로당')]:
            selected=[s for s in plan['selected'] if s['kind']==kind]
            if selected:
                a=coords(selected);ax.scatter(a[:,0],a[:,1],c=color,marker=marker,s=90,edgecolors='white',linewidths=.6,zorder=4,label=f'{label} {len(selected)}곳')
        ax.set_title(LABELS[key(plan)],fontsize=15,pad=12);ax.legend(loc='lower right',fontsize=9)
    fig.suptitle('10곳을 고른다면: 편의점과 경로당의 역할',fontsize=20)
    savefig(fig,'04_10곳_선정위치')

    # A single resident grid illustrates how the chosen site changes a route.
    with (ROOT/'results/facility_network.pkl').open('rb') as f:net=pickle.load(f)
    segments=np.array([np.asarray(s[2].coords) for s in net.segments])
    paths=[]
    for feature in examples:
        ll=np.array(feature['geometry']['coordinates']);paths.append(np.array(TO_XY.transform(ll[:,0],ll[:,1])).T)
    total=np.vstack(paths);lo=total.min(axis=0)-220;hi=total.max(axis=0)+220
    fig,ax=plt.subplots(figsize=(11,7.5));ax.set_aspect('equal');ax.axis('off')
    mask=(segments[:,:,0].max(axis=1)>=lo[0])&(segments[:,:,0].min(axis=1)<=hi[0])&(segments[:,:,1].max(axis=1)>=lo[1])&(segments[:,:,1].min(axis=1)<=hi[1])
    ax.add_collection(LineCollection(segments[mask],colors='#d9e3e8',linewidths=.8,zorder=1))
    for path,feature,color,label in zip(paths,examples,[RED,TEAL],['현재','추가 후']):
        ax.plot(path[:,0],path[:,1],color=color,lw=3,zorder=3,label=f"{label} {feature['properties']['length_m']:,.0f}m")
        ax.scatter(path[-1,0],path[-1,1],marker='s',s=95,color=color,zorder=5)
        ax.annotate(feature['properties']['destination_name'],path[-1],xytext=(8,10),textcoords='offset points',fontsize=11,bbox={'facecolor':'white','edgecolor':'none','alpha':.85},zorder=6)
    ax.scatter(paths[0][0,0],paths[0][0,1],s=110,c=BLUE,edgecolors='white',zorder=5)
    ax.annotate('인구 격자 중심점',paths[0][0],xytext=(8,-20),textcoords='offset points',bbox={'facecolor':'white','edgecolor':'none','alpha':.85},zorder=6)
    ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1]);ax.legend(loc='upper right',fontsize=12)
    ax.set_title('한 주민 격자에서 수거지까지 가는 길',fontsize=19,pad=18)
    ax.text(.98,.03,'OSM 보행도로 최단경로 추정',transform=ax.transAxes,ha='right',fontsize=9,color='#526975')
    savefig(fig,'05_주민격자_경로변화')

    areas=sorted(mix['administrative_results'],key=lambda a:a['before']['mean_m']-a['after']['mean_m'])
    other={a['name']:a for a in cv['administrative_results']}
    fig,ax=plt.subplots(figsize=(11,8));y=np.arange(len(areas))
    for offset,plan,color,label in [(-.18,cv,ORANGE,'편의점 10곳'),(.18,mix,TEAL,'혼합 10곳')]:
        by_name={a['name']:a for a in plan['administrative_results']}
        gain=[by_name[a['name']]['before']['mean_m']-by_name[a['name']]['after']['mean_m'] for a in areas]
        ax.barh(y+offset,gain,height=.34,color=color,label=label)
    ax.set_yticks(y,[a['name'] for a in areas]);ax.set_xlabel('현재 대비 평균 보행거리 감소 (m)');ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True);ax.legend(loc='lower right');ax.set_title('어느 지역의 이동거리를 줄이는가',fontsize=19,pad=17)
    fig.tight_layout();savefig(fig,'06_읍면동별_개선효과')

    # Upper bound with every eligible, connected candidate; not an installation proposal.
    eligible=np.array([is_screened_candidate(c) for c in candidates])
    potential=np.minimum(baseline,distance[:,nex:][:,eligible].min(axis=1))
    bound=metrics(potential[valid],population[valid])
    selected_export=[]
    for plan in r['results']:
        for site in plan['selected']:
            selected_export.append(dict(추가규모=plan['max_sites'],대안=LABELS[key(plan)],시설명=site['name'],구분='경로당' if site['kind']=='senior_centre' else '편의점',읍면동=site['dong'],주소=site['address'],경도=site['lon'],위도=site['lat'],후보ID=site['id']))
    with (OUT/'선정후보_규모별_대안별.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=selected_export[0].keys());w.writeheader();w.writerows(selected_export)
    checks=dict(population_optimization_executed=True,base_models=len(r['results']),all_base_optimal=all(p['solver']['optimal'] for p in r['results']),
        all_base_gap_zero=all(p['solver']['mip_gap']==0 for p in r['results']),
        same_population_all_plans=all(p['residents']['population']==r['baseline']['residents']['population'] for p in r['results']),
        existing_network_preserved=all(p['residents']['mean_m']<=r['baseline']['residents']['mean_m'] for p in r['results']),
        example_routes_match_matrix=route_audit['example_paths_match_distance_matrix'])
    assert all(v for k,v in checks.items() if isinstance(v,bool))
    write_json(OUT/'validation.json',checks)
    write_json(OUT/'all_candidate_upper_bound.json',bound)

    css='''*{box-sizing:border-box}body{margin:0;background:#f0f5f6;color:#173847;font-family:-apple-system,BlinkMacSystemFont,'Apple SD Gothic Neo','Malgun Gothic',sans-serif;line-height:1.8}main{max-width:1150px;margin:auto;background:white;padding:48px 62px}h1{font-size:38px;line-height:1.35;letter-spacing:-1px;margin:10px 0 22px}h2{font-size:26px;line-height:1.45;margin:0 0 18px}h3{font-size:19px}p{margin:12px 0 18px}section{padding:38px 0;border-top:1px solid #d9e4e8}.eyebrow{font-size:13px;letter-spacing:1px;color:#158f83;font-weight:700}.lead{font-size:19px}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:15px;margin:28px 0}.card{padding:20px;border:1px solid #dae5e9;border-radius:9px}.card b{font-size:29px;display:block}.card span{font-size:13px;color:#526b77}.note{font-size:14px;color:#526b77}.callout{background:#eaf6f2;border-left:4px solid #138f83;padding:20px 25px;margin:22px 0}.warning{background:#fff7e8;border-left-color:#df8b36}.flow{background:#173847;color:white;padding:23px;border-radius:9px;font-size:18px;text-align:center}.formula{background:#f3f6f8;padding:20px 24px;font-size:18px}figure{margin:24px 0}figure img{width:100%;display:block}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px;margin:18px 0}td,th{padding:11px 12px;border-bottom:1px solid #dce6e9;vertical-align:top;text-align:left}th{background:#eef4f6}details{border:1px solid #dce6e9;border-radius:8px;margin:16px 0;padding:14px 18px}summary{font-weight:600;cursor:pointer}a{color:#087f76}nav{display:flex;gap:17px;flex-wrap:wrap;margin:22px 0;font-size:14px}.badge{display:inline-block;background:#edf3f5;padding:4px 11px;border-radius:14px;font-size:12px}li{margin:8px 0}footer{color:#607783;font-size:13px;padding-top:20px}@media(max-width:750px){main{padding:28px 20px}h1{font-size:28px}.cards{grid-template-columns:1fr}.lead{font-size:17px}}@media print{body{background:white}main{padding:0}nav{display:none}figure{break-inside:avoid}}'''
    page=f'<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>제천시 폐의약품 수거망 · 주민 보행거리와 보완 입지 분석</title><style>{css}</style><main>'
    page+='<header><div class="eyebrow">JECHEON · RESIDENT ACCESSIBILITY · 2026.10.23</div><h1>도심의 편의점과 읍·면의 경로당,<br>주민의 배출 부담을 얼마나 줄일까</h1><p class="lead">제천시 기존 폐의약품 수거망을 유지하면서, 주민의 거주 분포와 보행도로를 연결해 추가 거점을 골랐다. 직선거리로 고른 입지와 도로거리로 고른 입지를 비교하고, 고령층을 더 배려하는 대안까지 계산했다.</p><span class="badge">2024 인구 · 2025 경계 · 2026 도로·점포 자료</span></header>'
    page+='<nav><a href="#answer">핵심 결과</a><a href="#method">계산 방법</a><a href="#locations">선정 위치</a><a href="#age">고령층</a><a href="#robustness">검증</a><a href="#data">자료·출처</a></nav>'
    page+=f'<div class="cards"><div class="card"><b>{nex}곳</b><span>기존 수거지 유지</span></div><div class="card"><b>{fmt(r["baseline"]["residents"]["population"])}명</b><span>도로 비교에 포함된 공개 격자인구 가중치</span></div><div class="card"><b>{len(r["results"])}개</b><span>추가 규모·선정 목표별 기본 결과</span></div></div>'
    page+='<section id="answer"><h2>1. 평균 이동거리와 가까운 배출 기회는 서로 다른 목표다</h2><p>거점 10곳을 추가하는 예시에서 편의점만 활용한 안과 편의점·경로당을 함께 활용한 안을 비교했다. 두 안 모두 기존 수거망에 비해 이동거리를 줄였다.</p>'
    page+=table(['10곳 추가 대안','편의점 / 경로당','전체 평균거리','고령층 평균거리','350m 커버율'],[
        ['현재 수거망','추가 없음',f'{r["baseline"]["residents"]["mean_m"]:,.0f}m',f'{r["baseline"]["elderly_lower"]["mean_m"]:,.0f}m',f'{r["baseline"]["residents"]["within_distance_percent"]["350"]:.1f}%']]+
        [[LABELS[key(p)],f'{p["selected_kinds"].get("convenience",0)} / {p["selected_kinds"].get("senior_centre",0)}',f'{p["residents"]["mean_m"]:,.0f}m',f'{p["elderly_lower"]["mean_m"]:,.0f}m',f'{p["residents"]["within_distance_percent"]["350"]:.1f}%'] for p in plans])
    page+='<div class="callout"><b>편의점안은 350m 안의 배출 기회를 넓히는 데, 혼합 효율안은 긴 이동거리를 줄이는 데 상대적으로 유리했다.</b><br>이 비교에서 혼합안의 전체 평균거리는 더 짧지만, 350m 커버율은 편의점안이 더 높다. 정책이 무엇을 우선하는지에 따라 선택할 안이 달라진다.</div>'
    page+=imgtag('02_10곳_대안별_효과비교')+'<p class="note">전체 인구는 100m 격자, 고령층은 집계구 대표점으로 평가했다. 각 열 안에서 대안의 개선을 비교하며 두 열의 절대값 차이를 연령별 실제 이동 차이로 해석하지 않는다. 10곳은 비교용 시나리오다.</p></section>'
    page+='<section id="method"><h2>2. 이번에는 실제로 어떻게 계산했나</h2><div class="flow">인구 격자 중심 → 가까운 보행도로 → 도로 최단경로 → 기존 수거지 또는 추가 거점</div><p>OSM에서 보행 금지·사유 접근 제한 도로를 걸러내고, 도로 구간에 주민과 시설을 연결했다. 연결 허용 거리는 80m이며, 시설·인구점과 도로 사이의 거리도 합산했다. 자동차 일방통행과 보행 방향 제한을 구분하고, 고가도로 교차를 임의로 연결하지 않았다.</p><p>시설별 임의 점수를 합산하는 대신, <b>각 주민 위치의 거리 감소에 공개 인구를 곱해 합한 값</b>이 가장 커지도록 거점을 선정했다. 여러 거점이 같은 주민에게 가까워도 최종적으로 가장 가까운 한 곳까지의 거리만 계산한다.</p><div class="formula">선정 목표: Σ (공개 인구 × 가장 가까운 수거지까지의 보행거리) 최소화<br>조건: 기존 수거지 유지 · 추가 거점 수 ≤ 5·10·15·20곳</div>'
    page+=imgtag('01_현재와_10곳추가_거리지도')
    route=route_audit['example']
    page+=f'<h3>한 격자에서 보면 더 직관적이다</h3><p>{esc(route["adm_name"])}의 한 인구 격자에서는 기존 <b>{esc(route["before_name"])}</b>까지 약 {route["before_m"]:,.0f}m를 이동해야 했다. 혼합안의 <b>{esc(route["after_name"])}</b>을 활용하면 약 {route["after_m"]:,.0f}m로 줄어든다.</p>'+imgtag('05_주민격자_경로변화')+'<p class="note">관측한 주민 동선이 아닌 도로 최단경로의 추정이다. 보행 신호·경사·미등록 지름길·시설의 실제 출입구는 포함하지 않았다.</p></section>'
    page+='<section id="locations"><h2>3. 어떤 거점을 고르며, 어느 지역이 달라지나</h2><p>리조트·휴게소·학사 등 출입 확인이 필요한 곳과 좌표 또는 출입구가 불분명한 후보 14개를 기본안에서 보류했다. 남은 후보에서도 도로 경로가 확인되는 위치를 사용했다. 경로당은 일반 주민에게 개방되는 수거 접점을 마련하고 협약을 맺는 도입 시나리오다.</p>'+imgtag('04_10곳_선정위치')
    page+='<p>혼합 효율안의 경로당 후보는 봉양읍 명암리, 수산면 하천리, 백운면 운학1리, 송학면 입석3리, 용두동 그린코아루A다. 편의점이 부족한 생활권을 보완하는 역할과 도심 주거지의 접근성을 높이는 역할이 함께 나타났다.</p>'
    for plan in plans:
        page+=f'<details><summary>{LABELS[key(plan)]} — 10곳 후보 명단</summary>'+table(['구분','시설명','읍면동','도로명주소'],[['경로당' if s['kind']=='senior_centre' else '편의점',s['name'],s['dong'],s['address']] for s in plan['selected']])+'</details>'
    page+=imgtag('06_읍면동별_개선효과')+'<p>용두동·송학면·수산면 등에서 평균 이동거리 감소가 컸다. 청풍면·한수면 등은 이번 10곳으로 거리가 줄지 않아 별도 보완 대상을 검토할 필요가 있다. 현재 모형은 전체 거리 감소를 우선하므로 지역별 최소 개선량을 보장하지 않는다.</p></section>'
    page+='<section><h2>4. 몇 곳을 추가할 때 효과가 커지나</h2>'+imgtag('03_추가규모별_효과')
    page+=table(['추가 수','편의점안 전체 평균','혼합 효율안 전체 평균','고령층 우선안 전체 / 고령층 평균'],[
        [str(k)+'곳',f'{next(p for p in r["results"] if p["max_sites"]==k and key(p)=="convenience_efficiency")["residents"]["mean_m"]:,.0f}m',f'{next(p for p in r["results"] if p["max_sites"]==k and key(p)=="mixed_efficiency")["residents"]["mean_m"]:,.0f}m',
         ' / '.join(f'{next(p for p in r["results"] if p["max_sites"]==k and key(p)=="mixed_elderly_priority")[m]["mean_m"]:,.0f}m' for m in ['residents','elderly_lower'])] for k in [5,10,15,20]])
    page+='<p>처음 5곳의 효과가 크고, 이후 거점을 늘릴수록 추가 개선폭은 줄어든다. 각 설치 규모에서 따로 최적화했으므로 5곳 명단이 10곳 명단에 모두 포함되는 단계별 설치 순서는 아니다. 실제 도입 규모는 시설별 설치·회수 비용을 연결한 뒤 정한다.</p></section>'
    page+='<section id="age"><h2>5. 고령층 변수는 선정에 직접 사용했다</h2><p>집계구별 65세 이상 인구를 가중치로 사용했다. 먼저 전체 인구의 거리 감소를 최대화한 뒤, 그 개선 효과의 <b>95% 이상을 유지</b>하면서 고령층의 거리를 최대한 줄이는 조합을 계산했다. 95%는 전체 효과와 고령층 편익의 절충을 살펴보기 위한 정책 시나리오다.</p>'
    poploss=fair['residents']['mean_m']-mix['residents']['mean_m'];eldergain=mix['elderly_lower']['mean_m']-fair['elderly_lower']['mean_m']
    page+=f'<div class="callout">10곳 기준으로 혼합 효율안에 비해 전체 평균거리가 약 <b>{poploss:.0f}m</b> 늘어나는 대신, 고령층 평균거리를 약 <b>{eldergain:.0f}m</b> 더 줄였다. 이때 조합은 편의점 5곳·경로당 5곳에서 편의점 2곳·경로당 8곳으로 바뀌었다.</div>'
    page+='<p>넓은 집계구의 대표점이 산지나 비주거지에 놓이는 문제를 줄이기 위해, 집계구 안의 공개 인구 격자 중 인구 가중 거리 합이 가장 작은 중심점을 대표점으로 삼았다. 고령인구를 100m 격자에 나눠 배치한 것은 아니다.</p>'
    page+=f'<p>같은 고령인구를 도형 내부점에 두면 {conn["groups"]["elderly_interior"]["comparable_weight"]:,.0f}명, 거주 분포 대표점에 두면 {conn["groups"]["elderly_medoid"]["comparable_weight"]:,.0f}명의 하한 가중치를 도로망에 연결할 수 있었다. 두 방식에서 모두 연결되는 집계구만 비교한 결과도 아래에 함께 제시했다.</p>'
    page+='<details><summary>대표점과 연령 비공개 값에 따른 결과</summary>'+table(['대표점','10곳 대안','동일 표본 고령인구 하한','현재 평균','추가 후 평균'],[
        ['도형 내부점' if a['point_method']=='interior' else '거주 분포 대표점',LABELS[a['candidate_group']+'_'+a['mode']],fmt(a['before']['population']),f'{a["before"]["mean_m"]:,.0f}m',f'{a["after"]["mean_m"]:,.0f}m'] for a in ext['representative_points']])
    page+='<p>대표점에 따라 대안 간 차이도 달라진다. 고령층 우선안은 두 대표점 방식에서 모두 혼합 효율안보다 고령층 거리를 줄였지만, 편의점안과의 우열은 표본·대표점에 영향을 받았다. 고령층 수치는 후속 현장 검증이 필요한 집계구 단위 추정이다.</p>'
    page+=table(['10곳 대안','고령인구 하한 가중 평균','상한 가중 평균'],[[LABELS[key(p)],f'{p["elderly_lower"]["mean_m"]:,.0f}m',f'{p["elderly_upper"]["mean_m"]:,.0f}m'] for p in plans])+'<p class="note">미공개 연령구간을 0~4명으로 두고 총인구와 나머지 연령을 이용해 상하한을 좁혔다. 주 선정은 하한 가중치, 상한 결과는 같은 선정안을 재평가한 값이다.</p></details></section>'
    page+='<section id="robustness"><h2>6. 직선거리·계단·도로 연결 조건을 바꿔 확인했다</h2>'+table(['후보 범위','직선·도로 선정 공통 거점','직선으로 고른 안의 도로 평균','도로로 고른 안의 도로 평균'],[
        ['편의점' if a['candidate_group']=='convenience' else '편의점+경로당',f'{a["common_selected"]} / 10곳',f'{a["actual_road_evaluation"]["after"]["mean_m"]:,.0f}m',f'{a["road_optimum_mean_m"]:,.0f}m'] for a in ext['straight_model_comparison']])
    page+='<p>혼합안은 도로거리로 바꾸면서 10곳 중 5곳이 달라졌고, 도로거리 기준 평균 이동부담이 약 20m 더 줄었다. 거점 구성의 변화에 비해 전체 평균의 추가 개선은 작았다. 도로망을 연결하는 가치는 우회와 연결 단절을 반영해 후보별 효과를 더 구체적으로 판단할 수 있다는 데 있다.</p>'
    page+='<details><summary>추가 검증 결과 보기</summary>'+table(['검증','확인 결과'],[
        ['계단 제외','기존망 경로를 잃는 공개 격자인구 가중치 28명. 같은 표본에서 10곳 대안의 평균거리는 각각 약 1.5m 증가.'],
        ['연결 허용치 80→120m',f'추가로 {ext["connector120"][0]["added_population"]:,.0f}명의 격자인구 가중치를 연결. 기존 공통 표본의 거리는 동일. 확장 표본에서 대안 순위 유지.'],
        ['500m 격자로 재평가','공개 통계와 대표점·도로 연결 표본이 함께 변함. 고정된 10곳 선정안을 별도 표본에서 평가.'],
        ['출입 확인 대상 포함','별도 전체 후보 계산을 보관. 기본안은 출입·좌표 확인 대상 14개 보류.'],
        ['계산 검증',f'기본 {checks["base_models"]}개 결과 모두 최적해 인증·최적성 격차 0. 선정 후 최단거리로 독립 재집계.'],
        ['지도 경계',f'이동부담·최장거리 상위 경로 표본 {len(route_audit["boundary_checked_paths"])}개 점검. 경계까지 최소 {route_audit["min_boundary_margin_m"]:,.0f}m. 전체 경로의 경계 독립성을 증명한 것은 아님.']])
    page+=table(['500m 격자 재평가','현재 평균','10곳 추가 후'],[[LABELS[a['candidate_group']+'_'+a['mode']],f'{a["before"]["mean_m"]:,.0f}m',f'{a["after"]["mean_m"]:,.0f}m'] for a in ext['grid500m']])+'</details></section>'
    page+='<section id="data"><h2>7. 분석 범위와 해석에 필요한 자료 정보</h2>'+table(['자료·범위','수량 또는 기준'],[
        ['제천시 공간 범위','17개 읍면동'],['기존 수거지','2025-08 보건소 안내 기반 73곳 · 좌표는 2026-09 심평원 자료 연결'],
        ['후보 원목록','편의점 138개 위치 · 경로당 341개 위치'],
        ['100m 격자 공개 인구 합계',f'{audit["grids"]["100"]["released_population"]:,.0f}명'],
        ['주 비교 표본',f'{conn["groups"]["grid100m"]["reachable_points"]:,}개 격자 중심 · 가중치 {conn["groups"]["grid100m"]["comparable_weight"]:,.0f}명'],
        ['도로 연결 미확보 가중치',f'{conn["groups"]["grid100m"]["unmatched_weight"]+conn["groups"]["grid100m"]["attached_but_unreachable_weight"]:,.0f}명 · 별도 집계'],
        ['집계구 총인구',f'{audit["elderly"]["census_areas"]}개 집계구 · {audit["elderly"]["total_population"]:,.0f}명'],
        ['65세 이상 공개 자료에서 산출한 범위',f'{audit["elderly"]["lower_sum"]:,.0f}~{audit["elderly"]["upper_sum"]:,.0f}명']])
    page+='<p>SGIS 격자인구에는 통계비밀 보호를 위한 값 조정(iLBA)이 적용된다. 작은 값은 0 또는 5로 대체되고 다른 값에도 조정이 있어, 100m 격자 합계와 집계구 총인구가 일치하지 않는다. 이 보고서의 인구 수치는 <b>공개 통계를 이용한 분석 가중치</b>이며, 주민 명부에서 직접 센 인원이 아니다. 값이 없는 격자를 0명으로 바꾸지 않았다.</p><p>이 결과는 수거 장소까지의 잠재적 보행 접근성을 계산한 것이다. 실제 배출량·이용률·수거 시간대·회수 비용은 측정하지 않았다. 기존 시설 3곳의 명부 간 주소 차이와 후보의 참여 의사·수거함 설치 공간은 도입 전 확인 항목으로 남아 있다.</p>'
    sources=[('SGIS 인구·경계 자료','https://sgis.mods.go.kr/view/pss/openDataIntrcn'),('SGIS 신청자료 다운로드·이용 안내','https://sgis.mods.go.kr/view/pss/downloadList'),('제천시 폐의약품 수거 안내','https://www.jecheon.go.kr/health/selectBbsNttView.do?key=944&bbsNo=170&nttNo=386840'),('제천시 경로당 현황','https://www.data.go.kr/data/15013784/fileData.do'),('소상공인시장진흥공단 상가정보','https://www.data.go.kr/data/15083033/fileData.do'),('심평원 병의원·약국 현황','https://www.data.go.kr/data/15051059/fileData.do'),('OpenStreetMap contributors · ODbL','https://www.openstreetmap.org/copyright')]
    page+='<h3>자료 출처</h3><ul>'+''.join(f'<li><a href="{url}" target="_blank">{esc(label)}</a></li>' for label,url in sources)+'</ul></section>'
    page+='<section><h2>이번 결과에서 이어갈 방향</h2><p><b>편의점의 가까운 생활권 서비스와 경로당의 먼 생활권 보완 효과를 함께 보여주는 방향</b>이 적합하다. 전체 평균거리만으로 우열을 정하기보다 350m 커버율, 먼 지역의 거리 감소, 고령층 효과를 나란히 제시하면 제천에 필요한 이유가 더 분명해진다.</p><p>실행 단계에서는 후보를 현장 확인한 뒤 시설별 참여 여부·설치비·회수비를 넣어 다시 계산한다. 시범사업을 순차 도입하려면 이미 설치한 거점을 고정하는 조건도 추가한다.</p></section><footer>제천시 폐의약품 수거망 · 주민 보행거리와 보완 입지 분석<br>2026.10.23 · 공개 자료를 이용한 정책 시나리오 · © OpenStreetMap contributors</footer></main></html>'
    target=OUT/'제천_주민보행거리와_보완거점_분석.html';target.write_text(page,encoding='utf-8')
    write_json(OUT/'report_summary.json',dict(title='제천시 폐의약품 수거망 주민 보행거리와 보완 입지 분석',
        baseline=r['baseline'],ten_sites=[dict(name=LABELS[key(p)],residents=p['residents'],elderly=p['elderly_lower'],selected_kinds=p['selected_kinds']) for p in plans],
        straight_comparison=ext['straight_model_comparison'],validation=checks,
        note='공개 인구 가중치·도로 최단경로 추정. 실제 이용량 및 현장 참여 미확인.'))
    from report_presentation import refine_report
    target.write_text(refine_report(page, read(OUT/'report_summary.json')),encoding='utf-8')
    print(target.resolve())


if __name__=='__main__':main()
