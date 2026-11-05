"""Readable, self-contained presentation of the calculated resident results."""
from pathlib import Path
from bs4 import BeautifulSoup


def refine_report(page, summary):
    soup = BeautifulSoup(page, 'html.parser')
    soup.style.string = Path(__file__).with_name('resident_report.css').read_text(encoding='utf-8')
    main = soup.main
    baseline = summary['baseline']['residents']['mean_m']
    cv, mixed, older = summary['ten_sites']
    reduction = 100 * (baseline - mixed['residents']['mean_m']) / baseline
    elder_gain = mixed['elderly']['mean_m'] - older['elderly']['mean_m']

    def fragment(markup):
        return BeautifulSoup(markup, 'html.parser')

    soup.header.clear()
    soup.header['class'] = 'hero'
    soup.header.append(fragment(f'''
      <div class="eyebrow">제천시 폐의약품 수거망 · 주민 보행거리 분석</div>
      <h1>폐의약품을 버리러 가는 길,<br>제천에서는 얼마나 줄일 수 있을까?</h1>
      <p class="lead">기존 수거지 73곳을 유지하고, 편의점과 경로당을 추가하는 대안을 비교했습니다.
      인구가 사는 위치와 보행도로를 연결해 <strong>어디에 설치해야 누구의 이동 부담이 줄어드는지</strong> 살펴봅니다.</p>
      <div class="hero-bottom"><span>17개 읍면동 · 추가 거점 5 / 10 / 15 / 20곳</span><span>분석일 2026.10.23</span></div>
    '''))

    old_cards = main.find('div', class_='cards')
    old_cards.clear()
    old_cards['class'] = 'result-cards'
    old_cards.append(fragment(f'''
      <div class="metric"><span class="metric-label">전체 이동거리 개선 · 혼합 10곳</span>
        <strong>{reduction:.1f}<small>% 감소</small></strong><p>{baseline:,.0f}m → {mixed['residents']['mean_m']:,.0f}m</p></div>
      <div class="metric"><span class="metric-label">350m 안의 배출 기회 · 편의점 10곳</span>
        <strong>{cv['residents']['within_distance_percent']['350']:.1f}<small>%</small></strong><p>현재 18.0%에서 개선</p></div>
      <div class="metric"><span class="metric-label">고령층 우선안 · 혼합 효율안 대비</span>
        <strong>{elder_gain:.0f}<small>m 추가 감소</small></strong><p>고령층의 추정 평균 보행거리</p></div>
    '''))
    soup.nav.extract()
    old_cards.insert_after(fragment('''<p class="metric-note">10곳 추가 시나리오의 서로 다른 지표입니다. 거리는 공개 인구를 가중치로 계산한 편도 보행거리이며, 실제 방문 실적은 아닙니다.</p>'''))

    sections = main.find_all('section', recursive=False)
    ids = ['answer','method','locations','scale','age','robustness','data','conclusion']
    short = ['핵심 결과','계산 방법','선정 위치','추가 규모','고령층','검증','자료·출처','해석과 제안']
    titles = [
        '평균거리와 가까운 배출 기회, 어느 쪽을 우선할까?',
        '주민이 사는 곳에서 수거지까지, 도로를 따라 계산합니다',
        '선정 위치를 보면 편의점과 경로당의 역할이 보입니다',
        '거점을 늘렸을 때 추가로 얻는 효과는 얼마나 될까?',
        '고령층을 더 배려하면 선정 결과도 달라집니다',
        '다른 조건에서도 결과가 유지되는지 확인했습니다',
        '자료의 범위와 계산에 사용한 가정',
        '이 분석으로 무엇을 말할 수 있을까?',
    ]
    for i,(section,id_,label,title) in enumerate(zip(sections,ids,short,titles),1):
        section['id'] = id_
        section.h2.string = title
        kicker = soup.new_tag('p', attrs={'class':'section-kicker'})
        kicker.string = f'{i:02d} / {label}'
        section.insert(0,kicker)
    nav = soup.new_tag('nav', attrs={'aria-label':'보고서 목차'})
    for id_,label in zip(ids,short):
        a = soup.new_tag('a',href='#'+id_); a.string=label;nav.append(a)
    sections[0].insert_before(nav)
    nav.insert_before(fragment('''<div class="opening-story">
      <p class="section-kicker">연구의 출발점</p><h2>생활권이 다르면, 필요한 보완 거점도 달라질 수 있습니다.</h2>
      <p>수거지가 여러 곳에 있어도 주민이 사는 곳과 멀리 떨어져 있다면 배출을 위한 이동이 남습니다.
      이번 분석에서는 기존 수거망의 공백을 먼저 살피고, 편의점만 추가하는 방법과 경로당을 함께 활용하는 방법을 비교했습니다.</p>
      <div class="questions"><div><span>어디가 먼가</span><p>기존 수거지까지의 보행거리</p></div>
      <div><span>어디를 더할까</span><p>주민의 거리 감소를 고려한 입지</p></div>
      <div><span>누가 혜택을 받나</span><p>읍면동별·고령층별 개선 효과</p></div></div>
    </div>'''))

    captions = {
      '02_10곳_대안별_효과비교':('대안별 접근성 비교','편의점안의 350m 커버율이 가장 높고, 혼합 효율안의 전체 평균거리가 가장 짧습니다.'),
      '01_현재와_10곳추가_거리지도':('현재 수거망과 혼합 10곳 추가 후의 거리','같은 인구 격자와 같은 색 범위를 사용했습니다. 붉을수록 수거지까지 멉니다.'),
      '05_주민격자_경로변화':('용두동의 한 인구 격자에서 본 경로 변화','인구 격자 중심에서 기존 수거지와 추가 거점까지의 도로 최단경로를 비교했습니다.'),
      '04_10곳_선정위치':('대안별 10곳의 선정 위치','주황색 원은 편의점, 초록색 삼각형은 경로당입니다. 시설 유형의 수는 최적화 결과로 결정됩니다.'),
      '06_읍면동별_개선효과':('읍면동별 평균 보행거리 감소','지역별 개선폭은 다릅니다. 10곳 추가 후에도 변화가 없는 지역이 남습니다.'),
      '03_추가규모별_효과':('5·10·15·20곳 추가에 따른 효과','각 규모에서 별도로 최적화한 결과이며, 앞서 고른 거점을 고정한 순차 설치 계획은 아닙니다.'),
    }
    for i,figure in enumerate(main.find_all('figure'),1):
        name = figure.img.get('alt','')
        title,detail = captions[name]
        figure.img['alt']=title
        figure.img['loading']='lazy'
        cap=soup.new_tag('figcaption')
        cap.append(fragment(f'<b>그림 {i}. {title}</b><span>{detail}</span><small>자료: 본 연구 분석. 지도 경계 SGIS · 보행도로 OpenStreetMap.</small>'))
        figure.append(cap)
        if name=='05_주민격자_경로변화':figure['class']='route-figure'

    # Make the difference between model improvement and field effectiveness explicit.
    sections[-1].clear()
    sections[-1].append(fragment(f'''
      <p class="section-kicker">08 / 해석과 제안</p><h2>입지 대안을 비교하는 근거는 생겼습니다.</h2>
      <p class="conclusion-lead">이번 결과의 핵심은 <strong>제천에서 어느 주민의 이동 부담을 먼저 줄일지 선택할 수 있다는 점</strong>입니다.</p>
      <div class="interpretation-grid">
        <div><h3>기존망의 보완 가능성</h3><p>혼합 10곳 추가 시 전체 평균 보행거리가 {baseline:,.0f}m에서 {mixed['residents']['mean_m']:,.0f}m로 줄었습니다. 어느 위치의 추가 거점이 도움이 되는지 검토할 수 있습니다.</p></div>
        <div><h3>목표에 따른 선택</h3><p>편의점안의 350m 커버율이 더 높고, 혼합안은 평균거리와 고령층 거리에서 이점이 나타났습니다. 한 지표만으로 대안을 정하기보다 지역별 혜택을 함께 비교할 필요가 있습니다.</p></div>
      </div>
      <div class="callout warning"><b>효과의 크기는 구분해서 읽습니다.</b><p>편의점안과 혼합 효율안의 전체 평균거리 차이는 약 17m입니다. 이 차이만으로 경로당 도입이 크게 우수하다고 단정하기 어렵습니다. 도로 최적화의 추가 효과도 혼합 10곳에서 약 20m로, 선정 위치의 변화와 구분해 제시합니다.</p></div>
      <p>고령층 결과는 집계구 대표점과 비교 표본에 영향을 받았습니다. 실제 배출량과 참여율 증가, 운영비 절감은 현장 자료로 확인할 과제입니다. 이번 비교는 모형과 공개 자료 안에서의 정책 시나리오 평가이며, 관측 실험에 대한 통계적 유의성 검정은 아닙니다.</p>
      <h3>다음 단계는 후보 확인과 운영 조건의 연결입니다.</h3>
      <p>편의점과 경로당의 참여 의사, 일반 주민의 출입, 설치 공간을 확인한 뒤 설치비와 회수비를 연결해 재계산합니다. 청풍면·한수면처럼 이번 10곳으로 개선되지 않는 지역도 별도 보완 대상을 검토합니다.</p>
      <div class="downloads"><a href="선정후보_규모별_대안별.csv" download>규모별 선정 후보 명단</a><a href="figures/02_10곳_대안별_효과비교.png" download>핵심 비교 그림</a></div>
    '''))
    # A complete document structure helps readers and local document tools.
    head=soup.new_tag('head')
    for tag in list(soup.html.find_all(['meta','title','style'],recursive=False)):head.append(tag.extract())
    body=soup.new_tag('body');body.append(main.extract())
    soup.html.clear();soup.html.append(head);soup.html.append(body)
    return str(soup)
