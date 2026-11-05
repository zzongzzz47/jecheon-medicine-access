"""A conservative name/metadata screen, not a claim of verified public access."""
import re


def review_reasons(row):
    reasons = []
    flags = row.get('flags', [])
    if row['kind'] == 'convenience':
        if '시설 출입 조건 확인' in flags or re.search(r'리조트|리솜|포레스트|휴게소|병원|구내|학사|도서관|학생회관|캠퍼스', row['name']):
            reasons.append('시설 내부·휴게소 등 일반 주민의 출입 확인 필요')
        if '층·출입구 위치 확인' in flags:
            reasons.append('층·출입구 확인 필요')
    if '동일 좌표 통합' in flags and len(set(s.strip() for s in row['name'].split('/'))) > 1:
        reasons.append('서로 다른 시설명이 동일 좌표에 있어 위치 확인 필요')
    return reasons


def is_screened_candidate(row):
    return not review_reasons(row)
