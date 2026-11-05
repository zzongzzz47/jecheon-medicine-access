"""Create auditable resident-to-facility walking distances after SGIS import."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from walking_network import WalkingNetwork, TO_XY
from prepare_population import write_json

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def xy(rows):
    return np.array(TO_XY.transform([r['lon'] for r in rows], [r['lat'] for r in rows])).T


def main(max_connector=80, compare_steps=True):
    started = time.monotonic()
    source = ROOT / 'data/processed/population'
    if not (source / 'grid100m.json').exists():
        raise FileNotFoundError('SGIS 자료를 먼저 prepare_population.py로 연결하세요. 아직 주민 분석을 실행하지 않았습니다.')
    output = ROOT / f'results/residents_connector{max_connector:g}'
    output.mkdir(parents=True, exist_ok=True)
    origins = []
    population_files = []
    for filename, group in [('grid100m.json', 'grid100m'), ('grid500m.json', 'grid500m'),
                            ('elderly_interior.json', 'elderly_interior'),
                            ('elderly_population_medoid.json', 'elderly_medoid')]:
        path = source / filename
        if not path.exists():
            continue
        population_files.append(path)
        for row in read(path):
            # Retain nonzero upper bounds even when the lower bound is zero.
            weight = row.get('elderly_upper') if group.startswith('elderly') else row.get('population')
            if weight is None or weight <= 0:
                continue
            origins.append({**row, 'id': f'{group}_{row["source_id"]}', 'group': group})
    candidates = read(ROOT / 'data/processed/candidates.json')
    existing = read(ROOT / 'data/processed/existing_inventory.json')
    if not origins:
        raise ValueError('분석 가능한 공개 인구가 없습니다.')
    metadata = read(ROOT / 'data/raw/jecheon_roads_osm.metadata.json')
    network = WalkingNetwork(ROOT / 'data/raw/jecheon_roads_osm.json', metadata['bbox'])
    network.attach(origins + existing + candidates, max_connector)
    print(f'수요점 {len(origins)}개, 기존 수거지 {len(existing)}곳, 후보 {len(candidates)}곳 거리 계산', flush=True)
    destinations = existing + candidates
    distances = network.distances([r['id'] for r in origins], [r['id'] for r in destinations])
    straight = cdist(xy(origins), xy(destinations))
    finite = np.isfinite(distances)
    if not np.all(distances[finite] + 1e-5 >= straight[finite]):
        raise AssertionError('도로 경로가 직선거리보다 짧습니다.')
    arrays = dict(walking=distances, straight=straight,
                  origin_ids=[r['id'] for r in origins], destination_ids=[r['id'] for r in destinations])
    if compare_steps:
        print('계단 제외 조건 계산', flush=True)
        arrays['walking_no_steps'] = network.distances([r['id'] for r in origins], [r['id'] for r in destinations], no_steps=True)
        if not np.all(arrays['walking_no_steps'] + 1e-5 >= distances):
            raise AssertionError('계단 제외 후 거리가 짧아졌습니다.')
    np.savez_compressed(output / 'distances.npz', **arrays)
    baseline = distances[:, :len(existing)].min(axis=1)
    groups = {}
    for group in sorted({r['group'] for r in origins}):
        idx = np.array([i for i, r in enumerate(origins) if r['group'] == group])
        rows = [origins[i] for i in idx]
        weights = np.array([r['elderly_lower'] if group.startswith('elderly') else r['population'] for r in rows])
        matched = np.array([network.snaps[r['id']]['matched'] for r in rows])
        reachable = np.isfinite(baseline[idx])
        groups[group] = dict(points=len(rows), published_weight=float(weights.sum()),
            connected_points=int(matched.sum()), reachable_points=int(reachable.sum()),
            comparable_weight=float(weights[reachable].sum()),
            unmatched_weight=float(weights[~matched].sum()),
            attached_but_unreachable_weight=float(weights[matched & ~reachable].sum()))
    quality = []
    for i, row in enumerate(origins):
        quality.append({**row, **network.snaps[row['id']],
                        'existing_reachable': bool(np.isfinite(baseline[i])),
                        'baseline_m': float(baseline[i]) if np.isfinite(baseline[i]) else None})
    write_json(output / 'origins.json', quality)
    write_json(output / 'existing.json', existing)
    write_json(output / 'candidates.json', [{**r, **network.snaps[r['id']]} for r in candidates])
    source_files = population_files + [ROOT / 'data/processed/candidates.json', ROOT / 'data/processed/existing_inventory.json', ROOT / 'data/raw/jecheon_roads_osm.json']
    write_json(output / 'connection_audit.json', dict(groups=groups, network=network.audit,
        missing_population_rule='비공개 셀 제외, 0명 대입 안 함',
        unreachable_rule='연결 실패·기존망 경로 미확보 수요는 별도 집계; 비교에서 소리 없이 제거하지 않음',
        no_steps_compared=compare_steps, elapsed_seconds=time.monotonic() - started,
        inputs=[dict(file=str(p.relative_to(ROOT)), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in source_files]))
    print(json.dumps(groups, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-connector', type=float, default=80)
    parser.add_argument('--skip-stairs-comparison', action='store_true')
    args = parser.parse_args()
    main(args.max_connector, not args.skip_stairs_comparison)
