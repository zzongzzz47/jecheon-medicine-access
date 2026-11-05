"""Compare supplementary sites on the same disclosed, routable resident sample.

All candidate participation is a prospective scenario. Siting does not verify
store consent or public access to a senior centre. Distances are one-way.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from optimize_locations import solve_siting
from prepare_population import write_json
from candidate_policy import is_screened_candidate, review_reasons

ROOT = Path(__file__).resolve().parents[1]


def metrics(distance, weights):
    valid = (weights > 0) & np.isfinite(distance)
    d, w = distance[valid], weights[valid]
    if w.sum() == 0:
        return dict(population=0, mean_m=None)
    order = np.argsort(d)
    p90 = float(d[order][np.searchsorted(np.cumsum(w[order]), .9 * w.sum())])
    return dict(population=float(w.sum()), mean_m=float(d @ w / w.sum()), p90_m=p90,
                within_distance_percent={str(radius): float(100 * w[d <= radius].sum() / w.sum()) for radius in [200, 350, 500, 1000]})


def main(connector=80, sizes=(5, 10, 15, 20), elderly_location='interior', retention=.95, time_limit=180, candidate_policy='screened'):
    folder = ROOT / f'results/residents_connector{connector:g}'
    arrays = np.load(folder / 'distances.npz')
    origins = json.loads((folder / 'origins.json').read_text())
    existing = json.loads((folder / 'existing.json').read_text())
    candidates = json.loads((folder / 'candidates.json').read_text())
    assert arrays['origin_ids'].tolist() == [r['id'] for r in origins]
    assert arrays['destination_ids'].tolist() == [r['id'] for r in existing + candidates]
    elderly_group = 'elderly_' + elderly_location
    all_distance = arrays['walking']
    baseline_all = all_distance[:, :len(existing)].min(axis=1)
    keep = np.array([r['group'] in {'grid100m', elderly_group} for r in origins]) & np.isfinite(baseline_all)
    rows = [r for r, use in zip(origins, keep) if use]
    baseline = baseline_all[keep]
    distances = all_distance[keep, len(existing):]
    population = np.array([r['population'] if r['group'] == 'grid100m' else 0. for r in rows])
    elderly = np.array([r['elderly_lower'] if r['group'] == elderly_group else 0. for r in rows])
    elderly_upper = np.array([r['elderly_upper'] if r['group'] == elderly_group else 0. for r in rows])
    if population.sum() == 0 or elderly.sum() == 0:
        raise ValueError('총인구·고령인구의 연결 표본을 먼저 점검하세요.')
    eligible = np.array([is_screened_candidate(r) or candidate_policy == 'all' for r in candidates])
    report = dict(population_optimization_run=True, elderly_point_method=elderly_location, candidate_policy=candidate_policy,
        candidate_review_exclusions=[dict(id=r['id'],name=r['name'],reasons=review_reasons(r)) for r in candidates if not is_screened_candidate(r)],
        senior_centre_scenario='일반 주민에게 개방되는 수거 접점을 마련하고 시설과 협약한다는 도입 조건',
        installation_unit='시설별 비용을 같게 둔 추가 거점 수 시나리오',
        sample_rule='기존 수거지까지 도로 경로가 확인된 동일 표본으로 모든 대안 비교',
        baseline=dict(residents=metrics(baseline, population), elderly_lower=metrics(baseline, elderly), elderly_upper=metrics(baseline, elderly_upper)),
        results=[])
    groups = [('convenience', np.array([r['kind'] == 'convenience' for r in candidates])),
              ('mixed', np.ones(len(candidates), dtype=bool))]
    for group, candidate_filter in groups:
        indices = np.flatnonzero(candidate_filter & eligible & np.any(np.isfinite(distances), axis=0))
        if not len(indices):
            raise ValueError(f'{group}: 연결된 후보 없음')
        for count in sizes:
            print(f'{group}: 추가 {count}곳 계산', flush=True)
            solved = solve_siting(baseline, distances[:, indices], population, count, elderly=elderly,
                retain_efficiency=retention if group == 'mixed' else None, time_limit=time_limit)
            for mode in ['efficiency', 'elderly_priority']:
                if mode not in solved:
                    continue
                result = solved[mode]
                selected = indices[result['selected_indices']]
                after = np.minimum(baseline, distances[:, selected].min(axis=1)) if len(selected) else baseline
                entry = dict(candidate_group=group, mode=mode, max_sites=count,
                    selected=[candidates[i] for i in selected], solver=result,
                    residents=metrics(after, population), elderly_lower=metrics(after, elderly), elderly_upper=metrics(after, elderly_upper))
                if not np.all(after <= baseline + 1e-6):
                    raise AssertionError('기존망 유지 조건인데 이동거리가 증가했습니다.')
                recalculated = float(population @ (baseline - after))
                if not np.isclose(recalculated, result['population_distance_reduction_person_m']):
                    raise AssertionError('선정 후 독립 재집계 불일치')
                report['results'].append(entry)
            # Save completed scenarios even if a later, harder model times out.
            write_json(folder / f'optimization_{elderly_location}_{candidate_policy}.json', report)
    print(f'완료: {folder / ("optimization_" + elderly_location + "_" + candidate_policy + ".json")}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--connector', type=float, default=80)
    parser.add_argument('--sizes', type=int, nargs='+', default=[5, 10, 15, 20])
    parser.add_argument('--elderly-location', choices=['interior', 'medoid'], default='interior')
    parser.add_argument('--retention', type=float, default=.95)
    parser.add_argument('--time-limit', type=float, default=180)
    parser.add_argument('--candidate-policy', choices=['screened', 'all'], default='screened')
    args = parser.parse_args()
    main(args.connector, args.sizes, args.elderly_location, args.retention, args.time_limit, args.candidate_policy)
