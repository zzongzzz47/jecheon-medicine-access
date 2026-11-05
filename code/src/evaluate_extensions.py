"""Distance-model comparison and fixed-plan robustness checks on real inputs."""
import json
from pathlib import Path
from collections import Counter

import numpy as np
from candidate_policy import is_screened_candidate
from optimize_locations import solve_siting
from prepare_population import write_json
from run_population_optimization import metrics

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'results/residents_connector80'


def read(path):
    return json.loads(path.read_text())


def evaluate(distance, existing_count, selected_columns, weights):
    baseline = distance[:, :existing_count].min(axis=1)
    after = np.minimum(baseline, distance[:, existing_count + selected_columns].min(axis=1)) if len(selected_columns) else baseline
    finite = np.isfinite(baseline)
    return dict(before=metrics(baseline[finite], weights[finite]), after=metrics(after[finite], weights[finite]))


def main():
    arrays = np.load(FOLDER / 'distances.npz')
    rows = read(FOLDER / 'origins.json')
    candidates = read(FOLDER / 'candidates.json')
    existing = read(FOLDER / 'existing.json')
    n = len(existing)
    primary = read(FOLDER / 'optimization_medoid_screened.json')
    plans = [p for p in primary['results'] if p['max_sites'] == 10]
    lookup = {r['id']: i for i, r in enumerate(candidates)}
    all_lookup = {r['id']: i for i, r in enumerate(rows)}
    pop = np.array([r.get('population') or 0. if r['group'] == 'grid100m' else 0. for r in rows])
    road = arrays['walking']; straight = arrays['straight']
    report = dict(primary_scenario_size=10, straight_model_comparison=[], stairs=[], representative_points=[],
                  grid500m=[], connector120=[], candidate_access_condition=[])
    valid = (pop > 0) & np.isfinite(road[:, :n].min(axis=1))
    for group in ['convenience', 'mixed']:
        eligible = np.array([is_screened_candidate(c) and (group == 'mixed' or c['kind'] == 'convenience') for c in candidates])
        eligible &= np.any(np.isfinite(road[valid, n:]), axis=0)
        ci = np.flatnonzero(eligible)
        print('Straight distance model:', group, flush=True)
        answer = solve_siting(straight[valid, :n].min(axis=1), straight[valid, n:][:, ci], pop[valid], 10, time_limit=150)['efficiency']
        selected = ci[answer['selected_indices']]
        road_plan = next(p for p in plans if p['candidate_group'] == group and p['mode'] == 'efficiency')
        road_selected = np.array([lookup[c['id']] for c in road_plan['selected']])
        road_eval = evaluate(road[valid], n, selected, pop[valid])
        report['straight_model_comparison'].append(dict(candidate_group=group, solver=answer,
            straight_selected_ids=[candidates[i]['id'] for i in selected],
            road_selected_ids=[candidates[i]['id'] for i in road_selected],
            common_selected=len(set(selected) & set(road_selected)),
            actual_road_evaluation=road_eval,
            road_optimum_mean_m=road_plan['residents']['mean_m'],
            advantage_of_road_model_m=road_eval['after']['mean_m'] - road_plan['residents']['mean_m']))

    for plan in plans:
        selected = np.array([lookup[c['id']] for c in plan['selected']])
        label = dict(candidate_group=plan['candidate_group'], mode=plan['mode'])
        no_steps = arrays['walking_no_steps']
        common = valid & np.isfinite(no_steps[:, :n].min(axis=1))
        report['stairs'].append(dict(**label, common_population=float(pop[common].sum()),
            lost_baseline_population=float(pop[valid & ~common].sum()),
            usual=evaluate(road[common], n, selected, pop[common]),
            without_steps=evaluate(no_steps[common], n, selected, pop[common])))
        p500 = np.array([r.get('population') or 0. if r['group'] == 'grid500m' else 0. for r in rows])
        sample500 = (p500 > 0) & np.isfinite(road[:, :n].min(axis=1))
        report['grid500m'].append(dict(**label, **evaluate(road[sample500], n, selected, p500[sample500]),
            note='500m 격자 중심·공개 통계·도로 연결 표본이 함께 바뀜. 비밀보호 효과만 따로 추정한 결과가 아님'))

        oa = {}
        for i, row in enumerate(rows):
            if row['group'].startswith('elderly'):
                oa.setdefault(row['source_id'], {})[row['group']] = i
        pairs = [a for a in oa.values() if len(a) == 2 and all(np.isfinite(road[i, :n]).any() for i in a.values())]
        for method in ['interior', 'medoid']:
            idx = np.array([a['elderly_' + method] for a in pairs])
            w = np.array([rows[i]['elderly_lower'] for i in idx])
            report['representative_points'].append(dict(**label, point_method=method, census_areas=len(pairs),
                **evaluate(road[idx], n, selected, w)))

    larger = ROOT / 'results/residents_connector120'
    if (larger / 'distances.npz').exists():
        d120 = np.load(larger / 'distances.npz')
        r120 = read(larger / 'origins.json')
        assert [r['id'] for r in r120] == [r['id'] for r in rows]
        m120 = d120['walking']
        common = valid & np.isfinite(m120[:, :n].min(axis=1))
        added = (pop > 0) & ~valid & np.isfinite(m120[:, :n].min(axis=1))
        for plan in plans:
            selected = np.array([lookup[c['id']] for c in plan['selected']])
            report['connector120'].append(dict(candidate_group=plan['candidate_group'], mode=plan['mode'],
                common_population=float(pop[common].sum()), added_population=float(pop[added].sum()),
                common80=evaluate(road[common], n, selected, pop[common]),
                common120=evaluate(m120[common], n, selected, pop[common]),
                enlarged_sample=evaluate(m120[(pop > 0)], n, selected, pop[(pop > 0)])))

    first = FOLDER / 'optimization_medoid.json'
    if first.exists():
        for p in read(first)['results']:
            if p['max_sites'] == 10:
                report['candidate_access_condition'].append(dict(candidate_group=p['candidate_group'], mode=p['mode'],
                    mean_m=p['residents']['mean_m'], elderly_mean_m=p['elderly_lower']['mean_m'],
                    review_candidates=[c['name'] for c in p['selected'] if not is_screened_candidate(c)]))

    # Report improvements by administrative area; all rows share the primary sample.
    for plan in primary['results']:
        selected = np.array([lookup[c['id']] for c in plan['selected']])
        baseline = road[:, :n].min(axis=1)
        after = np.minimum(baseline, road[:, n + selected].min(axis=1))
        plan['administrative_results'] = []
        for name in sorted({r['adm_name'] for r in rows}):
            mask = valid & np.array([r['adm_name'] == name for r in rows])
            if mask.any():
                plan['administrative_results'].append(dict(name=name, before=metrics(baseline[mask], pop[mask]), after=metrics(after[mask], pop[mask])))
        plan['selected_kinds'] = dict(Counter(c['kind'] for c in plan['selected']))
    write_json(FOLDER / 'optimization_medoid_screened.json', primary)
    write_json(FOLDER / 'extension_results.json', report)
    print('Extension checks complete', flush=True)


if __name__ == '__main__':
    main()
