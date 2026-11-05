"""Import requested SGIS files without inventing values for undisclosed cells.

Jecheon code 33030 and age codes were checked against the official SGIS
2025-06 administrative and statistics codebooks. Statistics use 2024 counts
re-tabulated to 2025 Q2 boundaries, as selected in the application.
"""
import argparse
import hashlib
import json
import unicodedata
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
from pyproj import CRS, Transformer
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/processed/population'
CRS_METRIC = CRS.from_epsg(5179)
CITY_CODE = '33030'
EXPECTED_ADM_COUNT = 17


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def unpack(path, target):
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError(f'압축파일 손상: {path.name}')
        for item in archive.infolist():
            name = item.filename
            if not item.flag_bits & 0x800:
                try:
                    name = name.encode('cp437').decode('cp949')
                except UnicodeError:
                    pass
            dest = target / unicodedata.normalize('NFC', name)
            if not dest.resolve().is_relative_to(target.resolve()):
                raise ValueError('압축파일 내부 경로 오류')
            if item.is_dir():
                dest.mkdir(parents=True, exist_ok=True)
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(archive.read(item))
    for nested in list(target.rglob('*.zip')):
        unpack(nested, nested.with_suffix(''))


def read_projected(path, bounds=None):
    source = pyogrio.read_info(path).get('crs')
    if not source:
        raise ValueError(f'{path.name}: 원본 .prj 좌표계를 확인해야 합니다.')
    options = {'engine': 'pyogrio'}
    if bounds is not None:
        options['bbox'] = Transformer.from_crs(CRS_METRIC, source, always_xy=True).transform_bounds(*bounds, densify_pts=21)
    return gpd.read_file(path, **options).to_crs(CRS_METRIC)


def read_table(path):
    for encoding in ('utf-8-sig', 'cp949'):
        try:
            table = pd.read_csv(path, header=None, names=['year', 'id', 'code', 'value'], dtype=str, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f'통계파일 문자 인코딩 확인 필요: {path}')
    table['value'] = pd.to_numeric(table['value'], errors='coerce')
    table = table.loc[table.year == '2024'].copy()
    return table


def unique_values(frame, keys):
    frame = frame.drop_duplicates()
    if frame.duplicated(keys).any():
        raise ValueError(f'동일 통계키에 서로 다른 값: {keys}')
    return frame


def elderly_bounds(age, total):
    """Exact bounds under the published census-cell suppression rule, 0..4."""
    codes = [f'in_age_{i:03d}' for i in range(1, 22)]
    age = age.reindex(index=total.index, columns=codes)
    if total.isna().any():
        raise ValueError('집계구 총인구 누락: 연령 비공개 구간을 계산할 수 없습니다.')
    known = age.sum(axis=1)
    max_all = known + 4 * age.isna().sum(axis=1)
    if ((total < known) | (total > max_all)).any():
        raise ValueError('총인구와 연령 합계 불일치: 연령코드·비공개 규칙 확인 필요')
    old, young = age[codes[13:]], age[codes[:13]]
    lower = np.maximum(old.sum(axis=1), total - young.sum(axis=1) - 4 * young.isna().sum(axis=1))
    upper = np.minimum(old.sum(axis=1) + 4 * old.isna().sum(axis=1), total - young.sum(axis=1))
    if (upper < lower).any():
        raise ValueError('고령인구 상하한 불일치')
    return lower, upper, old.notna().sum(axis=1)


def point_records(frame, geometry, kind):
    points = gpd.GeoSeries(geometry, index=frame.index, crs=frame.crs).to_crs(4326)
    records = []
    for (_, row), point in zip(frame.iterrows(), points):
        record = dict(id=f'{kind}_{row.id}', source_id=str(row.id), kind=kind,
                      lon=float(point.x), lat=float(point.y), adm_cd=str(row.adm_cd),
                      adm_name=row.ADM_NM, reference_year=2024)
        for col in ['population', 'elderly_lower', 'elderly_upper', 'census_population', 'published_age_bins']:
            if col in row:
                record[col] = None if pd.isna(row[col]) else float(row[col])
        records.append(record)
    return records


def population_medoid(points, weights):
    """An inhabited grid centre minimizing weighted straight distances.

    This is a location proxy for a census area, not an allocation of older
    residents to grids. Its alternative is the polygon's interior point.
    """
    xy = np.array([[point.x, point.y] for point in points])
    return points[int(np.argmin(cdist(xy, xy) @ np.asarray(weights)))]


def main(archives):
    OUT.mkdir(parents=True, exist_ok=True)
    extracted = []
    manifests = []
    for archive in archives:
        archive = archive.resolve()
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        folder = ROOT / '.cache/sgis_jecheon' / digest[:12]
        unpack(archive, folder)
        extracted.append(folder)
        manifests.append(dict(file=archive.name, sha256=digest))
    shapes = [p for folder in extracted for p in folder.rglob('*.shp')]
    admin_paths = [p for p in shapes if p.stem.startswith(('bnd_dong_33_', 'bnd_dong_33030_'))]
    census_paths = [p for p in shapes if p.stem.startswith(('bnd_oa_33_', 'bnd_oa_33030_'))]
    if len(admin_paths) != 1 or len(census_paths) != 1:
        raise ValueError('충북 2025년 2분기 읍면동·집계구 경계를 각각 하나씩 제공하세요.')
    admin = read_projected(admin_paths[0])
    admin['ADM_CD'] = admin.ADM_CD.astype(str)
    admin = admin.loc[admin.ADM_CD.str.startswith(CITY_CODE)].copy()
    if len(admin) != EXPECTED_ADM_COUNT:
        raise ValueError(f'제천시 읍면동 경계 확인 필요: {len(admin)}개')
    polygon = admin.geometry.union_all()
    tables = []
    for folder in extracted:
        for path in folder.rglob('*.csv'):
            if '인구' in path.name or path.name.startswith(('in_', 'to_in')):
                tables.append((path.name, read_table(path)))
    audit = dict(statistics_year=2024, boundary='2025_2Q', inputs=manifests,
                 administrative_names=admin.ADM_NM.tolist(), grids={})
    admin.to_file(OUT / 'population.gpkg', layer='admin', driver='GPKG')
    grids_by_size = {}
    for resolution in (100, 500):
        suffix = f'_{resolution}M'
        paths = [p for p in shapes if p.stem.upper().endswith(suffix)]
        selected_tables = [t for name, t in tables if suffix in name.upper()]
        if not paths or not selected_tables:
            if resolution == 100:
                raise ValueError('100m 인구 통계와 격자 경계가 필요합니다.')
            continue
        values = pd.concat(selected_tables, ignore_index=True)
        values = unique_values(values.loc[values.code == 'to_in_001', ['id', 'value']], ['id'])
        parts = [read_projected(p, polygon.bounds) for p in paths]
        grids = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry='geometry').rename(columns={'GRID_CD': 'id'})
        grids = grids.drop_duplicates('id')
        centres = gpd.GeoDataFrame({'id': grids.id}, geometry=grids.geometry.centroid, crs=grids.crs)
        mapped = gpd.sjoin(centres, admin[['ADM_CD', 'ADM_NM', 'geometry']], how='inner', predicate='within')
        if mapped.id.duplicated().any():
            raise ValueError('인구 중심점과 읍면동 경계의 중복 연결')
        grids = grids.merge(mapped[['id', 'ADM_CD', 'ADM_NM']], on='id').rename(columns={'ADM_CD': 'adm_cd'})
        grids = grids.merge(values.rename(columns={'value': 'population'}), on='id', how='left')
        records = point_records(grids, grids.geometry.centroid, f'grid{resolution}m')
        write_json(OUT / f'grid{resolution}m.json', records)
        grids.to_file(OUT / 'population.gpkg', layer=f'grid{resolution}m', driver='GPKG')
        audit['grids'][str(resolution)] = dict(cells=len(grids), published_cells=int(grids.population.notna().sum()),
            missing_cells=int(grids.population.isna().sum()), released_population=float(grids.population.sum()),
            boundary_rule='격자 중심점이 제천시 읍면동 내부인 격자',
            missing_rule='미공개·누락 값을 0명으로 대체하지 않음')
        grids_by_size[resolution] = grids

    census = read_projected(census_paths[0], polygon.bounds)
    code_col = next(c for c in census if c.upper() in ('TOT_OA_CD', 'OA_CD'))
    census = census.rename(columns={code_col: 'id', 'ADM_CD': 'adm_cd'})
    census['id'] = census.id.astype(str)
    census['adm_cd'] = census.adm_cd.astype(str)
    census = census.merge(admin[['ADM_CD', 'ADM_NM']].rename(columns={'ADM_CD': 'adm_cd'}), on='adm_cd')
    census_tables = [t for name, t in tables if not any(f'_{s}' in name.upper() for s in ('100M', '500M', '1K', '10K', '100K'))]
    if not census_tables:
        raise ValueError('집계구 성연령별인구·총인구 자료가 필요합니다.')
    ct = pd.concat(census_tables, ignore_index=True)
    ct = unique_values(ct.loc[ct.id.isin(census.id), ['id', 'code', 'value']], ['id', 'code'])
    age = ct.loc[ct.code.str.startswith('in_age_')].pivot(index='id', columns='code', values='value')
    total = ct.loc[ct.code == 'to_in_001'].set_index('id').value.reindex(census.id)
    lower, upper, known_bins = elderly_bounds(age, total)
    census['elderly_lower'] = lower.to_numpy()
    census['elderly_upper'] = upper.to_numpy()
    census['census_population'] = total.to_numpy()
    census['published_age_bins'] = known_bins.to_numpy()
    geometric = census.geometry.representative_point()
    write_json(OUT / 'elderly_interior.json', point_records(census, geometric, 'elderly'))

    # Use all-resident location information only to test the area-point assumption.
    inhabited = grids_by_size[100].loc[grids_by_size[100].population > 0].copy()
    inhabited.geometry = inhabited.geometry.centroid
    joined = gpd.sjoin(inhabited[['population', 'geometry']], census[['id', 'geometry']], how='inner', predicate='within')
    if joined.index.duplicated().any():
        raise ValueError('격자 중심점이 둘 이상의 집계구와 연결됨')
    alternate = []
    location_methods = []
    for i, row in census.iterrows():
        inside = joined.loc[joined.id == row.id]
        if len(inside):
            alternate.append(population_medoid(inside.geometry.tolist(), inside.population.to_numpy()))
            location_methods.append('공개 인구가 있는 100m 격자 중심점 중 인구 가중 거리 합 최소점')
        else:
            alternate.append(geometric.loc[i])
            location_methods.append('공개 인구 격자 중심점이 없어 집계구 내부점 사용')
    records = point_records(census, alternate, 'elderly')
    for record, method in zip(records, location_methods):
        record['representative_method'] = method
    write_json(OUT / 'elderly_population_medoid.json', records)
    census.to_file(OUT / 'population.gpkg', layer='census_ages', driver='GPKG')
    audit['elderly'] = dict(census_areas=len(census), total_population=float(total.sum()),
        lower_sum=float(lower.sum()), upper_sum=float(upper.sum()),
        exact_age_count_areas=int((lower == upper).sum()),
        age_rule='65세 이상 8개 연령구간. 미공개 구간 0~4명, 총인구·나머지 연령으로 범위 축소',
        interior_point_file='elderly_interior.json', settlement_point_file='elderly_population_medoid.json',
        no_published_grid_fallback=sum('없어' in s for s in location_methods),
        location_note='집계구별 고령인구를 한 대표점에 두는 근사. 100m 격자에 고령인구를 배분하지 않음')
    write_json(ROOT / 'results/population_input_audit.json', audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('archives', nargs='+', type=Path)
    main(parser.parse_args().archives)
