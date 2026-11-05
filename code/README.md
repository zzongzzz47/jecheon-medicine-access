# 사용한 코드

| 순서 | 역할 | 파일 |
|---|---|---|
| 1 | 기존·후보 시설 정리 | `src/prepare_facilities.py` |
| 2 | 보행도로 수집 | `src/fetch_roads.py` |
| 3 | SGIS 인구와 경계 연결 | `src/prepare_population.py` |
| 4 | 도로 연결과 거리 행렬 | `src/connect_population.py` |
| 5 | 최적화 | `src/run_population_optimization.py` |
| 6 | 검증과 보고서 | `src/check_methods.py`, `src/build_resident_report.py` |

```sh
cd code
python -m pip install -r requirements.txt
# SGIS 원본과 시설·도로 자료를 먼저 준비한 뒤 실행
python src/connect_population.py --max-connector 80
python src/run_population_optimization.py --connector 80 --elderly-location medoid --candidate-policy screened
python src/build_resident_report.py
```

`prepare_population.py`에는 인구와 경계 ZIP의 경로를 인자로 전달합니다. 구체적인 입력은 스크립트와 `../docs/DATA.md`를 확인하세요. 초기 도로 파일럿(`run_network_pilot.py`)은 주민 인구를 연결한 최종 분석과 구분됩니다. 원자료를 포함하지 않아 위 세 줄만으로 재현이 완료되지는 않습니다.
