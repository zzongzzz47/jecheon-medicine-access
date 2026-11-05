"""Fetch the public OSM road extract once, retaining query and provenance."""
from pathlib import Path
import datetime, hashlib, json, time
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw'
BBOX = [127.90, 36.80, 128.39, 37.30]  # W,S,E,N; facilities plus external buffer

def main():
    dest = RAW / 'jecheon_roads_osm.json'
    if dest.exists():
        data = json.loads(dest.read_text())
        if data.get('elements') and not data.get('remark'):
            print('Using cached road extract', len(data['elements']), flush=True)
            return
    w,s,e,n = BBOX
    query = f'[out:json][timeout:90];way["highway"]({s},{w},{n},{e});out body;>;out body qt;'
    url = 'https://overpass-api.de/api/interpreter'
    start = time.monotonic()
    response = requests.post(url, data={'data': query}, timeout=110,
        headers={'User-Agent':'JecheonPublicAccessibilityResearch/1.0'})
    response.raise_for_status()
    data = response.json()
    if data.get('remark') or not data.get('elements'):
        raise ValueError(data.get('remark', 'No OSM elements'))
    dest.write_bytes(response.content)
    metadata = dict(source_url=url, query=query, bbox=BBOX,
        retrieved_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        bytes=len(response.content), sha256=hashlib.sha256(response.content).hexdigest(),
        elements=len(data['elements']), osm3s=data.get('osm3s'),
        license='© OpenStreetMap contributors, ODbL 1.0',
        license_url='https://www.openstreetmap.org/copyright',
        elapsed_seconds=time.monotonic()-start)
    dest.with_suffix('.metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(json.dumps(metadata, ensure_ascii=False), flush=True)

if __name__ == '__main__':
    main()
