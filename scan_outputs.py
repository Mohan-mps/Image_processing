import json
from pathlib import Path

folders = ['output', 'output_batch_0', 'output_batch_1', 'output_batch_2']
seen = {}
for f in folders:
    for p in sorted(Path(f).glob('*.json')):
        if '_report' in p.stem:
            continue
        if p.stem not in seen:
            seen[p.stem] = p

print(f"Total unique JSONs: {len(seen)}")

has_nc_crop  = 0
has_nc_text  = 0
no_nc        = 0

for stem, p in seen.items():
    d = json.loads(p.read_text(encoding='utf-8'))
    nc_det  = d.get('non_chart_detections', [])
    nc_text = d.get('non_chart_text_extraction', {}).get('extracted_text', [])
    nc_crops = list(Path('non_chart_crops').glob(f'{stem}_region_*.png'))
    if nc_crops:
        has_nc_crop += 1
    if nc_text:
        has_nc_text += 1
    if not nc_crops and not nc_text:
        no_nc += 1

print(f"Images with non-chart crops : {has_nc_crop}")
print(f"Images with non-chart text  : {has_nc_text}")
print(f"Images with neither         : {no_nc}")
