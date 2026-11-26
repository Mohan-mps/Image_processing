import json

d = json.loads(open('output/ac6c03182__Im3.json', encoding='utf-8').read())
measurements = d.get('measurements', [])
nc_texts     = d.get('non_chart_text_extraction', {}).get('extracted_text', [])

# Build lookup: object_id -> measurement
meas_map = {m['object_id']: m for m in measurements}

print(f"Total measurements : {len(measurements)}")
print(f"Total OCR items    : {len(nc_texts)}")
print()

# Check fields available in measurements
if measurements:
    print("Fields in a measurement record:")
    for k, v in measurements[0].items():
        print(f"  {k}: {v}")
print()

# Cross-reference with non-chart text
print("Sample cross-reference (first 5 OCR items):")
for t in nc_texts[:5]:
    oid  = t['object_id']
    meas = meas_map.get(oid)
    if meas:
        print(f"  id={oid} text={t['text']!r:20s} "
              f"width_px={meas['width_px']} height_px={meas['adjusted_height_px']} "
              f"width_pts={meas['width_points']} height_pts={meas['height_points']}")
    else:
        print(f"  id={oid} text={t['text']!r:20s}  NO measurement found")
