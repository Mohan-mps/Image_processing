import json

d = json.loads(open('output/ac6c03182__Im3.json', encoding='utf-8').read())

# Check measurements — full record for a tick-related item
measurements = d.get('measurements', [])
print("=== Full measurement record sample ===")
for m in measurements[:3]:
    print(json.dumps(m, indent=2))
    print()

# Check x_tick_ocr and y_tick_ocr for pts fields
chart = d['charts'][1]
print("=== x_tick_ocr full record sample ===")
for item in chart.get('x_tick_ocr', [])[:3]:
    print(json.dumps(item, indent=2))
    print()

print("=== y_tick_ocr full record sample ===")
for item in chart.get('y_tick_ocr', [])[:3]:
    print(json.dumps(item, indent=2))
    print()
