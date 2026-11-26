import json

d = json.loads(open('output/ac6c03182__Im3.json', encoding='utf-8').read())
chart = d['charts'][1]  # chart 2 — numeric ticks easier to verify

print("VLM x_axis_tick:", chart['x_axis_tick'])
print("VLM y_axis_tick:", chart['y_axis_tick'])
print()
print(f"x_tick_ocr count: {len(chart.get('x_tick_ocr', []))}")
print(f"y_tick_ocr count: {len(chart.get('y_tick_ocr', []))}")
print()

# Check if x_axis_tick values appear verbatim in x_tick_ocr
print("=== x_tick_ocr texts ===")
for item in chart.get('x_tick_ocr', []):
    print(f"  text={item['text']!r:15s}  w={item['bbox_in_crop']['width']}  h={item['bbox_in_crop']['height']}")

print()
print("=== y_tick_ocr texts ===")
for item in chart.get('y_tick_ocr', []):
    print(f"  text={item['text']!r:15s}  w={item['bbox_in_crop']['width']}  h={item['bbox_in_crop']['height']}")
