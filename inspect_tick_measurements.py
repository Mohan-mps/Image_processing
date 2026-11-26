import json

d = json.loads(open('output/ac6c03182__Im3.json', encoding='utf-8').read())
charts = d.get('charts', [])

# Check what fields exist inside x_tick_ocr / y_tick_ocr
print("=== Chart tick OCR structure ===")
for chart in charts[:2]:
    cid = chart['chart_id']
    print(f"\nChart {cid}:")
    print(f"  x_axis_tick : {chart.get('x_axis_tick')}")
    print(f"  y_axis_tick : {chart.get('y_axis_tick')}")
    print(f"\n  x_tick_ocr sample:")
    for item in chart.get('x_tick_ocr', [])[:3]:
        print(f"    {json.dumps(item, indent=6)}")
    print(f"\n  y_tick_ocr sample:")
    for item in chart.get('y_tick_ocr', [])[:3]:
        print(f"    {json.dumps(item, indent=6)}")
    print(f"\n  All keys in chart: {list(chart.keys())}")
