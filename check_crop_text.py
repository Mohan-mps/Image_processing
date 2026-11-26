import json

d = json.loads(open("output/ac6c03182__Im3.json").read())
crop = {'x1': 0, 'y1': 365, 'x2': 368, 'y2': 1466}
texts = d['non_chart_text_extraction']['extracted_text']

inside = [t for t in texts if
    t['bounding_box']['x_min'] >= crop['x1'] and
    t['bounding_box']['x_max'] <= crop['x2'] and
    t['bounding_box']['y_min'] >= crop['y1'] and
    t['bounding_box']['y_max'] <= crop['y2']]

print(f"Total OCR items: {len(texts)}")
print(f"Inside crop bbox: {len(inside)}")
print()
for t in inside:
    print(f"  id={t['object_id']}  text={t['text']!r}  type={t['content_type']}  conf_det={t['detection_confidence']:.2f}  conf_rec={t['recognition_confidence']:.2f}")
