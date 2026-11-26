"""Debug: show OCR text vs crop bounding boxes for all images with non-chart crops."""
import json
from pathlib import Path

OUTPUT_FOLDERS = ["output", "output_batch_0", "output_batch_1", "output_batch_2"]

seen = {}
for folder in OUTPUT_FOLDERS:
    for p in sorted(Path(folder).glob("*.json")):
        if "_report" in p.stem: continue
        if p.stem not in seen:
            seen[p.stem] = p

for stem, json_path in sorted(seen.items()):
    data         = json.loads(json_path.read_text(encoding="utf-8"))
    nc_detects   = data.get("non_chart_detections", [])
    nc_crops     = sorted(Path("non_chart_crops").glob(f"{stem}_region_*.png"))
    all_texts    = data.get("non_chart_text_extraction", {}).get("extracted_text", [])

    if not nc_crops or not all_texts:
        continue

    print(f"\n{'='*70}")
    print(f"IMAGE: {stem}  |  total OCR items: {len(all_texts)}")

    for idx, crop_path in enumerate(nc_crops, start=1):
        det = nc_detects[idx-1] if idx-1 < len(nc_detects) else {}
        cb  = det.get("crop_bounding_box", {})
        bb  = det.get("bounding_box", {})
        print(f"\n  Region {idx}: {crop_path.name}")
        print(f"    detection bounding_box     : {bb}")
        print(f"    detection crop_bounding_box: {cb}")

        if cb:
            x1,y1,x2,y2 = cb["x1"],cb["y1"],cb["x2"],cb["y2"]
            inside = [t for t in all_texts
                      if t["bounding_box"]["x_min"] >= x1
                      and t["bounding_box"]["x_max"] <= x2
                      and t["bounding_box"]["y_min"] >= y1
                      and t["bounding_box"]["y_max"] <= y2]
            print(f"    strict filter  → {len(inside)} items")

            # Try relaxed: just y overlap
            inside_y = [t for t in all_texts
                        if t["bounding_box"]["y_max"] >= y1
                        and t["bounding_box"]["y_min"] <= y2]
            print(f"    y-only  filter → {len(inside_y)} items")

            # Try relaxed: x AND y overlap (not fully contained)
            inside_overlap = [t for t in all_texts
                              if t["bounding_box"]["x_min"] < x2
                              and t["bounding_box"]["x_max"] > x1
                              and t["bounding_box"]["y_min"] < y2
                              and t["bounding_box"]["y_max"] > y1]
            print(f"    overlap filter → {len(inside_overlap)} items")
            for t in inside_overlap[:10]:
                print(f"      text={t['text']!r:30s}  bbox={t['bounding_box']}")
        else:
            print(f"    NO crop_bounding_box found!")
            print(f"    detection keys: {list(det.keys())}")

        # Show a sample of all text bboxes to compare
        if not cb:
            print(f"    Sample OCR bboxes (first 5):")
            for t in all_texts[:5]:
                print(f"      text={t['text']!r:20s}  bbox={t['bounding_box']}")
