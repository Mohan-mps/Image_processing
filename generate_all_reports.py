"""Generate one Excel report per processed image covering:
   Sheet 1 - Overview        : original image + summary
   Sheet 2 - Cropped Charts  : each chart crop with x/y tick values
   Sheet 3 - Non-Chart Crops : crop image + only the text visible inside it
"""

import json
from pathlib import Path
from PIL import Image as PILImage

from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side

# ── Folder layout ──────────────────────────────────────────────────────────────
OUTPUT_FOLDERS   = ["output", "output_batch_0", "output_batch_1", "output_batch_2"]
INPUT_DIR        = Path("input")
CROP_DIR         = Path("cropped_charts")
NON_CHART_DIR    = Path("non_chart_crops")
REPORTS_DIR      = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

# ── Display sizes ──────────────────────────────────────────────────────────────
ORIG_W,  ORIG_H  = 500, 350   # original image
CROP_W,  CROP_H  = 280, 200   # chart crop
NON_DISPLAY_W    = 350         # non-chart crop width (height auto from aspect ratio)
ROW_PX           = 15

# ── Helpers ────────────────────────────────────────────────────────────────────
def px_to_pt(px):        return px * 0.75
def col_w(px):           return px / 7.0
def set_rh(ws, row, px): ws.row_dimensions[row].height = px_to_pt(px)

def thin_border():
    s = Side(style="thin", color="BBBBBB")
    return Border(left=s, right=s, top=s, bottom=s)

def style_header(cell, text, bg="1F4E79"):
    cell.value = text
    cell.font  = Font(bold=True, color="FFFFFF", size=11)
    cell.fill  = PatternFill("solid", fgColor=bg)
    cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

def style_label(cell, text, bg="D6E4F0"):
    cell.value = text
    cell.font  = Font(bold=True, size=10)
    cell.fill  = PatternFill("solid", fgColor=bg)
    cell.alignment = Alignment(horizontal="center", vertical="center")

def embed(ws, path, cell_ref, w, h):
    img        = XLImage(str(path))
    img.width  = w
    img.height = h
    ws.add_image(img, cell_ref)

def aspect_size(path, target_w):
    with PILImage.open(str(path)) as im:
        ow, oh = im.size
    return target_w, int(target_w * oh / ow)

def collect_json_map():
    """Return {stem: Path} — first folder wins for duplicates."""
    seen = {}
    for folder in OUTPUT_FOLDERS:
        for p in sorted(Path(folder).glob("*.json")):
            if "_report" in p.stem:
                continue
            if p.stem not in seen:
                seen[p.stem] = p
    return seen

def filter_texts_for_crop(all_texts, crop_bb, crop_path=None):
    """Return OCR items overlapping the crop region.
    Falls back to returning all texts when crop_bb is empty.
    Uses overlap (not strict containment) to handle border-touching text.
    """
    x1 = crop_bb.get("x1")
    y1 = crop_bb.get("y1")
    x2 = crop_bb.get("x2")
    y2 = crop_bb.get("y2")

    if None in (x1, y1, x2, y2):
        return all_texts  # no spatial reference — return all

    return [t for t in all_texts
            if t["bounding_box"]["x_min"] < x2
            and t["bounding_box"]["x_max"] > x1
            and t["bounding_box"]["y_min"] < y2
            and t["bounding_box"]["y_max"] > y1]

# ── Per-image report ───────────────────────────────────────────────────────────
def build_report(stem, json_path):
    data   = json.loads(json_path.read_text(encoding="utf-8"))
    charts = data.get("charts", [])

    input_img    = INPUT_DIR / data["source_image"]
    nc_crops     = sorted(NON_CHART_DIR.glob(f"{stem}_region_*.png"))
    nc_detects   = data.get("non_chart_detections", [])
    all_nc_texts = data.get("non_chart_text_extraction", {}).get("extracted_text", [])

    wb = Workbook()

    # ── Sheet 1 · Overview ─────────────────────────────────────────────────────
    ws = wb.active
    ws.title = "Overview"
    ws.column_dimensions["A"].width = col_w(ORIG_W + 20)
    ws.column_dimensions["B"].width = 32

    ws.merge_cells("A1:B1")
    style_header(ws["A1"], f"Image Report  —  {data['source_image']}")
    set_rh(ws, 1, 30)

    style_label(ws["A2"], "Original Input Image")
    set_rh(ws, 2, 20)

    img_rows = int(ORIG_H / ROW_PX) + 1
    if input_img.exists():
        embed(ws, input_img, "A3", ORIG_W, ORIG_H)
    else:
        ws["A3"].value = "Image file not found"
        ws["A3"].font  = Font(color="FF0000")
    for r in range(3, 3 + img_rows):
        set_rh(ws, r, ROW_PX)

    sr = 3 + img_rows + 1
    style_label(ws[f"A{sr}"], "Summary")
    ws[f"B{sr}"].value = "Value"
    ws[f"B{sr}"].font  = Font(bold=True)
    for i, (k, v) in enumerate([
        ("Source image",          data["source_image"]),
        ("Image dimensions",      f"{data['image_width']} × {data['image_height']} px"),
        ("Total charts detected", len(charts)),
        ("Non-chart crops",       len(nc_crops)),
        ("Non-chart text items",  len(all_nc_texts)),
    ]):
        r = sr + 1 + i
        ws[f"A{r}"].value = k;  ws[f"A{r}"].font = Font(bold=True, size=10)
        ws[f"B{r}"].value = str(v); ws[f"B{r}"].font = Font(size=10)
        set_rh(ws, r, 18)

    # ── Sheet 2 · Cropped Charts ───────────────────────────────────────────────
    ws2 = wb.create_sheet("Cropped Charts")
    ws2.column_dimensions["A"].width = 10
    ws2.column_dimensions["B"].width = col_w(CROP_W + 20)
    ws2.column_dimensions["C"].width = 36
    ws2.column_dimensions["D"].width = 26

    for col, text in [("A","Chart #"), ("B","Cropped Chart"),
                      ("C","X-Axis Ticks"), ("D","Y-Axis Ticks")]:
        style_header(ws2[f"{col}1"], text)
    set_rh(ws2, 1, 25)

    cur = 2
    if charts:
        for chart in charts:
            cid     = chart["chart_id"]
            x_ticks = chart.get("x_axis_tick", [])
            y_ticks = chart.get("y_axis_tick", [])
            crop_f  = CROP_DIR / f"{stem}_chart_{cid:03d}.png"

            block = max(int(CROP_H / ROW_PX) + 1,
                        len(x_ticks) + 2,
                        len(y_ticks) + 2)
            for r in range(cur, cur + block): set_rh(ws2, r, ROW_PX)

            # Chart # label
            ws2.merge_cells(start_row=cur, start_column=1,
                            end_row=cur + block - 1, end_column=1)
            c = ws2.cell(cur, 1, f"Chart\n{cid}")
            c.font      = Font(bold=True, size=11)
            c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            c.fill      = PatternFill("solid", fgColor="EBF5FB")
            c.border    = thin_border()

            # Crop image
            if crop_f.exists():
                embed(ws2, crop_f, f"B{cur}", CROP_W, CROP_H)
            else:
                ws2.cell(cur, 2, "Image not found").font = Font(color="FF0000")

            # X ticks
            ws2.cell(cur, 3, "X-Axis Ticks").font = Font(bold=True, size=10, color="1F4E79")
            ws2.cell(cur, 3).fill = PatternFill("solid", fgColor="D6EAF8")
            ws2.cell(cur, 3).alignment = Alignment(horizontal="center")
            for i, tick in enumerate(x_ticks):
                c = ws2.cell(cur + 1 + i, 3, tick)
                c.font = Font(size=9)
                c.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)
                c.border = thin_border()

            # Y ticks
            ws2.cell(cur, 4, "Y-Axis Ticks").font = Font(bold=True, size=10, color="1F4E79")
            ws2.cell(cur, 4).fill = PatternFill("solid", fgColor="D5F5E3")
            ws2.cell(cur, 4).alignment = Alignment(horizontal="center")
            for i, tick in enumerate(y_ticks):
                c = ws2.cell(cur + 1 + i, 4, tick)
                c.font = Font(size=9)
                c.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)
                c.border = thin_border()

            cur += block + 1
    else:
        ws2["A2"].value = "No charts detected in this image."
        ws2["A2"].font  = Font(italic=True, color="888888")

    # ── Sheet 3 · Non-Chart Crops ──────────────────────────────────────────────
    ws3 = wb.create_sheet("Non-Chart Crops")
    ws3.column_dimensions["A"].width = 12
    ws3.column_dimensions["B"].width = col_w(NON_DISPLAY_W + 20)
    ws3.column_dimensions["C"].width = 6
    ws3.column_dimensions["D"].width = 36
    ws3.column_dimensions["E"].width = 18
    ws3.column_dimensions["F"].width = 14
    ws3.column_dimensions["G"].width = 14

    for col, text in [("A","Region"), ("B","Non-Chart Crop"), ("C","#"),
                      ("D","Extracted Text"), ("E","Content Type"),
                      ("F","Det. Conf"), ("G","Rec. Conf")]:
        style_header(ws3[f"{col}1"], text)
    set_rh(ws3, 1, 25)

    if nc_crops:
        cur3 = 2
        for idx, crop_path in enumerate(nc_crops, start=1):
            # aspect-correct size
            nw, nh = aspect_size(crop_path, NON_DISPLAY_W)

            # filter text to only items inside this crop
            det_bb = (nc_detects[idx - 1].get("crop_bounding_box", {})
                      if idx - 1 < len(nc_detects) else {})
            if det_bb:
                texts = filter_texts_for_crop(all_nc_texts, det_bb)
            else:
                texts = filter_texts_for_crop(all_nc_texts, {})

            block = max(int(nh / ROW_PX) + 1, len(texts) + 2, 3)
            for r in range(cur3, cur3 + block): set_rh(ws3, r, ROW_PX)

            # Region label
            ws3.merge_cells(start_row=cur3, start_column=1,
                            end_row=cur3 + block - 1, end_column=1)
            c = ws3.cell(cur3, 1, f"Region\n{idx}")
            c.font      = Font(bold=True, size=11)
            c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            c.fill      = PatternFill("solid", fgColor="FDFEFE")
            c.border    = thin_border()

            # Crop image
            embed(ws3, crop_path, f"B{cur3}", nw, nh)

            if texts:
                # Column sub-headers
                for col, lbl, bg in [("C","#","E8DAEF"), ("D","Text","E8DAEF"),
                                     ("E","Type","E8DAEF"), ("F","Det.","E8DAEF"),
                                     ("G","Rec.","E8DAEF")]:
                    c = ws3[f"{col}{cur3}"]
                    c.value     = lbl
                    c.font      = Font(bold=True, size=9, color="4A235A")
                    c.fill      = PatternFill("solid", fgColor=bg)
                    c.alignment = Alignment(horizontal="center", vertical="center")
                    c.border    = thin_border()

                for i, item in enumerate(texts):
                    r = cur3 + 1 + i

                    c = ws3.cell(r, 3, item["object_id"])
                    c.font = Font(size=9); c.alignment = Alignment(horizontal="center")
                    c.border = thin_border()

                    c = ws3.cell(r, 4, item["text"])
                    c.font = Font(size=9); c.alignment = Alignment(horizontal="left", wrap_text=True)
                    c.border = thin_border()

                    c = ws3.cell(r, 5, item.get("content_type", ""))
                    c.font = Font(size=9); c.alignment = Alignment(horizontal="center")
                    c.border = thin_border()

                    det_c = item.get("detection_confidence", 0)
                    c = ws3.cell(r, 6, round(det_c, 4))
                    c.font = Font(size=9, color="276221" if det_c >= 0.8 else "CC3300")
                    c.alignment = Alignment(horizontal="center"); c.border = thin_border()

                    rec_c = item.get("recognition_confidence", 0)
                    c = ws3.cell(r, 7, round(rec_c, 4))
                    c.font = Font(size=9, color="276221" if rec_c >= 0.8 else "CC3300")
                    c.alignment = Alignment(horizontal="center"); c.border = thin_border()
            else:
                c = ws3.cell(cur3, 3, "No text detected inside this crop")
                c.font = Font(italic=True, size=9, color="888888")
                ws3.merge_cells(start_row=cur3, start_column=3,
                                end_row=cur3, end_column=7)

            cur3 += block + 1
    else:
        ws3["A2"].value = "No non-chart regions detected for this image."
        ws3["A2"].font  = Font(italic=True, color="888888")

    out = REPORTS_DIR / f"{stem}_report.xlsx"
    wb.save(str(out))
    return out, len(charts), len(nc_crops), len(all_nc_texts)


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    json_map = collect_json_map()
    total    = len(json_map)
    print(f"Generating reports for {total} images → folder: {REPORTS_DIR}/")
    print("─" * 60)

    for i, (stem, json_path) in enumerate(sorted(json_map.items()), start=1):
        try:
            out, n_charts, n_nc, n_txt = build_report(stem, json_path)
            print(f"[{i:2d}/{total}] ✓ {stem}  "
                  f"({n_charts} charts, {n_nc} non-chart crops, {n_txt} text items)")
        except Exception as e:
            print(f"[{i:2d}/{total}] ✗ {stem}  ERROR: {e}")

    print("─" * 60)
    print(f"Done! Reports saved in '{REPORTS_DIR}/'")
