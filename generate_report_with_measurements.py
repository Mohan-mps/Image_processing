"""Generate Excel report for ONE image with measurement columns (width/height px & points)
alongside extracted text in the Non-Chart Crops sheet.
"""

import json
from pathlib import Path
from PIL import Image as PILImage

from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side

# ── Config ─────────────────────────────────────────────────────────────────────
IMAGE_STEM    = "ac6c03182__Im3"
INPUT_IMAGE   = Path("input")          / f"{IMAGE_STEM}.jpg"
JSON_FILE     = Path("output")         / f"{IMAGE_STEM}.json"
CROP_DIR      = Path("cropped_charts")
NON_CHART_DIR = Path("non_chart_crops")
OUTPUT_XLSX   = Path("reports")        / f"{IMAGE_STEM}_report_v5.xlsx"
Path("reports").mkdir(exist_ok=True)

ORIG_W, ORIG_H   = 420, 300
CROP_W, CROP_H   = 280, 200
NON_DISPLAY_W    = 350
ROW_PX           = 15

# ── Helpers ────────────────────────────────────────────────────────────────────
def px_pt(px):         return px * 0.75
def col_w(px):         return px / 7.0
def set_rh(ws, r, px): ws.row_dimensions[r].height = px_pt(px)

def thin():
    s = Side(style="thin", color="BBBBBB")
    return Border(left=s, right=s, top=s, bottom=s)

def embed(ws, path, cell_ref, w, h):
    img = XLImage(str(path))
    img.width, img.height = w, h
    ws.add_image(img, cell_ref)

def aspect(path, target_w):
    with PILImage.open(str(path)) as im:
        ow, oh = im.size
    return target_w, int(target_w * oh / ow)

def hdr(cell, text, bg="1F4E79"):
    cell.value = text
    cell.font  = Font(bold=True, color="FFFFFF", size=10)
    cell.fill  = PatternFill("solid", fgColor=bg)
    cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    cell.border = thin()

def label_cell(cell, text, bg="2E4057"):
    cell.value = text
    cell.font  = Font(bold=True, color="FFFFFF", size=10)
    cell.fill  = PatternFill("solid", fgColor=bg)
    cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    cell.border = thin()

def filter_texts(all_texts, crop_bb, crop_path=None):
    x1 = crop_bb.get("x1")
    y1 = crop_bb.get("y1")
    x2 = crop_bb.get("x2")
    y2 = crop_bb.get("y2")
    if None in (x1, y1, x2, y2):
        return all_texts
    return [t for t in all_texts
            if t["bounding_box"]["x_min"] < x2
            and t["bounding_box"]["x_max"] > x1
            and t["bounding_box"]["y_min"] < y2
            and t["bounding_box"]["y_max"] > y1]

# ── Load JSON ──────────────────────────────────────────────────────────────────
data         = json.loads(JSON_FILE.read_text(encoding="utf-8"))
charts       = data.get("charts", [])
nc_crops     = sorted(NON_CHART_DIR.glob(f"{IMAGE_STEM}_region_*.png"))
nc_detects   = data.get("non_chart_detections", [])
all_nc_texts = data.get("non_chart_text_extraction", {}).get("extracted_text", [])
measurements = data.get("measurements", [])
dpi          = data.get("measurement_scale", {}).get("dpi", 96.0)

# Build measurement lookup: object_id -> measurement record
meas_map = {m["object_id"]: m for m in measurements}

# ── Workbook ───────────────────────────────────────────────────────────────────
wb = Workbook()

# ═══════════════════════════════════════════════════════════════════════════════
# SHEET 1 — Overview
# ═══════════════════════════════════════════════════════════════════════════════
ws = wb.active
ws.title = "Overview"
ws.column_dimensions["A"].width = col_w(ORIG_W + 20)
ws.column_dimensions["B"].width = 32

ws.merge_cells("A1:B1")
hdr(ws["A1"], f"Image Report  —  {data['source_image']}", bg="0D1B2A")
set_rh(ws, 1, 30)

label_cell(ws["A2"], "Original Input Image", bg="1B4F72")
set_rh(ws, 2, 20)

img_rows = int(ORIG_H / ROW_PX) + 1
if INPUT_IMAGE.exists():
    embed(ws, INPUT_IMAGE, "A3", ORIG_W, ORIG_H)
for r in range(3, 3 + img_rows):
    set_rh(ws, r, ROW_PX)

sr = 3 + img_rows + 1
label_cell(ws[f"A{sr}"], "Summary", bg="1B4F72")
ws[f"B{sr}"].value = "Value"; ws[f"B{sr}"].font = Font(bold=True)
for i, (k, v) in enumerate([
    ("Source image",          data["source_image"]),
    ("Image dimensions",      f"{data['image_width']} × {data['image_height']} px"),
    ("Total charts detected", len(charts)),
    ("Non-chart crops",       len(nc_crops)),
    ("Non-chart text items",  len(all_nc_texts)),
    ("Measurement records",   len(measurements)),
]):
    r = sr + 1 + i
    ws[f"A{r}"].value = k;   ws[f"A{r}"].font = Font(bold=True, size=10)
    ws[f"B{r}"].value = str(v); ws[f"B{r}"].font = Font(size=10)
    set_rh(ws, r, 18)

# ═══════════════════════════════════════════════════════════════════════════════
# SHEET 2 — Cropped Charts + Tick Measurements
# ═══════════════════════════════════════════════════════════════════════════════
ws2 = wb.create_sheet("Cropped Charts")

# Columns:
# A=Chart#  B=Crop  C=X Tick  D=W(px)  E=H(px)  F=Y Tick  G=W(px)  H=H(px)
ws2.column_dimensions["A"].width = 10
ws2.column_dimensions["B"].width = col_w(CROP_W + 20)
ws2.column_dimensions["C"].width = 28
ws2.column_dimensions["D"].width = 10
ws2.column_dimensions["E"].width = 10
ws2.column_dimensions["F"].width = 20
ws2.column_dimensions["G"].width = 10
ws2.column_dimensions["H"].width = 10

# Group header row 1
for c1, c2, text, bg in [
    (1, 2, "Chart",         "1A5276"),
    (3, 5, "X-Axis Ticks",  "1F4E79"),
    (6, 8, "Y-Axis Ticks",  "145A32"),
]:
    ws2.merge_cells(start_row=1, start_column=c1, end_row=1, end_column=c2)
    hdr(ws2.cell(1, c1), text, bg=bg)
set_rh(ws2, 1, 20)

# Column header row 2
for col, text, bg in [
    (1, "Chart #",   "1A5276"), (2, "Crop Image", "1A5276"),
    (3, "Tick Value","1F4E79"), (4, "W (pt)",     "1F4E79"), (5, "H (pt)", "1F4E79"),
    (6, "Tick Value","145A32"), (7, "W (pt)",     "145A32"), (8, "H (pt)", "145A32"),
]:
    hdr(ws2.cell(2, col), text, bg=bg)
set_rh(ws2, 2, 22)

def build_ocr_lookup(ocr_items, dpi):
    """Build dict: text -> (width_pts, height_pts) from bbox_in_crop converted to points."""
    factor = 72.0 / dpi if dpi else 0.75
    lookup = {}
    for item in ocr_items:
        t    = item["text"]
        bbox = item.get("bbox_in_crop", {})
        conf = item.get("detection_confidence", 0)
        w_px = bbox.get("width")
        h_px = bbox.get("height")
        w_pt = round(w_px * factor, 2) if w_px is not None else "—"
        h_pt = round(h_px * factor, 2) if h_px is not None else "—"
        if t not in lookup or conf > lookup[t][2]:
            lookup[t] = (w_pt, h_pt, conf)
    return {k: (v[0], v[1]) for k, v in lookup.items()}

cur = 3
for chart in charts:
    cid     = chart["chart_id"]
    x_ticks = chart.get("x_axis_tick", [])
    y_ticks = chart.get("y_axis_tick", [])
    crop_f  = CROP_DIR / f"{IMAGE_STEM}_chart_{cid:03d}.png"

    x_ocr_map = build_ocr_lookup(chart.get("x_tick_ocr", []), dpi)
    y_ocr_map = build_ocr_lookup(chart.get("y_tick_ocr", []), dpi)

    block = max(int(CROP_H / ROW_PX) + 1,
                len(x_ticks) + 2,
                len(y_ticks) + 2)
    for r in range(cur, cur + block): set_rh(ws2, r, ROW_PX)

    # Chart # label
    ws2.merge_cells(start_row=cur, start_column=1,
                    end_row=cur + block - 1, end_column=1)
    c = ws2.cell(cur, 1, f"Chart\n{cid}")
    c.font = Font(bold=True, size=11)
    c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    c.fill = PatternFill("solid", fgColor="EBF5FB"); c.border = thin()

    # Crop image
    if crop_f.exists():
        embed(ws2, crop_f, f"B{cur}", CROP_W, CROP_H)
    else:
        ws2.cell(cur, 2, "not found").font = Font(color="FF0000")

    # Sub-headers
    for col, lbl in [(3,"Tick Value"),(4,"W(pt)"),(5,"H(pt)")]:
        c = ws2.cell(cur, col, lbl)
        c.font = Font(bold=True, size=8, color="1F4E79")
        c.fill = PatternFill("solid", fgColor="D6EAF8")
        c.alignment = Alignment(horizontal="center"); c.border = thin()

    for i, tick in enumerate(x_ticks):
        r = cur + 1 + i
        w_pt, h_pt = x_ocr_map.get(tick, ("—", "—"))
        c = ws2.cell(r, 3, tick)
        c.font = Font(size=9); c.alignment = Alignment(horizontal="left", wrap_text=True)
        c.border = thin()
        for col, val in [(4, w_pt), (5, h_pt)]:
            c = ws2.cell(r, col, val)
            c.font = Font(size=9, color="1A5276")
            c.alignment = Alignment(horizontal="center"); c.border = thin()

    for col, lbl in [(6,"Tick Value"),(7,"W(pt)"),(8,"H(pt)")]:
        c = ws2.cell(cur, col, lbl)
        c.font = Font(bold=True, size=8, color="145A32")
        c.fill = PatternFill("solid", fgColor="D5F5E3")
        c.alignment = Alignment(horizontal="center"); c.border = thin()

    for i, tick in enumerate(y_ticks):
        r = cur + 1 + i
        w_pt, h_pt = y_ocr_map.get(tick, ("—", "—"))
        c = ws2.cell(r, 6, tick)
        c.font = Font(size=9); c.alignment = Alignment(horizontal="left", wrap_text=True)
        c.border = thin()
        for col, val in [(7, w_pt), (8, h_pt)]:
            c = ws2.cell(r, col, val)
            c.font = Font(size=9, color="145A32")
            c.alignment = Alignment(horizontal="center"); c.border = thin()

    cur += block + 1

# ═══════════════════════════════════════════════════════════════════════════════
# SHEET 3 — Non-Chart Crops + Extracted Text + Measurements
# ═══════════════════════════════════════════════════════════════════════════════
ws3 = wb.create_sheet("Non-Chart Crops")

# Columns:
# A=Region  B=Crop image  C=#  D=Text  E=Type  F=Det  G=Rec
# H=W(px)   I=H(px)       J=W(pts) K=H(pts)
ws3.column_dimensions["A"].width = 14
ws3.column_dimensions["B"].width = col_w(NON_DISPLAY_W + 20)
ws3.column_dimensions["C"].width = 6
ws3.column_dimensions["D"].width = 32
ws3.column_dimensions["E"].width = 16
ws3.column_dimensions["F"].width = 10
ws3.column_dimensions["G"].width = 10
ws3.column_dimensions["H"].width = 10
ws3.column_dimensions["I"].width = 10
ws3.column_dimensions["J"].width = 11
ws3.column_dimensions["K"].width = 11

# Header row — two-level: group headers first
# Row 1: merged group headers
groups = [
    (1,  2,  "Region & Crop",           "1A5276"),
    (3,  7,  "OCR Extraction",          "117A65"),
    (8, 11,  "Measurements (from JSON)","76448A"),
]
for c1, c2, text, bg in groups:
    ws3.merge_cells(start_row=1, start_column=c1, end_row=1, end_column=c2)
    hdr(ws3.cell(1, c1), text, bg=bg)
set_rh(ws3, 1, 22)

# Row 2: individual column headers
col_headers = [
    (1, "Region",        "1A5276"),
    (2, "Crop Image",    "1A5276"),
    (3, "#",             "117A65"),
    (4, "Extracted Text","117A65"),
    (5, "Content Type",  "117A65"),
    (6, "Det. Conf",     "117A65"),
    (7, "Rec. Conf",     "117A65"),
    (8, "Width (pt)",    "76448A"),
    (9, "Height (pt)",   "76448A"),
    (10,"Area (pt²)",    "76448A"),
    (11,"DPI",           "76448A"),
]
for col, text, bg in col_headers:
    hdr(ws3.cell(2, col), text, bg=bg)
set_rh(ws3, 2, 22)

cur3 = 3
for idx, crop_path in enumerate(nc_crops, start=1):
    nw, nh = aspect(crop_path, NON_DISPLAY_W)
    det    = nc_detects[idx - 1] if idx - 1 < len(nc_detects) else {}
    det_bb = det.get("crop_bounding_box", {})
    texts  = filter_texts(all_nc_texts, det_bb, crop_path)

    block = max(int(nh / ROW_PX) + 1, len(texts) + 2, 3)
    for r in range(cur3, cur3 + block): set_rh(ws3, r, ROW_PX)

    # Region label
    ws3.merge_cells(start_row=cur3, start_column=1,
                    end_row=cur3 + block - 1, end_column=1)
    c = ws3.cell(cur3, 1, f"Region\n{idx}")
    c.font = Font(bold=True, size=11)
    c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    c.fill = PatternFill("solid", fgColor="D5F5E3"); c.border = thin()

    # Crop image
    embed(ws3, crop_path, f"B{cur3}", nw, nh)

    if texts:
        # Sub-headers row
        sub_hdr_colors = {
            3:"EAFAF1", 4:"EAFAF1", 5:"EAFAF1", 6:"EAFAF1", 7:"EAFAF1",
            8:"F4ECF7",  9:"F4ECF7", 10:"F4ECF7", 11:"F4ECF7"
        }
        sub_hdr_labels = {
            3:"#", 4:"Text", 5:"Type", 6:"Det.", 7:"Rec.",
            8:"W(pt)", 9:"H(pt)", 10:"Area(pt²)", 11:"DPI"
        }
        for col in range(3, 12):
            c = ws3.cell(cur3, col, sub_hdr_labels[col])
            c.font = Font(bold=True, size=8)
            c.fill = PatternFill("solid", fgColor=sub_hdr_colors[col])
            c.alignment = Alignment(horizontal="center", vertical="center")
            c.border = thin()

        for i, item in enumerate(texts):
            r    = cur3 + 1 + i
            oid  = item["object_id"]
            meas = meas_map.get(oid, {})

            # #
            c = ws3.cell(r, 3, oid)
            c.font = Font(size=9); c.alignment = Alignment(horizontal="center")
            c.border = thin()

            # Text
            c = ws3.cell(r, 4, item["text"])
            c.font = Font(size=9)
            c.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)
            c.border = thin()

            # Content type
            c = ws3.cell(r, 5, item.get("content_type", ""))
            c.font = Font(size=9); c.alignment = Alignment(horizontal="center")
            c.border = thin()

            # Det conf
            det_c = item.get("detection_confidence", 0)
            c = ws3.cell(r, 6, round(det_c, 3))
            c.font = Font(size=9, color="276221" if det_c >= 0.8 else "CC3300")
            c.alignment = Alignment(horizontal="center"); c.border = thin()

            # Rec conf
            rec_c = item.get("recognition_confidence", 0)
            c = ws3.cell(r, 7, round(rec_c, 3))
            c.font = Font(size=9, color="276221" if rec_c >= 0.8 else "CC3300")
            c.alignment = Alignment(horizontal="center"); c.border = thin()

            # ── Measurements ───────────────────────────────────────────────
            w_pt   = meas.get("width_points",  "—")
            h_pt   = meas.get("height_points", "—")
            area   = meas.get("area_points_squared", "—")
            m_dpi  = meas.get("dpi", "—")

            for col, val in [(8, w_pt), (9, h_pt), (10, area), (11, m_dpi)]:
                c = ws3.cell(r, col, val)
                c.font = Font(size=9, color="4A235A")
                c.alignment = Alignment(horizontal="center")
                c.border = thin()
    else:
        c = ws3.cell(cur3, 3, "No text detected inside crop")
        c.font = Font(italic=True, size=9, color="888888")
        ws3.merge_cells(start_row=cur3, start_column=3,
                        end_row=cur3, end_column=11)

    cur3 += block + 1

# ── Save ───────────────────────────────────────────────────────────────────────
wb.save(str(OUTPUT_XLSX))
print(f"✓ Saved: {OUTPUT_XLSX}")
print(f"  Charts: {len(charts)}  |  Non-chart crops: {len(nc_crops)}"
      f"  |  Text items: {len(all_nc_texts)}  |  Measurements: {len(measurements)}")
