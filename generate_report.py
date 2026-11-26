"""Generate an Excel report for one image showing original, cropped charts with ticks, and non-chart crops."""

import json
from pathlib import Path

from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils import get_column_letter

# ── Paths ──────────────────────────────────────────────────────────────────────
IMAGE_STEM   = "ac6c03182__Im3"
INPUT_IMAGE  = Path("input")        / f"{IMAGE_STEM}.jpg"
JSON_FILE    = Path("output")       / f"{IMAGE_STEM}.json"
CROP_DIR     = Path("cropped_charts")
NON_CHART_DIR= Path("non_chart_crops")
OUTPUT_XLSX  = Path("output")       / f"{IMAGE_STEM}_report_v2.xlsx"

# ── Sizing ─────────────────────────────────────────────────────────────────────
ORIG_W, ORIG_H   = 420, 300   # px for original image in sheet
CROP_W, CROP_H   = 280, 200   # px for each chart crop
NON_DISPLAY_W    = 350         # target display width for non-chart crop (height auto from aspect ratio)
ROW_PX           = 15         # Excel row height in points ≈ pixels

def px_to_pt(px):
    return px * 0.75

def col_width(px):
    return px / 7.0

def style_header(cell, text, bg="1F4E79"):
    cell.value = text
    cell.font = Font(bold=True, color="FFFFFF", size=11)
    cell.fill = PatternFill("solid", fgColor=bg)
    cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

def style_label(cell, text, bg="D6E4F0"):
    cell.value = text
    cell.font = Font(bold=True, size=10)
    cell.fill = PatternFill("solid", fgColor=bg)
    cell.alignment = Alignment(horizontal="center", vertical="center")

def style_tick(cell, text):
    cell.value = text
    cell.font = Font(size=9)
    cell.alignment = Alignment(horizontal="left", vertical="top", wrap_text=True)

def thin_border():
    s = Side(style="thin", color="AAAAAA")
    return Border(left=s, right=s, top=s, bottom=s)

def set_row_height(ws, row, px):
    ws.row_dimensions[row].height = px_to_pt(px)

def embed_image(ws, img_path, cell, w, h):
    img = XLImage(str(img_path))
    img.width  = w
    img.height = h
    ws.add_image(img, cell)

# ── Load JSON ──────────────────────────────────────────────────────────────────
data   = json.loads(JSON_FILE.read_text())
charts = data["charts"]

# ── Non-chart crops for this image ────────────────────────────────────────────
non_chart_crops = sorted(NON_CHART_DIR.glob(f"{IMAGE_STEM}_region_*.png"))

# ── Workbook ───────────────────────────────────────────────────────────────────
wb = Workbook()

# ═══════════════════════════════════════════════════════════════════════════════
#  SHEET 1 — Overview
# ═══════════════════════════════════════════════════════════════════════════════
ws_ov = wb.active
ws_ov.title = "Overview"

# Column widths
ws_ov.column_dimensions["A"].width = col_width(ORIG_W + 20)
ws_ov.column_dimensions["B"].width = 30

# Title row
ws_ov.merge_cells("A1:B1")
style_header(ws_ov["A1"], f"Image Report  —  {IMAGE_STEM}.jpg")
set_row_height(ws_ov, 1, 30)

# Original image label
style_label(ws_ov["A2"], "Original Input Image")
set_row_height(ws_ov, 2, 20)

# Original image
embed_image(ws_ov, INPUT_IMAGE, "A3", ORIG_W, ORIG_H)
img_rows = int(ORIG_H / ROW_PX) + 1
for r in range(3, 3 + img_rows):
    set_row_height(ws_ov, r, ROW_PX)

# Summary below image
summary_row = 3 + img_rows + 1
style_label(ws_ov["A" + str(summary_row)], "Summary")
ws_ov["B" + str(summary_row)].value = "Value"
ws_ov["B" + str(summary_row)].font  = Font(bold=True)

rows_data = [
    ("Source image",          data["source_image"]),
    ("Image dimensions",      f"{data['image_width']} × {data['image_height']} px"),
    ("Total charts detected", len(charts)),
    ("Non-chart regions",     len(non_chart_crops)),
]
for i, (k, v) in enumerate(rows_data):
    r = summary_row + 1 + i
    ws_ov[f"A{r}"].value = k
    ws_ov[f"A{r}"].font  = Font(bold=True, size=10)
    ws_ov[f"B{r}"].value = str(v)
    ws_ov[f"B{r}"].font  = Font(size=10)
    set_row_height(ws_ov, r, 18)

# ═══════════════════════════════════════════════════════════════════════════════
#  SHEET 2 — Cropped Charts + Ticks
# ═══════════════════════════════════════════════════════════════════════════════
ws_ch = wb.create_sheet("Cropped Charts")

# Column layout: | Chart# | Crop image | X ticks | Y ticks |
ws_ch.column_dimensions["A"].width = 10
ws_ch.column_dimensions["B"].width = col_width(CROP_W + 20)
ws_ch.column_dimensions["C"].width = 35
ws_ch.column_dimensions["D"].width = 25

# Header row
for col, text in [("A","Chart #"), ("B","Cropped Chart"), ("C","X-Axis Ticks"), ("D","Y-Axis Ticks")]:
    style_header(ws_ch[f"{col}1"], text)
set_row_height(ws_ch, 1, 25)

current_row = 2
for chart in charts:
    chart_id  = chart["chart_id"]
    x_ticks   = chart.get("x_axis_tick", [])
    y_ticks   = chart.get("y_axis_tick", [])
    crop_file = CROP_DIR / f"{IMAGE_STEM}_chart_{chart_id:03d}.png"

    # How many rows this chart block needs
    block_rows = max(int(CROP_H / ROW_PX) + 1, len(x_ticks) + 2, len(y_ticks) + 2)

    # Set row heights for this block
    for r in range(current_row, current_row + block_rows):
        set_row_height(ws_ch, r, ROW_PX)

    # Chart # label (merge vertically)
    ws_ch.merge_cells(
        start_row=current_row, start_column=1,
        end_row=current_row + block_rows - 1, end_column=1
    )
    cell = ws_ch.cell(current_row, 1, f"Chart\n{chart_id}")
    cell.font      = Font(bold=True, size=11)
    cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    cell.fill      = PatternFill("solid", fgColor="EBF5FB")
    cell.border    = thin_border()

    # Crop image
    if crop_file.exists():
        embed_image(ws_ch, crop_file, f"B{current_row}", CROP_W, CROP_H)
    else:
        ws_ch.cell(current_row, 2, "Image not found").font = Font(color="FF0000")

    # X ticks header + values
    ws_ch.cell(current_row, 3, "X-Axis Ticks").font = Font(bold=True, size=10, color="1F4E79")
    ws_ch.cell(current_row, 3).fill = PatternFill("solid", fgColor="D6EAF8")
    ws_ch.cell(current_row, 3).alignment = Alignment(horizontal="center")
    for i, tick in enumerate(x_ticks):
        c = ws_ch.cell(current_row + 1 + i, 3, tick)
        c.font      = Font(size=9)
        c.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)
        c.border    = thin_border()

    # Y ticks header + values
    ws_ch.cell(current_row, 4, "Y-Axis Ticks").font = Font(bold=True, size=10, color="1F4E79")
    ws_ch.cell(current_row, 4).fill = PatternFill("solid", fgColor="D5F5E3")
    ws_ch.cell(current_row, 4).alignment = Alignment(horizontal="center")
    for i, tick in enumerate(y_ticks):
        c = ws_ch.cell(current_row + 1 + i, 4, tick)
        c.font      = Font(size=9)
        c.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)
        c.border    = thin_border()

    # Divider row between charts
    current_row += block_rows + 1

# ═══════════════════════════════════════════════════════════════════════════════
#  SHEET 3 — Non-Chart Crops + Extracted Text
# ═══════════════════════════════════════════════════════════════════════════════
ws_nc = wb.create_sheet("Non-Chart Crops")

# Column layout: | Region# | Crop image | # | Text | Type | Det.Conf | Rec.Conf |
ws_nc.column_dimensions["A"].width = 12
ws_nc.column_dimensions["B"].width = col_width(NON_DISPLAY_W + 20)
ws_nc.column_dimensions["C"].width = 6
ws_nc.column_dimensions["D"].width = 36
ws_nc.column_dimensions["E"].width = 18
ws_nc.column_dimensions["F"].width = 14
ws_nc.column_dimensions["G"].width = 14

# Header row
for col, text in [("A","Region"), ("B","Non-Chart Crop"), ("C","#"),
                  ("D","Extracted Text"), ("E","Content Type"),
                  ("F","Det. Conf"), ("G","Rec. Conf")]:
    style_header(ws_nc[f"{col}1"], text)
set_row_height(ws_nc, 1, 25)

# Pull extracted text from JSON — only items whose bbox falls inside the crop region
extracted_texts_all = data.get("non_chart_text_extraction", {}).get("extracted_text", [])
non_chart_detections = data.get("non_chart_detections", [])

if non_chart_crops:
    current_row = 2
    for idx, crop_path in enumerate(non_chart_crops, start=1):
        # Get the crop bounding box for this region from the JSON
        detection = non_chart_detections[idx - 1] if idx - 1 < len(non_chart_detections) else {}
        cb = detection.get("crop_bounding_box", {})
        cx1, cy1 = cb.get("x1", 0), cb.get("y1", 0)
        cx2, cy2 = cb.get("x2", 99999), cb.get("y2", 99999)

        # Filter only OCR items that fall inside this crop's bounding box
        extracted_texts = [
            t for t in extracted_texts_all
            if t["bounding_box"]["x_min"] >= cx1
            and t["bounding_box"]["x_max"] <= cx2
            and t["bounding_box"]["y_min"] >= cy1
            and t["bounding_box"]["y_max"] <= cy2
        ]
        # Compute display size preserving aspect ratio
        from PIL import Image as PILImage
        with PILImage.open(str(crop_path)) as pil_img:
            orig_w, orig_h = pil_img.size
        ratio   = orig_h / orig_w
        NON_W   = NON_DISPLAY_W
        NON_H   = int(NON_W * ratio)

        # How many rows: at least enough for the image, or enough for all text rows
        text_rows = max(len(extracted_texts), 1)
        block_rows = max(int(NON_H / ROW_PX) + 1, text_rows + 1)

        for r in range(current_row, current_row + block_rows):
            set_row_height(ws_nc, r, ROW_PX)

        # Region label (merged)
        ws_nc.merge_cells(
            start_row=current_row, start_column=1,
            end_row=current_row + block_rows - 1, end_column=1
        )
        cell = ws_nc.cell(current_row, 1, f"Region\n{idx}")
        cell.font      = Font(bold=True, size=11)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.fill      = PatternFill("solid", fgColor="FDFEFE")
        cell.border    = thin_border()

        # Crop image
        embed_image(ws_nc, crop_path, f"B{current_row}", NON_W, NON_H)

        # Extracted text column headers
        for col, label, bg in [("C","#","E8DAEF"), ("D","Text","E8DAEF"),
                                ("E","Type","E8DAEF"), ("F","Det.","E8DAEF"),
                                ("G","Rec.","E8DAEF")]:
            c = ws_nc[f"{col}{current_row}"]
            c.value     = label
            c.font      = Font(bold=True, size=9, color="4A235A")
            c.fill      = PatternFill("solid", fgColor=bg)
            c.alignment = Alignment(horizontal="center", vertical="center")
            c.border    = thin_border()

        # Text rows
        for i, item in enumerate(extracted_texts):
            r = current_row + 1 + i

            # #
            c = ws_nc.cell(r, 3, item["object_id"])
            c.font = Font(size=9); c.alignment = Alignment(horizontal="center")
            c.border = thin_border()

            # Text value
            c = ws_nc.cell(r, 4, item["text"])
            c.font = Font(size=9); c.alignment = Alignment(horizontal="left", wrap_text=True)
            c.border = thin_border()

            # Content type
            c = ws_nc.cell(r, 5, item.get("content_type", ""))
            c.font = Font(size=9); c.alignment = Alignment(horizontal="center")
            c.border = thin_border()

            # Detection confidence
            det_conf = item.get("detection_confidence", 0)
            c = ws_nc.cell(r, 6, round(det_conf, 4))
            c.font = Font(size=9, color="276221" if det_conf >= 0.8 else "CC3300")
            c.alignment = Alignment(horizontal="center")
            c.border = thin_border()

            # Recognition confidence
            rec_conf = item.get("recognition_confidence", 0)
            c = ws_nc.cell(r, 7, round(rec_conf, 4))
            c.font = Font(size=9, color="276221" if rec_conf >= 0.8 else "CC3300")
            c.alignment = Alignment(horizontal="center")
            c.border = thin_border()

        current_row += block_rows + 1
else:
    ws_nc["A2"].value = "No non-chart regions detected for this image."
    ws_nc["A2"].font  = Font(italic=True, color="888888")

# ── Save ───────────────────────────────────────────────────────────────────────
wb.save(OUTPUT_XLSX)
print(f"✓ Report saved to: {OUTPUT_XLSX}")
print(f"  Sheets: Overview | Cropped Charts ({len(charts)} charts) | Non-Chart Crops ({len(non_chart_crops)} regions, {len(extracted_texts)} text items)")
