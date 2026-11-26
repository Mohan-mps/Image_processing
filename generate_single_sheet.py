"""Generate a SINGLE Excel file with ONE sheet containing all 50 images.
Each image block contains:
  - Original image
  - Each chart crop with X / Y tick values
  - Non-chart crop(s) with filtered extracted text
Blocks are stacked vertically, separated by a coloured divider row.
"""

import json
from pathlib import Path
from PIL import Image as PILImage

from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side

# ── Paths ──────────────────────────────────────────────────────────────────────
OUTPUT_FOLDERS = ["output", "output_batch_0", "output_batch_1", "output_batch_2"]
INPUT_DIR      = Path("input")
CROP_DIR       = Path("cropped_charts")
NON_CHART_DIR  = Path("non_chart_crops")
OUT_FILE       = Path("reports") / "all_images_report.xlsx"
Path("reports").mkdir(exist_ok=True)

# ── Display sizes ──────────────────────────────────────────────────────────────
ORIG_W,  ORIG_H  = 420, 300
CROP_W,  CROP_H  = 240, 180
NON_DISPLAY_W    = 300
ROW_PX           = 15

# ── Columns ───────────────────────────────────────────────────────────────────
# A        B            C          D            E           F        G
# Label | Orig/Crop | Chart#  | X-Ticks    | Y-Ticks   | Det.    | Rec.
COL_LABEL   = 1   # A
COL_IMG     = 2   # B
COL_CHART   = 3   # C
COL_X       = 4   # D
COL_Y       = 5   # E
COL_TYPE    = 5   # E  (reused for non-chart text type)
COL_DET     = 6   # F
COL_REC     = 7   # G

# ── Helpers ────────────────────────────────────────────────────────────────────
def px_pt(px):        return px * 0.75
def col_w(px):        return px / 7.0
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

def divider_row(ws, row, ncols=7, text=""):
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=ncols)
    c = ws.cell(row, 1, text)
    c.fill = PatternFill("solid", fgColor="2E4057")
    c.font = Font(bold=True, color="FFFFFF", size=11)
    c.alignment = Alignment(horizontal="left", vertical="center")
    set_rh(ws, row, 22)

def collect_json_map():
    seen = {}
    for folder in OUTPUT_FOLDERS:
        for p in sorted(Path(folder).glob("*.json")):
            if "_report" in p.stem:
                continue
            if p.stem not in seen:
                seen[p.stem] = p
    return dict(sorted(seen.items()))

def filter_texts(all_texts, crop_bb, crop_path=None):
    """Return OCR items overlapping the crop region.
    Falls back to the crop image file dimensions when crop_bb is empty.
    Uses overlap (not strict containment) to handle border-touching text.
    """
    x1 = crop_bb.get("x1")
    y1 = crop_bb.get("y1")
    x2 = crop_bb.get("x2")
    y2 = crop_bb.get("y2")

    # If bbox missing, infer from the crop_path filename pattern or use all texts
    if None in (x1, y1, x2, y2):
        if crop_path is None:
            return all_texts   # no info — return everything
        # crop image is a sub-region of the original; use all OCR text since
        # we have no spatial reference — just return all items
        return all_texts

    # Overlap filter: text bbox overlaps the crop region
    return [t for t in all_texts
            if t["bounding_box"]["x_min"] < x2
            and t["bounding_box"]["x_max"] > x1
            and t["bounding_box"]["y_min"] < y2
            and t["bounding_box"]["y_max"] > y1]

# ── Build workbook ─────────────────────────────────────────────────────────────
wb = Workbook()
ws = wb.active
ws.title = "All Images"

# Column widths
ws.column_dimensions["A"].width = 18          # label
ws.column_dimensions["B"].width = col_w(ORIG_W + 10)  # image
ws.column_dimensions["C"].width = 10          # chart#
ws.column_dimensions["D"].width = 32          # x-ticks / text
ws.column_dimensions["E"].width = 24          # y-ticks / type
ws.column_dimensions["F"].width = 12          # det conf
ws.column_dimensions["G"].width = 12          # rec conf

# Global header (frozen after row 1)
from openpyxl.utils import get_column_letter
for col, txt in [(1,"Image / Section"), (2,"Visual"), (3,"Chart #"),
                 (4,"X-Axis Ticks / Extracted Text"), (5,"Y-Axis Ticks / Content Type"),
                 (6,"Det. Conf"), (7,"Rec. Conf")]:
    hdr(ws.cell(1, col), txt, bg="0D1B2A")
set_rh(ws, 1, 28)
ws.freeze_panes = "A2"

cur = 2   # current row pointer

json_map = collect_json_map()
total    = len(json_map)

for img_num, (stem, json_path) in enumerate(json_map.items(), start=1):
    data         = json.loads(json_path.read_text(encoding="utf-8"))
    charts       = data.get("charts", [])
    input_img    = INPUT_DIR / data["source_image"]
    nc_crops     = sorted(NON_CHART_DIR.glob(f"{stem}_region_*.png"))
    nc_detects   = data.get("non_chart_detections", [])
    all_nc_texts = data.get("non_chart_text_extraction", {}).get("extracted_text", [])

    # ── Divider row ────────────────────────────────────────────────────────────
    divider_row(ws, cur,
                text=f"  [{img_num}/{total}]  {data['source_image']}   "
                     f"({len(charts)} charts, {len(nc_crops)} non-chart crops)")
    cur += 1

    # ── Original image ─────────────────────────────────────────────────────────
    orig_rows = int(ORIG_H / ROW_PX) + 1
    ws.merge_cells(start_row=cur, start_column=COL_LABEL,
                   end_row=cur + orig_rows - 1, end_column=COL_LABEL)
    label_cell(ws.cell(cur, COL_LABEL), "Original\nImage", bg="1B4F72")

    if input_img.exists():
        embed(ws, input_img, f"B{cur}", ORIG_W, ORIG_H)
    else:
        ws.cell(cur, COL_IMG, "Image not found").font = Font(color="FF0000")

    # Summary in cols C-G
    summary = [
        ("Dimensions",  f"{data['image_width']} × {data['image_height']} px"),
        ("Charts",      str(len(charts))),
        ("Non-chart crops", str(len(nc_crops))),
        ("OCR text items",  str(len(all_nc_texts))),
    ]
    for i, (k, v) in enumerate(summary):
        r = cur + i
        ws.cell(r, COL_CHART, k).font = Font(bold=True, size=9)
        ws.cell(r, COL_CHART).alignment = Alignment(horizontal="right")
        ws.cell(r, COL_X, v).font = Font(size=9)

    for r in range(cur, cur + orig_rows):
        set_rh(ws, r, ROW_PX)
    cur += orig_rows + 1

    # ── Cropped charts ─────────────────────────────────────────────────────────
    if charts:
        # Sub-header
        for col, txt, bg in [(COL_LABEL,"Chart Crops","1A5276"),
                              (COL_IMG,"Crop Image","1A5276"),
                              (COL_CHART,"Chart #","1A5276"),
                              (COL_X,"X-Axis Ticks","1A5276"),
                              (COL_Y,"Y-Axis Ticks","1A5276")]:
            hdr(ws.cell(cur, col), txt, bg=bg)
        set_rh(ws, cur, 20)
        cur += 1

        for chart in charts:
            cid     = chart["chart_id"]
            x_ticks = chart.get("x_axis_tick", [])
            y_ticks = chart.get("y_axis_tick", [])
            crop_f  = CROP_DIR / f"{stem}_chart_{cid:03d}.png"

            block = max(int(CROP_H / ROW_PX) + 1,
                        len(x_ticks) + 2, len(y_ticks) + 2)
            for r in range(cur, cur + block):
                set_rh(ws, r, ROW_PX)

            # Label
            ws.merge_cells(start_row=cur, start_column=COL_LABEL,
                           end_row=cur + block - 1, end_column=COL_LABEL)
            c = ws.cell(cur, COL_LABEL, f"Chart {cid}")
            c.font = Font(bold=True, size=10)
            c.fill = PatternFill("solid", fgColor="D6EAF8")
            c.alignment = Alignment(horizontal="center", vertical="center")
            c.border = thin()

            # Crop image
            if crop_f.exists():
                embed(ws, crop_f, f"B{cur}", CROP_W, CROP_H)
            else:
                ws.cell(cur, COL_IMG, "not found").font = Font(color="FF0000")

            # Chart #
            ws.merge_cells(start_row=cur, start_column=COL_CHART,
                           end_row=cur + block - 1, end_column=COL_CHART)
            c = ws.cell(cur, COL_CHART, str(cid))
            c.font = Font(bold=True, size=14)
            c.alignment = Alignment(horizontal="center", vertical="center")
            c.fill = PatternFill("solid", fgColor="EBF5FB")
            c.border = thin()

            # X ticks header
            c = ws.cell(cur, COL_X, "X-Axis Ticks")
            c.font = Font(bold=True, size=9, color="1F4E79")
            c.fill = PatternFill("solid", fgColor="D6EAF8")
            c.alignment = Alignment(horizontal="center")
            c.border = thin()
            for i, tick in enumerate(x_ticks):
                c = ws.cell(cur + 1 + i, COL_X, tick)
                c.font = Font(size=9)
                c.alignment = Alignment(horizontal="left", wrap_text=True)
                c.border = thin()

            # Y ticks header
            c = ws.cell(cur, COL_Y, "Y-Axis Ticks")
            c.font = Font(bold=True, size=9, color="145A32")
            c.fill = PatternFill("solid", fgColor="D5F5E3")
            c.alignment = Alignment(horizontal="center")
            c.border = thin()
            for i, tick in enumerate(y_ticks):
                c = ws.cell(cur + 1 + i, COL_Y, tick)
                c.font = Font(size=9)
                c.alignment = Alignment(horizontal="left", wrap_text=True)
                c.border = thin()

            cur += block + 1
    else:
        c = ws.cell(cur, COL_LABEL, "No charts detected")
        c.font = Font(italic=True, size=9, color="888888")
        set_rh(ws, cur, 18)
        cur += 1

    # ── Non-chart crops ────────────────────────────────────────────────────────
    if nc_crops:
        # Sub-header
        for col, txt, bg in [(COL_LABEL,"Non-Chart","117A65"),
                              (COL_IMG,"Crop Image","117A65"),
                              (COL_CHART,"#","117A65"),
                              (COL_X,"Extracted Text","117A65"),
                              (COL_Y,"Content Type","117A65"),
                              (COL_DET,"Det. Conf","117A65"),
                              (COL_REC,"Rec. Conf","117A65")]:
            hdr(ws.cell(cur, col), txt, bg=bg)
        set_rh(ws, cur, 20)
        cur += 1

        for idx, crop_path in enumerate(nc_crops, start=1):
            nw, nh = aspect(crop_path, NON_DISPLAY_W)
            det_bb = (nc_detects[idx - 1].get("crop_bounding_box", {})
                      if idx - 1 < len(nc_detects) else {})
            texts  = filter_texts(all_nc_texts, det_bb, crop_path)

            block = max(int(nh / ROW_PX) + 1, len(texts) + 2, 3)
            for r in range(cur, cur + block):
                set_rh(ws, r, ROW_PX)

            # Label
            ws.merge_cells(start_row=cur, start_column=COL_LABEL,
                           end_row=cur + block - 1, end_column=COL_LABEL)
            c = ws.cell(cur, COL_LABEL, f"Non-Chart\nRegion {idx}")
            c.font = Font(bold=True, size=10)
            c.fill = PatternFill("solid", fgColor="D5F5E3")
            c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            c.border = thin()

            # Crop image
            embed(ws, crop_path, f"B{cur}", nw, nh)

            if texts:
                # text sub-header row
                for col, lbl in [(COL_CHART,"#"), (COL_X,"Text"),
                                 (COL_Y,"Type"), (COL_DET,"Det."), (COL_REC,"Rec.")]:
                    c = ws.cell(cur, col, lbl)
                    c.font = Font(bold=True, size=8, color="186A3B")
                    c.fill = PatternFill("solid", fgColor="EAFAF1")
                    c.alignment = Alignment(horizontal="center")
                    c.border = thin()

                for i, item in enumerate(texts):
                    r = cur + 1 + i
                    c = ws.cell(r, COL_CHART, item["object_id"])
                    c.font = Font(size=8); c.alignment = Alignment(horizontal="center")
                    c.border = thin()

                    c = ws.cell(r, COL_X, item["text"])
                    c.font = Font(size=9); c.alignment = Alignment(horizontal="left", wrap_text=True)
                    c.border = thin()

                    c = ws.cell(r, COL_Y, item.get("content_type", ""))
                    c.font = Font(size=8); c.alignment = Alignment(horizontal="center")
                    c.border = thin()

                    det_c = item.get("detection_confidence", 0)
                    c = ws.cell(r, COL_DET, round(det_c, 3))
                    c.font = Font(size=8, color="276221" if det_c >= 0.8 else "CC3300")
                    c.alignment = Alignment(horizontal="center"); c.border = thin()

                    rec_c = item.get("recognition_confidence", 0)
                    c = ws.cell(r, COL_REC, round(rec_c, 3))
                    c.font = Font(size=8, color="276221" if rec_c >= 0.8 else "CC3300")
                    c.alignment = Alignment(horizontal="center"); c.border = thin()
            else:
                c = ws.cell(cur, COL_X, "No text detected inside crop")
                c.font = Font(italic=True, size=9, color="888888")
                ws.merge_cells(start_row=cur, start_column=COL_X,
                               end_row=cur, end_column=COL_REC)

            cur += block + 1

    cur += 1   # blank spacer between images

# ── Save ───────────────────────────────────────────────────────────────────────
wb.save(str(OUT_FILE))
print(f"✓ Saved: {OUT_FILE}  ({cur} rows total,  {total} images)")
