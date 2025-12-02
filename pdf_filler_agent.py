
"""
PDF Filler Agent 
- Widgets retrieved fresh from PDF pages
- Works with merged schema (widget_pdf_name)
"""

import json
import fitz  # PyMuPDF
from pathlib import Path
import argparse
import logging
import base64
import io
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - pdf_filler_agent - %(levelname)s - %(message)s"
)
logger = logging.getLogger("pdf_filler_agent")


# ---------------------------------------------------------
# Load JSON
# ---------------------------------------------------------
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------
# Signature image insertion
# ---------------------------------------------------------
def insert_signature(page, rect, base64_data):
    try:
        if base64_data.startswith("data:"):
            base64_data = base64_data.split(",", 1)[1]

        img_bytes = base64.b64decode(base64_data)

        # Validate image
        Image.open(io.BytesIO(img_bytes))

        page.insert_image(rect, stream=img_bytes)
        return True
    except Exception as e:
        logger.error(f"Signature error: {e}")
        return False


# ---------------------------------------------------------
# Find widget on page fresh (critical fix)
# ---------------------------------------------------------
def get_widget_from_page(page, widget_name):
    widgets = page.widgets()
    if not widgets:
        return None

    for w in widgets:
        if w.field_name == widget_name:
            return w
    return None


# ---------------------------------------------------------
# Main filler logic
# ---------------------------------------------------------
def fill_pdf(pdf_path, schema_path, data_path, output_path):

    logger.info(f"Loading merged schema: {schema_path}")
    schema = load_json(schema_path)

    logger.info(f"Loading user data: {data_path}")
    data = load_json(data_path)

    # Open PDF
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    logger.info(f"Opened PDF: {pdf_path} ({total_pages} pages)")

    fields = schema.get("fields", [])
    logger.info(f"Merged schema contains {len(fields)} fields")

    filled = 0
    skipped_no_widget = 0
    skipped_no_data = 0

    # ---------------------------------------------------------
    # Create a lightweight index of widget names and their pages
    # ---------------------------------------------------------
    logger.info("Indexing PDF widget names…")

    widget_index = {}  # { "1-1-1": [page numbers] }

    for page_num in range(total_pages):
        page = doc[page_num]
        widgets = page.widgets()
        if not widgets:
            continue

        for w in widgets:
            wname = w.field_name
            if wname not in widget_index:
                widget_index[wname] = []
            widget_index[wname].append(page_num)

    logger.info(f"Indexed {len(widget_index)} unique widget names")


    # ---------------------------------------------------------
    # Fill each semantic+widget field
    # ---------------------------------------------------------
    for f in fields:
        fid = f.get("id")
        wname = f.get("widget_pdf_name")
        ftype = f.get("type")
        rect = f.get("widget_rect")
        target_page = f.get("page")

        value = data.get(fid)

        logger.info(f"Processing id='{fid}', widget='{wname}', type={ftype}")

        # No data provided
        if value is None:
            skipped_no_data += 1
            logger.warning(f"No data provided for '{fid}'")
            continue

        # No widget in merged schema
        if not wname:
            skipped_no_widget += 1
            logger.warning(f"No widget assigned in schema for '{fid}'")
            continue

        # Widget mapping
        pages_with_widget = widget_index.get(wname)
        if not pages_with_widget:
            skipped_no_widget += 1
            logger.warning(f"Widget '{wname}' not found in actual PDF for '{fid}'")
            continue

        # Use schema page if provided, else first page from index
        if target_page:
            page_number = target_page - 1
        else:
            page_number = pages_with_widget[0]

        page = doc[page_number]

        # Retrieve widget FRESH from page (critical)
        widget = get_widget_from_page(page, wname)
        if not widget:
            skipped_no_widget += 1
            logger.warning(f"Widget '{wname}' missing on page {page_number+1}")
            continue

        # ---------------------------------------------------------
        # Signature field
        # ---------------------------------------------------------
        if ftype == "signature":
            if rect:
                sig_rect = fitz.Rect(rect["x0"], rect["y0"], rect["x1"], rect["y1"])
                ok = insert_signature(page, sig_rect, value)
                if ok:
                    filled += 1
            continue

        # ---------------------------------------------------------
        # Checkbox / select
        # ---------------------------------------------------------
        if ftype in ("checkbox", "selection", "grouped_choices"):
            widget.field_value = str(value)
            widget.update()
            filled += 1
            continue

        # ---------------------------------------------------------
        # Text fields
        # ---------------------------------------------------------
        try:
            widget.field_value = str(value)
            widget.update()
            filled += 1
        except Exception as e:
            logger.error(f"Error filling field '{fid}' on page {page_number+1}: {e}")


    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    doc.save(output_path)
    doc.close()

    logger.info("=======================================")
    logger.info(f"FILL SUMMARY")
    logger.info(f"  Filled fields       : {filled}")
    logger.info(f"  Skipped no data     : {skipped_no_data}")
    logger.info(f"  Skipped no widget   : {skipped_no_widget}")
    logger.info(f"  Output PDF          : {output_path}")
    logger.info("=======================================")

    return True


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------
def cli():
    parser = argparse.ArgumentParser(description="Fill PDF using merged schema")

    parser.add_argument("--pdf", required=True, help="Input PDF")
    parser.add_argument("--schema", required=True, help="Merged schema JSON")
    parser.add_argument("--data", required=True, help="User input JSON")
    parser.add_argument("--out", required=True, help="Output filled PDF path")

    args = parser.parse_args()

    fill_pdf(args.pdf, args.schema, args.data, args.out)


if __name__ == "__main__":
    cli()
