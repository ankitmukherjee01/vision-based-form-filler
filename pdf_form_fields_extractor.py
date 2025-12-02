"""
PDF Form Fields Extractor
Extracts all fillable field information from unlocked PDFs and saves to JSON.

"""

import os
import json
from pathlib import Path
import re
import fitz  # PyMuPDF

INPUT_DIR = Path("forms")           
OUTPUT_DIR = Path("2_fields_json")  

OUTPUT_DIR.mkdir(exist_ok=True)

# Field type mapping
FIELD_TYPE_MAP = {
    fitz.PDF_WIDGET_TYPE_BUTTON: "Button",
    fitz.PDF_WIDGET_TYPE_CHECKBOX: "CheckBox",
    fitz.PDF_WIDGET_TYPE_COMBOBOX: "ComboBox",
    fitz.PDF_WIDGET_TYPE_LISTBOX: "ListBox",
    fitz.PDF_WIDGET_TYPE_RADIOBUTTON: "RadioButton",
    fitz.PDF_WIDGET_TYPE_SIGNATURE: "Signature",
    fitz.PDF_WIDGET_TYPE_TEXT: "Text",
    fitz.PDF_WIDGET_TYPE_UNKNOWN: "Unknown"
}


# ------------- label / name helpers ------------- #

def normalize_label(text: str) -> str:
    """Normalize text for comparison / pattern checks."""
    if not text:
        return ""
    text = text.lower()
    # remove leading numbering like "1.", "2)", "b.", etc.
    text = re.sub(r"^[0-9a-z]+\s*[\.\)\-:]\s*", "", text)
    # keep only letters, numbers, and spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # collapse spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def to_snake_case_from_label(text: str) -> str:
    """
    Turn a label like '2. First name (print)' into 'first_name'.
    This is your Option A: snake_case.
    """
    norm = normalize_label(text)
    if not norm:
        return ""

    # Some special patterns you probably care about
    t = norm

    # Very specific patterns first
    if "medicare" in t and "number" in t:
        return "medicare_number"
    if "social security" in t or "ssn" in t:
        return "ssn"
    if "first name" in t:
        return "first_name"
    if "middle name" in t:
        return "middle_name"
    if "last name" in t or "surname" in t:
        return "last_name"
    if "zip code" in t or ("zip" in t and "code" in t):
        return "zip_code"
    if "postal code" in t:
        return "postal_code"
    if "city" in t and "state" not in t:
        # bare city
        return "city"
    if "state" in t:
        return "state"
    if "phone" in t or "telephone" in t:
        return "phone_number"
    if "email" in t or "e mail" in t:
        return "email_address"
    if "date of birth" in t or "dob" in t:
        return "date_of_birth"
    if "signature" in t and "applicant" in t:
        return "signature_of_applicant"
    if "signature" in t:
        return "signature"
    if "address" in t and "mailing" in t:
        return "mailing_address"
    if "address" in t:
        return "address"

    # Fallback: take first 2–3 words as the base
    words = t.split()
    if not words:
        return ""

    # You can tune how many words to keep
    max_words = 3
    words = words[:max_words]

    base = "_".join(words)
    # final cleanup: keep only a-z0-9 and underscore
    base = re.sub(r"[^a-z0-9_]", "", base)
    return base


def generate_logical_field_name(field_info: dict, used_names: dict) -> str:
    """
    Generate a semantic snake_case name from any available label/context.
    Keeps uniqueness by appending _2, _3, ... when needed.
    Priority:
    1) field_context_on_pdf (TU)
    2) field_context_detected (nearby label)
    3) field_context_all_directions (left/right/top/bottom)
    4) fallback: original widget field_name
    """
    candidates = []

    # Highest priority: built-in label from PDF (/TU)
    if field_info.get("field_context_on_pdf"):
        candidates.append(field_info["field_context_on_pdf"])

    # Next: best detected label
    if field_info.get("field_context_detected"):
        candidates.append(field_info["field_context_detected"])

    # Then: any direction text
    directions = field_info.get("field_context_all_directions") or {}
    for txt in directions.values():
        if txt:
            candidates.append(txt)

    logical_name = ""

    for text in candidates:
        logical_name = to_snake_case_from_label(text)
        if logical_name:
            break

    # Fallback to original field_name_raw (still we try to snake_case it)
    if not logical_name:
        raw = field_info.get("field_name_raw") or ""
        # turn "1-1-1" into "field_1_1_1" instead of empty
        raw_norm = re.sub(r"[^a-z0-9]+", "_", raw.lower()).strip("_")
        logical_name = raw_norm or "field"

    # Ensure uniqueness per-document
    base = logical_name
    if base in used_names:
        used_names[base] += 1
        logical_name = f"{base}_{used_names[base]}"
    else:
        used_names[base] = 1

    return logical_name


# ------------- text extraction helpers ------------- #

def extract_nearby_text(page, widget, direction="left", search_distance=200):
    """
    Extract text near a field widget in a specific direction.

    Args:
        page: PyMuPDF page object
        widget: Widget object
        direction: "left", "right", "top", or "bottom"
        search_distance: How many points to search in the specified direction

    Returns:
        str: Nearby text, cleaned up
    """
    field_rect = widget.rect

    # Define search rectangle based on direction
    if direction == "left":
        # Search to the left of the field
        search_rect = fitz.Rect(
            max(0, field_rect.x0 - search_distance),
            field_rect.y0 - 5,  # Small vertical margin
            field_rect.x0,
            field_rect.y1 + 5
        )
    elif direction == "right":
        # Search to the right of the field
        search_rect = fitz.Rect(
            field_rect.x1,
            field_rect.y0 - 5,
            min(page.rect.width, field_rect.x1 + search_distance),
            field_rect.y1 + 5
        )
    elif direction == "top":
        # Search below the field (PyMuPDF Y increases downward)
        search_rect = fitz.Rect(
            field_rect.x0 - 5,
            max(0, field_rect.y1),
            field_rect.x1 + 5,
            min(page.rect.height, field_rect.y1 + search_distance)
        )
    elif direction == "bottom":
        # Search above the field
        search_rect = fitz.Rect(
            field_rect.x0 - 5,
            max(0, field_rect.y0 - search_distance),
            field_rect.x1 + 5,
            field_rect.y0
        )
    else:
        return ""

    # Extract text from the search area
    try:
        text = page.get_text("text", clip=search_rect)
        # Clean up: remove extra whitespace, newlines
        text = " ".join(text.split())
        return text.strip()
    except Exception:
        return ""


def find_best_label(page, widget):
    """
    Find the best label for a field by searching in all directions and choosing the closest.

    Args:
        page: PyMuPDF page object
        widget: Widget object

    Returns:
        str: Best label found
    """
    field_type = widget.field_type
    field_rect = widget.rect

    # Search in all directions and collect results with distances
    direction_results = []

    directions = ["left", "right", "top", "bottom"]
    search_distances = {
        "left": 200,
        "right": 200,
        "top": 150,
        "bottom": 150
    }

    for direction in directions:
        text = extract_nearby_text(page, widget, direction, search_distances[direction])
        if text and len(text.strip()) > 0:
            # Calculate approximate distance (simplified)
            if direction in ["left", "right"]:
                distance = abs(field_rect.x0 - (
                    field_rect.x0 - search_distances[direction]
                    if direction == "left" else field_rect.x1 + search_distances[direction]
                ))
            else:
                distance = abs(field_rect.y0 - (
                    field_rect.y1 + search_distances[direction]
                    if direction == "top" else field_rect.y0 - search_distances[direction]
                ))

            direction_results.append({
                "text": text.strip(),
                "direction": direction,
                "distance": distance
            })

    if not direction_results:
        return ""

    # For checkboxes and radio buttons, prioritize text to the RIGHT
    if field_type in [fitz.PDF_WIDGET_TYPE_CHECKBOX, fitz.PDF_WIDGET_TYPE_RADIOBUTTON]:
        # Look for right direction first
        right_results = [r for r in direction_results if r["direction"] == "right"]
        if right_results:
            return right_results[0]["text"]

        # Then try left
        left_results = [r for r in direction_results if r["direction"] == "left"]
        if left_results:
            return left_results[0]["text"]

    # For text fields, prioritize LEFT then TOP, but choose closest
    else:
        left_results = [r for r in direction_results if r["direction"] == "left"]
        if left_results:
            closest_left = min(left_results, key=lambda x: x["distance"])
            return closest_left["text"]

        top_results = [r for r in direction_results if r["direction"] == "top"]
        if top_results:
            closest_top = min(top_results, key=lambda x: x["distance"])
            return closest_top["text"]

    # Fallback: return the closest result overall
    closest = min(direction_results, key=lambda x: x["distance"])
    return closest["text"]


# ------------- main extraction per widget ------------- #

def extract_field_info(widget, page, used_names: dict):
    """
    Extract detailed information from a form field widget.

    Args:
        widget: PyMuPDF widget object
        page: PyMuPDF page object (for nearby text extraction)
        used_names: dict used to ensure unique logical names within the doc

    Returns:
        dict: Field information
    """
    # Get built-in label (from /TU entry)
    builtin_label = widget.field_label if hasattr(widget, "field_label") else None

    # Extract context from PDF text
    context_all_directions = {}
    nearby_label = None

    # Always try to detect context (not only when builtin missing)
    directions = ["left", "right", "top", "bottom"]
    for direction in directions:
        text = extract_nearby_text(page, widget, direction, search_distance=200)
        if text and len(text.strip()) > 0:
            context_all_directions[direction] = text.strip()

    # Get best single label
    nearby_label = find_best_label(page, widget)

    # Base info
    field_info = {
        # raw pdf widget name
        "field_name_raw": widget.field_name,

        # we'll fill 'field_name' with semantic snake_case later
        "field_name": None,

        "field_context_on_pdf": builtin_label,       # /TU
        "field_context_detected": nearby_label or None,
        "field_context_all_directions": context_all_directions,
        "field_type": FIELD_TYPE_MAP.get(widget.field_type, "Unknown"),
        "field_type_code": widget.field_type,
        "field_value": widget.field_value,
        "field_flags": widget.field_flags,
        "rect": {
            "x0": round(widget.rect.x0, 2),
            "y0": round(widget.rect.y0, 2),
            "x1": round(widget.rect.x1, 2),
            "y1": round(widget.rect.y1, 2),
            "width": round(widget.rect.width, 2),
            "height": round(widget.rect.height, 2),
        },
    }

    # Type-specific properties
    if widget.field_type == fitz.PDF_WIDGET_TYPE_TEXT:
        field_info["text_format"] = (
            widget.text_format if hasattr(widget, "text_format") else None
        )
        field_info["text_maxlen"] = (
            widget.text_maxlen if hasattr(widget, "text_maxlen") else None
        )

    elif widget.field_type in [fitz.PDF_WIDGET_TYPE_COMBOBOX, fitz.PDF_WIDGET_TYPE_LISTBOX]:
        field_info["choice_values"] = (
            widget.choice_values if hasattr(widget, "choice_values") else None
        )

    elif widget.field_type == fitz.PDF_WIDGET_TYPE_CHECKBOX:
        field_info["is_checked"] = bool(widget.field_value)

    # Basic flags
    field_info["is_readonly"] = bool(widget.field_flags & (1 << 0)) if widget.field_flags else False
    field_info["is_required"] = bool(widget.field_flags & (1 << 1)) if widget.field_flags else False
    field_info["is_no_export"] = bool(widget.field_flags & (1 << 2)) if widget.field_flags else False

    # Button states
    if hasattr(widget, "button_states"):
        field_info["button_states"] = widget.button_states()

    # Now generate semantic logical field_name (snake_case)
    logical_name = generate_logical_field_name(field_info, used_names)
    field_info["field_name"] = logical_name

    return field_info


def extract_fields_from_pdf(pdf_path):
    """
    Extract all form fields from a PDF.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        dict: PDF information and fields
    """
    pdf_info = {
        "filename": pdf_path.name,
        "filepath": str(pdf_path),
        "total_pages": 0,
        "total_fields": 0,
        "pages": [],
    }

    used_names = {}  # track logical names for this doc to keep them unique

    try:
        doc = fitz.open(pdf_path)
        pdf_info["total_pages"] = len(doc)

        # Iterate through all pages
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_info = {
                "page_number": page_num + 1,
                "fields": [],
            }

            # Get all widgets (form fields) on this page
            widgets = page.widgets()

            if widgets:
                for widget in widgets:
                    field_info = extract_field_info(widget, page, used_names)
                    field_info["page"] = page_num + 1
                    page_info["fields"].append(field_info)
                    pdf_info["total_fields"] += 1

            # Only add page if it has fields
            if page_info["fields"]:
                pdf_info["pages"].append(page_info)

        doc.close()

    except Exception as e:
        pdf_info["error"] = str(e)

    return pdf_info


def main():
    """Main function to extract fields from all PDFs."""

    # Check if input directory exists
    if not INPUT_DIR.exists():
        print(f"❌ Input directory '{INPUT_DIR}' not found!")
        return

    # Get all PDF files from input directory
    pdf_files = list(INPUT_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in '{INPUT_DIR}'")
        return

    print(f"Found {len(pdf_files)} PDF file(s) to process...\n")

    total_fields = 0
    success_count = 0

    # Process each PDF file
    for pdf_file in pdf_files:
        print(f"Processing: {pdf_file.name}")

        pdf_data = extract_fields_from_pdf(pdf_file)

        if "error" in pdf_data:
            print(f"  ❌ Error: {pdf_data['error']}")
        else:
            print(
                f"  ✓ Extracted {pdf_data['total_fields']} fields from {pdf_data['total_pages']} pages"
            )
            total_fields += pdf_data["total_fields"]
            success_count += 1

        # Save JSON file for this PDF
        output_file = OUTPUT_DIR / f"{pdf_file.stem}_fields.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(pdf_data, f, indent=2, ensure_ascii=False)

        print(f"  💾 Saved to: {output_file.name}\n")

    # Print summary
    print("=" * 60)
    print("Field extraction complete!")
    print(f"  Total PDFs processed: {success_count}/{len(pdf_files)}")
    print(f"  Total fields extracted: {total_fields}")
    print(f"\nJSON files saved to: {OUTPUT_DIR.absolute()}")


if __name__ == "__main__":
    main()
