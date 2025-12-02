"""
- Uses semantic labels + semantic snake_case field_name
- Uses widget field_name (semantic) + field_name_raw (PDF name)
- Primary merge key = exact semantic field_name match
- Fallback = label similarity match
"""

import json
from pathlib import Path
from difflib import SequenceMatcher
import re


BASE_DIR = Path(__file__).parent.resolve()

SEMANTIC_DIR = BASE_DIR / "3_semantic_schemas"
WIDGET_DIR = BASE_DIR / "2_fields_json"
OUTPUT_DIR = BASE_DIR / "4_merged_schema"


# ----------------- helpers ----------------- #

def log(msg: str):
    print(f"[MERGER] {msg}")


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def similarity(a: str, b: str) -> float:
    a_norm = normalize_text(a)
    b_norm = normalize_text(b)
    if not a_norm or not b_norm:
        return 0.0
    return SequenceMatcher(None, a_norm, b_norm).ratio()


# ----------------- load data ----------------- #

def load_semantic_fields(form_name: str):
    path = SEMANTIC_DIR / f"{form_name}_semantic.json"
    if not path.exists():
        raise FileNotFoundError(f"Semantic file not found: {path}")

    log(f"Loading semantic schema: {path.name}")
    with open(path, "r", encoding="utf-8") as f:
        root = json.load(f)

    return root.get("fields", []), root


def load_widget_fields(form_name: str):
    path = WIDGET_DIR / f"{form_name}_fields.json"
    if not path.exists():
        raise FileNotFoundError(f"Widget fields file not found: {path}")

    log(f"Loading widget schema: {path.name}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    widgets = []
    for page in data.get("pages", []):
        page_num = page.get("page_number")

        for w in page.get("fields", []):
            widgets.append({
                "page": page_num,
                "field_name": w.get("field_name"),             # semantic name (snake_case)
                "field_name_raw": w.get("field_name_raw"),     # original PDF name "1-1-1"
                "context_detected": w.get("field_context_detected") or "",
                "context_on_pdf": w.get("field_context_on_pdf") or "",
                "rect": w.get("rect") or {},
                "raw": w,
            })

    return widgets, data


# ----------------- NEW MATCHING LOGIC ----------------- #

def choose_widget_label_candidate(w):
    return w.get("context_detected") or w.get("context_on_pdf") or ""


def find_best_widget(sf, widgets, used_indices, threshold=0.45):
    """
    v3 Matching priority:

    1. Semantic field_name == widget field_name  (Perfect match)
    2. Same page + fuzzy label match
    3. Any page + fuzzy label match
    """

    sf_name = sf.get("id", "")
    sf_label = sf.get("label", "")
    sf_page = sf.get("page")

    # ---------- EXACT semantic name match ---------- #
    for i, w in enumerate(widgets):
        if i in used_indices:
            continue

        if sf_name and w.get("field_name") == sf_name:
            return i, 1.0  # perfect match

    # ---------- SAME PAGE + LABEL SIMILARITY ---------- #
    best_i = None
    best_score = 0

    for i, w in enumerate(widgets):
        if i in used_indices:
            continue

        if sf_page and w["page"] != sf_page:
            continue

        score = similarity(sf_label, choose_widget_label_candidate(w))
        if score > best_score:
            best_score = score
            best_i = i

    if best_i is not None and best_score >= threshold:
        return best_i, best_score

    # ---------- ANY PAGE (fallback) ---------- #
    for i, w in enumerate(widgets):
        if i in used_indices:
            continue

        score = similarity(sf_label, choose_widget_label_candidate(w))
        if score > best_score:
            best_score = score
            best_i = i

    if best_i is not None and best_score >= threshold:
        return best_i, best_score

    return None, best_score


# ----------------- merge ----------------- #

def merge_form(form_name: str):
    semantic_fields, semantic_root = load_semantic_fields(form_name)
    widgets, widget_root = load_widget_fields(form_name)

    used = set()
    merged = []

    log(f"Merging {form_name}: {len(semantic_fields)} semantic fields, {len(widgets)} widgets")

    for sf in semantic_fields:
        idx, score = find_best_widget(sf, widgets, used)

        if idx is not None:
            w = widgets[idx]
            used.add(idx)

            merged.append({
                # --- semantic part ---
                "id": sf.get("id"),
                "label": sf.get("label"),
                "type": sf.get("type"),
                "required": sf.get("required", False),
                "options": sf.get("options"),
                "format_hint": sf.get("format_hint"),
                "group": sf.get("group"),
                "depends_on": sf.get("depends_on"),
                "depends_value": sf.get("depends_value"),
                "max_length": sf.get("max_length"),
                "validation": sf.get("validation"),
                "parts": sf.get("parts"),

                # --- widget mapping ---
                "widget_field_name": w.get("field_name"),           # semantic widget name
                "widget_pdf_name": w.get("field_name_raw"),         # important for filler agent
                "widget_rect": w.get("rect"),
                "page": w.get("page"),

                # --- debug ---
                "_match_score": round(score, 3),
                "_widget_label": choose_widget_label_candidate(w),
            })

        else:
            # Keep semantic-only field
            sf["widget_field_name"] = None
            sf["widget_pdf_name"] = None
            sf["widget_rect"] = None
            sf["_match_score"] = 0
            merged.append(sf)

    # write
    OUTPUT_DIR.mkdir(exist_ok=True)
    out = OUTPUT_DIR / f"{form_name}_merged.json"

    final = {
        "form_name": semantic_root.get("form_name", form_name),
        "total_pages": widget_root.get("total_pages"),
        "fields": merged,
        "page_results": semantic_root.get("page_results"),
    }

    with open(out, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    log(f"✓ Saved merged schema: {out}")


def main():
    semantic_files = list(SEMANTIC_DIR.glob("*_semantic.json"))
    if not semantic_files:
        log("No semantic schemas found.")
        return

    for f in semantic_files:
        form_name = f.stem.replace("_semantic", "")
        log(f"—— Merging {form_name} ——")
        merge_form(form_name)


if __name__ == "__main__":
    main()
