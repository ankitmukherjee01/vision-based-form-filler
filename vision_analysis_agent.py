"""
Vision Analysis Agent (Semantic Only)
Analyzes form images to detect fields, extract labels, classify types,
and generate semantic form schema WITHOUT any coordinates.
"""

import os
import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image

load_dotenv()
logger = logging.getLogger(__name__)


class VisionAnalysisAgent:
    """Analyzes form images using Gemini to extract SEMANTIC form schema (no coords)."""

    MAX_IMAGE_PIXELS = 20_000_000
    SUPPORTED_IMAGE_FORMATS = ["PNG", "JPEG", "JPG"]

    DEFAULT_PREFERRED_MODELS = [
        "gemini-1.5-pro-latest",
        "gemini-1.5-flash-latest",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "gemini-pro-vision",
    ]
    DEFAULT_FALLBACK_MODELS = [
        "gemini-1.5-pro-latest",
        "gemini-1.5-flash-latest",
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        check_models: bool = True,
        preferred_models: Optional[List[str]] = None,
        fallback_models: Optional[List[str]] = None,
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found.")

        self.preferred_models = preferred_models or self.DEFAULT_PREFERRED_MODELS
        self.fallback_models = fallback_models or self.DEFAULT_FALLBACK_MODELS

        genai.configure(api_key=self.api_key)

        if check_models:
            try:
                available = [
                    m.name.replace("models/", "")
                    for m in genai.list_models()
                    if "generateContent" in m.supported_generation_methods
                ]

                selected = None
                for m in self.preferred_models:
                    if m in available:
                        selected = m
                        break

                if not selected:
                    gemini_models = [m for m in available if "gemini" in m.lower()]
                    selected = gemini_models[0] if gemini_models else available[0]

                self.model = genai.GenerativeModel(selected)
                logger.info(f"VisionAnalysisAgent using model: {selected}")
            except Exception as e:
                logger.warning(f"Model check failed: {e}, using fallback.")
                self._initialize_fallback_model()
        else:
            self._initialize_fallback_model()

    def _initialize_fallback_model(self):
        last_error = None
        for m in self.fallback_models:
            try:
                self.model = genai.GenerativeModel(m)
                logger.info(f"Fallback Gemini model selected: {m}")
                return
            except Exception as e:
                last_error = e
        raise ValueError(f"Could not initialize any Gemini model: {last_error}")

    def _create_analysis_prompt(self) -> str:
        """
        IMPORTANT: Put your semantic-only prompt here.
        – NO bbox
        – NO widget_rect
        – Only semantic fields: id, label, type, required, options, format_hint, group, depends_on, depends_value, max_length, validation, parts (if you want).
        And return ONLY:
        {
          "fields": [ ... ]
        }
        """
        return """
You are a form analysis expert. Analyze this form image and extract all FORM FIELDS with SEMANTIC properties only.

Do NOT output any coordinates. Do NOT output bbox, widget_rect, x0/y0, or geometry.

For each field, extract:
- id: machine friendly identifier from label (snake_case)
- label: full human-readable label text with context
- type: one of [text, number, checkbox, selection, grouped_choices, date, signature]
- required: true/false (be aggressive)
- options: for radio groups / dropdowns
- format_hint: patterns like XXX-XX-XXXX, MM/DD/YYYY, etc.
- group: one of [name, address, contact, date_range, other] or null
- depends_on: id of controlling field if conditional, else null
- depends_value: value that activates it (e.g. "Yes", true), else null
- max_length: estimated characters or null
- validation: one of [numeric, alphabetic, alphanumeric, email, phone, ssn, date] or null
- parts: null OR semantic multi-part info (if clearly multi-part), but WITHOUT coordinates.

Return ONLY this JSON:

{
  "fields": [
    {
      "id": "field_identifier_made_from_label",
      "label": "Complete label text",
      "type": "text|number|checkbox|selection|grouped_choices|date|signature",
      "required": true|false,
      "options": ["opt1","opt2"] or null,
      "format_hint": "XXX-XX-XXXX" or null,
      "group": "name|address|contact|date_range|other" or null,
      "depends_on": "field_id" or null,
      "depends_value": "Yes|No|true|false|value" or null,
      "max_length": 50 or null,
      "validation": "numeric|alphabetic|alphanumeric|email|phone|ssn|date" or null,
      "parts": null
    }
  ]
}

Return ONLY valid JSON. No explanation, no markdown, no comments.
"""

    def _post_process_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure basic structure and defaults of the semantic schema."""
        if not isinstance(schema, dict):
            schema = {"fields": []}
        fields = schema.get("fields", [])
        if not isinstance(fields, list):
            fields = []
        schema["fields"] = fields

        for f in fields:
            f.setdefault("id", "")
            f.setdefault("label", "")
            f.setdefault("type", "text")
            f.setdefault("required", False)
            f.setdefault("options", None)
            f.setdefault("format_hint", None)
            f.setdefault("group", None)
            f.setdefault("depends_on", None)
            f.setdefault("depends_value", None)
            f.setdefault("max_length", None)
            f.setdefault("validation", None)
            f.setdefault("parts", None)
        return schema

    def analyze_form_image(self, image_path: Path) -> Dict[str, Any]:
        """Analyze a single image and return semantic schema."""
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        logger.info(f"Analyzing image: {image_path}")

        image = Image.open(image_path)
        if image.format not in self.SUPPORTED_IMAGE_FORMATS:
            raise ValueError(f"Unsupported image format: {image.format}")

        if image.mode != "RGB":
            image = image.convert("RGB")

        prompt = self._create_analysis_prompt()
        response = self.model.generate_content([prompt, image])

        text = response.text.strip()
        # strip markdown code fences if present
        text = re.sub(r"```(?:json)?\s*([\s\S]*?)```", r"\1", text).strip()

        try:
            schema = json.loads(text)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error from Gemini: {e}")
            logger.error(f"Raw response (truncated): {text[:500]}")
            raise

        schema = self._post_process_schema(schema)
        return schema

    def analyze_form_pages(self, form_name: str, image_paths: List[Path]) -> Dict[str, Any]:
        """
        Analyze multiple page images of the same form.
        Returns a single combined semantic schema:
        {
          "form_name": ...,
          "total_pages": N,
          "fields": [...],
          "page_results": [...]
        }
        """
        logger.info(f"Analyzing form '{form_name}' with {len(image_paths)} page(s)")
        all_fields: List[Dict[str, Any]] = []
        page_results: List[Dict[str, Any]] = []

        for idx, img_path in enumerate(sorted(image_paths), start=1):
            try:
                page_schema = self.analyze_form_image(img_path)
                fields = page_schema.get("fields", [])

                # tag fields with page index (semantic only; no coords)
                for f in fields:
                    f["page"] = idx
                    all_fields.append(f)

                page_results.append({
                    "page": idx,
                    "status": "success",
                    "fields": len(fields)
                })
                logger.info(f"Page {idx}: {len(fields)} semantic field(s)")
            except Exception as e:
                logger.error(f"Failed to analyze page {idx}: {e}")
                page_results.append({
                    "page": idx,
                    "status": "failed",
                    "error": str(e)
                })

        return {
            "form_name": form_name,
            "total_pages": len(image_paths),
            "fields": all_fields,
            "page_results": page_results
        }

    def save_schema(self, schema: Dict[str, Any], output_path: Path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(schema, f, indent=2, ensure_ascii=False)
        logger.info(f"Semantic schema saved: {output_path}")


def main(
    images_root: str = "output/images",
    semantic_dir: str = "3_semantic_schemas",
    image_pattern: str = "page_*.png"
):
    logger.info("=" * 60)
    logger.info("Vision Semantic Analysis Agent - START")
    logger.info("=" * 60)

    try:
        agent = VisionAnalysisAgent()
    except Exception as e:
        logger.error(f"Failed to init VisionAnalysisAgent: {e}")
        return

    images_root = Path(images_root)
    semantic_dir = Path(semantic_dir)
    semantic_dir.mkdir(parents=True, exist_ok=True)

    if not images_root.exists():
        logger.error(f"Images root not found: {images_root}")
        return

    form_dirs = [d for d in images_root.iterdir() if d.is_dir()]
    if not form_dirs:
        logger.warning(f"No form dirs found under {images_root}")
        return

    for form_dir in form_dirs:
        form_name = form_dir.name
        image_paths = sorted(form_dir.glob(image_pattern))
        if not image_paths:
            logger.warning(f"No page images found under {form_dir}")
            continue

        try:
            semantic_schema = agent.analyze_form_pages(form_name, image_paths)
            out_path = semantic_dir / f"{form_name}_semantic.json"
            agent.save_schema(semantic_schema, out_path)
        except Exception as e:
            logger.error(f"Failed analyzing form {form_name}: {e}")

    logger.info("Vision Semantic Analysis - DONE")


if __name__ == "__main__":
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("project.log", encoding="utf-8"),
                logging.StreamHandler()
            ]
        )
    main()
