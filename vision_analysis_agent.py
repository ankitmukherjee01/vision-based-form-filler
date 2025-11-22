"""
Vision Analysis Agent (Micro-Agent 1 - Form Analyzer)
Analyzes form images to detect fields, extract labels, classify types, and generate form schema.
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

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)


class VisionAnalysisAgent:
    """Analyzes form images using Gemini 1.5 Pro Vision to extract form schema."""
    
    # Configuration constants
    MERGE_Y_THRESHOLD = 20  # Pixels for same-row detection
    MERGE_X_GAP_THRESHOLD = 50  # Pixels for horizontal proximity
    MAX_IMAGE_PIXELS = 20_000_000  # 20MP limit for image size
    SUPPORTED_IMAGE_FORMATS = ['PNG', 'JPEG', 'JPG']
    
    # Model configuration
    DEFAULT_PREFERRED_MODELS = [
        'gemini-1.5-pro-latest',
        'gemini-1.5-flash-latest',
        'gemini-1.5-pro',
        'gemini-1.5-flash',
        'gemini-pro-vision'
    ]
    DEFAULT_FALLBACK_MODELS = [
        'gemini-1.5-pro-latest',
        'gemini-1.5-flash-latest'
    ]
    
    # Multi-part field detection keywords (used for spatial merging logic)
    MULTI_PART_KEYWORDS = [
        "medicare", "ssn", "social security", "phone", "telephone",
        "date", "dob", "birth", "zip", "postal"
    ]
    
    # Field type constants
    FIELD_TYPES = {
        "text", "number", "checkbox", "selection", 
        "grouped_choices", "date", "signature"
    }
    
    # Validation types
    VALIDATION_TYPES = {
        "numeric", "alphabetic", "alphanumeric", 
        "email", "phone", "ssn", "date"
    }
    
    # Group names
    GROUP_NAMES = {
        "name", "address", "contact", "date_range"
    }
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        merge_y_threshold: int = None,
        merge_x_gap_threshold: int = None,
        check_models: bool = True,
        preferred_models: Optional[List[str]] = None,
        fallback_models: Optional[List[str]] = None
    ):
        """
        Initialize the Vision Analysis Agent.
        
        Args:
            api_key: Gemini API key (defaults to GEMINI_API_KEY from .env)
            merge_y_threshold: Y-coordinate threshold for same-row detection (default: 20)
            merge_x_gap_threshold: X-coordinate gap threshold for horizontal proximity (default: 50)
            check_models: Whether to check available models on init (default: True)
            preferred_models: List of preferred model names in order (default: class constant)
            fallback_models: List of fallback model names to try (default: class constant)
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. Please set it in .env file or pass as argument."
            )
        
        # Set configurable thresholds
        self.merge_y_threshold = merge_y_threshold or self.MERGE_Y_THRESHOLD
        self.merge_x_gap_threshold = merge_x_gap_threshold or self.MERGE_X_GAP_THRESHOLD
        
        # Set model preferences
        self.preferred_models = preferred_models or self.DEFAULT_PREFERRED_MODELS
        self.fallback_models = fallback_models or self.DEFAULT_FALLBACK_MODELS
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        
        # Get available models and find the best one for vision
        if check_models:
            try:
                available_models = [
                    m.name.replace('models/', '') 
                    for m in genai.list_models() 
                    if 'generateContent' in m.supported_generation_methods
                ]
                logger.debug(f"Available models: {available_models}")
                
                # Find the first available preferred model
                selected_model = None
                for preferred in self.preferred_models:
                    if preferred in available_models:
                        selected_model = preferred
                        break
                
                # If no preferred model found, use the first available model with 'gemini' in name
                if not selected_model:
                    gemini_models = [m for m in available_models if 'gemini' in m.lower()]
                    if gemini_models:
                        selected_model = gemini_models[0]
                    else:
                        selected_model = available_models[0] if available_models else None
                
                if not selected_model:
                    raise ValueError("No available Gemini models found")
                
                self.model = genai.GenerativeModel(selected_model)
                logger.info(f"Vision Analysis Agent initialized with {selected_model}")
                
            except Exception as e:
                logger.warning(f"Error checking available models: {e}. Using fallback.")
                self._initialize_fallback_model()
        else:
            # Skip model checking, use fallback directly
            self._initialize_fallback_model()
    
    def _initialize_fallback_model(self):
        """Initialize model using fallback method."""
        last_error = None
        for fallback_model in self.fallback_models:
            try:
                self.model = genai.GenerativeModel(fallback_model)
                logger.info(f"Vision Analysis Agent initialized with {fallback_model} (fallback)")
                return
            except Exception as e:
                last_error = e
                continue
        
        raise ValueError(
            f"Could not initialize any Gemini model. Last error: {last_error}. "
            "Please check your API key and model availability."
        )
    
    def _create_analysis_prompt(self) -> str:
        """
        Create the enhanced prompt for form analysis.
        
        Returns:
            Analysis prompt string
        """
        return """You are a form analysis expert. Analyze this form image and extract all form fields with comprehensive properties.

CRITICAL REQUIREMENTS - You must extract ALL of the following for each field:

1. FIELD DETECTION: Text fields, checkboxes, radio groups, signature fields, tables, dropdowns
2. LABEL EXTRACTION: The complete text label/question near each field with full context
3. FIELD TYPE: text, number, checkbox, selection, grouped_choices, date, signature
4. BOUNDING BOX: Accurate pixel coordinates [x1, y1, x2, y2] where (x1,y1) is top-left and (x2,y2) is bottom-right
   - CRITICAL: All coordinates MUST be relative to the FULL ORIGINAL IMAGE dimensions
   - The image you receive is the complete original image. Report coordinates relative to this full image.
   - If the image appears cropped or resized in your view, you must still report coordinates relative to the original full image dimensions.
   - All x coordinates must be between 0 and the image width, all y coordinates must be between 0 and the image height.
5. REQUIRED STATUS: You MUST mark as required if you see:
   - Asterisks (*) next to the field
   - "Required" text anywhere near the field
   - Fields that are clearly mandatory: signatures, dates signed, names, SSN, Medicare numbers, IDs
   - Fields marked with "must", "shall", or similar mandatory language
   - When in doubt, mark as required - be aggressive with required field detection
6. OPTIONS: For grouped_choices and selection fields, you MUST extract ALL visible options:
   - Read every option text carefully
   - Include ALL options in the array (e.g., ["Yes", "No"], ["Single", "Married", "Divorced", "Widowed"])
   - Do NOT use default options - extract what you actually see
   - If you cannot see all options, include what you can see
7. FORMAT HINTS: You MUST detect and include format patterns:
   - SSN/Social Security: "XXX-XX-XXXX" (always 11 characters: 3-2-4 pattern)
   - Phone/Telephone: "(XXX) XXX-XXXX" or "XXX-XXX-XXXX" (10 digits)
   - Date: "MM/DD/YYYY" or "MM/YYYY" (check the form for the exact format)
   - ZIP/Postal Code: "XXXXX" or "XXXXX-XXXX" (5 or 9 digits)
   - Medicare Number: Typically "XXX-XX-XXXX" (11 characters: 3-2-4 pattern) or check the form
   - Include format_hint for ANY field that has visible formatting guides or patterns
   - CRITICAL: For multi-part fields, INFER the format from the structure even if dashes/separators aren't visible:
     * 3-part field with ~11 total characters = "XXX-XX-XXXX" (Medicare/SSN pattern)
     * 3-part field with ~10 total characters = "XXX-XXX-XXXX" (phone pattern)
     * 2-part date field = "MM/YYYY" or "MM/DD" (check context)
     * 3-part date field = "MM/DD/YYYY"
     * 5-part field with ~5 total characters = "XXXXX" (ZIP code)
   - When you detect a multi-part field, you MUST infer and include the format_hint based on the number of parts and typical patterns
8. FIELD GROUPING: You MUST identify related fields and assign them to groups:
   - Name fields (first, middle, last, suffix): group = "name"
   - Address fields (street, city, state, zip): group = "address"
   - Date ranges (start date, end date): group = "date_range"
   - Contact fields (phone, email): group = "contact"
   - If fields are clearly related but don't fit above categories, use group = "other"
   - Standalone fields should have group = null
9. CONDITIONAL LOGIC: You MUST identify conditional fields:
   - Look for phrases like "If yes", "If no", "If checked", "If applicable", "When", "Complete if"
   - Identify the controlling field (the Yes/No question or checkbox that controls this field)
   - Set "depends_on" to the ID of the controlling field
   - Set "depends_value" to the value that triggers this field ("Yes", "No", true for checkboxes, or the specific option value)
   - If a field appears only when another field has a specific value, mark it as conditional
10. MAX LENGTH: Estimate character limits based on visible field size and any visible constraints
11. VALIDATION RULES: Detect and include validation patterns:
   - "numbers only" or numeric-only fields: validation = "numeric"
   - "letters only" or alphabetic-only fields: validation = "alphabetic"
   - Alphanumeric fields: validation = "alphanumeric"
   - Email fields: validation = "email"
   - Phone fields: validation = "phone"
   - SSN fields: validation = "ssn"
   - Date fields: validation = "date"

FIELD TYPE GUIDELINES:
- Checkboxes: type = "checkbox"
- Radio buttons: type = "grouped_choices" (MUST include complete "options" array)
- Dropdowns: type = "selection" (include "options" array if visible)
- Date fields: type = "date" (MUST include format in "format_hint")
- Signature fields: type = "signature"
- Numeric-only fields: type = "number"
- Text fields: type = "text"
- SSN, phone, zip can be "number" or "text" depending on whether they accept formatting

MULTI-PART FIELDS:
If a single label has multiple input boxes (e.g., Medicare Number split into 3 boxes, SSN split into XXX-XX-XXXX, Date split into MM/DD/YYYY), create ONE field entry with:
- A single "id" and "label"
- "bbox" should be the combined/overall bounding box covering all parts
- "parts" array containing objects with "bbox" and "part_index" for each individual box
- Example: {"id": "medicare_number", "label": "Medicare Number", "bbox": [combined], "parts": [{"bbox": [box1], "part_index": 1}, {"bbox": [box2], "part_index": 2}, {"bbox": [box3], "part_index": 3}]}

OTHER GUIDELINES:
- Bounding boxes must be precise pixel coordinates
- Extract full labels with context (e.g., "1. Medicare Number" not just "Medicare Number")
- For tables, extract each fillable cell as a separate field
- Use descriptive IDs based on the label (e.g., "first_name", "date_of_birth", "ssn", "zip_code")

Return ONLY a valid JSON object with this exact structure:
{
  "fields": [
    {
      "id": "field_identifier_made_from_label",
      "label": "Complete extracted label text with context",
      "type": "text|number|checkbox|selection|grouped_choices|date|signature",
      "bbox": [x1, y1, x2, y2],
      "required": true|false,
      "options": ["option1", "option2"] (ONLY for grouped_choices and selection, null otherwise),
      "format_hint": "XXX-XX-XXXX" (for formatted fields, null otherwise),
      "group": "name|address|contact|date_range|other" (for related fields, null if standalone),
      "depends_on": "field_id" (if conditional, null otherwise),
      "depends_value": "Yes|No|value|true" (the value that triggers this field, null otherwise),
      "max_length": 50 (estimated character limit, null if unknown),
      "validation": "numeric|alphabetic|alphanumeric|email|phone|ssn|date" (null if none),
      "parts": [{"bbox": [x1, y1, x2, y2], "part_index": 1}, ...] (only for multi-part fields, null otherwise)
    }
  ]
}

IMPORTANT: 
- Extract ALL options for grouped_choices and selection fields - do not use defaults
- Identify ALL conditional relationships - look carefully for "if", "when", "complete if" phrases
- Assign groups to related fields - be thorough with grouping
- Include format_hint for ANY field with visible formatting OR for multi-part fields (infer from structure)
- For multi-part fields, ALWAYS infer and include format_hint based on number of parts and typical patterns
- Be aggressive with required field detection
- Return ONLY the JSON, no additional text or explanation."""
    
    def _post_process_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Post-process schema: validate structure and ensure consistent defaults.
        Trusts the agent's output for semantic decisions (groups, formats, required, conditionals).
        
        Args:
            schema: Raw schema from Gemini
            
        Returns:
            Validated schema with consistent structure
        """
        # Validate schema structure
        if not isinstance(schema, dict):
            logger.warning("Schema is not a dictionary. Creating new schema.")
            schema = {"fields": []}
        
        if "fields" not in schema:
            logger.warning("Schema missing 'fields' key. Adding empty fields array.")
            schema["fields"] = []
        
        fields = schema.get("fields", [])
        
        if not isinstance(fields, list):
            logger.warning("Fields is not a list. Converting to list.")
            fields = []
            schema["fields"] = fields
        
        # Only ensure consistent structure - trust agent's semantic decisions
        for field in fields:
            # Ensure all optional fields exist with None defaults for consistency
            # This ensures the schema structure is always consistent
            field.setdefault("options", None)
            field.setdefault("format_hint", None)
            field.setdefault("group", None)
            field.setdefault("depends_on", None)
            field.setdefault("depends_value", None)
            field.setdefault("max_length", None)
            field.setdefault("validation", None)
            field.setdefault("parts", None)
            field.setdefault("required", False)  # Default to False if not specified
        
        # Remove internal metadata (_image_info) from schema - it's extracted separately
        if "_image_info" in schema:
            del schema["_image_info"]
        
        # Note: Multi-part field merging is done later in analyze_form_pages
        # after page numbers are assigned
        
        return schema
    
    def _merge_multipart_fields(self, fields: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect and merge fields that appear to be parts of a multi-part field.
        
        Args:
            fields: List of field dictionaries
            
        Returns:
            Updated fields with multi-part fields merged
        """
        # Group fields by page for processing
        fields_by_page = {}
        for field in fields:
            page = field.get("page", 1)
            if page not in fields_by_page:
                fields_by_page[page] = []
            fields_by_page[page].append(field)
        
        merged_fields = []
        processed_ids = set()
        
        for page, page_fields in fields_by_page.items():
            # Sort fields by y-coordinate (top to bottom), then x-coordinate (left to right)
            sorted_fields = sorted(
                page_fields,
                key=lambda f: (f.get("bbox", [0, 0, 0, 0])[1], f.get("bbox", [0, 0, 0, 0])[0])
            )
            
            i = 0
            while i < len(sorted_fields):
                field = sorted_fields[i]
                
                if field.get("id") in processed_ids:
                    i += 1
                    continue
                
                # Check if this field might be part of a multi-part field
                field_id = field.get("id", "").lower()
                field_label = field.get("label", "").lower()
                
                # Look for fields with similar labels/IDs that are close together
                potential_parts = [field]
                
                # Check for common multi-part patterns
                is_multipart_candidate = any(
                    kw in field_id or kw in field_label 
                    for kw in self.MULTI_PART_KEYWORDS
                )
                
                if is_multipart_candidate:
                    # Look for nearby fields with similar characteristics
                    field_bbox = field.get("bbox", [])
                    if len(field_bbox) == 4:
                        field_y = field_bbox[1]  # Top y-coordinate
                        field_x_end = field_bbox[2]  # Right x-coordinate
                        
                        # Check fields to the right (within same row, close horizontally)
                        for j in range(i + 1, len(sorted_fields)):
                            other_field = sorted_fields[j]
                            other_bbox = other_field.get("bbox", [])
                            
                            if len(other_bbox) != 4:
                                continue
                            
                            other_y = other_bbox[1]
                            other_x = other_bbox[0]  # Left x-coordinate
                            
                            # Check if fields are on the same row (similar y-coordinates)
                            y_diff = abs(other_y - field_y)
                            x_gap = other_x - field_x_end
                            
                            # Same row (within threshold) and close horizontally (within threshold)
                            if y_diff < self.merge_y_threshold and 0 < x_gap < self.merge_x_gap_threshold:
                                # Check if they have similar labels/IDs
                                other_id = other_field.get("id", "").lower()
                                other_label = other_field.get("label", "").lower()
                                
                                # Similar field characteristics
                                if (field_id.split("_")[0] == other_id.split("_")[0] or
                                    any(kw in other_id or kw in other_label for kw in self.MULTI_PART_KEYWORDS)):
                                    potential_parts.append(other_field)
                                    field_x_end = other_bbox[2]  # Update for next iteration
                                else:
                                    break
                            elif y_diff > self.merge_y_threshold:  # Different row, stop looking
                                break
                
                # If we found multiple parts, merge them
                if len(potential_parts) > 1:
                    # Create merged field
                    merged_field = potential_parts[0].copy()
                    
                    # Calculate combined bounding box
                    all_bboxes = [f.get("bbox", []) for f in potential_parts if len(f.get("bbox", [])) == 4]
                    if all_bboxes:
                        min_x = min(b[0] for b in all_bboxes)
                        min_y = min(b[1] for b in all_bboxes)
                        max_x = max(b[2] for b in all_bboxes)
                        max_y = max(b[3] for b in all_bboxes)
                        merged_field["bbox"] = [min_x, min_y, max_x, max_y]
                    
                    # Create parts array
                    merged_field["parts"] = [
                        {
                            "bbox": f.get("bbox"),
                            "part_index": idx + 1
                        }
                        for idx, f in enumerate(potential_parts)
                    ]
                    
                    merged_fields.append(merged_field)
                    
                    # Mark all parts as processed
                    for part in potential_parts:
                        processed_ids.add(part.get("id"))
                    
                    logger.debug(
                        f"Merged {len(potential_parts)} fields into multi-part field: {merged_field.get('id')}"
                    )
                else:
                    # Single field, add as-is
                    merged_fields.append(field)
                    processed_ids.add(field.get("id"))
                
                i += 1
        
        # Add any fields that weren't processed (shouldn't happen, but safety check)
        for field in fields:
            if field.get("id") not in processed_ids:
                merged_fields.append(field)
                processed_ids.add(field.get("id"))  # Mark as processed to prevent duplicates
        
        return merged_fields
    
    def analyze_form_image(self, image_path: Path) -> Dict[str, Any]:
        """
        Analyze a single form image and extract form schema.
        
        Args:
            image_path: Path to the form image
            
        Returns:
            Dictionary containing the form schema
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        logger.info(f"Analyzing form image: {image_path.name}")
        
        try:
            # Load and validate image
            image = Image.open(image_path)
            
            # Validate image format
            if image.format not in self.SUPPORTED_IMAGE_FORMATS:
                raise ValueError(
                    f"Unsupported image format: {image.format}. "
                    f"Supported formats: {', '.join(self.SUPPORTED_IMAGE_FORMATS)}"
                )
            
            # Validate image size
            image_pixels = image.size[0] * image.size[1]
            if image_pixels > self.MAX_IMAGE_PIXELS:
                logger.warning(
                    f"Large image detected: {image.size} ({image_pixels:,} pixels). "
                    f"Consider resizing for better performance. Max recommended: {self.MAX_IMAGE_PIXELS:,}"
                )
            
            # Convert to RGB if necessary (some formats like PNG with transparency)
            if image.mode != 'RGB':
                logger.debug(f"Converting image from {image.mode} to RGB")
                image = image.convert('RGB')
            
            # Log image dimensions being sent to Gemini
            original_width, original_height = image.size
            logger.info(f"Image dimensions sent to Gemini: {original_width}x{original_height} pixels")
            
            # Create prompt
            prompt = self._create_analysis_prompt()
            
            # Call Gemini Vision API
            logger.info("Sending image to Gemini 1.5 Pro for analysis...")
            response = self.model.generate_content([prompt, image])
            
            # Extract JSON from response
            response_text = response.text.strip()
            
            # Clean up response (remove markdown code blocks more robustly)
            # Remove markdown code blocks using regex
            response_text = re.sub(
                r'```(?:json)?\s*\n?(.*?)\n?```',
                r'\1',
                response_text,
                flags=re.DOTALL
            ).strip()
            
            # Remove any leading/trailing whitespace and newlines
            response_text = response_text.strip()
            
            # Parse JSON to analyze bounding boxes and infer processed dimensions
            try:
                parsed_response = json.loads(response_text)
                fields = parsed_response.get("fields", [])
                
                # Initialize min/max coordinates for crop detection
                min_x = min_y = float('inf')
                max_x = max_y = 0
                all_coords = []

                if not fields:
                    # No fields extracted, can't infer dimensions
                    logger.warning("No fields in response, cannot infer processed dimensions")
                    processed_width = original_width
                    processed_height = original_height
                    crop_region = None
                else:
                    # First pass: Collect all coordinates to check for normalization
                    for field in fields:
                        if not isinstance(field, dict): continue
                        bbox = field.get("bbox", [])
                        if bbox and len(bbox) >= 4:
                            all_coords.extend(bbox)
                        parts = field.get("parts")
                        if parts and isinstance(parts, list):
                            for part in parts:
                                if isinstance(part, dict) and part.get("bbox"):
                                    all_coords.extend(part.get("bbox"))

                    # Detect if coordinates are normalized (0-1000)
                    is_normalized = False
                    if all_coords:
                        max_coord_val = max(all_coords)
                        # If all coords are <= 1000 but image is significantly larger, assuming normalized
                        if max_coord_val <= 1000 and (original_width > 1000 or original_height > 1000):
                            logger.info(f"Detected normalized coordinates (0-1000) in response (max val: {max_coord_val}). Scaling to image dimensions ({original_width}x{original_height}).")
                            is_normalized = True
                            scale_x = original_width / 1000.0
                            scale_y = original_height / 1000.0
                            
                            # Fix coordinates in place
                            for field in fields:
                                if "bbox" in field and field["bbox"]:
                                    b = field["bbox"]
                                    field["bbox"] = [int(b[0]*scale_x), int(b[1]*scale_y), int(b[2]*scale_x), int(b[3]*scale_y)]
                                if "parts" in field and field["parts"]:
                                    for part in field["parts"]:
                                        if "bbox" in part and part["bbox"]:
                                            b = part["bbox"]
                                            part["bbox"] = [int(b[0]*scale_x), int(b[1]*scale_y), int(b[2]*scale_x), int(b[3]*scale_y)]

                    # Second pass: Find min/max bounding box coordinates to detect crop region (using fixed coords)
                    for field in fields:
                        bbox = field.get("bbox", [])
                        if bbox and len(bbox) >= 4:
                            min_x = min(min_x, bbox[0])
                            min_y = min(min_y, bbox[1])
                            max_x = max(max_x, bbox[2])
                            max_y = max(max_y, bbox[3])
                        
                        parts = field.get("parts")
                        if parts and isinstance(parts, list):
                            for part in parts:
                                part_bbox = part.get("bbox", [])
                                if part_bbox and len(part_bbox) >= 4:
                                    min_x = min(min_x, part_bbox[0])
                                    min_y = min(min_y, part_bbox[1])
                                    max_x = max(max_x, part_bbox[2])
                                    max_y = max(max_y, part_bbox[3])
                    
                    # Detect if cropping occurred
                    if max_x > 0 and max_y > 0 and min_x != float('inf'):
                        # Add a small margin (5%)
                        crop_right = int(max_x * 1.05)
                        crop_bottom = int(max_y * 1.05)
                        crop_left = max(0, int(min_x * 0.95))
                        crop_top = max(0, int(min_y * 0.95))
                        
                        crop_width = crop_right - crop_left
                        crop_height = crop_bottom - crop_top
                        
                        width_ratio = crop_width / original_width if original_width > 0 else 1.0
                        height_ratio = crop_height / original_height if original_height > 0 else 1.0
                        
                        original_aspect = original_width / original_height if original_height > 0 else 1.0
                        crop_aspect = crop_width / crop_height if crop_height > 0 else 1.0
                        aspect_diff = abs(original_aspect - crop_aspect)
                        
                        # Detect cropping: Removed the logic checking (crop_left > 10) as that triggers false positives on full pages with margins
                        is_cropped = (
                            (width_ratio < 0.8 or height_ratio < 0.8) or  # Significant size reduction (relaxed to 0.8)
                            (aspect_diff > 0.2) # Significant Aspect ratio mismatch
                        )
                        
                        if is_cropped:
                            crop_region = {
                                "offset_x": crop_left,
                                "offset_y": crop_top,
                                "width": crop_width,
                                "height": crop_height,
                                "detected": True,
                                "aspect_ratio_original": round(original_aspect, 3),
                                "aspect_ratio_crop": round(crop_aspect, 3),
                                "size_ratio_width": round(width_ratio, 3),
                                "size_ratio_height": round(height_ratio, 3)
                            }
                            processed_width = crop_width
                            processed_height = crop_height
                            
                            logger.warning(
                                f"CROP DETECTED! Original: {original_width}x{original_height} "
                                f"(aspect {original_aspect:.3f}), "
                                f"Crop region: {crop_left},{crop_top} to {crop_right},{crop_bottom} "
                                f"({crop_width}x{crop_height}, aspect {crop_aspect:.3f})"
                            )
                        else:
                            crop_region = {
                                "offset_x": 0,
                                "offset_y": 0,
                                "width": original_width,
                                "height": original_height,
                                "detected": False
                            }
                            processed_width = original_width
                            processed_height = original_height
                            logger.debug(f"Bounding boxes match original dimensions - no cropping detected")
                    else:
                        crop_region = {
                            "offset_x": 0,
                            "offset_y": 0,
                            "width": original_width,
                            "height": original_height,
                            "detected": False
                        }
                        processed_width = original_width
                        processed_height = original_height
                        logger.warning("Could not infer processed dimensions from bounding boxes, using original")
                
                # Store info
                parsed_response["_image_info"] = {
                    "original_dimensions": {"width": original_width, "height": original_height},
                    "processed_dimensions": {"width": processed_width, "height": processed_height},
                    "crop_region": crop_region if 'crop_region' in locals() else {
                        "offset_x": 0, "offset_y": 0, "width": original_width, "height": original_height, "detected": False
                    },
                    "bbox_range": {
                        "min_x": int(min_x) if min_x != float('inf') else 0,
                        "min_y": int(min_y) if min_y != float('inf') else 0,
                        "max_x": int(max_x),
                        "max_y": int(max_y)
                    }
                }
                
                # Re-serialize to JSON string
                response_text = json.dumps(parsed_response, ensure_ascii=False)
                
            except json.JSONDecodeError:
                logger.warning("Could not parse response to infer dimensions, will parse later")
            except Exception as e:
                logger.warning(f"Error inferring processed dimensions: {e}")
            
            # If still wrapped in code blocks, try simple split
            if response_text.startswith("```"):
                if "```json" in response_text:
                    parts = response_text.split("```json", 1)
                    if len(parts) > 1:
                        response_text = parts[1].split("```", 1)[0].strip()
                elif "```" in response_text:
                    parts = response_text.split("```", 1)
                    if len(parts) > 1:
                        response_text = parts[1].rsplit("```", 1)[0].strip()
            
            # Parse JSON
            try:
                schema = json.loads(response_text)
                
                # Extract image info BEFORE post-processing (which removes it)
                image_info = schema.get("_image_info", {})
                
                # Post-process and validate schema
                schema = self._post_process_schema(schema)
                
                # Re-attach image info after post-processing
                if image_info:
                    schema["_image_info"] = image_info
                
                logger.info(f"Successfully extracted {len(schema.get('fields', []))} fields")
                return schema
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                logger.error(f"Response text: {response_text[:500]}")
                raise ValueError(f"Invalid JSON response from Gemini: {e}")
                
        except Exception as e:
            logger.error(f"Error analyzing image {image_path}: {e}")
            raise
    
    def analyze_form_pages(
        self,
        form_name: str,
        image_paths: List[Path]
    ) -> Dict[str, Any]:
        """
        Analyze multiple pages of a form and combine into a single schema.
        
        Args:
            form_name: Name of the form
            image_paths: List of paths to form page images
            
        Returns:
            Combined form schema with all fields from all pages
        """
        logger.info(f"Analyzing form '{form_name}' with {len(image_paths)} page(s)")
        
        all_fields = []
        page_results = []  # Track success/failure per page
        image_dimensions = {}  # Store image dimensions for coordinate scaling
        
        for i, image_path in enumerate(image_paths, start=1):
            logger.info(f"Processing page {i}/{len(image_paths)}: {image_path.name}")
            
            try:
                # Load image to get dimensions before analysis
                image = Image.open(image_path)
                img_width, img_height = image.size
                image_dimensions[f"page_{i}"] = {
                    "width": img_width,
                    "height": img_height
                }
                logger.debug(f"Page {i} image dimensions: {img_width}x{img_height}")
                
                page_schema = self.analyze_form_image(image_path)
                fields = page_schema.get("fields", [])
                
                # Extract image info (processed dimensions and crop region) if available
                image_info = page_schema.get("_image_info", {})
                logger.debug(f"Page {i}: Extracted _image_info: {image_info}")
                if image_info:
                    processed_dims = image_info.get("processed_dimensions", {})
                    original_dims = image_info.get("original_dimensions", {})
                    crop_region = image_info.get("crop_region")
                    logger.debug(f"Page {i}: processed_dims={processed_dims}, original_dims={original_dims}, crop_region={crop_region}")
                    if processed_dims and processed_dims.get("width") and processed_dims.get("height"):
                        # Store processed dimensions for this page
                        image_dimensions[f"page_{i}"]["processed_width"] = processed_dims.get("width")
                        image_dimensions[f"page_{i}"]["processed_height"] = processed_dims.get("height")
                        
                        # Store crop region if available
                        if crop_region:
                            image_dimensions[f"page_{i}"]["crop_region"] = crop_region
                            if crop_region.get("detected"):
                                logger.info(
                                    f"Page {i}: Processed dimensions {processed_dims.get('width')}x{processed_dims.get('height')}, "
                                    f"CROP DETECTED: offset=({crop_region.get('offset_x')},{crop_region.get('offset_y')}), "
                                    f"size={crop_region.get('width')}x{crop_region.get('height')}"
                                )
                            else:
                                logger.info(
                                    f"Page {i}: Processed dimensions inferred as "
                                    f"{processed_dims.get('width')}x{processed_dims.get('height')} "
                                    f"(original: {original_dims.get('width')}x{original_dims.get('height')}, no crop)"
                                )
                        else:
                            logger.info(
                                f"Page {i}: Processed dimensions inferred as "
                                f"{processed_dims.get('width')}x{processed_dims.get('height')} "
                                f"(original: {original_dims.get('width')}x{original_dims.get('height')})"
                            )
                    else:
                        logger.warning(f"Page {i}: Processed dimensions missing or invalid in _image_info")
                else:
                    logger.warning(f"Page {i}: No _image_info found in page_schema")
                
                # Add page number and clean up field IDs
                existing_ids = {f.get("id") for f in all_fields}
                
                for field in fields:
                    field["page"] = i
                    
                    # Clean up field ID: remove existing page suffix if present
                    original_id = field.get("id", "")
                    if "_page_" in original_id:
                        # Remove the _page_X suffix
                        field["id"] = original_id.rsplit("_page_", 1)[0]
                    
                    # For multi-page forms, ensure uniqueness by adding page suffix to ID
                    if len(image_paths) > 1:
                        # Check if ID exists in previously processed fields
                        base_id = field["id"]
                        if base_id in existing_ids:
                            field["id"] = f"{base_id}_page_{i}"
                            # Ensure the new ID is also unique
                            counter = 1
                            while field["id"] in existing_ids:
                                field["id"] = f"{base_id}_page_{i}_{counter}"
                                counter += 1
                        # Add to existing_ids set for next iteration
                        existing_ids.add(field["id"])
                    
                    all_fields.append(field)
                
                logger.info(f"Page {i}: Extracted {len(fields)} fields")
                page_results.append({"page": i, "status": "success", "fields_count": len(fields)})
                
            except Exception as e:
                logger.error(f"Failed to analyze page {i}: {e}")
                page_results.append({"page": i, "status": "failed", "error": str(e)})
                # Continue with other pages
                continue
        
        # Detect and merge multi-part fields (after page numbers are assigned)
        # This uses spatial logic that's difficult to prompt reliably
        all_fields = self._merge_multipart_fields(all_fields)
        
        # Create combined schema with page results
        combined_schema = {
            "form_name": form_name,
            "total_pages": len(image_paths),
            "fields": all_fields,
            "page_results": page_results,  # Track which pages succeeded/failed
            "image_dimensions": image_dimensions  # Store image dimensions for coordinate scaling
        }
        
        successful_pages = sum(1 for r in page_results if r.get("status") == "success")
        logger.info(
            f"Form '{form_name}': Total {len(all_fields)} fields extracted across "
            f"{successful_pages}/{len(image_paths)} successful page(s)"
        )
        return combined_schema
    
    def save_schema(self, schema: Dict[str, Any], output_path: Path):
        """
        Save form schema to JSON file.
        
        Args:
            schema: Form schema dictionary
            output_path: Path to save the JSON file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(schema, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Schema saved to: {output_path}")


def main(
    images_dir: str = "output/images",
    schemas_dir: str = "output/schemas",
    image_pattern: str = "page_*.png"
):
    """
    Main function to run the vision analysis agent.
    
    Args:
        images_dir: Directory containing form image directories (default: "output/images")
        schemas_dir: Directory to save schemas (default: "output/schemas")
        image_pattern: Pattern to match page images (default: "page_*.png")
    """
    logger.info("=" * 60)
    logger.info("Vision Analysis Agent - Starting")
    logger.info("=" * 60)
    
    # Initialize the agent
    try:
        agent = VisionAnalysisAgent()
    except ValueError as e:
        logger.error(f"Initialization failed: {e}")
        return
    
    # Process forms from images directory
    images_dir = Path(images_dir)
    schemas_dir = Path(schemas_dir)
    schemas_dir.mkdir(parents=True, exist_ok=True)
    
    if not images_dir.exists():
        logger.warning(f"Images directory not found: {images_dir}")
        logger.info("Please run pdf_ingestion.py first to generate images.")
        return
    
    # Find all form directories
    form_dirs = [d for d in images_dir.iterdir() if d.is_dir()]
    
    if not form_dirs:
        logger.warning(f"No form directories found in {images_dir}")
        return
    
    logger.info(f"Found {len(form_dirs)} form(s) to analyze")
    
    # Process each form
    results = {}
    for form_dir in form_dirs:
        form_name = form_dir.name
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing form: {form_name}")
        logger.info(f"{'='*60}")
        
        # Get all page images for this form
        image_paths = sorted(form_dir.glob(image_pattern))
        
        if not image_paths:
            logger.warning(f"No page images found in {form_dir}")
            continue
        
        try:
            # Analyze form
            schema = agent.analyze_form_pages(form_name, image_paths)
            
            # Save schema
            schema_path = schemas_dir / f"{form_name}_schema.json"
            agent.save_schema(schema, schema_path)
            
            results[form_name] = {
                "status": "success",
                "fields_count": len(schema.get("fields", [])),
                "pages": schema.get("total_pages", 0),
                "schema_path": str(schema_path)
            }
            
        except Exception as e:
            logger.error(f"Failed to process form {form_name}: {e}")
            results[form_name] = {
                "status": "failed",
                "error": str(e)
            }
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Vision Analysis Summary")
    logger.info("=" * 60)
    
    for form_name, result in results.items():
        if result["status"] == "success":
            logger.info(
                f"[SUCCESS] - {form_name}: {result['fields_count']} fields, "
                f"{result['pages']} page(s)"
            )
        else:
            logger.info(f"[FAILED] - {form_name}: {result.get('error', 'Unknown error')}")
    
    logger.info(f"\nSchemas saved to: {schemas_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    # Configure logging (reuse from pdf_ingestion if available)
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('project.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
    
    main()