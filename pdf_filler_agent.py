"""
PDF Filler Agent (Action Layer) - Image-Based Implementation
Micro-Agent 3: Fills PDF forms with user data based on schema.

This is an image-based solution that:
- Loads images from PDF ingestion step
- Draws text, checkmarks, and signatures on images using PIL
- Converts filled images back to PDF

Responsibilities:
- Load original PDF page images
- Place text into each bounding box with proper alignment
- Draw checkmarks for checkboxes
- Handle signature fields (text or image)
- Handle multi-part fields (SSN, Medicare, dates)
- Apply conditional field logic
- Combine pages and produce final filled PDF
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from PIL import Image, ImageDraw, ImageFont
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('project.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PDFFillerAgent:
    """Fills PDF forms with user data based on schema using image-based approach."""
    
    # Default configuration
    DEFAULT_FONT_SIZE = 12
    DEFAULT_FONT_FAMILY = "arial.ttf"  # Will try to load system fonts
    DEFAULT_TEXT_COLOR = (0, 0, 0)  # Black
    CHECKBOX_CHECK_COLOR = (0, 0, 0)  # Black
    CHECKBOX_LINE_WIDTH = 3
    
    def __init__(
        self,
        images_dir: str = "output/images",
        output_dir: str = "output/filled_pdfs",
        font_size: int = None,
        font_path: Optional[str] = None,
        text_color: Tuple[int, int, int] = None,
        checkbox_color: Tuple[int, int, int] = None,
        signature_image_dir: Optional[str] = None
    ):
        """
        Initialize the PDF Filler Agent.
        
        Args:
            images_dir: Directory containing form images from ingestion step
            output_dir: Directory to save filled PDFs
            font_size: Default font size for text (default: 12)
            font_path: Path to TrueType font file (None = use default)
            text_color: RGB tuple for text color (default: (0, 0, 0) black)
            checkbox_color: RGB tuple for checkbox checkmark (default: (0, 0, 0) black)
            signature_image_dir: Directory containing signature images (optional)
        """
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.font_size = font_size or self.DEFAULT_FONT_SIZE
        self.text_color = text_color or self.DEFAULT_TEXT_COLOR
        self.checkbox_color = checkbox_color or self.CHECKBOX_CHECK_COLOR
        self.signature_image_dir = Path(signature_image_dir) if signature_image_dir else None
        
        # Load font
        self.font = self._load_font(font_path)
        
        logger.info(f"PDF Filler Agent initialized. Images directory: {self.images_dir}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def _load_font(self, font_path: Optional[str] = None) -> Optional[ImageFont.FreeTypeFont]:
        """
        Load font for text rendering.
        
        Args:
            font_path: Path to font file (None = try to load system default)
            
        Returns:
            PIL ImageFont object or None
        """
        if font_path:
            try:
                font = ImageFont.truetype(font_path, self.font_size)
                logger.info(f"Loaded font from: {font_path}")
                return font
            except Exception as e:
                logger.warning(f"Failed to load font from {font_path}: {e}")
        
        # Try to load common system fonts (prioritize scalable TrueType fonts)
        font_paths = [
            "C:/Windows/Fonts/arial.ttf",  # Windows
            "C:/Windows/Fonts/Arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",  # macOS
            "/System/Library/Fonts/Supplemental/Arial.ttf",  # macOS
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",  # Linux
            "arial.ttf",
            "Arial.ttf",
        ]
        
        for path in font_paths:
            try:
                font = ImageFont.truetype(path, self.font_size)
                logger.info(f"Loaded system font: {path}")
                return font
            except:
                continue
        
        # Fallback to default font (not ideal, but better than nothing)
        try:
            font = ImageFont.load_default()
            logger.warning("Using default PIL font (may have sizing limitations)")
            return font
        except:
            logger.warning("Could not load any font, text rendering may be limited")
            return None
    
    def _get_font_at_size(self, size: int) -> Optional[ImageFont.FreeTypeFont]:
        """
        Get font at a specific size (for dynamic sizing).
        
        Args:
            size: Font size in points
            
        Returns:
            PIL ImageFont object at specified size
        """
        if not self.font:
            return ImageFont.load_default()
        
        # If font has a path, load it at the new size
        if hasattr(self.font, 'path'):
            try:
                return ImageFont.truetype(self.font.path, size)
            except:
                pass
        
        # For default font, we can't resize, so return as-is
        return self.font
    
    def load_schema(self, schema_path: str) -> Dict[str, Any]:
        """Load form schema from JSON file."""
        schema_path = Path(schema_path)
        if not schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_path}")
        
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema = json.load(f)
        
        logger.info(f"Loaded schema: {schema.get('form_name', 'unknown')} with {len(schema.get('fields', []))} fields")
        return schema
    
    def load_user_data(self, user_data_path: str) -> Dict[str, Any]:
        """Load user data from JSON file."""
        user_data_path = Path(user_data_path)
        if not user_data_path.exists():
            raise FileNotFoundError(f"User data file not found: {user_data_path}")
        
        with open(user_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract user_data if nested
        if 'user_data' in data:
            user_data = data['user_data']
        else:
            user_data = data
        
        logger.info(f"Loaded user data with {len(user_data)} fields")
        return user_data
    
    def _get_image_path(self, form_name: str, page_num: int) -> Path:
        """
        Get path to page image from ingestion step.
        
        Args:
            form_name: Name of the form
            page_num: Page number (1-indexed)
            
        Returns:
            Path to image file
        """
        image_path = self.images_dir / form_name / f"page_{page_num:03d}.png"
        if not image_path.exists():
            # Try lowercase
            image_path = self.images_dir / form_name.lower() / f"page_{page_num:03d}.png"
        return image_path
    
    def _should_fill_field(self, field: Dict[str, Any], user_data: Dict[str, Any]) -> bool:
        """
        Determine if a field should be filled based on conditional logic.
        
        Args:
            field: Field definition from schema
            user_data: User data dictionary
            
        Returns:
            True if field should be filled, False otherwise
        """
        # Check if field has dependencies
        depends_on = field.get('depends_on')
        depends_value = field.get('depends_value')
        
        if depends_on is None:
            return True  # No dependencies, always fill if value exists
        
        # Check if controlling field has the required value
        controlling_value = user_data.get(depends_on)
        
        if controlling_value is None:
            return False
        
        # Handle string matching (case-insensitive)
        if isinstance(controlling_value, str) and isinstance(depends_value, str):
            return controlling_value.lower() == depends_value.lower()
        
        return controlling_value == depends_value
    
    def _get_field_value(self, field_id: str, user_data: Dict[str, Any]) -> Optional[Any]:
        """Get field value from user data, handling None/null values."""
        value = user_data.get(field_id)
        
        # Return None for null/None values
        if value is None:
            return None
        
        # Convert to string for text fields
        if isinstance(value, (int, float, bool)):
            return str(value)
        
        return value
    
    def _format_value_for_field(self, value: str, field: Dict[str, Any]) -> str:
        """
        Format value according to field's format_hint.
        
        Args:
            value: Raw value to format
            field: Field definition from schema
            
        Returns:
            Formatted value string
        """
        if not value:
            return ""
        
        format_hint = field.get('format_hint')
        if not format_hint:
            return value
        
        # Remove existing formatting
        clean_value = ''.join(c for c in value if c.isalnum())
        
        # Apply format based on hint
        if format_hint == "XXX-XX-XXXX":  # SSN/Medicare
            # Extract only alphanumeric, take first 9 characters
            clean_value = ''.join(c for c in value if c.isalnum())[:9]
            if len(clean_value) >= 9:
                return f"{clean_value[:3]}-{clean_value[3:5]}-{clean_value[5:9]}"
            elif len(clean_value) >= 5:
                return f"{clean_value[:3]}-{clean_value[3:5]}-{clean_value[5:]}"
            elif len(clean_value) > 0:
                # Partial value - return as-is
                return value
            else:
                return value
        
        elif format_hint == "XXXXX":  # ZIP code
            return clean_value[:5]
        
        elif format_hint == "(XXX) XXX-XXXX":  # Phone
            # Extract only digits, take first 10
            clean_value = ''.join(c for c in value if c.isdigit())[:10]
            if len(clean_value) >= 10:
                return f"({clean_value[:3]}) {clean_value[3:6]}-{clean_value[6:10]}"
            elif len(clean_value) > 0:
                # Partial phone number - format what we have
                return value
            else:
                return value
        
        elif format_hint in ["MM/YYYY", "MM/DD/YYYY"]:
            # Date formatting - assume value is already in correct format or clean it
            return value
        
        return value
    
    def _calculate_font_size(
        self,
        text: str,
        bbox: List[int],
        max_font_size: int = 20,
        min_font_size: int = 8
    ) -> int:
        """
        Calculate appropriate font size to fit text within bounding box.
        
        Args:
            text: Text to fit
            bbox: Bounding box [x1, y1, x2, y2]
            max_font_size: Maximum font size
            min_font_size: Minimum font size
            
        Returns:
            Appropriate font size
        """
        if not text:
            return self.font_size
        
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        if width <= 0 or height <= 0:
            return self.font_size
        
        # Try different font sizes (binary search for efficiency)
        best_size = min_font_size
        low, high = min_font_size, max_font_size
        
        while low <= high:
            size = (low + high) // 2
            try:
                test_font = self._get_font_at_size(size)
                if not test_font:
                    break
                
                # Get text bounding box
                # Try textbbox first (Pillow 9.2+), fallback to getbbox
                try:
                    # Create a temporary draw object to check for textbbox
                    temp_draw = ImageDraw.Draw(Image.new('RGB', (100, 100)))
                    if hasattr(temp_draw, 'textbbox'):
                        bbox_text = temp_draw.textbbox((0, 0), text, font=test_font)
                    else:
                        bbox_text = test_font.getbbox(text)
                    text_width = bbox_text[2] - bbox_text[0]
                    text_height = bbox_text[3] - bbox_text[1]
                except:
                    # Ultimate fallback
                    bbox_text = test_font.getbbox(text)
                    text_width = bbox_text[2] - bbox_text[0]
                    text_height = bbox_text[3] - bbox_text[1]
                
                # Check if text fits (with 10% margin)
                if text_width <= width * 0.9 and text_height <= height * 0.8:
                    best_size = size
                    low = size + 1  # Try larger
                else:
                    high = size - 1  # Try smaller
            except Exception as e:
                # If font loading fails, break and use best_size found so far
                break
        
        return best_size
    
    def _draw_text_field(
        self,
        image: Image.Image,
        draw: ImageDraw.ImageDraw,
        field: Dict[str, Any],
        value: str,
        font_size: Optional[int] = None
    ):
        """
        Draw text into a field's bounding box.
        
        Args:
            image: PIL Image object
            draw: PIL ImageDraw object
            field: Field definition from schema
            value: Text value to place
            font_size: Optional font size (auto-calculated if None)
        """
        if not value:
            return
        
        bbox = field['bbox']
        x1, y1, x2, y2 = bbox
        
        # Verify coordinates are within image bounds
        img_width, img_height = image.size
        x1 = max(0, min(x1, img_width))
        y1 = max(0, min(y1, img_height))
        x2 = max(0, min(x2, img_width))
        y2 = max(0, min(y2, img_height))
        
        if x2 <= x1 or y2 <= y1:
            logger.warning(f"Invalid bbox for field {field.get('id')}: {bbox}")
            return
        
        # Calculate font size if not provided
        # Use the original bbox dimensions (before clamping) for font size calculation
        # to get accurate sizing based on the intended field size
        if font_size is None:
            # Use original bbox for font size calculation (before clamping)
            original_bbox = field['bbox']
            font_size = self._calculate_font_size(value, original_bbox)
        
        # Format value if needed
        formatted_value = self._format_value_for_field(value, field)
        
        # Get font at calculated size
        font = self._get_font_at_size(font_size)
        if not font:
            font = ImageFont.load_default()
        
        # Calculate text position (left-aligned, vertically centered)
        # Use textbbox for accurate measurements (Pillow 9.2+)
        try:
            if hasattr(draw, 'textbbox'):
                # textbbox gives accurate bounding box
                bbox_text = draw.textbbox((0, 0), formatted_value, font=font)
                text_width = bbox_text[2] - bbox_text[0]
                text_height = bbox_text[3] - bbox_text[1]
                # Account for the offset (textbbox includes baseline)
                text_x = x1
                text_y = y1 + (y2 - y1 - text_height) // 2 - bbox_text[1]  # Adjust for baseline
            else:
                # Fallback for older Pillow versions
                bbox_text = font.getbbox(formatted_value)
                text_width = bbox_text[2] - bbox_text[0]
                text_height = bbox_text[3] - bbox_text[1]
                text_x = x1
                text_y = y1 + (y2 - y1 - text_height) // 2
        except Exception as e:
            # Ultimate fallback
            logger.warning(f"Could not calculate text position for field {field.get('id')}: {e}")
            text_x = x1
            text_y = y1 + (y2 - y1) // 2
        
        # Draw text
        try:
            draw.text(
                (text_x, text_y),
                formatted_value,
                font=font,
                fill=self.text_color
            )
        except Exception as e:
            logger.warning(f"Failed to draw text for field {field.get('id')}: {e}")
    
    def _draw_checkbox(
        self,
        image: Image.Image,
        draw: ImageDraw.ImageDraw,
        field: Dict[str, Any],
        checked: bool
    ):
        """
        Draw a checkmark in a checkbox field.
        
        Args:
            image: PIL Image object
            draw: PIL ImageDraw object
            field: Field definition from schema
            checked: Whether checkbox should be checked
        """
        if not checked:
            return
        
        bbox = field['bbox']
        x1, y1, x2, y2 = bbox
        
        # Verify coordinates are within image bounds
        img_width, img_height = image.size
        x1 = max(0, min(x1, img_width))
        y1 = max(0, min(y1, img_height))
        x2 = max(0, min(x2, img_width))
        y2 = max(0, min(y2, img_height))
        
        if x2 <= x1 or y2 <= y1:
            logger.warning(f"Invalid bbox for checkbox field {field.get('id')}: {bbox}")
            return
        
        # Calculate checkmark size (80% of box size)
        width = x2 - x1
        height = y2 - y1
        margin = min(width, height) * 0.15
        
        # Draw checkmark using lines
        # Checkmark: from bottom-left to center, then to top-right
        check_x1 = x1 + margin
        check_y1 = y1 + height * 0.6
        check_x2 = x1 + width * 0.4
        check_y2 = y1 + height * 0.5
        check_x3 = x2 - margin
        check_y3 = y1 + margin
        
        # Draw checkmark as two lines
        draw.line(
            [(check_x1, check_y1), (check_x2, check_y2)],
            fill=self.checkbox_color,
            width=self.CHECKBOX_LINE_WIDTH
        )
        draw.line(
            [(check_x2, check_y2), (check_x3, check_y3)],
            fill=self.checkbox_color,
            width=self.CHECKBOX_LINE_WIDTH
        )
    
    def _draw_signature(
        self,
        image: Image.Image,
        draw: ImageDraw.ImageDraw,
        field: Dict[str, Any],
        value: str
    ):
        """
        Draw signature in signature field (text or image).
        
        Args:
            image: PIL Image object
            draw: PIL ImageDraw object
            field: Field definition from schema
            value: Signature value (text or path to image)
        """
        if not value:
            return
        
        bbox = field['bbox']
        x1, y1, x2, y2 = bbox
        
        # Verify coordinates are within image bounds
        img_width, img_height = image.size
        x1 = max(0, min(x1, img_width))
        y1 = max(0, min(y1, img_height))
        x2 = max(0, min(x2, img_width))
        y2 = max(0, min(y2, img_height))
        
        if x2 <= x1 or y2 <= y1:
            logger.warning(f"Invalid bbox for signature field {field.get('id')}: {bbox}")
            return
        
        # Check if value is a path to an image file
        sig_path = Path(value)
        if sig_path.exists() or (self.signature_image_dir and (self.signature_image_dir / value).exists()):
            try:
                # Load signature image
                if sig_path.exists():
                    sig_image = Image.open(sig_path)
                else:
                    sig_image = Image.open(self.signature_image_dir / value)
                
                # Resize to fit bounding box
                sig_image.thumbnail((x2 - x1, y2 - y1), Image.Resampling.LANCZOS)
                
                # Paste onto main image
                sig_x = x1 + (x2 - x1 - sig_image.width) // 2
                sig_y = y1 + (y2 - y1 - sig_image.height) // 2
                image.paste(sig_image, (sig_x, sig_y), sig_image if sig_image.mode == 'RGBA' else None)
                logger.info(f"Inserted signature image for field {field.get('id')}")
                return
            except Exception as e:
                logger.warning(f"Failed to insert signature image: {e}, falling back to text")
        
        # Fallback to text signature
        self._draw_text_field(image, draw, field, value)
    
    def _draw_multipart_field(
        self,
        image: Image.Image,
        draw: ImageDraw.ImageDraw,
        field: Dict[str, Any],
        value: str
    ):
        """
        Draw text into a multi-part field (e.g., SSN, Medicare number).
        
        Args:
            image: PIL Image object
            draw: PIL ImageDraw object
            field: Field definition with parts array
            value: Full value to split across parts
        """
        if not value or not field.get('parts'):
            # Fallback to regular text field
            self._draw_text_field(image, draw, field, value)
            return
        
        # Format value according to format_hint
        formatted_value = self._format_value_for_field(value, field)
        
        # Split value based on format_hint
        format_hint = field.get('format_hint', '')
        parts = field['parts']
        
        if format_hint == "XXX-XX-XXXX":
            # Remove dashes and split
            clean_value = ''.join(c for c in formatted_value if c.isalnum())
            # Handle both 9-digit and longer values (take first 9 digits)
            if len(clean_value) >= 9:
                part_values = [
                    clean_value[:3],
                    clean_value[3:5],
                    clean_value[5:9]
                ]
            else:
                # Fallback: split evenly
                part_values = self._split_value_evenly(clean_value, len(parts))
        elif format_hint == "MM/YYYY":
            # Split on /
            if '/' in formatted_value:
                part_values = formatted_value.split('/')
            else:
                clean_value = ''.join(c for c in formatted_value if c.isalnum())
                part_values = [
                    clean_value[:2] if len(clean_value) >= 2 else clean_value,
                    clean_value[2:] if len(clean_value) > 2 else ""
                ]
        elif format_hint == "MM/DD/YYYY":
            # Split on /
            if '/' in formatted_value:
                part_values = formatted_value.split('/')
            else:
                clean_value = ''.join(c for c in formatted_value if c.isalnum())
                if len(clean_value) >= 8:
                    part_values = [
                        clean_value[:2],
                        clean_value[2:4],
                        clean_value[4:8]
                    ]
                else:
                    part_values = self._split_value_evenly(clean_value, 3)
        elif format_hint == "XXXXX":
            # Split ZIP code into individual digits
            clean_value = ''.join(c for c in formatted_value if c.isdigit())
            part_values = list(clean_value[:5].ljust(5, ' '))
        else:
            # Default: split evenly
            clean_value = ''.join(c for c in formatted_value if c.isalnum())
            part_values = self._split_value_evenly(clean_value, len(parts))
        
        # Calculate a consistent font size for all parts based on the largest part
        # This ensures uniform appearance across multi-part fields
        max_part_width = 0
        max_part_height = 0
        longest_part_value = max(part_values, key=len) if part_values else "X"
        
        for part in parts:
            part_bbox = part['bbox']
            part_width = part_bbox[2] - part_bbox[0]
            part_height = part_bbox[3] - part_bbox[1]
            max_part_width = max(max_part_width, part_width)
            max_part_height = max(max_part_height, part_height)
        
        # Use the largest part's dimensions to calculate font size
        # Create a bbox at origin for font size calculation (position doesn't matter for size)
        largest_part_bbox = [0, 0, max_part_width, max_part_height]
        consistent_font_size = self._calculate_font_size(
            longest_part_value,
            largest_part_bbox
        )
        
        logger.debug(
            f"Multi-part field '{field.get('id')}': "
            f"Largest part dimensions: {max_part_width}x{max_part_height}, "
            f"Longest value: '{longest_part_value}', "
            f"Calculated font size: {consistent_font_size}"
        )
        
        # Draw each part in its bounding box with consistent font size
        for i, part in enumerate(parts):
            if i < len(part_values) and part_values[i]:
                part_bbox = part['bbox']
                part_field = {**field, 'bbox': part_bbox}
                # Use the consistent font size for all parts
                self._draw_text_field(image, draw, part_field, part_values[i], font_size=consistent_font_size)
    
    def _split_value_evenly(self, value: str, num_parts: int) -> List[str]:
        """Split value evenly across number of parts."""
        if not value or num_parts == 0:
            return [""] * num_parts
        
        part_length = len(value) // num_parts
        remainder = len(value) % num_parts
        
        parts = []
        start = 0
        for i in range(num_parts):
            length = part_length + (1 if i < remainder else 0)
            parts.append(value[start:start + length])
            start += length
        
        return parts
    
    def _scale_field_coordinates(
        self,
        field: Dict[str, Any],
        scale_x: float,
        scale_y: float
    ) -> Dict[str, Any]:
        """
        Scale field coordinates to match actual image dimensions.
        
        Args:
            field: Field definition with bbox coordinates
            scale_x: Horizontal scaling factor
            scale_y: Vertical scaling factor
            
        Returns:
            Field with scaled coordinates
        """
        if abs(scale_x - 1.0) < 0.001 and abs(scale_y - 1.0) < 0.001:
            # No scaling needed
            return field
        
        # Create a copy of the field
        scaled_field = field.copy()
        
        # Scale main bbox
        if 'bbox' in scaled_field:
            bbox = scaled_field['bbox']
            scaled_field['bbox'] = [
                int(bbox[0] * scale_x),  # x1
                int(bbox[1] * scale_y),  # y1
                int(bbox[2] * scale_x),  # x2
                int(bbox[3] * scale_y)   # y2
            ]
        
        # Scale parts if present
        if 'parts' in scaled_field and scaled_field['parts']:
            scaled_parts = []
            for part in scaled_field['parts']:
                scaled_part = part.copy()
                if 'bbox' in scaled_part:
                    part_bbox = scaled_part['bbox']
                    scaled_part['bbox'] = [
                        int(part_bbox[0] * scale_x),  # x1
                        int(part_bbox[1] * scale_y),  # y1
                        int(part_bbox[2] * scale_x),  # x2
                        int(part_bbox[3] * scale_y)   # y2
                    ]
                scaled_parts.append(scaled_part)
            scaled_field['parts'] = scaled_parts
        
        return scaled_field
    
    def fill_pdf(
        self,
        form_name: str,
        schema_path: str,
        user_data_path: str,
        output_filename: Optional[str] = None
    ) -> str:
        """
        Fill a PDF form with user data based on schema.
        
        Args:
            form_name: Name of the form (used to locate images)
            schema_path: Path to schema JSON file
            user_data_path: Path to user data JSON file
            output_filename: Optional output filename (auto-generated if None)
            
        Returns:
            Path to filled PDF file
        """
        logger.info("=" * 60)
        logger.info("PDF Filler Agent - Starting (Image-Based)")
        logger.info("=" * 60)
        
        # Load schema and user data
        schema = self.load_schema(schema_path)
        user_data = self.load_user_data(user_data_path)
        
        logger.info(f"Filling form: {form_name}")
        
        # Get total pages from schema
        total_pages = schema.get('total_pages', 1)
        logger.info(f"Form has {total_pages} total pages")
        
        # Store reference image dimensions from schema (if available)
        # This helps us scale coordinates if images were resized
        reference_dimensions = schema.get('image_dimensions', {})
        
        # Group fields by page
        fields_by_page: Dict[int, List[Dict[str, Any]]] = {}
        for field in schema.get('fields', []):
            page_num = field.get('page', 1)
            if page_num not in fields_by_page:
                fields_by_page[page_num] = []
            fields_by_page[page_num].append(field)
        
        # Process ALL pages from 1 to total_pages (not just pages with fields)
        filled_images = []
        total_fields_filled = 0
        
        for page_num in range(1, total_pages + 1):
            # Load page image
            image_path = self._get_image_path(form_name, page_num)
            if not image_path.exists():
                logger.warning(f"Image not found for page {page_num}: {image_path}")
                # Still add a placeholder to maintain page order
                # Try to create a blank image with same dimensions as other pages
                if filled_images:
                    # Use dimensions from previous page
                    blank_image = Image.new('RGB', filled_images[0].size, color='white')
                    filled_images.append(blank_image)
                    logger.warning(f"Created blank placeholder for missing page {page_num}")
                continue
            
            # Get fields for this page (may be empty)
            fields = fields_by_page.get(page_num, [])
            logger.info(f"Processing page {page_num} with {len(fields)} fields")
            
            # Load image
            image = Image.open(image_path)
            # Convert to RGB if necessary (for PDF conversion)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Get actual image dimensions
            img_width, img_height = image.size
            logger.info(f"Page {page_num} image dimensions: {img_width}x{img_height}")
            
            # Check if we need to scale coordinates
            # Get reference dimensions for this page (if stored in schema)
            page_ref_key = f"page_{page_num}"
            scale_x = scale_y = 1.0
            
            if page_ref_key in reference_dimensions:
                # Calculate actual max bounding box coordinates from fields on this page
                # This is more accurate than inferred processed dimensions
                page_fields = [f for f in fields if f.get('page') == page_num]
                actual_max_x = 0
                actual_max_y = 0
                
                for field in page_fields:
                    # Check main bbox
                    bbox = field.get('bbox', [])
                    if len(bbox) >= 4:
                        actual_max_x = max(actual_max_x, bbox[2])
                        actual_max_y = max(actual_max_y, bbox[3])
                    
                    # Check parts bboxes
                    parts = field.get('parts', [])
                    if parts:
                        for part in parts:
                            part_bbox = part.get('bbox', [])
                            if len(part_bbox) >= 4:
                                actual_max_x = max(actual_max_x, part_bbox[2])
                                actual_max_y = max(actual_max_y, part_bbox[3])
                
                # Prefer processed dimensions (what Gemini actually saw) over original
                processed_width = reference_dimensions[page_ref_key].get('processed_width')
                processed_height = reference_dimensions[page_ref_key].get('processed_height')
                
                if processed_width and processed_height:
                    # Prefer actual max bbox coordinates over processed dimensions for accuracy
                    # The processed dimensions may include a margin that causes misalignment
                    if actual_max_x > 0 and actual_max_y > 0:
                        # Use actual max coordinates with a small margin (2% instead of 5%)
                        # This should be more accurate than the inferred processed dimensions
                        ref_width = int(actual_max_x * 1.02)
                        ref_height = int(actual_max_y * 1.02)
                        logger.info(
                            f"Page {page_num}: Using actual max bbox coordinates ({actual_max_x}x{actual_max_y}) "
                            f"with 2% margin ({ref_width}x{ref_height}) instead of processed dimensions "
                            f"({processed_width}x{processed_height}) for more accurate scaling"
                        )
                    else:
                        # Fallback to processed dimensions if we can't determine actual max
                        ref_width = processed_width
                        ref_height = processed_height
                        logger.info(
                            f"Page {page_num}: Using processed dimensions {ref_width}x{ref_height} "
                            f"(could not determine actual max bbox)"
                        )
                else:
                    # Fallback to original dimensions
                    ref_width = reference_dimensions[page_ref_key].get('width')
                    ref_height = reference_dimensions[page_ref_key].get('height')
                    logger.info(
                        f"Page {page_num}: Using original dimensions {ref_width}x{ref_height} "
                        f"(no processed dimensions found)"
                    )
                
                if ref_width and ref_height:
                    scale_x = img_width / ref_width
                    scale_y = img_height / ref_height
                    if abs(scale_x - 1.0) > 0.01 or abs(scale_y - 1.0) > 0.01:
                        logger.info(
                            f"Page {page_num} coordinate scaling: "
                            f"Reference (Gemini processed): {ref_width}x{ref_height}, "
                            f"Actual (loaded image): {img_width}x{img_height}. "
                            f"Scaling factors: {scale_x:.3f}x (horizontal), {scale_y:.3f}x (vertical)"
                        )
                    else:
                        logger.debug(f"Page {page_num} image dimensions match reference - no scaling needed")
            else:
                # No reference dimensions in schema (old schema format)
                # Log a warning but proceed - coordinates should match if images are unchanged
                logger.debug(
                    f"Page {page_num}: No reference dimensions in schema. "
                    f"Assuming coordinates match image size {img_width}x{img_height}"
                )
            
            draw = ImageDraw.Draw(image)
            
            page_fields_filled = 0
            for field in fields:
                field_id = field.get('id')
                field_type = field.get('type')
                
                # Check if field should be filled
                if not self._should_fill_field(field, user_data):
                    continue
                
                # Get field value
                value = self._get_field_value(field_id, user_data)
                if value is None:
                    continue
                
                # Scale field coordinates if needed
                scaled_field = self._scale_field_coordinates(field, scale_x, scale_y)
                
                # Debug logging for coordinate scaling (first few fields and multi-part fields)
                if page_fields_filled < 5 or field.get('parts'):
                    original_bbox = field.get('bbox', [])
                    scaled_bbox = scaled_field.get('bbox', [])
                    logger.info(
                        f"Field '{field_id}': Original bbox {original_bbox} -> "
                        f"Scaled bbox {scaled_bbox} (scale: {scale_x:.3f}x{scale_y:.3f})"
                    )
                    if field.get('parts'):
                        for i, part in enumerate(field.get('parts', [])):
                            orig_part_bbox = part.get('bbox', [])
                            scaled_part = scaled_field.get('parts', [])[i] if i < len(scaled_field.get('parts', [])) else {}
                            scaled_part_bbox = scaled_part.get('bbox', [])
                            logger.info(
                                f"  Part {i+1}: Original {orig_part_bbox} -> Scaled {scaled_part_bbox}"
                            )
                
                # Fill field based on type
                try:
                    if field_type == 'checkbox':
                        checked = bool(value) if isinstance(value, bool) else str(value).lower() in ['true', 'yes', '1', 'checked']
                        self._draw_checkbox(image, draw, scaled_field, checked)
                        page_fields_filled += 1
                    
                    elif field_type == 'signature':
                        self._draw_signature(image, draw, scaled_field, str(value))
                        page_fields_filled += 1
                    
                    elif field_type in ['text', 'date', 'number']:
                        if field.get('parts'):
                            # Multi-part field
                            self._draw_multipart_field(image, draw, scaled_field, str(value))
                        else:
                            # Regular text field
                            self._draw_text_field(image, draw, scaled_field, str(value))
                        page_fields_filled += 1
                    
                    elif field_type == 'grouped_choices':
                        # For radio buttons/choices, draw the selected option as text
                        if value in field.get('options', []):
                            self._draw_text_field(image, draw, scaled_field, str(value))
                            page_fields_filled += 1
                    
                    else:
                        logger.warning(f"Unknown field type: {field_type} for field {field_id}")
                
                except Exception as e:
                    logger.error(f"Error filling field {field_id}: {e}")
            
            total_fields_filled += page_fields_filled
            filled_images.append(image)
            logger.info(f"Filled {page_fields_filled} fields on page {page_num}")
        
        # Generate output filename
        if output_filename is None:
            output_filename = f"{form_name}_filled.pdf"
        
        output_path = self.output_dir / output_filename
        
        # Save as PDF
        if filled_images:
            # Save first image as PDF
            filled_images[0].save(
                output_path,
                "PDF",
                resolution=300.0,
                save_all=True,
                append_images=filled_images[1:] if len(filled_images) > 1 else []
            )
        
        logger.info("=" * 60)
        logger.info(f"PDF Filler Agent - Complete")
        logger.info(f"Total fields filled: {total_fields_filled}")
        logger.info(f"Output saved to: {output_path}")
        logger.info("=" * 60)
        
        return str(output_path)


def main():
    """Main function to run the PDF Filler Agent."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fill PDF forms with user data (image-based)')
    parser.add_argument('--form-name', required=True, help='Name of the form (e.g., cms-40b-508c-2025)')
    parser.add_argument('--schema', required=True, help='Path to schema JSON file')
    parser.add_argument('--data', required=True, help='Path to user data JSON file')
    parser.add_argument('--output', help='Output filename (optional)')
    parser.add_argument('--images-dir', default='output/images', help='Directory containing form images')
    parser.add_argument('--output-dir', default='output/filled_pdfs', help='Output directory for filled PDFs')
    
    args = parser.parse_args()
    
    # Initialize agent
    agent = PDFFillerAgent(
        images_dir=args.images_dir,
        output_dir=args.output_dir
    )
    
    # Fill PDF
    output_path = agent.fill_pdf(
        form_name=args.form_name,
        schema_path=args.schema,
        user_data_path=args.data,
        output_filename=args.output
    )
    
    print(f"\n[SUCCESS] Filled PDF saved to: {output_path}")


if __name__ == "__main__":
    main()

