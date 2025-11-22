

import json
import logging
import os
import io
from pathlib import Path
from typing import Dict, List, Optional, Any
from PIL import Image, ImageFont, ImageDraw
import fitz  # PyMuPDF (used only for stitching images back to PDF)

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
    """
    Fills form data directly onto the high-res images used for analysis,
    then converts those images back to a PDF.
    """
    
    # Default configuration
    DEFAULT_FONT_SIZE = 30  # Base font size for 300 DPI images
    DEFAULT_TEXT_COLOR = (0, 0, 0, 255)  # Black
    CHECKBOX_COLOR = (0, 0, 0, 255)
    CHECKBOX_LINE_WIDTH = 4
    
    # "Comb" fields (Character count for spacing)
    COMB_FIELDS = {
        "medicare": 11, "state": 2, "zip": 5, "postal": 5, "year": 4,
        "date_of_birth": 8, "dob": 8, "phone": 10, "telephone": 10,
        "mobile": 10, "ssn": 9, "social": 9
    }

    def __init__(
        self,
        forms_dir: str = "forms",
        output_dir: str = "output/filled_pdfs",
        images_dir: str = "output/images",
        font_path: str = "arial.ttf", 
        signature_image_dir: Optional[str] = None
    ):
        self.forms_dir = Path(forms_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir = Path(images_dir)
        
        self.font_path = self._resolve_font_path(font_path)
        self.signature_image_dir = Path(signature_image_dir) if signature_image_dir else None
        
        logger.info(f"Image-Based PDF Filler Agent initialized")

    def _resolve_font_path(self, font_name: str) -> str:
        """Find a usable font file."""
        if os.path.exists(font_name): return font_name
        # Windows System Fonts
        win_path = os.path.join("C:\\Windows\\Fonts", font_name)
        if os.path.exists(win_path): return win_path
        return None # Pillow will fallback to default

    def load_schema(self, schema_path: str) -> Dict[str, Any]:
        with open(schema_path, 'r', encoding='utf-8') as f: return json.load(f)
    
    def load_user_data(self, user_data_path: str) -> Dict[str, Any]:
        with open(user_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('user_data', data)

    def _should_fill_field(self, field: Dict[str, Any], user_data: Dict[str, Any]) -> bool:
        depends_on = field.get('depends_on')
        depends_value = field.get('depends_value')
        if depends_on is None: return True
        
        controlling_value = user_data.get(depends_on)
        if controlling_value is None: return False
        
        if isinstance(controlling_value, str) and isinstance(depends_value, str):
            return controlling_value.lower() == depends_value.lower()
        return controlling_value == depends_value
    
    def _get_field_value(self, field_id: str, user_data: Dict[str, Any]) -> Optional[Any]:
        value = user_data.get(field_id)
        if value is None: return None
        if isinstance(value, (int, float, bool)): return str(value)
        return value

    def _get_font(self, size: int) -> ImageFont.ImageFont:
        if self.font_path:
            try: return ImageFont.truetype(self.font_path, size)
            except IOError: pass
        return ImageFont.load_default()

    def _calculate_font_size(self, text: str, width: int, height: int) -> int:
        """Iteratively find the largest font that fits."""
        if not text: return self.DEFAULT_FONT_SIZE
        size = int(height * 0.6) # Start modest (60% of box height)
        min_size = 14
        
        if self.font_path:
            while size > min_size:
                font = ImageFont.truetype(self.font_path, size)
                bbox = font.getbbox(text)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                if text_w < (width * 0.95) and text_h < (height * 0.9):
                    return size
                size -= 2
        return max(size, min_size)

    def _draw_checkbox(self, draw: ImageDraw.Draw, rect: List[int], checked: bool):
        if not checked: return
        x1, y1, x2, y2 = rect
        width, height = x2 - x1, y2 - y1
        cx, cy = x1 + width / 2, y1 + height / 2
        size = min(width, height) * 0.35
        draw.line([(cx - size, cy - size), (cx + size, cy + size)], fill=self.CHECKBOX_COLOR, width=self.CHECKBOX_LINE_WIDTH)
        draw.line([(cx + size, cy - size), (cx - size, cy + size)], fill=self.CHECKBOX_COLOR, width=self.CHECKBOX_LINE_WIDTH)

    def _draw_signature(self, img: Image.Image, rect: List[int], value: str):
        if not value: return
        sig_path = Path(value)
        if not sig_path.exists() and self.signature_image_dir:
            sig_path = self.signature_image_dir / value
            
        if sig_path.exists():
            try:
                sig_img = Image.open(sig_path).convert("RGBA")
                target_w, target_h = rect[2] - rect[0], rect[3] - rect[1]
                sig_img.thumbnail((target_w, target_h), Image.Resampling.LANCZOS)
                sig_w, sig_h = sig_img.size
                paste_x = int(rect[0] + (target_w - sig_w) / 2)
                paste_y = int(rect[1] + (target_h - sig_h) / 2) # Center vertically for sigs
                img.paste(sig_img, (paste_x, paste_y), sig_img)
                return
            except Exception: pass
        
        # Fallback
        draw = ImageDraw.Draw(img)
        self._draw_standard_text(draw, rect, f"/s/ {value}")

    def _draw_comb_text(self, draw: ImageDraw.Draw, rect: List[int], value: str, max_chars: int):
        """Draws text spaced evenly for comb fields."""
        x1, y1, x2, y2 = rect
        total_width = x2 - x1
        slot_width = total_width / max_chars
        clean_value = "".join([c for c in value if c.isalnum()])
        
        # Font calculation
        font_size = min(int((y2 - y1) * 0.5), int(slot_width * 0.7))
        font = self._get_font(font_size)
        
        for i, char in enumerate(clean_value):
            if i >= max_chars: break
            slot_center_x = x1 + (i * slot_width) + (slot_width / 2)
            # Align near bottom to avoid overlaps
            draw_y_base = y2 - ((y2 - y1) * 0.15) 
            
            bbox = font.getbbox(char)
            char_w = bbox[2] - bbox[0]
            char_h = bbox[3] - bbox[1]
            
            draw_x = slot_center_x - (char_w / 2)
            draw_y = draw_y_base - char_h
            
            draw.text((draw_x, draw_y), char, font=font, fill=self.DEFAULT_TEXT_COLOR)

    def _draw_standard_text(self, draw: ImageDraw.Draw, rect: List[int], value: str):
        """Draws text aligned to the BOTTOM-LEFT to avoid overlapping labels at the top."""
        x1, y1, x2, y2 = rect
        width = x2 - x1
        height = y2 - y1
        
        font_size = self._calculate_font_size(value, width, height)
        font = self._get_font(font_size)
        
        bbox = font.getbbox(value)
        text_h = bbox[3] - bbox[1]
        
        # Horizontal: Left align with padding
        draw_x = x1 + (width * 0.02) + 5
        
        # Vertical: Bottom align with padding (approx 15% from bottom)
        # This pushes text down, away from the "First Name" label at the top
        draw_y = y2 - (height * 0.15) - text_h
        
        # Safety check: if box is tiny, just center it
        if draw_y < y1:
            draw_y = y1 + (height - text_h) / 2
        
        draw.text((draw_x, draw_y), value, font=font, fill=self.DEFAULT_TEXT_COLOR)

    def fill_pdf(self, form_name: str, schema_path: str, user_data_path: str, output_filename: Optional[str] = None) -> str:
        logger.info(f"Starting filling process for {form_name}")
        schema = self.load_schema(schema_path)
        user_data = self.load_user_data(user_data_path)
        
        form_images_path = self.images_dir / form_name
        if not form_images_path.exists(): form_images_path = self.images_dir

        fields_by_page = {}
        for field in schema.get('fields', []):
            fields_by_page.setdefault(field.get('page', 1), []).append(field)
        
        filled_images = []
        total_pages = schema.get('total_pages', 1)
        
        for i in range(1, total_pages + 1):
            # Locate Image
            page_img_path = list(form_images_path.glob(f"page_{i:03d}.*"))
            if not page_img_path: page_img_path = list(form_images_path.glob(f"page_{i}.*"))
            if not page_img_path:
                logger.error(f"Missing image for page {i}"); continue
                
            img = Image.open(page_img_path[0]).convert("RGBA")
            draw = ImageDraw.Draw(img)
            
            for field in fields_by_page.get(i, []):
                if not self._should_fill_field(field, user_data): continue
                value = self._get_field_value(field['id'], user_data)
                if not value: continue
                
                bbox = field.get('bbox')
                if not bbox or len(bbox) < 4: continue
                
                try:
                    ftype = field.get('type')
                    fid = field.get('id', '').lower()
                    
                    # Check for Comb field
                    is_comb = False
                    comb_chars = 0
                    for key, count in self.COMB_FIELDS.items():
                        if key in fid:
                            is_comb = True; comb_chars = count; break
                    
                    if ftype == 'checkbox':
                        self._draw_checkbox(draw, bbox, str(value).lower() in ['true', 'yes', '1', 'checked'])
                    elif ftype == 'signature':
                        self._draw_signature(img, bbox, str(value))
                    elif is_comb and ftype != 'date':
                        self._draw_comb_text(draw, bbox, str(value), comb_chars)
                    else:
                        self._draw_standard_text(draw, bbox, str(value))
                except Exception as e:
                    logger.error(f"Error filling {field.get('id')}: {e}")

            filled_images.append(img.convert("RGB"))

        # Stitch to PDF
        out_name = output_filename or f"{form_name}_filled.pdf"
        out_path = self.output_dir / out_name
        doc = fitz.open()
        for f_img in filled_images:
            img_byte_arr = io.BytesIO()
            f_img.save(img_byte_arr, format='JPEG', quality=95)
            page = doc.new_page(width=f_img.width, height=f_img.height)
            page.insert_image(page.rect, stream=img_byte_arr.getvalue())
            
        doc.save(out_path)
        doc.close()
        logger.info(f"Saved to: {out_path}")
        return str(out_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--form-name', required=True)
    parser.add_argument('--schema', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output')
    args = parser.parse_args()
    
    agent = PDFFillerAgent()
    agent.fill_pdf(args.form_name, args.schema, args.data, args.output)