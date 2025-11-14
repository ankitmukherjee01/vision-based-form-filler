"""
PDF Ingestion Module
Converts PDF pages to high-resolution images for vision analysis.
"""

import os
import logging
from pathlib import Path
from typing import List, Optional
from pdf2image import convert_from_path
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)


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


class PDFIngestionModule:
    """Handles conversion of PDF pages to high-resolution images."""
    
    def __init__(
        self,
        output_dir: str = "output/images",
        dpi: int = 300,
        fmt: str = "PNG",
        poppler_path: Optional[str] = None
    ):
        """
        Initialize the PDF ingestion module.
        
        Args:
            output_dir: Directory to save converted images
            dpi: Resolution for image conversion (default: 300 for high-res)
            fmt: Image format (PNG, JPEG, etc.)
            poppler_path: Path to poppler bin directory (auto-detected if None)
        """
        self.output_dir = Path(output_dir)
        self.dpi = dpi
        self.fmt = fmt
        
        # Auto-detect poppler path if not provided
        if poppler_path is None:
            poppler_path = self._detect_poppler_path()
        
        self.poppler_path = poppler_path
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory set to: {self.output_dir}")
        if self.poppler_path:
            logger.info(f"Using Poppler from: {self.poppler_path}")
    
    def _detect_poppler_path(self) -> Optional[str]:
        """
        Auto-detect poppler path from dependencies folder.
        
        Returns:
            Path to poppler bin directory, or None if not found
        """
        # Check for poppler in dependencies folder
        dependencies_dir = Path("dependencies")
        if dependencies_dir.exists():
            # Look for poppler-* directories
            poppler_dirs = list(dependencies_dir.glob("poppler-*/Library/bin"))
            if poppler_dirs:
                poppler_path = str(poppler_dirs[0])
                logger.info(f"Auto-detected Poppler path: {poppler_path}")
                return poppler_path
        
        logger.warning("Poppler not found in dependencies folder. Will try system PATH.")
        return None
    
    def convert_pdf_to_images(
        self,
        pdf_path: str,
        output_prefix: Optional[str] = None
    ) -> List[str]:
        """
        Convert a PDF file to high-resolution images (one per page).
        
        Args:
            pdf_path: Path to the PDF file
            output_prefix: Optional prefix for output filenames
            
        Returns:
            List of paths to the generated image files
            
        Raises:
            FileNotFoundError: If PDF file doesn't exist
            PDFInfoNotInstalledError: If poppler is not installed
            PDFSyntaxError: If PDF is corrupted
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        logger.info(f"Processing PDF: {pdf_path.name}")
        
        # Generate output prefix from PDF filename if not provided
        if output_prefix is None:
            output_prefix = pdf_path.stem
        
        # Create form-specific subdirectory
        form_output_dir = self.output_dir / output_prefix
        form_output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Form output directory: {form_output_dir}")
        
        try:
            # Convert PDF to images
            convert_kwargs = {
                "dpi": self.dpi,
                "fmt": self.fmt
            }
            
            # Add poppler_path if available
            if self.poppler_path:
                convert_kwargs["poppler_path"] = self.poppler_path
            
            images = convert_from_path(pdf_path, **convert_kwargs)
            
            logger.info(f"Converted {len(images)} page(s) from {pdf_path.name}")
            
            # Save each page as an image in the form-specific directory
            image_paths = []
            for i, image in enumerate(images, start=1):
                image_filename = f"page_{i:03d}.{self.fmt.lower()}"
                image_path = form_output_dir / image_filename
                
                image.save(image_path, self.fmt)
                image_paths.append(str(image_path))
                
                logger.info(f"Saved page {i} as: {form_output_dir.name}/{image_filename}")
            
            logger.info(f"Successfully converted {pdf_path.name} to {len(image_paths)} image(s) in {form_output_dir}")
            return image_paths
            
        except PDFInfoNotInstalledError:
            error_msg = (
                "Poppler not installed. Please install poppler-utils:\n"
                "  Windows: Download from https://github.com/oschwartz10612/poppler-windows/releases\n"
                "  macOS: brew install poppler\n"
                "  Linux: sudo apt-get install poppler-utils"
            )
            logger.error(error_msg)
            raise
            
        except PDFSyntaxError as e:
            logger.error(f"PDF syntax error: {e}")
            raise
            
        except PDFPageCountError as e:
            logger.error(f"Could not count PDF pages: {e}")
            raise
            
        except Exception as e:
            logger.error(f"Unexpected error processing PDF: {e}")
            raise
    
    def process_directory(
        self,
        directory: str,
        pattern: str = "*.pdf"
    ) -> dict:
        """
        Process all PDFs in a directory.
        
        Args:
            directory: Directory containing PDF files
            pattern: File pattern to match (default: *.pdf)
            
        Returns:
            Dictionary mapping PDF filenames to lists of image paths
        """
        directory = Path(directory)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        pdf_files = list(directory.glob(pattern))
        logger.info(f"Found {len(pdf_files)} PDF file(s) in {directory}")
        
        results = {}
        
        for pdf_file in pdf_files:
            try:
                image_paths = self.convert_pdf_to_images(pdf_file)
                results[pdf_file.name] = image_paths
            except Exception as e:
                logger.error(f"Failed to process {pdf_file.name}: {e}")
                results[pdf_file.name] = []
        
        return results


def main():
    """Main function to run the PDF ingestion module."""
    logger.info("=" * 60)
    logger.info("PDF Ingestion Module - Starting")
    logger.info("=" * 60)
    
    # Initialize the ingestion module
    ingestion_module = PDFIngestionModule(
        output_dir="output/images",
        dpi=300,
        fmt="PNG"
    )
    
    # Process PDFs in the forms directory
    forms_dir = Path("forms")
    
    if not forms_dir.exists():
        logger.warning(f"Forms directory not found: {forms_dir}")
        logger.info("Creating forms directory...")
        forms_dir.mkdir(exist_ok=True)
        logger.info("Please add PDF files to the forms directory and run again.")
        return
    
    # Process all PDFs
    results = ingestion_module.process_directory(forms_dir)
    
    # Summary
    logger.info("=" * 60)
    logger.info("PDF Ingestion Summary")
    logger.info("=" * 60)
    
    total_pages = 0
    for pdf_name, image_paths in results.items():
        page_count = len(image_paths)
        total_pages += page_count
        status = "[SUCCESS]" if page_count > 0 else "[FAILED]"
        form_name = Path(pdf_name).stem
        logger.info(f"{status} - {pdf_name}: {page_count} page(s) -> {form_name}/")
    
    logger.info(f"Total pages processed: {total_pages}")
    logger.info(f"Images organized by form in: {ingestion_module.output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

