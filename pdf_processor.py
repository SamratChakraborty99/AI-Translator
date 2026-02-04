"""
PDF Text Extraction Utility
Extracts and processes text from PDF files with OCR support for scanned PDFs
"""
import io
import logging
import os
from typing import Optional
from PyPDF2 import PdfReader

from config import settings

logger = logging.getLogger(__name__)

# Try to import OCR libraries
OCR_AVAILABLE = False
try:
    import pytesseract
    from pdf2image import convert_from_bytes
    from PIL import Image
    OCR_AVAILABLE = True
    
    # Set Tesseract path for Windows (common installation paths)
    tesseract_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        r"C:\Users\{}\AppData\Local\Tesseract-OCR\tesseract.exe".format(os.getenv('USERNAME', '')),
    ]
    
    for path in tesseract_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            break
    
    logger.info("OCR support enabled with pytesseract")
except ImportError as e:
    logger.warning(f"OCR libraries not available: {e}. OCR for scanned PDFs will be disabled.")


class PDFProcessor:
    """Handles PDF file processing and text extraction with OCR fallback"""
    
    MAX_FILE_SIZE = settings.MAX_FILE_SIZE_MB * 1024 * 1024  # Convert to bytes
    
    @staticmethod
    def validate_file(file_content: bytes, filename: str) -> tuple[bool, Optional[str]]:
        """
        Validate the uploaded file
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check file extension
        if not filename.lower().endswith('.pdf'):
            return False, "Only PDF files are allowed"
        
        # Check file size
        if len(file_content) > PDFProcessor.MAX_FILE_SIZE:
            return False, f"File size exceeds maximum limit of {settings.MAX_FILE_SIZE_MB}MB"
        
        # Check if file is empty
        if len(file_content) == 0:
            return False, "Uploaded file is empty"
        
        return True, None
    
    @staticmethod
    def extract_text_with_pypdf2(file_content: bytes) -> str:
        """Extract text using PyPDF2 (for text-based PDFs)"""
        pdf_file = io.BytesIO(file_content)
        reader = PdfReader(pdf_file)
        
        text_parts = []
        for page_num, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            except Exception as e:
                logger.warning(f"Error extracting text from page {page_num + 1}: {str(e)}")
                continue
        
        return "\n\n".join(text_parts)
    
    @staticmethod
    def extract_text_with_ocr(file_content: bytes) -> str:
        """Extract text using OCR (for scanned/image-based PDFs)"""
        if not OCR_AVAILABLE:
            raise ValueError(
                "This PDF appears to be scanned/image-based. "
                "OCR support is not available. Please install Tesseract OCR."
            )
        
        try:
            logger.info("Using OCR to extract text from scanned PDF...")
            
            # Poppler path for Windows (installed via winget)
            poppler_path = r"C:\Users\Samrat\AppData\Local\Microsoft\WinGet\Packages\oschwartz10612.Poppler_Microsoft.Winget.Source_8wekyb3d8bbwe\poppler-25.07.0\Library\bin"
            
            # Convert PDF pages to images
            images = convert_from_bytes(file_content, dpi=300, poppler_path=poppler_path)
            
            text_parts = []
            for page_num, image in enumerate(images):
                try:
                    # Extract text using Tesseract OCR with multiple language support
                    # eng = English, hin = Hindi, script/Devanagari = Devanagari script
                    # Using multiple languages to auto-detect the content
                    page_text = pytesseract.image_to_string(image, lang='eng+hin+mar+san+ben+guj+tam+tel+kan+mal+pan+ori+urd')
                    if page_text.strip():
                        text_parts.append(page_text)
                    logger.info(f"OCR completed for page {page_num + 1}")
                except Exception as e:
                    logger.warning(f"OCR error on page {page_num + 1}: {str(e)}")
                    continue
            
            return "\n\n".join(text_parts)
            
        except Exception as e:
            logger.error(f"OCR processing failed: {str(e)}")
            raise ValueError(f"OCR processing failed: {str(e)}")
    
    @staticmethod
    def extract_text(file_content: bytes) -> str:
        """
        Extract text from a PDF file with OCR fallback
        
        Args:
            file_content: PDF file content as bytes
        
        Returns:
            Extracted text from the PDF
        """
        try:
            # First, try PyPDF2 for text-based PDFs
            logger.info("Attempting text extraction with PyPDF2...")
            text = PDFProcessor.extract_text_with_pypdf2(file_content)
            cleaned_text = PDFProcessor.clean_text(text)
            
            # If we got meaningful text, return it
            if cleaned_text.strip() and len(cleaned_text.strip()) > 50:
                logger.info(f"Successfully extracted {len(cleaned_text)} characters with PyPDF2")
                return cleaned_text
            
            # If text is too short or empty, try OCR
            logger.info("Minimal text found with PyPDF2, attempting OCR...")
            ocr_text = PDFProcessor.extract_text_with_ocr(file_content)
            cleaned_ocr_text = PDFProcessor.clean_text(ocr_text)
            
            if cleaned_ocr_text.strip():
                logger.info(f"Successfully extracted {len(cleaned_ocr_text)} characters with OCR")
                return cleaned_ocr_text
            
            raise ValueError("No text could be extracted from the PDF (tried both text extraction and OCR)")
            
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise ValueError(f"Failed to extract text from PDF: {str(e)}")
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean and normalize extracted text
        
        Args:
            text: Raw extracted text
        
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Strip whitespace
            line = line.strip()
            # Skip empty lines if the previous line was also empty
            if line or (cleaned_lines and cleaned_lines[-1]):
                cleaned_lines.append(line)
        
        # Join lines and remove excessive newlines
        cleaned_text = '\n'.join(cleaned_lines)
        
        # Replace multiple spaces with single space
        while '  ' in cleaned_text:
            cleaned_text = cleaned_text.replace('  ', ' ')
        
        # Replace multiple newlines with double newline
        while '\n\n\n' in cleaned_text:
            cleaned_text = cleaned_text.replace('\n\n\n', '\n\n')
        
        return cleaned_text.strip()


# Global processor instance
pdf_processor = PDFProcessor()
