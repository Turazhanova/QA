# ocr_utils.py
from __future__ import annotations
from typing import Optional, List
import io
import pytesseract
from PIL import Image
import pypdfium2 as pdfium

# OCR for a single PIL image
def ocr_image(img: Image.Image, lang: str = "eng") -> str:
    return pytesseract.image_to_string(img, lang=lang) or ""

# OCR for image file path (png/jpg/jpeg)
def ocr_image_file(path: str, lang: str = "eng") -> str:
    with Image.open(path) as im:
        return ocr_image(im, lang=lang)

# Render a PDF page to a PIL image (high DPI for better OCR)
def _render_pdf_pages_to_images(pdf_path: str, scale: float = 2.0) -> List[Image.Image]:
    pdf = pdfium.PdfDocument(pdf_path)
    images: List[Image.Image] = []
    for i in range(len(pdf)):
        page = pdf[i]
        pil_image = page.render(scale=scale).to_pil()
        images.append(pil_image)
    return images

# OCR a full PDF via rendering pages -> OCR
def ocr_pdf(pdf_path: str, lang: str = "eng") -> str:
    pages = _render_pdf_pages_to_images(pdf_path, scale=2.0)
    texts = []
    for img in pages:
        texts.append(ocr_image(img, lang=lang))
    return "\n".join(texts).strip()
