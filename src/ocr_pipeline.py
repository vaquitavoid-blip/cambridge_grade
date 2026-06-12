# src/ocr_pipeline.py
# ─────────────────────────────────────────────────────────────────────────────
# OCR Pipeline — extracts clean essay text from uploaded PDFs and images
# Uses pdfplumber for digital PDFs (no OCR needed, perfect accuracy)
# Falls back to pytesseract OCR for scanned PDFs and photos of handwriting
# ─────────────────────────────────────────────────────────────────────────────

import re
import sys
import tempfile
from pathlib import Path
from typing import Optional

# ── Optional imports — graceful fallback if not installed ─────────────────────
try:
    import pdfplumber
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

try:
    from PIL import Image
    import pytesseract
    OCR_SUPPORT = True
except ImportError:
    OCR_SUPPORT = False

try:
    import fitz  # PyMuPDF — converts PDF pages to images for OCR
    PYMUPDF_SUPPORT = True
except ImportError:
    PYMUPDF_SUPPORT = False


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────

def extract_text_from_file(file_path: str) -> tuple[str, str]:
    """
    Extracts essay text from a PDF or image file.

    Returns:
        (extracted_text, method_used)
        method_used is one of: 'pdf_digital', 'pdf_ocr', 'image_ocr', 'error'
    """
    path = Path(file_path)
    ext  = path.suffix.lower()

    if ext == ".pdf":
        return _extract_from_pdf(file_path)
    elif ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"):
        return _extract_from_image(file_path)
    else:
        return "", "error: unsupported file type"


def extract_text_from_bytes(file_bytes: bytes, filename: str) -> tuple[str, str]:
    """
    Same as extract_text_from_file but takes raw bytes (for Streamlit uploads).
    Writes to a temp file, extracts, then cleans up.
    """
    ext = Path(filename).suffix.lower()
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        text, method = extract_text_from_file(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return text, method


# ─────────────────────────────────────────────────────────────────────────────
# PDF EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def _extract_from_pdf(path: str) -> tuple[str, str]:
    """
    Strategy:
    1. Try pdfplumber to extract digital text (typed PDFs — perfect quality)
    2. If text is empty/garbage, fall back to OCR via PyMuPDF + pytesseract
    """
    # Attempt 1: digital text extraction
    if PDF_SUPPORT:
        text = _pdfplumber_extract(path)
        if _is_good_text(text):
            return _clean_text(text), "pdf_digital"

    # Attempt 2: OCR (scanned PDF / photo of paper)
    if PYMUPDF_SUPPORT and OCR_SUPPORT:
        text = _pdf_ocr_extract(path)
        if text:
            return _clean_text(text), "pdf_ocr"

    if not PDF_SUPPORT:
        return "", "error: pdfplumber not installed. Run: pip install pdfplumber"
    if not OCR_SUPPORT:
        return "", "error: pytesseract not installed. Run: pip install pytesseract pillow"

    return "", "error: could not extract text from PDF"


def _pdfplumber_extract(path: str) -> str:
    """Extract text from a digitally-typed PDF."""
    pages_text = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text)
    return "\n\n".join(pages_text)


def _pdf_ocr_extract(path: str) -> str:
    """
    Convert each PDF page to an image, then run Tesseract OCR.
    Used for scanned/photographed exam scripts.
    """
    doc   = fitz.open(path)
    pages_text = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        # Render at 300 DPI for good OCR accuracy
        mat  = fitz.Matrix(300 / 72, 300 / 72)
        pix  = page.get_pixmap(matrix=mat)

        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img = _preprocess_for_ocr(img)

        text = pytesseract.image_to_string(
            img,
            config="--oem 3 --psm 6",  # OEM 3 = best engine, PSM 6 = assume uniform block of text
        )
        pages_text.append(text)

    doc.close()
    return "\n\n".join(pages_text)


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def _extract_from_image(path: str) -> tuple[str, str]:
    """Extract text from a photo of a handwritten or printed essay."""
    if not OCR_SUPPORT:
        return "", "error: pytesseract not installed. Run: pip install pytesseract pillow"

    img  = Image.open(path)
    img  = _preprocess_for_ocr(img)
    text = pytesseract.image_to_string(
        img,
        config="--oem 3 --psm 6",
    )
    return _clean_text(text), "image_ocr"


def _preprocess_for_ocr(img: "Image.Image") -> "Image.Image":
    """
    Improves OCR accuracy by:
    - Converting to grayscale
    - Upscaling small images
    - Enhancing contrast
    """
    from PIL import ImageEnhance, ImageFilter

    # Convert to grayscale
    img = img.convert("L")

    # Upscale if image is too small (OCR struggles under 1000px wide)
    w, h = img.size
    if w < 1200:
        scale = 1200 / w
        img   = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    # Sharpen slightly
    img = img.filter(ImageFilter.SHARPEN)

    # Boost contrast
    enhancer = ImageEnhance.Contrast(img)
    img      = enhancer.enhance(1.5)

    return img


# ─────────────────────────────────────────────────────────────────────────────
# TEXT CLEANING
# ─────────────────────────────────────────────────────────────────────────────

def _is_good_text(text: str) -> bool:
    """
    Returns True if the extracted text looks like real essay content.
    Rejects: empty strings, strings with too many non-ASCII chars (garbled OCR),
    or strings that are mostly whitespace/numbers (table of contents, etc.)
    """
    if not text or len(text.strip()) < 100:
        return False

    # Check ratio of normal ASCII letters to total characters
    letters = sum(1 for c in text if c.isalpha())
    total   = len(text.replace(" ", "").replace("\n", ""))
    if total == 0:
        return False

    ratio = letters / total
    return ratio > 0.6  # At least 60% of chars should be letters


def _clean_text(text: str) -> str:
    """
    Cleans OCR/PDF output into readable essay text:
    - Removes page numbers, headers, footers
    - Fixes hyphenated line breaks (e.g. "govern-\nment" → "government")
    - Normalises whitespace
    - Removes common OCR artefacts
    """
    if not text:
        return ""

    # Fix hyphenated line breaks
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Remove standalone page numbers (lines with just a number)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)

    # Remove common header/footer patterns
    text = re.sub(r"(Cambridge International|UCLES|Page \d+|Turn over)", "", text, flags=re.IGNORECASE)

    # Collapse multiple blank lines into one paragraph break
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Fix OCR artefacts: replace common misreads
    replacements = {
        "|":   "I",    # vertical bar often misread as capital I
        "l ":  "I ",   # lowercase l at start sometimes = I
        " 0 ": " o ",  # zero vs letter o in context
    }
    for wrong, right in replacements.items():
        text = text.replace(wrong, right)

    # Normalise smart quotes to regular quotes
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')

    # Strip leading/trailing whitespace per line
    lines = [line.rstrip() for line in text.splitlines()]
    text  = "\n".join(lines).strip()

    return text


# ─────────────────────────────────────────────────────────────────────────────
# QUALITY REPORT
# ─────────────────────────────────────────────────────────────────────────────

def ocr_quality_report(text: str, method: str) -> dict:
    """
    Returns a quality report so the UI can warn users if OCR was poor.
    """
    word_count = len(text.split())
    has_econ   = any(w in text.lower() for w in [
        "demand", "supply", "market", "price", "inflation",
        "gdp", "unemployment", "fiscal", "monetary", "elasticity",
    ])

    quality = "good"
    warnings = []

    if method in ("pdf_ocr", "image_ocr"):
        if word_count < 80:
            quality = "poor"
            warnings.append("Very few words extracted — image may be blurry or too dark.")
        elif word_count < 150:
            quality = "fair"
            warnings.append("Fewer words than expected — check the extracted text looks correct.")

        if not has_econ:
            warnings.append("No economics keywords detected — make sure this is an economics essay.")

    return {
        "method":     method,
        "word_count": word_count,
        "quality":    quality,
        "warnings":   warnings,
    }