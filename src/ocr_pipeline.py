# src/ocr_pipeline.py
# ─────────────────────────────────────────────────────────────────────────────
# OCR Pipeline — extracts clean essay text from uploaded PDFs and images
#
# Strategy:
# 1. Digital PDFs -> pdfplumber (perfect accuracy, instant)
# 2. Scanned PDFs / photos of PRINTED text -> Tesseract OCR (good accuracy)
# 3. Handwriting -> Tesseract is weak; if GOOGLE_VISION_API_KEY is configured,
#    use Google Cloud Vision instead (much better for handwriting, free tier
#    covers ~1000 images/month). Otherwise falls back to Tesseract with a
#    warning so the UI can tell the user to verify/retype.
#
# Supports multiple images in one submission (e.g. a multi-page handwritten
# essay photographed page by page) — combine_images_to_text() handles this.
# ─────────────────────────────────────────────────────────────────────────────

import re
import os
import sys
import base64
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

try:
    import requests
    REQUESTS_SUPPORT = True
except ImportError:
    REQUESTS_SUPPORT = False


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

def _get_google_vision_key() -> Optional[str]:
    """
    Checks for a Google Cloud Vision API key in (in order):
    1. Streamlit secrets (st.secrets['google_vision_api_key'])
    2. Environment variable GOOGLE_VISION_API_KEY
    Returns None if not configured — Tesseract is used instead.
    """
    try:
        import streamlit as st
        if "google_vision_api_key" in st.secrets:
            return st.secrets["google_vision_api_key"]
    except Exception:
        pass
    return os.environ.get("GOOGLE_VISION_API_KEY")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXTRACTOR — single file
# ─────────────────────────────────────────────────────────────────────────────

def extract_text_from_file(file_path: str) -> tuple[str, str]:
    """
    Extracts essay text from a PDF or image file.

    Returns:
        (extracted_text, method_used)
        method_used is one of:
          'pdf_digital', 'pdf_ocr', 'image_ocr_tesseract',
          'image_ocr_vision', 'error'
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
# MULTI-IMAGE SUPPORT — combine several uploaded pages into one essay
# ─────────────────────────────────────────────────────────────────────────────

def extract_text_from_multiple_files(files: list[tuple[bytes, str]]) -> tuple[str, str, list[dict]]:
    """
    Processes multiple uploaded files (e.g. several photos of essay pages,
    in order) and combines them into one essay text.

    Args:
        files: list of (file_bytes, filename) tuples, in page order

    Returns:
        (combined_text, overall_method, per_file_reports)
        overall_method is the method used for the FIRST file (representative)
        per_file_reports is a list of {filename, method, word_count} dicts
    """
    page_texts = []
    reports    = []
    methods_used = []

    for file_bytes, filename in files:
        text, method = extract_text_from_bytes(file_bytes, filename)
        page_texts.append(text.strip())
        methods_used.append(method)
        reports.append({
            "filename":   filename,
            "method":     method,
            "word_count": len(text.split()),
        })

    combined = "\n\n".join(p for p in page_texts if p)
    combined = _clean_text(combined)

    # Pick the most "serious" method as representative for quality reporting
    overall_method = methods_used[0] if methods_used else "error"

    return combined, overall_method, reports


# ─────────────────────────────────────────────────────────────────────────────
# PDF EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def _extract_from_pdf(path: str) -> tuple[str, str]:
    """
    Strategy:
    1. Try pdfplumber to extract digital text (typed PDFs — perfect quality)
    2. If text is empty/garbage, fall back to OCR via PyMuPDF + pytesseract/Vision
    """
    if PDF_SUPPORT:
        text = _pdfplumber_extract(path)
        if _is_good_text(text):
            return _clean_text(text), "pdf_digital"

    if PYMUPDF_SUPPORT and OCR_SUPPORT:
        text, method = _pdf_ocr_extract(path)
        if text:
            return _clean_text(text), method

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


def _pdf_ocr_extract(path: str) -> tuple[str, str]:
    """
    Convert each PDF page to an image, then OCR it (Vision or Tesseract).
    Used for scanned/photographed exam scripts.
    """
    doc = fitz.open(path)
    pages_text = []
    method_used = "image_ocr_tesseract"

    for page_num in range(len(doc)):
        page = doc[page_num]
        mat  = fitz.Matrix(300 / 72, 300 / 72)  # 300 DPI
        pix  = page.get_pixmap(matrix=mat)
        img  = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        text, method = _ocr_image(img)
        method_used = method
        pages_text.append(text)

    doc.close()
    return "\n\n".join(pages_text), "pdf_ocr" if method_used == "image_ocr_tesseract" else "pdf_ocr_vision"


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def _extract_from_image(path: str) -> tuple[str, str]:
    """Extract text from a photo of a handwritten or printed essay."""
    if not OCR_SUPPORT:
        return "", "error: pytesseract not installed. Run: pip install pytesseract pillow"

    img = Image.open(path)
    text, method = _ocr_image(img)
    return _clean_text(text), method


def _ocr_image(img: "Image.Image") -> tuple[str, str]:
    """
    Runs OCR on a single PIL image, using the best available engine:
    - Google Cloud Vision (if API key configured) — much better for handwriting
    - Tesseract with multiple preprocessing/PSM attempts, best result chosen
    """
    vision_key = _get_google_vision_key()
    if vision_key and REQUESTS_SUPPORT:
        text = _vision_ocr(img, vision_key)
        if text:
            return text, "image_ocr_vision"
        # If Vision fails (quota/error), fall through to Tesseract

    if not OCR_SUPPORT:
        return "", "error: pytesseract not installed"

    return _tesseract_ocr_best_effort(img), "image_ocr_tesseract"


def _tesseract_ocr_best_effort(img: "Image.Image") -> str:
    """
    Runs Tesseract with several preprocessing variants and PSM modes,
    returns the result with the most extracted words (a rough proxy
    for "most successfully read").
    """
    candidates = []

    preprocessed_variants = {
        "standard":   _preprocess_for_ocr(img, mode="standard"),
        "binarized":  _preprocess_for_ocr(img, mode="binarized"),
        "high_contrast": _preprocess_for_ocr(img, mode="high_contrast"),
    }

    psm_modes = ["6", "3", "4"]  # uniform block, full auto, single column

    for variant_name, variant_img in preprocessed_variants.items():
        for psm in psm_modes:
            try:
                text = pytesseract.image_to_string(
                    variant_img,
                    config=f"--oem 3 --psm {psm}",
                )
                word_count = len(text.split())
                candidates.append((word_count, text))
            except Exception:
                continue

    if not candidates:
        return ""

    # Pick the candidate with the most extracted words
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _vision_ocr(img: "Image.Image", api_key: str) -> str:
    """
    Sends an image to Google Cloud Vision API for OCR.
    Much stronger on handwriting than Tesseract.
    Requires a free Google Cloud API key with Vision API enabled.
    """
    import io

    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=90)
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"
    payload = {
        "requests": [{
            "image": {"content": img_b64},
            "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
        }]
    }

    try:
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        annotation = data.get("responses", [{}])[0].get("fullTextAnnotation", {})
        return annotation.get("text", "")
    except Exception as e:
        print(f"[ocr_pipeline] Google Vision OCR failed: {e}")
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def _preprocess_for_ocr(img: "Image.Image", mode: str = "standard") -> "Image.Image":
    """
    Improves OCR accuracy with several preprocessing strategies.

    mode:
      "standard"      - grayscale, upscale, sharpen, mild contrast boost
      "binarized"      - converts to pure black/white using adaptive threshold
                          (helps with shadows/uneven lighting on photos)
      "high_contrast"  - aggressive contrast + sharpening for faint pencil writing
    """
    from PIL import ImageEnhance, ImageFilter
    import numpy as np

    img = img.convert("L")  # grayscale

    # Upscale if image is too small (OCR struggles under ~1500px wide)
    w, h = img.size
    target_w = 1800
    if w < target_w:
        scale = target_w / w
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    if mode == "standard":
        img = img.filter(ImageFilter.SHARPEN)
        img = ImageEnhance.Contrast(img).enhance(1.5)

    elif mode == "binarized":
        # Adaptive-ish thresholding using local mean
        arr = np.array(img).astype("float32")
        # Simple global threshold based on image's own mean brightness
        threshold = arr.mean() * 0.85
        binary = (arr > threshold) * 255
        img = Image.fromarray(binary.astype("uint8"))

    elif mode == "high_contrast":
        img = ImageEnhance.Contrast(img).enhance(2.2)
        img = ImageEnhance.Sharpness(img).enhance(2.0)
        img = img.filter(ImageFilter.SHARPEN)

    return img


# ─────────────────────────────────────────────────────────────────────────────
# TEXT CLEANING
# ─────────────────────────────────────────────────────────────────────────────

def _is_good_text(text: str) -> bool:
    """
    Returns True if the extracted text looks like real essay content.
    """
    if not text or len(text.strip()) < 100:
        return False

    letters = sum(1 for c in text if c.isalpha())
    total   = len(text.replace(" ", "").replace("\n", ""))
    if total == 0:
        return False

    ratio = letters / total
    return ratio > 0.6


def _clean_text(text: str) -> str:
    """
    Cleans OCR/PDF output into readable essay text.
    """
    if not text:
        return ""

    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"(Cambridge International|UCLES|Page \d+|Turn over)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n{3,}", "\n\n", text)

    replacements = {
        "|":   "I",
        "l ":  "I ",
        " 0 ": " o ",
    }
    for wrong, right in replacements.items():
        text = text.replace(wrong, right)

    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')

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

    quality  = "good"
    warnings = []

    is_ocr = method in ("pdf_ocr", "image_ocr_tesseract", "image_ocr_vision", "pdf_ocr_vision")

    if is_ocr:
        if word_count < 80:
            quality = "poor"
            warnings.append("Very few words extracted — image may be blurry, dark, or hard to read.")
        elif word_count < 150:
            quality = "fair"
            warnings.append("Fewer words than expected — please check the extracted text carefully.")

        if not has_econ:
            warnings.append("No economics keywords detected — make sure this is an economics essay.")

        if method == "image_ocr_tesseract":
            warnings.append(
                "OCR used (Tesseract) — this works best with neat, printed handwriting or typed text. "
                "If the text below looks garbled, please correct it manually or type the essay instead."
            )

    return {
        "method":     method,
        "word_count": word_count,
        "quality":    quality,
        "warnings":   warnings,
    }