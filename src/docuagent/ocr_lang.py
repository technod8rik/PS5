"""OCR and language identification for document elements."""

import os
import urllib.request
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import cv2
import numpy as np
from dataclasses import dataclass
from paddleocr import PaddleOCR

from .layout import LayoutBox


@dataclass
class Element:
    """Document element with content and metadata."""
    page: int
    cls: str  # Text/Title/List/Table/Figure
    bbox: Tuple[int, int, int, int]  # HBB [x, y, w, h]
    content: str | None  # text for textual classes, else None
    language: str | None
    meta: dict  # optional: confidences, etc.


class OCRLang:
    """OCR and language identification for document elements."""
    
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.ocr_lang = cfg["ocr"]["lang"]
        self.langid_model_path = cfg["langid"]["model_path"]
        self.debug_dir = Path(cfg["io"]["debug_dir"])
        
        # Initialize PaddleOCR
        self.ocr = PaddleOCR(
            use_angle_cls=True, 
            lang=self.ocr_lang, 
            show_log=False
        )
        
        # Initialize language ID model
        self.langid_model = self._load_langid_model()
    
    def _load_langid_model(self):
        """Load fastText language ID model."""
        model_path = Path(self.langid_model_path)
        
        # Download model if not present
        if not model_path.exists():
            print(f"[INFO] Downloading fastText language ID model to {model_path}")
            model_path.parent.mkdir(parents=True, exist_ok=True)
            self._download_fasttext_model(model_path)
        
        try:
            import fasttext
            return fasttext.load_model(str(model_path))
        except ImportError:
            print("[WARN] fasttext not available. Install with: pip install fasttext")
            return None
    
    def _download_fasttext_model(self, model_path: Path):
        """Download fastText lid.176.bin model."""
        url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
        try:
            urllib.request.urlretrieve(url, str(model_path))
            print(f"[INFO] Downloaded language ID model to {model_path}")
        except Exception as e:
            print(f"[ERROR] Failed to download language ID model: {e}")
            raise RuntimeError("Could not download language ID model")
    
    def run(self, image_bgr: np.ndarray, boxes: List[LayoutBox], page_idx: int) -> List[Element]:
        """Run OCR and language ID on layout boxes.
        
        Args:
            image_bgr: Input image in BGR format
            boxes: Layout detection results
            page_idx: Page index
            
        Returns:
            List of Element objects with OCR text and language codes
        """
        elements = []
        
        for box in boxes:
            # Crop image to box region
            x, y, w, h = box.bbox
            crop = image_bgr[y:y+h, x:x+w]
            
            if crop.size == 0:
                continue
            
            # Process based on element type
            if box.class_name in ["Text", "Title", "List"]:
                # Run OCR for textual elements
                element = self._process_textual_element(crop, box, page_idx)
            else:
                # Create stub element for non-textual elements
                element = Element(
                    page=page_idx,
                    cls=box.class_name,
                    bbox=box.bbox,
                    content=None,
                    language=None,
                    meta={"source": "layout_detection", "confidence": box.score}
                )
            
            elements.append(element)
        
        # Sort by reading order (page, y, x)
        elements.sort(key=lambda e: (e.page, e.bbox[1], e.bbox[0]))
        
        # Save debug OCR results
        self._save_debug_ocr(elements, page_idx)
        
        return elements
    
    def _process_textual_element(self, crop: np.ndarray, box: LayoutBox, page_idx: int) -> Element:
        """Process textual element with OCR and language ID."""
        # Run OCR on crop
        ocr_result = self.ocr.ocr(crop, cls=True)
        
        # Extract text lines
        text_lines = []
        if ocr_result and ocr_result[0]:
            for line in ocr_result[0]:
                if line and len(line) >= 2:
                    text = line[1][0]
                    confidence = line[1][1]
                    if text and text.strip():
                        text_lines.append((text.strip(), confidence))
        
        # Combine text lines
        if text_lines:
            combined_text = " ".join([line[0] for line in text_lines])
            avg_confidence = sum(line[1] for line in text_lines) / len(text_lines)
        else:
            combined_text = ""
            avg_confidence = 0.0
        
        # Language identification
        language, lang_confidence = self._identify_language(combined_text)
        
        return Element(
            page=page_idx,
            cls=box.class_name,
            bbox=box.bbox,
            content=combined_text if combined_text else None,
            language=language,
            meta={
                "source": "paddle_ocr",
                "confidence": avg_confidence,
                "text_lines": len(text_lines),
                "layout_confidence": box.score,
                "lang_confidence": lang_confidence
            }
        )
    
    def _identify_language(self, text: str) -> Tuple[Optional[str], float]:
        """Identify language of text using fastText.
        
        Returns:
            Tuple of (language_code, confidence)
        """
        if not text or not self.langid_model:
            return None, 0.0
        
        try:
            # Clean text for language ID
            clean_text = text.strip()
            if len(clean_text) < 3:  # Too short for reliable language ID
                return None, 0.0
            
            # Get language prediction
            prediction = self.langid_model.predict(clean_text)
            if prediction and len(prediction) > 0:
                # Extract language code and confidence
                lang_code = prediction[0][0].replace("__label__", "")
                confidence = prediction[0][1] if len(prediction[0]) > 1 else 1.0
                
                # Threshold for confidence
                if confidence < 0.55:
                    return "unknown", confidence
                
                return lang_code, confidence
        except Exception as e:
            print(f"[WARN] Language ID failed: {e}")
        
        return None, 0.0
    
    def _save_debug_ocr(self, elements: List[Element], page_idx: int):
        """Save debug OCR results as TSV."""
        debug_path = self.debug_dir / f"page_{page_idx}_ocr.tsv"
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(debug_path, 'w', encoding='utf-8') as f:
            f.write("class\tbbox\tcontent\tlanguage\tconfidence\n")
            for element in elements:
                bbox_str = f"{element.bbox[0]},{element.bbox[1]},{element.bbox[2]},{element.bbox[3]}"
                content = element.content or ""
                language = element.language or ""
                confidence = element.meta.get("confidence", 0.0)
                
                f.write(f"{element.cls}\t{bbox_str}\t{content}\t{language}\t{confidence:.3f}\n")
        
        print(f"[DEBUG] Saved OCR results to {debug_path}")


def crop_image_by_bbox(image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Crop image by bounding box.
    
    Args:
        image: Input image
        bbox: Bounding box as (x, y, w, h)
        
    Returns:
        Cropped image
    """
    x, y, w, h = bbox
    return image[y:y+h, x:x+w]

#working
