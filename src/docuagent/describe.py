"""VLM-based description of non-text elements (tables, figures, charts)."""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np
from PIL import Image
import torch

from .ocr_lang import Element


class Describer:
    """VLM-based description of non-text elements."""
    
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.model_name = cfg["vlm"]["model"]
        self.batch_size = cfg["describe"]["batch_size"]
        self.device = cfg["device"]
        self.debug_dir = Path(cfg["io"]["debug_dir"])
        
        # Initialize VLM model
        self.model = self._load_vlm_model()
    
    def _load_vlm_model(self):
        """Load VLM model for image description."""
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
            from transformers import pipeline
            
            print(f"[INFO] Loading VLM model: {self.model_name}")
            
            # Load model and processor
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            processor = AutoProcessor.from_pretrained(self.model_name)
            
            # Create pipeline
            vlm_pipeline = pipeline(
                "image-to-text",
                model=model,
                processor=processor,
                device=0 if self.device == "cuda" else -1,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            return vlm_pipeline
            
        except ImportError:
            print("[WARN] transformers not available. Install with: pip install transformers")
            return None
        except Exception as e:
            print(f"[WARN] Failed to load VLM model: {e}")
            return None
    
    def describe_elements(self, image_bgr: np.ndarray, elements: List[Element]) -> List[Element]:
        """Generate descriptions for non-text elements.
        
        Args:
            image_bgr: Input image in BGR format
            elements: List of elements (some may already have content)
            
        Returns:
            Updated elements with descriptions for non-text elements
        """
        if not self.model:
            print("[WARN] VLM model not available, skipping descriptions")
            return elements
        
        # Filter elements that need descriptions
        non_text_elements = [e for e in elements if e.cls in ["Table", "Figure"] and e.content is None]
        
        if not non_text_elements:
            return elements
        
        print(f"[INFO] Describing {len(non_text_elements)} non-text elements")
        
        # Process in batches
        updated_elements = []
        for i in range(0, len(non_text_elements), self.batch_size):
            batch = non_text_elements[i:i + self.batch_size]
            batch_descriptions = self._describe_batch(image_bgr, batch)
            
            for element, description in zip(batch, batch_descriptions):
                # Update element with description
                updated_element = Element(
                    page=element.page,
                    cls=element.cls,
                    bbox=element.bbox,
                    content=description,
                    language=element.language,
                    meta={**element.meta, "source": "vlm_description"}
                )
                updated_elements.append(updated_element)
        
        # Replace non-text elements in original list
        result = []
        non_text_set = set(id(e) for e in non_text_elements)
        
        for element in elements:
            if id(element) in non_text_set:
                # Find corresponding updated element
                for updated in updated_elements:
                    if (updated.page == element.page and 
                        updated.bbox == element.bbox and 
                        updated.cls == element.cls):
                        result.append(updated)
                        break
                else:
                    result.append(element)  # Fallback if not found
            else:
                result.append(element)
        
        return result
    
    def _describe_batch(self, image_bgr: np.ndarray, elements: List[Element]) -> List[Optional[str]]:
        """Describe a batch of non-text elements."""
        descriptions = []
        
        for element in elements:
            # Crop image to element region
            x, y, w, h = element.bbox
            crop = image_bgr[y:y+h, x:x+w]
            
            if crop.size == 0:
                descriptions.append(None)
                continue
            
            # Convert BGR to RGB for PIL
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(crop_rgb)
            
            # Generate description based on element type
            if element.cls == "Table":
                description = self._describe_table(pil_image)
            elif element.cls == "Figure":
                description = self._describe_figure(pil_image)
            else:
                description = None
            
            descriptions.append(description)
        
        return descriptions
    
    def _describe_table(self, image: Image.Image) -> Optional[str]:
        """Describe a table element."""
        prompt = ("Describe this table briefly: what variables, units, and notable values/patterns appear? "
                 "Focus on the structure, column headers, and key data points.")
        
        try:
            result = self.model(image, prompt=prompt, max_new_tokens=150)
            if result and len(result) > 0:
                description = result[0]["generated_text"]
                # Clean up the description
                description = description.strip()
                if description.startswith(prompt):
                    description = description[len(prompt):].strip()
                return description if description else None
        except Exception as e:
            print(f"[WARN] Table description failed: {e}")
        
        return None
    
    def _describe_figure(self, image: Image.Image) -> Optional[str]:
        """Describe a figure element."""
        prompt = ("Describe this image. If it is a chart/map/diagram, summarize axes, trends, and key takeaways. "
                 "Keep the description concise and informative.")
        
        try:
            result = self.model(image, prompt=prompt, max_new_tokens=150)
            if result and len(result) > 0:
                description = result[0]["generated_text"]
                # Clean up the description
                description = description.strip()
                if description.startswith(prompt):
                    description = description[len(prompt):].strip()
                return description if description else None
        except Exception as e:
            print(f"[WARN] Figure description failed: {e}")
        
        return None
    
    def _save_debug_descriptions(self, elements: List[Element], page_idx: int):
        """Save debug descriptions."""
        debug_path = self.debug_dir / f"page_{page_idx}_descriptions.txt"
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(debug_path, 'w', encoding='utf-8') as f:
            f.write(f"Page {page_idx} - Element Descriptions\n")
            f.write("=" * 50 + "\n\n")
            
            for i, element in enumerate(elements):
                if element.cls in ["Table", "Figure"] and element.content:
                    f.write(f"Element {i+1}: {element.cls}\n")
                    f.write(f"BBox: {element.bbox}\n")
                    f.write(f"Description: {element.content}\n")
                    f.write("-" * 30 + "\n\n")
        
        print(f"[DEBUG] Saved descriptions to {debug_path}")


def resize_image_for_vlm(image: Image.Image, max_size: int = 1024) -> Image.Image:
    """Resize image for VLM processing while maintaining aspect ratio."""
    width, height = image.size
    
    if max(width, height) <= max_size:
        return image
    
    if width > height:
        new_width = max_size
        new_height = int(height * max_size / width)
    else:
        new_height = max_size
        new_width = int(width * max_size / height)
    
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
