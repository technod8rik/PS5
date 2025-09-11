"""Layout detection with PP-DocLayout-L and YOLOv10 backends."""

import os
from pathlib import Path
from typing import List, Tuple, Dict, Any
import cv2
import numpy as np
from dataclasses import dataclass


class LayoutBox:
    """Layout detection result with HBB format."""
    def __init__(self, class_name: str, bbox: Tuple[int, int, int, int], score: float):
        self.class_name = class_name
        self.bbox = bbox  # [x, y, w, h]
        self.score = score


class LayoutDetector:
    """Layout detection with pluggable backends."""
    
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.model_name = cfg["layout"]["model"]
        self.conf_threshold = cfg["layout"]["conf_threshold"]
        self.classes = cfg["layout"]["classes"]
        self.debug_dir = Path(cfg["io"]["debug_dir"])
        
        # Initialize backend
        if self.model_name == "pp_doclayout_l":
            self._init_pp_doclayout()
        elif self.model_name == "yolov10":
            self._init_yolov10()
        else:
            raise ValueError(f"Unsupported layout model: {self.model_name}")
    
    def _init_pp_doclayout(self):
        """Initialize PP-DocLayout-L backend."""
        try:
            # Try to import PaddleDetection
            from paddledet.core.workspace import load_config, merge_config
            from paddledet.engine import Trainer
            self.backend = "pp_doclayout"
            print("[INFO] Using PP-DocLayout-L backend")
        except ImportError:
            print("[WARN] PaddleDetection not available, falling back to YOLOv10")
            self._init_yolov10()
    
    def _init_yolov10(self):
        """Initialize YOLOv10 backend via ultralytics."""
        try:
            from ultralytics import YOLO
            
            # Check for trained weights
            weights_dir = Path(self.cfg["paths"]["weights_dir"])
            trained_weights = weights_dir / "doclayout_yolov10_best.pt"
            
            if trained_weights.exists():
                print(f"[INFO] Loading trained weights: {trained_weights}")
                self.model = YOLO(str(trained_weights))
            else:
                print(f"[WARN] Trained weights not found at {trained_weights}")
                print("[WARN] Using pre-trained model. Run 'docuagent train yolo' to train custom weights.")
                self.model = YOLO('yolov10n.pt')  # Lightweight model
            
            self.backend = "yolov10"
            print("[INFO] Using YOLOv10 backend")
        except ImportError:
            raise RuntimeError("ultralytics not available. Install with: pip install ultralytics")
    
    def detect(self, image_bgr: np.ndarray, page_idx: int) -> List[LayoutBox]:
        """Detect layout elements and return HBB boxes in reading order.
        
        Args:
            image_bgr: Input image in BGR format
            page_idx: Page index for debugging
            
        Returns:
            List of LayoutBox objects sorted by reading order (top-left to bottom-right)
        """
        if self.backend == "pp_doclayout":
            boxes = self._detect_pp_doclayout(image_bgr)
        else:  # yolov10
            boxes = self._detect_yolov10(image_bgr)
        
        # Apply NMS
        boxes = self._apply_nms(boxes)
        
        # Sort by reading order (y, x)
        boxes.sort(key=lambda b: (b.bbox[1], b.bbox[0]))
        
        # Save debug overlay
        self._save_debug_overlay(image_bgr, boxes, page_idx)
        
        return boxes
    
    def _detect_pp_doclayout(self, image_bgr: np.ndarray) -> List[LayoutBox]:
        """PP-DocLayout-L detection (stub implementation)."""
        # This is a stub - in production, implement actual PP-DocLayout-L inference
        # For now, return empty list
        return []
    
    def _detect_yolov10(self, image_bgr: np.ndarray) -> List[LayoutBox]:
        """YOLOv10 detection with layout class mapping."""
        results = self.model(image_bgr, conf=self.conf_threshold)
        boxes = []
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    # Convert to HBB format
                    x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                    
                    # Map class to layout class (simplified mapping)
                    class_name = self._map_yolo_to_layout_class(cls)
                    
                    if class_name in self.classes:
                        boxes.append(LayoutBox(class_name, (x, y, w, h), float(conf)))
        
        return boxes
    
    def _map_yolo_to_layout_class(self, yolo_cls: int) -> str:
        """Map YOLO class to layout class."""
        # Simplified mapping - in production, use proper class mapping
        class_mapping = {
            0: "Text",
            1: "Title", 
            2: "List",
            3: "Table",
            4: "Figure"
        }
        return class_mapping.get(yolo_cls, "Text")
    
    def _apply_nms(self, boxes: List[LayoutBox]) -> List[LayoutBox]:
        """Apply Non-Maximum Suppression to remove overlapping boxes."""
        if not boxes:
            return boxes
        
        # Convert to format expected by OpenCV NMS
        boxes_array = np.array([[b.bbox[0], b.bbox[1], b.bbox[0] + b.bbox[2], b.bbox[1] + b.bbox[3]] 
                               for b in boxes])
        scores = np.array([b.score for b in boxes])
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            boxes_array.tolist(), 
            scores.tolist(), 
            self.conf_threshold, 
            0.4  # NMS threshold
        )
        
        if len(indices) > 0:
            indices = indices.flatten()
            return [boxes[i] for i in indices]
        else:
            return []
    
    def _save_debug_overlay(self, image_bgr: np.ndarray, boxes: List[LayoutBox], page_idx: int):
        """Save debug overlay with detected boxes."""
        debug_path = self.debug_dir / f"page_{page_idx}_layout.jpg"
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create overlay
        overlay = image_bgr.copy()
        
        # Color mapping for different classes
        colors = {
            "Text": (0, 255, 0),      # Green
            "Title": (255, 0, 0),     # Blue
            "List": (0, 0, 255),      # Red
            "Table": (255, 255, 0),   # Cyan
            "Figure": (255, 0, 255)   # Magenta
        }
        
        for box in boxes:
            x, y, w, h = box.bbox
            color = colors.get(box.class_name, (128, 128, 128))
            
            # Draw rectangle
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            label = f"{box.class_name}: {box.score:.2f}"
            cv2.putText(overlay, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        cv2.imwrite(str(debug_path), overlay)
        print(f"[DEBUG] Saved layout overlay to {debug_path}")


def quad_to_hbb(quad: np.ndarray) -> Tuple[int, int, int, int]:
    """Convert quadrilateral to horizontal bounding box.
    
    Args:
        quad: Array of 4 points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        
    Returns:
        HBB as (x, y, w, h)
    """
    xs = quad[:, 0]
    ys = quad[:, 1]
    x, y = int(xs.min()), int(ys.min())
    w, h = int(xs.max() - xs.min()), int(ys.max() - ys.min())
    return x, y, w, h
