"""Utility functions for document processing."""

import os
import math
from pathlib import Path
from typing import List, Tuple, Optional, Union
import cv2
import numpy as np
import fitz  # PyMuPDF
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp


def pdf_to_images(pdf_path: Union[str, Path], dpi: int = 220, 
                  page_range: Optional[Tuple[int, int]] = None) -> List[np.ndarray]:
    """Convert PDF to images with specified DPI and page range.
    
    Args:
        pdf_path: Path to PDF file
        dpi: DPI for rasterization
        page_range: Optional (start, end) page range (0-indexed)
        
    Returns:
        List of images as numpy arrays in BGR format
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    doc = fitz.open(str(pdf_path))
    images = []
    
    start_page = 0
    end_page = len(doc)
    
    if page_range:
        start_page, end_page = page_range
        end_page = min(end_page, len(doc))
    
    for pno in range(start_page, end_page):
        page = doc[pno]
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
        images.append(img[:, :, ::-1])  # RGB→BGR for OpenCV consistency
    
    doc.close()
    return images


def image_to_images(image_path: Union[str, Path]) -> List[np.ndarray]:
    """Load single image as list.
    
    Args:
        image_path: Path to image file
        
    Returns:
        List containing single image as numpy array
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    
    return [img]


def deskew_and_denoise(img: np.ndarray) -> np.ndarray:
    """Fast deskew via Hough lines + mild denoise.
    
    Args:
        img: Input image
        
    Returns:
        Processed image
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Detect skew using Hough lines
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=150, minLineLength=150, maxLineGap=20)
    
    angle_deg = 0.0
    if lines is not None and len(lines) > 0:
        angles = []
        for x1, y1, x2, y2 in lines[:, 0, :]:
            ang = math.degrees(math.atan2((y2 - y1), (x2 - x1)))
            if -45 < ang < 45:
                angles.append(ang)
        if angles:
            angle_deg = float(np.median(angles))
    
    # Apply rotation if significant skew detected
    if abs(angle_deg) > 0.2:
        h, w = gray.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    # Apply mild denoising
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img


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


def crop_image_by_bbox(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """Crop image by bounding box.
    
    Args:
        image: Input image
        bbox: Bounding box as (x, y, w, h)
        
    Returns:
        Cropped image
    """
    x, y, w, h = bbox
    return image[y:y+h, x:x+w]


def resize_image(image: np.ndarray, max_size: int = 1024) -> np.ndarray:
    """Resize image while maintaining aspect ratio.
    
    Args:
        image: Input image
        max_size: Maximum dimension
        
    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    
    if max(h, w) <= max_size:
        return image
    
    if w > h:
        new_w = max_size
        new_h = int(h * max_size / w)
    else:
        new_h = max_size
        new_w = int(w * max_size / h)
    
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)


def process_pages_parallel(process_func, pages_data: List, num_workers: int = 4) -> List:
    """Process pages in parallel using multiprocessing.
    
    Args:
        process_func: Function to process each page
        pages_data: List of page data
        num_workers: Number of worker processes
        
    Returns:
        List of results
    """
    if num_workers <= 1:
        return [process_func(data) for data in pages_data]
    
    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_data = {executor.submit(process_func, data): data for data in pages_data}
        
        for future in as_completed(future_to_data):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"[ERROR] Page processing failed: {e}")
                results.append(None)
    
    return results


def set_deterministic_seeds(seed: int = 42):
    """Set deterministic seeds for reproducibility.
    
    Args:
        seed: Random seed
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    
    try:
        import random
        random.seed(seed)
    except ImportError:
        pass


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_stem(path: Union[str, Path]) -> str:
    """Get file stem without extension.
    
    Args:
        path: File path
        
    Returns:
        File stem
    """
    return Path(path).stem


def is_image_file(path: Union[str, Path]) -> bool:
    """Check if file is a supported image format.
    
    Args:
        path: File path
        
    Returns:
        True if supported image format
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    return Path(path).suffix.lower() in image_extensions


def is_pdf_file(path: Union[str, Path]) -> bool:
    """Check if file is a PDF.
    
    Args:
        path: File path
        
    Returns:
        True if PDF file
    """
    return Path(path).suffix.lower() == '.pdf'
