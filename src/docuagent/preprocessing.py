"""Document preprocessing utilities."""

from pathlib import Path
from typing import List, Union, Optional, Tuple
import cv2
import numpy as np

from .utils import pdf_to_images, image_to_images, deskew_and_denoise


class DocumentPreprocessor:
    """Document preprocessing pipeline."""
    
    def __init__(self, dpi: int = 220, enable_deskew: bool = True):
        """Initialize preprocessor.
        
        Args:
            dpi: DPI for PDF rasterization
            enable_deskew: Whether to enable deskewing
        """
        self.dpi = dpi
        self.enable_deskew = enable_deskew
    
    def process_document(self, input_path: Union[str, Path], 
                        page_range: Optional[Tuple[int, int]] = None) -> List[np.ndarray]:
        """Process document and return preprocessed images.
        
        Args:
            input_path: Path to input document (PDF or image)
            page_range: Optional page range for PDFs
            
        Returns:
            List of preprocessed images
        """
        input_path = Path(input_path)
        
        # Load images based on file type
        if input_path.suffix.lower() == '.pdf':
            images = pdf_to_images(input_path, dpi=self.dpi, page_range=page_range)
        else:
            images = image_to_images(input_path)
        
        # Preprocess each image
        processed_images = []
        for img in images:
            processed = self._preprocess_image(img)
            processed_images.append(processed)
        
        return processed_images
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess single image.
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image
        """
        # Apply deskewing and denoising if enabled
        if self.enable_deskew:
            image = deskew_and_denoise(image)
        
        # Additional preprocessing can be added here
        # e.g., contrast enhancement, noise reduction, etc.
        
        return image
    
    def save_processed_images(self, images: List[np.ndarray], output_dir: Path, 
                             doc_name: str) -> List[Path]:
        """Save processed images to disk.
        
        Args:
            images: List of processed images
            output_dir: Output directory
            doc_name: Document name (without extension)
            
        Returns:
            List of saved image paths
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_paths = []
        
        for i, img in enumerate(images):
            output_path = output_dir / f"{doc_name}_page_{i:03d}.jpg"
            cv2.imwrite(str(output_path), img)
            saved_paths.append(output_path)
        
        return saved_paths
