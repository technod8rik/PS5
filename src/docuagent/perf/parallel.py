"""Parallel processing for per-page operations."""

import multiprocessing as mp
from multiprocessing import Pool, Process, Queue
import time
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import cv2
from PIL import Image
import torch


@dataclass
class PageTask:
    """Represents a task for processing a single page."""
    page_id: int
    image_path: str
    image_data: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None


@dataclass
class PageResult:
    """Represents the result of processing a single page."""
    page_id: int
    success: bool
    result: Any
    error: Optional[str] = None
    processing_time: float = 0.0


class ParallelProcessor:
    """Parallel processor for per-page operations."""
    
    def __init__(self, num_workers: int = 4, use_gpu: bool = True):
        """Initialize parallel processor.
        
        Args:
            num_workers: Number of worker processes
            use_gpu: Whether to use GPU (requires proper CUDA setup)
        """
        self.num_workers = min(num_workers, mp.cpu_count())
        self.use_gpu = use_gpu
        
        # Set multiprocessing start method
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # Already set
    
    def process_pages(
        self,
        pages: List[PageTask],
        process_func: Callable[[PageTask], PageResult],
        batch_size: Optional[int] = None
    ) -> List[PageResult]:
        """Process pages in parallel.
        
        Args:
            pages: List of page tasks
            process_func: Function to process each page
            batch_size: Optional batch size for processing
            
        Returns:
            List of page results
        """
        if not pages:
            return []
        
        print(f"[INFO] Processing {len(pages)} pages with {self.num_workers} workers")
        
        if batch_size is None:
            batch_size = max(1, len(pages) // self.num_workers)
        
        # Process in batches
        results = []
        for i in range(0, len(pages), batch_size):
            batch = pages[i:i + batch_size]
            batch_results = self._process_batch(batch, process_func)
            results.extend(batch_results)
        
        return results
    
    def _process_batch(
        self,
        batch: List[PageTask],
        process_func: Callable[[PageTask], PageResult]
    ) -> List[PageResult]:
        """Process a batch of pages."""
        if len(batch) == 1:
            # Single page - process directly
            return [process_func(batch[0])]
        
        # Multiple pages - use multiprocessing
        with Pool(self.num_workers) as pool:
            results = pool.map(process_func, batch)
        
        return results


def create_page_task(
    page_id: int,
    image_path: str,
    load_image: bool = False
) -> PageTask:
    """Create a page task.
    
    Args:
        page_id: Unique page identifier
        image_path: Path to image file
        load_image: Whether to load image data into memory
        
    Returns:
        PageTask object
    """
    task = PageTask(
        page_id=page_id,
        image_path=image_path,
        metadata={}
    )
    
    if load_image:
        try:
            image = cv2.imread(image_path)
            if image is not None:
                task.image_data = image
            else:
                print(f"Warning: Could not load image {image_path}")
        except Exception as e:
            print(f"Warning: Error loading image {image_path}: {e}")
    
    return task


def process_layout_detection(task: PageTask) -> PageResult:
    """Process layout detection for a single page.
    
    This is a worker function that should be called from a separate process.
    """
    start_time = time.time()
    
    try:
        # Load image if not already loaded
        if task.image_data is None:
            image = cv2.imread(task.image_path)
            if image is None:
                raise ValueError(f"Could not load image: {task.image_path}")
        else:
            image = task.image_data
        
        # Initialize layout detector (this would be done in the main process)
        # For now, we'll simulate the detection
        from ..layout import LayoutDetector
        
        # Create a minimal config for the detector
        config = {
            'layout': {
                'model': 'pp_doclayout_l',
                'conf_threshold': 0.25,
                'classes': ['Text', 'Title', 'List', 'Table', 'Figure']
            },
            'device': 'cpu'  # Use CPU in worker processes
        }
        
        detector = LayoutDetector(config)
        layout_boxes = detector.detect(image, task.page_id)
        
        result = {
            'layout_boxes': layout_boxes,
            'image_shape': image.shape,
            'page_id': task.page_id
        }
        
        processing_time = time.time() - start_time
        
        return PageResult(
            page_id=task.page_id,
            success=True,
            result=result,
            processing_time=processing_time
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        return PageResult(
            page_id=task.page_id,
            success=False,
            result=None,
            error=str(e),
            processing_time=processing_time
        )


def process_ocr_lang(task: PageTask) -> PageResult:
    """Process OCR and language identification for a single page.
    
    This is a worker function that should be called from a separate process.
    """
    start_time = time.time()
    
    try:
        # Load image if not already loaded
        if task.image_data is None:
            image = cv2.imread(task.image_path)
            if image is None:
                raise ValueError(f"Could not load image: {task.image_path}")
        else:
            image = task.image_data
        
        # Get layout boxes from metadata
        layout_boxes = task.metadata.get('layout_boxes', [])
        
        # Initialize OCR and language ID (this would be done in the main process)
        from ..ocr_lang import OCRLang
        
        # Create a minimal config
        config = {
            'ocr': {'lang': 'ml'},
            'langid': {'model_path': '.cache/lid.176.bin'},
            'device': 'cpu'  # Use CPU in worker processes
        }
        
        ocr_lang = OCRLang(config)
        elements = ocr_lang.run(image, layout_boxes, task.page_id)
        
        result = {
            'elements': elements,
            'page_id': task.page_id
        }
        
        processing_time = time.time() - start_time
        
        return PageResult(
            page_id=task.page_id,
            success=True,
            result=result,
            processing_time=processing_time
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        return PageResult(
            page_id=task.page_id,
            success=False,
            result=None,
            error=str(e),
            processing_time=processing_time
        )


def process_vlm_descriptions(task: PageTask) -> PageResult:
    """Process VLM descriptions for a single page.
    
    This is a worker function that should be called from a separate process.
    """
    start_time = time.time()
    
    try:
        # Load image if not already loaded
        if task.image_data is None:
            image = cv2.imread(task.image_path)
            if image is None:
                raise ValueError(f"Could not load image: {task.image_path}")
        else:
            image = task.image_data
        
        # Get elements from metadata
        elements = task.metadata.get('elements', [])
        
        # Initialize describer (this would be done in the main process)
        from ..describe import Describer
        
        # Create a minimal config
        config = {
            'vlm': {'model': 'Qwen/Qwen2-VL-7B-Instruct'},
            'describe': {'batch_size': 2},
            'device': 'cpu'  # Use CPU in worker processes
        }
        
        describer = Describer(config)
        updated_elements = describer.describe_elements(image, elements)
        
        result = {
            'elements': updated_elements,
            'page_id': task.page_id
        }
        
        processing_time = time.time() - start_time
        
        return PageResult(
            page_id=task.page_id,
            success=True,
            result=result,
            processing_time=processing_time
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        return PageResult(
            page_id=task.page_id,
            success=False,
            result=None,
            error=str(e),
            processing_time=processing_time
        )


def process_pipeline(
    image_paths: List[str],
    num_workers: int = 4,
    use_gpu: bool = True,
    enable_caching: bool = True
) -> List[PageResult]:
    """Process a complete pipeline for multiple images.
    
    Args:
        image_paths: List of image file paths
        num_workers: Number of worker processes
        use_gpu: Whether to use GPU
        enable_caching: Whether to enable caching
        
    Returns:
        List of final page results
    """
    print(f"[INFO] Processing {len(image_paths)} images with {num_workers} workers")
    
    processor = ParallelProcessor(num_workers, use_gpu)
    
    # Create page tasks
    pages = []
    for i, image_path in enumerate(image_paths):
        task = create_page_task(i, image_path, load_image=False)
        pages.append(task)
    
    # Step 1: Layout detection
    print("[INFO] Step 1: Layout detection")
    layout_results = processor.process_pages(pages, process_layout_detection)
    
    # Update page tasks with layout results
    for result in layout_results:
        if result.success:
            page = pages[result.page_id]
            page.metadata['layout_boxes'] = result.result['layout_boxes']
    
    # Step 2: OCR and language ID
    print("[INFO] Step 2: OCR and language identification")
    ocr_results = processor.process_pages(pages, process_ocr_lang)
    
    # Update page tasks with OCR results
    for result in ocr_results:
        if result.success:
            page = pages[result.page_id]
            page.metadata['elements'] = result.result['elements']
    
    # Step 3: VLM descriptions
    print("[INFO] Step 3: VLM descriptions")
    vlm_results = processor.process_pages(pages, process_vlm_descriptions)
    
    return vlm_results


def batch_ocr_crops(
    crops: List[Tuple[np.ndarray, int, int]],  # (crop_image, page_id, crop_id)
    num_workers: int = 4
) -> List[Dict[str, Any]]:
    """Batch process OCR crops to reduce engine initialization overhead.
    
    Args:
        crops: List of (crop_image, page_id, crop_id) tuples
        num_workers: Number of worker processes
        
    Returns:
        List of OCR results
    """
    print(f"[INFO] Batch processing {len(crops)} OCR crops")
    
    # Group crops by page to minimize engine reinitialization
    crops_by_page = {}
    for crop_image, page_id, crop_id in crops:
        if page_id not in crops_by_page:
            crops_by_page[page_id] = []
        crops_by_page[page_id].append((crop_image, crop_id))
    
    # Process each page's crops together
    all_results = []
    for page_id, page_crops in crops_by_page.items():
        print(f"[INFO] Processing {len(page_crops)} crops for page {page_id}")
        
        # Initialize OCR once per page
        from ..ocr_lang import OCRLang
        config = {'ocr': {'lang': 'ml'}, 'device': 'cpu'}
        ocr_lang = OCRLang(config)
        
        # Process all crops for this page
        for crop_image, crop_id in page_crops:
            try:
                # Run OCR on crop
                result = ocr_lang.run_crop(crop_image)
                
                all_results.append({
                    'page_id': page_id,
                    'crop_id': crop_id,
                    'result': result,
                    'success': True
                })
            except Exception as e:
                all_results.append({
                    'page_id': page_id,
                    'crop_id': crop_id,
                    'result': None,
                    'success': False,
                    'error': str(e)
                })
    
    return all_results


def main():
    """Command line interface for parallel processing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Parallel processing utilities")
    parser.add_argument("--images", nargs='+', required=True, help="Image files to process")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU processing")
    parser.add_argument("--output", help="Output directory for results")
    
    args = parser.parse_args()
    
    # Process images
    results = process_pipeline(
        args.images,
        num_workers=args.workers,
        use_gpu=args.use_gpu
    )
    
    # Print results
    successful = sum(1 for r in results if r.success)
    total_time = sum(r.processing_time for r in results)
    
    print(f"\n[INFO] Processing complete:")
    print(f"  Successful: {successful}/{len(results)}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average time per page: {total_time/len(results):.2f}s")
    
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results
        import json
        results_data = [
            {
                'page_id': r.page_id,
                'success': r.success,
                'processing_time': r.processing_time,
                'error': r.error
            }
            for r in results
        ]
        
        with open(output_dir / "processing_results.json", 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"[INFO] Results saved to {output_dir}")


if __name__ == "__main__":
    main()
