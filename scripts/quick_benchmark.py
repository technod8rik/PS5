#!/usr/bin/env python3
"""Quick benchmark script for DocuAgent performance testing."""

import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import argparse

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from docuagent.config import load_config
from docuagent.preprocessing import DocumentPreprocessor
from docuagent.layout import LayoutDetector
from docuagent.ocr_lang import OCRLang
from docuagent.describe import Describer
from docuagent.compile_json import to_standard_json
from docuagent.perf.cache import OCRCache, VLMCache
from docuagent.perf.parallel import ParallelProcessor, process_pipeline


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    stage: str
    duration: float
    throughput: float  # pages per second
    memory_usage: float  # MB
    success: bool
    error: str = None


def benchmark_preprocessing(pdf_path: str, config: Dict[str, Any]) -> BenchmarkResult:
    """Benchmark document preprocessing."""
    print("[BENCHMARK] Preprocessing...")
    
    start_time = time.time()
    start_memory = get_memory_usage()
    
    try:
        preprocessor = DocumentPreprocessor(dpi=config["dpi"])
        images = preprocessor.process_document(pdf_path)
        
        duration = time.time() - start_time
        end_memory = get_memory_usage()
        memory_usage = end_memory - start_memory
        
        return BenchmarkResult(
            stage="preprocessing",
            duration=duration,
            throughput=len(images) / duration if duration > 0 else 0,
            memory_usage=memory_usage,
            success=True
        )
        
    except Exception as e:
        duration = time.time() - start_time
        return BenchmarkResult(
            stage="preprocessing",
            duration=duration,
            throughput=0,
            memory_usage=0,
            success=False,
            error=str(e)
        )


def benchmark_layout_detection(images: List, config: Dict[str, Any]) -> BenchmarkResult:
    """Benchmark layout detection."""
    print("[BENCHMARK] Layout detection...")
    
    start_time = time.time()
    start_memory = get_memory_usage()
    
    try:
        detector = LayoutDetector(config)
        all_boxes = []
        
        for i, image in enumerate(images):
            boxes = detector.detect(image, i)
            all_boxes.append(boxes)
        
        duration = time.time() - start_time
        end_memory = get_memory_usage()
        memory_usage = end_memory - start_memory
        
        return BenchmarkResult(
            stage="layout_detection",
            duration=duration,
            throughput=len(images) / duration if duration > 0 else 0,
            memory_usage=memory_usage,
            success=True
        )
        
    except Exception as e:
        duration = time.time() - start_time
        return BenchmarkResult(
            stage="layout_detection",
            duration=duration,
            throughput=0,
            memory_usage=0,
            success=False,
            error=str(e)
        )


def benchmark_ocr_lang(images: List, layout_boxes: List, config: Dict[str, Any]) -> BenchmarkResult:
    """Benchmark OCR and language identification."""
    print("[BENCHMARK] OCR and language ID...")
    
    start_time = time.time()
    start_memory = get_memory_usage()
    
    try:
        ocr_lang = OCRLang(config)
        all_elements = []
        
        for i, (image, boxes) in enumerate(zip(images, layout_boxes)):
            elements = ocr_lang.run(image, boxes, i)
            all_elements.append(elements)
        
        duration = time.time() - start_time
        end_memory = get_memory_usage()
        memory_usage = end_memory - start_memory
        
        return BenchmarkResult(
            stage="ocr_lang",
            duration=duration,
            throughput=len(images) / duration if duration > 0 else 0,
            memory_usage=memory_usage,
            success=True
        )
        
    except Exception as e:
        duration = time.time() - start_time
        return BenchmarkResult(
            stage="ocr_lang",
            duration=duration,
            throughput=0,
            memory_usage=0,
            success=False,
            error=str(e)
        )


def benchmark_vlm_descriptions(images: List, elements: List, config: Dict[str, Any]) -> BenchmarkResult:
    """Benchmark VLM description generation."""
    print("[BENCHMARK] VLM descriptions...")
    
    start_time = time.time()
    start_memory = get_memory_usage()
    
    try:
        describer = Describer(config)
        all_updated_elements = []
        
        for i, (image, page_elements) in enumerate(zip(images, elements)):
            if any(e.cls in ["Table", "Figure"] for e in page_elements):
                updated_elements = describer.describe_elements(image, page_elements)
                all_updated_elements.append(updated_elements)
            else:
                all_updated_elements.append(page_elements)
        
        duration = time.time() - start_time
        end_memory = get_memory_usage()
        memory_usage = end_memory - start_memory
        
        return BenchmarkResult(
            stage="vlm_descriptions",
            duration=duration,
            throughput=len(images) / duration if duration > 0 else 0,
            memory_usage=memory_usage,
            success=True
        )
        
    except Exception as e:
        duration = time.time() - start_time
        return BenchmarkResult(
            stage="vlm_descriptions",
            duration=duration,
            throughput=0,
            memory_usage=0,
            success=False,
            error=str(e)
        )


def benchmark_parallel_processing(image_paths: List[str], num_workers: int = 4) -> BenchmarkResult:
    """Benchmark parallel processing."""
    print(f"[BENCHMARK] Parallel processing with {num_workers} workers...")
    
    start_time = time.time()
    start_memory = get_memory_usage()
    
    try:
        results = process_pipeline(image_paths, num_workers=num_workers, use_gpu=False)
        
        duration = time.time() - start_time
        end_memory = get_memory_usage()
        memory_usage = end_memory - start_memory
        
        successful = sum(1 for r in results if r.success)
        
        return BenchmarkResult(
            stage="parallel_processing",
            duration=duration,
            throughput=len(image_paths) / duration if duration > 0 else 0,
            memory_usage=memory_usage,
            success=successful == len(image_paths)
        )
        
    except Exception as e:
        duration = time.time() - start_time
        return BenchmarkResult(
            stage="parallel_processing",
            duration=duration,
            throughput=0,
            memory_usage=0,
            success=False,
            error=str(e)
        )


def benchmark_caching(image_paths: List[str]) -> Tuple[BenchmarkResult, BenchmarkResult]:
    """Benchmark caching performance."""
    print("[BENCHMARK] Caching performance...")
    
    # Test without cache
    start_time = time.time()
    start_memory = get_memory_usage()
    
    try:
        # Simulate OCR processing without cache
        ocr_cache = OCRCache()
        vlm_cache = VLMCache()
        
        # Clear caches
        ocr_cache.clear_ocr_cache()
        vlm_cache.clear_vlm_cache()
        
        # Process images (this would normally do OCR/VLM)
        for image_path in image_paths:
            # Simulate cache miss
            result = ocr_cache.get_ocr_result(image_path)
            if result is None:
                # Simulate processing
                fake_result = {"text": "fake ocr result", "confidence": 0.9}
                ocr_cache.set_ocr_result(image_path, "ml", fake_result)
        
        no_cache_duration = time.time() - start_time
        no_cache_memory = get_memory_usage() - start_memory
        
        no_cache_result = BenchmarkResult(
            stage="no_cache",
            duration=no_cache_duration,
            throughput=len(image_paths) / no_cache_duration if no_cache_duration > 0 else 0,
            memory_usage=no_cache_memory,
            success=True
        )
        
    except Exception as e:
        no_cache_result = BenchmarkResult(
            stage="no_cache",
            duration=0,
            throughput=0,
            memory_usage=0,
            success=False,
            error=str(e)
        )
    
    # Test with cache
    start_time = time.time()
    start_memory = get_memory_usage()
    
    try:
        # Process images again (should hit cache)
        for image_path in image_paths:
            result = ocr_cache.get_ocr_result(image_path)
            # Should get cached result
        
        with_cache_duration = time.time() - start_time
        with_cache_memory = get_memory_usage() - start_memory
        
        with_cache_result = BenchmarkResult(
            stage="with_cache",
            duration=with_cache_duration,
            throughput=len(image_paths) / with_cache_duration if with_cache_duration > 0 else 0,
            memory_usage=with_cache_memory,
            success=True
        )
        
    except Exception as e:
        with_cache_result = BenchmarkResult(
            stage="with_cache",
            duration=0,
            throughput=0,
            memory_usage=0,
            success=False,
            error=str(e)
        )
    
    return no_cache_result, with_cache_result


def get_memory_usage() -> float:
    """Get current memory usage in MB."""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


def run_benchmark(
    pdf_path: str,
    config_path: str = "configs/default.yaml",
    output_path: str = None,
    num_workers: int = 4
) -> List[BenchmarkResult]:
    """Run complete benchmark suite."""
    print(f"[BENCHMARK] Starting benchmark for {pdf_path}")
    
    # Load configuration
    config = load_config(config_path)
    
    results = []
    
    # Benchmark preprocessing
    preprocess_result = benchmark_preprocessing(pdf_path, config)
    results.append(preprocess_result)
    
    if not preprocess_result.success:
        print(f"[ERROR] Preprocessing failed: {preprocess_result.error}")
        return results
    
    # Get preprocessed images
    preprocessor = DocumentPreprocessor(dpi=config["dpi"])
    images = preprocessor.process_document(pdf_path)
    
    # Benchmark layout detection
    layout_result = benchmark_layout_detection(images, config)
    results.append(layout_result)
    
    if not layout_result.success:
        print(f"[ERROR] Layout detection failed: {layout_result.error}")
        return results
    
    # Get layout boxes
    detector = LayoutDetector(config)
    layout_boxes = []
    for i, image in enumerate(images):
        boxes = detector.detect(image, i)
        layout_boxes.append(boxes)
    
    # Benchmark OCR and language ID
    ocr_result = benchmark_ocr_lang(images, layout_boxes, config)
    results.append(ocr_result)
    
    if not ocr_result.success:
        print(f"[ERROR] OCR failed: {ocr_result.error}")
        return results
    
    # Get OCR elements
    ocr_lang = OCRLang(config)
    all_elements = []
    for i, (image, boxes) in enumerate(zip(images, layout_boxes)):
        elements = ocr_lang.run(image, boxes, i)
        all_elements.append(elements)
    
    # Benchmark VLM descriptions
    vlm_result = benchmark_vlm_descriptions(images, all_elements, config)
    results.append(vlm_result)
    
    # Benchmark parallel processing
    image_paths = [f"temp_page_{i}.jpg" for i in range(len(images))]
    parallel_result = benchmark_parallel_processing(image_paths, num_workers)
    results.append(parallel_result)
    
    # Benchmark caching
    no_cache_result, with_cache_result = benchmark_caching(image_paths)
    results.extend([no_cache_result, with_cache_result])
    
    # Print results
    print_benchmark_results(results)
    
    # Save results
    if output_path:
        save_benchmark_results(results, output_path)
    
    return results


def print_benchmark_results(results: List[BenchmarkResult]):
    """Print benchmark results."""
    print("\n" + "="*60)
    print(" BENCHMARK RESULTS")
    print("="*60)
    
    total_duration = sum(r.duration for r in results if r.success)
    
    for result in results:
        status = "✓" if result.success else "✗"
        print(f"{status} {result.stage:20} | {result.duration:6.2f}s | {result.throughput:6.2f} pages/s | {result.memory_usage:6.1f} MB")
        
        if not result.success and result.error:
            print(f"    Error: {result.error}")
    
    print("-"*60)
    print(f"Total duration: {total_duration:.2f}s")
    print(f"Average throughput: {sum(r.throughput for r in results if r.success) / len([r for r in results if r.success]):.2f} pages/s")


def save_benchmark_results(results: List[BenchmarkResult], output_path: str):
    """Save benchmark results to JSON file."""
    results_data = [
        {
            'stage': r.stage,
            'duration': r.duration,
            'throughput': r.throughput,
            'memory_usage': r.memory_usage,
            'success': r.success,
            'error': r.error
        }
        for r in results
    ]
    
    with open(output_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"[INFO] Benchmark results saved to {output_path}")


def main():
    """Command line interface for benchmarking."""
    parser = argparse.ArgumentParser(description="Benchmark DocuAgent performance")
    parser.add_argument("--pdf", required=True, help="PDF file to benchmark")
    parser.add_argument("--config", default="configs/default.yaml", help="Configuration file")
    parser.add_argument("--out", help="Output JSON file for results")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    
    args = parser.parse_args()
    
    if not Path(args.pdf).exists():
        print(f"[ERROR] PDF file not found: {args.pdf}")
        return 1
    
    results = run_benchmark(args.pdf, args.config, args.out, args.workers)
    
    # Check if any benchmarks failed
    failed = [r for r in results if not r.success]
    if failed:
        print(f"\n[WARNING] {len(failed)} benchmarks failed")
        return 1
    
    print("\n[INFO] All benchmarks completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
