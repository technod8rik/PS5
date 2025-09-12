"""
RTL (Right-to-Left) and rotation detection module.

This module provides lightweight detection of:
- Dominant script direction (Arabic/Urdu/Persian vs others)
- Page rotation/skew detection
- RTL-specific preprocessing recommendations
"""

import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import random
from collections import Counter
import re

from ..ocr_lang import OCRLang
from ..config import load_config


def check_rtl_rotation(
    images_dir: str,
    sample_rate: float = 0.03,
    output_path: Optional[str] = None,
    config_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Check for RTL scripts and rotation issues in images.
    
    Args:
        images_dir: Path to images directory
        sample_rate: Fraction of images to sample for analysis
        output_path: Optional path to save results
        config_path: Optional path to config file
        
    Returns:
        Dict with RTL and rotation analysis results
    """
    images_path = Path(images_dir)
    if not images_path.exists():
        raise ValueError(f"Images directory not found: {images_dir}")
    
    # Load config
    cfg = load_config(config_path)
    
    # Get image files
    image_files = list(images_path.glob("*.jpg")) + list(images_path.glob("*.jpeg")) + list(images_path.glob("*.png"))
    
    # Sample images for analysis
    sample_size = max(1, int(len(image_files) * sample_rate))
    sampled_files = random.sample(image_files, min(sample_size, len(image_files)))
    
    print(f"[RTL] Analyzing {len(sampled_files)} images (sample rate: {sample_rate:.1%})")
    
    # Initialize results
    results = {
        "total_images": len(image_files),
        "sampled_images": len(sampled_files),
        "rtl_analysis": {
            "rtl_pages": [],
            "ltr_pages": [],
            "mixed_pages": [],
            "script_distribution": {},
            "confidence_scores": []
        },
        "rotation_analysis": {
            "skewed_pages": [],
            "rotation_angles": [],
            "skew_threshold": 1.5,
            "recommendations": []
        },
        "summary": {
            "rtl_percentage": 0.0,
            "skewed_percentage": 0.0,
            "needs_preprocessing": False
        }
    }
    
    # Initialize OCR for text analysis
    ocr_lang = None
    try:
        ocr_lang = OCRLang(cfg)
    except Exception as e:
        print(f"[WARN] Could not initialize OCR: {e}")
    
    # Analyze each sampled image
    for img_path in tqdm(sampled_files, desc="Analyzing RTL/rotation"):
        try:
            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            # RTL analysis
            rtl_result = _analyze_rtl_script(img, img_path, ocr_lang)
            if rtl_result:
                results["rtl_analysis"]["rtl_pages"].append(rtl_result) if rtl_result["is_rtl"] else results["rtl_analysis"]["ltr_pages"].append(rtl_result)
                if rtl_result["is_mixed"]:
                    results["rtl_analysis"]["mixed_pages"].append(rtl_result)
            
            # Rotation analysis
            rotation_result = _analyze_rotation(img, img_path)
            if rotation_result:
                results["rotation_analysis"]["rotation_angles"].append(rotation_result["angle"])
                if abs(rotation_result["angle"]) > results["rotation_analysis"]["skew_threshold"]:
                    results["rotation_analysis"]["skewed_pages"].append(rotation_result)
        
        except Exception as e:
            print(f"[WARN] Error analyzing {img_path}: {e}")
            continue
    
    # Calculate summary statistics
    total_analyzed = len(results["rtl_analysis"]["rtl_pages"]) + len(results["rtl_analysis"]["ltr_pages"])
    if total_analyzed > 0:
        results["summary"]["rtl_percentage"] = len(results["rtl_analysis"]["rtl_pages"]) / total_analyzed * 100
        results["summary"]["skewed_percentage"] = len(results["rotation_analysis"]["skewed_pages"]) / len(sampled_files) * 100
    
    # Generate recommendations
    results["rotation_analysis"]["recommendations"] = _generate_rotation_recommendations(results)
    results["summary"]["needs_preprocessing"] = (
        results["summary"]["rtl_percentage"] > 20 or 
        results["summary"]["skewed_percentage"] > 10
    )
    
    # Save results
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"[RTL] Results saved to {output_path}")
    
    return results


def _analyze_rtl_script(img: np.ndarray, img_path: Path, ocr_lang: Optional[OCRLang]) -> Optional[Dict[str, Any]]:
    """Analyze if image contains RTL script."""
    result = {
        "file": str(img_path),
        "is_rtl": False,
        "is_mixed": False,
        "confidence": 0.0,
        "script_analysis": {},
        "text_samples": []
    }
    
    # Method 1: OCR-based analysis
    if ocr_lang:
        try:
            # Run OCR on a sample region (center crop)
            h, w = img.shape[:2]
            center_crop = img[h//4:3*h//4, w//4:3*w//4]
            
            # Simple OCR to get text
            from paddleocr import PaddleOCR
            ocr = PaddleOCR(use_angle_cls=True, lang='ml', show_log=False)
            ocr_results = ocr.ocr(center_crop)
            
            if ocr_results and ocr_results[0]:
                text_samples = []
                for line in ocr_results[0]:
                    if line and len(line) >= 2:
                        text = line[1][0]
                        confidence = line[1][1]
                        if confidence > 0.5:  # Only high-confidence text
                            text_samples.append(text)
                
                if text_samples:
                    result["text_samples"] = text_samples
                    rtl_analysis = _analyze_text_direction(text_samples)
                    result.update(rtl_analysis)
        
        except Exception as e:
            print(f"[WARN] OCR analysis failed for {img_path}: {e}")
    
    # Method 2: Visual analysis (fallback)
    if not result["text_samples"]:
        visual_analysis = _analyze_visual_rtl_indicators(img)
        result.update(visual_analysis)
    
    return result


def _analyze_text_direction(text_samples: List[str]) -> Dict[str, Any]:
    """Analyze text direction from OCR samples."""
    rtl_chars = 0
    ltr_chars = 0
    total_chars = 0
    
    # RTL Unicode ranges
    rtl_ranges = [
        (0x0590, 0x05FF),  # Hebrew
        (0x0600, 0x06FF),  # Arabic
        (0x0750, 0x077F),  # Arabic Supplement
        (0x08A0, 0x08FF),  # Arabic Extended-A
        (0xFB1D, 0xFDFF),  # Arabic Presentation Forms-A
        (0xFE70, 0xFEFF),  # Arabic Presentation Forms-B
        (0x10A00, 0x10A5F),  # Kharoshthi
        (0x10A60, 0x10A7F),  # Old South Arabian
        (0x10B00, 0x10B3F),  # Inscriptional Parthian
        (0x10B40, 0x10B5F),  # Inscriptional Pahlavi
        (0x10B60, 0x10B7F),  # Psalter Pahlavi
        (0x10C00, 0x10C4F),  # Old Turkic
        (0x10C80, 0x10CFF),  # Old Hungarian
        (0x10D00, 0x10D3F),  # Hanifi Rohingya
        (0x10E80, 0x10EBF),  # Yezidi
        (0x10F00, 0x10F2F),  # Sogdian
        (0x10F30, 0x10F6F),  # Old Sogdian
        (0x10F70, 0x10FAF),  # Old Uyghur
        (0x10FB0, 0x10FDF),  # Chorasmian
        (0x10FE0, 0x10FFF),  # Elymaic
        (0x1E800, 0x1E8DF),  # Mende Kikakui
        (0x1E900, 0x1E95F),  # Adlam
        (0x1EC70, 0x1ECBF),  # Mro
        (0x1ED00, 0x1ED4F),  # Bassa Vah
        (0x1EE00, 0x1EEFF),  # Arabic Mathematical Alphabetic Symbols
    ]
    
    for text in text_samples:
        for char in text:
            char_code = ord(char)
            total_chars += 1
            
            # Check if character is in RTL range
            is_rtl = any(start <= char_code <= end for start, end in rtl_ranges)
            if is_rtl:
                rtl_chars += 1
            else:
                ltr_chars += 1
    
    if total_chars == 0:
        return {"is_rtl": False, "is_mixed": False, "confidence": 0.0}
    
    rtl_ratio = rtl_chars / total_chars
    ltr_ratio = ltr_chars / total_chars
    
    is_rtl = rtl_ratio > 0.6
    is_mixed = 0.3 < rtl_ratio < 0.7
    confidence = max(rtl_ratio, ltr_ratio)
    
    return {
        "is_rtl": is_rtl,
        "is_mixed": is_mixed,
        "confidence": confidence,
        "script_analysis": {
            "rtl_chars": rtl_chars,
            "ltr_chars": ltr_chars,
            "total_chars": total_chars,
            "rtl_ratio": rtl_ratio,
            "ltr_ratio": ltr_ratio
        }
    }


def _analyze_visual_rtl_indicators(img: np.ndarray) -> Dict[str, Any]:
    """Analyze visual indicators of RTL script."""
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
    
    if lines is None:
        return {"is_rtl": False, "is_mixed": False, "confidence": 0.0}
    
    # Analyze line directions
    horizontal_lines = 0
    vertical_lines = 0
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        
        if abs(angle) < 15:  # Horizontal
            horizontal_lines += 1
        elif abs(angle - 90) < 15 or abs(angle + 90) < 15:  # Vertical
            vertical_lines += 1
    
    # RTL scripts often have more vertical text flow
    total_lines = horizontal_lines + vertical_lines
    if total_lines == 0:
        return {"is_rtl": False, "is_mixed": False, "confidence": 0.0}
    
    vertical_ratio = vertical_lines / total_lines
    is_rtl = vertical_ratio > 0.3  # Threshold for RTL indication
    confidence = vertical_ratio
    
    return {
        "is_rtl": is_rtl,
        "is_mixed": False,
        "confidence": confidence,
        "script_analysis": {
            "horizontal_lines": horizontal_lines,
            "vertical_lines": vertical_lines,
            "vertical_ratio": vertical_ratio
        }
    }


def _analyze_rotation(img: np.ndarray, img_path: Path) -> Optional[Dict[str, Any]]:
    """Analyze image rotation/skew."""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is None:
            return None
        
        # Calculate angles
        angles = []
        for line in lines:
            rho, theta = line[0]
            angle = theta * 180 / np.pi
            # Normalize to [-90, 90]
            if angle > 90:
                angle -= 180
            angles.append(angle)
        
        if not angles:
            return None
        
        # Calculate median angle (more robust than mean)
        median_angle = np.median(angles)
        
        # Calculate confidence based on angle consistency
        angle_std = np.std(angles)
        confidence = max(0, 1 - angle_std / 45)  # Higher confidence for more consistent angles
        
        return {
            "file": str(img_path),
            "angle": median_angle,
            "confidence": confidence,
            "angle_std": angle_std,
            "line_count": len(lines)
        }
    
    except Exception as e:
        print(f"[WARN] Rotation analysis failed for {img_path}: {e}")
        return None


def _generate_rotation_recommendations(results: Dict[str, Any]) -> List[str]:
    """Generate recommendations based on rotation analysis."""
    recommendations = []
    
    skewed_count = len(results["rotation_analysis"]["skewed_pages"])
    total_analyzed = results["sampled_images"]
    
    if skewed_count > 0:
        skewed_percentage = skewed_count / total_analyzed * 100
        
        if skewed_percentage > 20:
            recommendations.append("High percentage of skewed pages detected. Consider implementing automatic deskewing.")
        elif skewed_percentage > 10:
            recommendations.append("Moderate percentage of skewed pages detected. Manual review recommended.")
        else:
            recommendations.append("Low percentage of skewed pages detected. Current preprocessing may be sufficient.")
        
        # Specific angle recommendations
        angles = results["rotation_analysis"]["rotation_angles"]
        if angles:
            avg_angle = np.mean(angles)
            if abs(avg_angle) > 5:
                recommendations.append(f"Average rotation angle: {avg_angle:.1f}°. Consider applying global rotation correction.")
    
    rtl_percentage = results["summary"]["rtl_percentage"]
    if rtl_percentage > 20:
        recommendations.append("High percentage of RTL pages detected. Consider RTL-specific preprocessing.")
    elif rtl_percentage > 5:
        recommendations.append("Some RTL pages detected. Mixed script handling may be needed.")
    
    if not recommendations:
        recommendations.append("No significant rotation or RTL issues detected.")
    
    return recommendations


def generate_rtl_report(results: Dict[str, Any], output_path: str) -> None:
    """Generate a human-readable RTL/rotation report."""
    report_lines = [
        "# RTL and Rotation Analysis Report",
        "",
        f"**Total images:** {results['total_images']}",
        f"**Sampled images:** {results['sampled_images']}",
        f"**RTL pages:** {len(results['rtl_analysis']['rtl_pages'])} ({results['summary']['rtl_percentage']:.1f}%)",
        f"**Skewed pages:** {len(results['rotation_analysis']['skewed_pages'])} ({results['summary']['skewed_percentage']:.1f}%)",
        "",
        "## RTL Analysis",
        ""
    ]
    
    # RTL analysis
    if results['rtl_analysis']['rtl_pages']:
        report_lines.extend([
            "### RTL Pages",
            f"**Count:** {len(results['rtl_analysis']['rtl_pages'])}",
            ""
        ])
        
        for page in results['rtl_analysis']['rtl_pages'][:5]:
            report_lines.extend([
                f"- **File:** {page['file']}",
                f"  - **Confidence:** {page['confidence']:.2f}",
                f"  - **Mixed:** {page['is_mixed']}",
                ""
            ])
    
    # Rotation analysis
    report_lines.extend([
        "## Rotation Analysis",
        f"**Skew threshold:** {results['rotation_analysis']['skew_threshold']}°",
        ""
    ])
    
    if results['rotation_analysis']['skewed_pages']:
        report_lines.extend([
            "### Skewed Pages",
            f"**Count:** {len(results['rotation_analysis']['skewed_pages'])}",
            ""
        ])
        
        for page in results['rotation_analysis']['skewed_pages'][:5]:
            report_lines.extend([
                f"- **File:** {page['file']}",
                f"  - **Angle:** {page['angle']:.1f}°",
                f"  - **Confidence:** {page['confidence']:.2f}",
                ""
            ])
    
    # Recommendations
    report_lines.extend([
        "## Recommendations",
        ""
    ])
    
    for rec in results['rotation_analysis']['recommendations']:
        report_lines.append(f"- {rec}")
    
    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"[RTL] Report saved to {output_path}")
