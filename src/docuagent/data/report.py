"""
Pre-upload report generator module.

This module generates comprehensive reports including:
- Dataset statistics and histograms
- Quality analysis results
- PII findings
- RTL/rotation analysis
- Class imbalance visualization
- Sample montages
"""

import json
import csv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import cv2
from PIL import Image
import seaborn as sns
from collections import Counter
import random


def generate_preupload_report(
    audit_results: Dict[str, Any],
    pii_results: Dict[str, Any],
    rtl_results: Dict[str, Any],
    license_results: Dict[str, Any],
    output_dir: str,
    sample_images: List[str] = None
) -> str:
    """
    Generate comprehensive pre-upload report.
    
    Args:
        audit_results: Results from dataset audit
        pii_results: Results from PII scan
        rtl_results: Results from RTL/rotation check
        license_results: Results from license check
        output_dir: Output directory for report and assets
        sample_images: Optional list of sample image paths
        
    Returns:
        Path to generated report
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create assets directory
    assets_dir = output_path / "preupload_assets"
    assets_dir.mkdir(exist_ok=True)
    
    print(f"[REPORT] Generating pre-upload report in {output_dir}")
    
    # Generate report sections
    report_sections = []
    
    # 1. Executive Summary
    report_sections.append(_generate_executive_summary(
        audit_results, pii_results, rtl_results, license_results
    ))
    
    # 2. Dataset Statistics
    report_sections.append(_generate_dataset_stats(audit_results, assets_dir))
    
    # 3. Quality Analysis
    report_sections.append(_generate_quality_analysis(audit_results, assets_dir))
    
    # 4. PII Analysis
    report_sections.append(_generate_pii_analysis(pii_results, assets_dir))
    
    # 5. RTL/Rotation Analysis
    report_sections.append(_generate_rtl_analysis(rtl_results, assets_dir))
    
    # 6. License Analysis
    report_sections.append(_generate_license_analysis(license_results, assets_dir))
    
    # 7. Class Distribution
    report_sections.append(_generate_class_distribution(audit_results, assets_dir))
    
    # 8. Sample Montages
    if sample_images:
        report_sections.append(_generate_sample_montages(sample_images, assets_dir))
    
    # 9. Recommendations
    report_sections.append(_generate_recommendations(
        audit_results, pii_results, rtl_results, license_results
    ))
    
    # Combine all sections
    full_report = "\n\n".join(report_sections)
    
    # Save report
    report_path = output_path / "preupload_report.md"
    with open(report_path, 'w') as f:
        f.write(full_report)
    
    print(f"[REPORT] Report saved to {report_path}")
    return str(report_path)


def _generate_executive_summary(
    audit_results: Dict[str, Any],
    pii_results: Dict[str, Any],
    rtl_results: Dict[str, Any],
    license_results: Dict[str, Any]
) -> str:
    """Generate executive summary section."""
    # Calculate key metrics
    total_images = audit_results.get("counts", {}).get("images", 0)
    total_labels = audit_results.get("counts", {}).get("labels", 0)
    
    # Quality issues
    image_issues = len(audit_results.get("image_sanity", {}).get("unreadable", []))
    label_issues = len(audit_results.get("label_sanity", {}).get("invalid_coords", []))
    duplicates = len(audit_results.get("duplicates", []))
    
    # PII issues
    pii_findings = pii_results.get("summary", {}).get("total_findings", 0)
    
    # RTL/Rotation issues
    rtl_percentage = rtl_results.get("summary", {}).get("rtl_percentage", 0)
    skewed_percentage = rtl_results.get("summary", {}).get("skewed_percentage", 0)
    
    # License status
    has_license = license_results.get("compliance_status", {}).get("has_license", False)
    
    # Overall status
    critical_issues = image_issues + label_issues + pii_findings
    needs_attention = rtl_percentage > 20 or skewed_percentage > 10 or not has_license
    
    status = "✅ READY" if critical_issues == 0 and not needs_attention else "⚠️ NEEDS ATTENTION"
    
    summary = f"""# Pre-Upload Dataset Report

## Executive Summary

**Status:** {status}

### Key Metrics
- **Total Images:** {total_images:,}
- **Total Labels:** {total_labels:,}
- **Critical Issues:** {critical_issues}
- **Needs Attention:** {'Yes' if needs_attention else 'No'}

### Issues Summary
- **Broken Images:** {image_issues}
- **Invalid Labels:** {label_issues}
- **Duplicates:** {duplicates}
- **PII Findings:** {pii_findings}
- **RTL Pages:** {rtl_percentage:.1f}%
- **Skewed Pages:** {skewed_percentage:.1f}%
- **License Status:** {'✅ Found' if has_license else '❌ Missing'}

### Quick Assessment
"""
    
    if critical_issues == 0:
        summary += "- ✅ No critical data quality issues found\n"
    else:
        summary += f"- ❌ {critical_issues} critical issues need immediate attention\n"
    
    if needs_attention:
        summary += "- ⚠️ Dataset needs preprocessing or compliance review\n"
    else:
        summary += "- ✅ Dataset appears ready for training\n"
    
    return summary


def _generate_dataset_stats(audit_results: Dict[str, Any], assets_dir: Path) -> str:
    """Generate dataset statistics section."""
    counts = audit_results.get("counts", {})
    imbalance = audit_results.get("imbalance", {})
    
    stats = f"""## Dataset Statistics

### Basic Counts
- **Images:** {counts.get('images', 0):,}
- **Labels:** {counts.get('labels', 0):,}
- **Missing Images:** {counts.get('missing_image', 0):,}
- **Missing Labels:** {counts.get('missing_label', 0):,}
- **Empty Labels:** {counts.get('empty_label', 0):,}

### Class Distribution
"""
    
    # Add class distribution table
    class_counts = imbalance.get("class_counts", {})
    class_percentages = imbalance.get("class_percentages", {})
    
    if class_counts:
        stats += "| Class | Count | Percentage |\n"
        stats += "|-------|-------|------------|\n"
        
        for class_name, count in class_counts.items():
            percentage = class_percentages.get(class_name, 0)
            stats += f"| {class_name} | {count:,} | {percentage:.1f}% |\n"
    
    # Generate class distribution chart
    if class_counts:
        _generate_class_distribution_chart(class_counts, assets_dir / "class_distribution.png")
        stats += f"\n![Class Distribution](preupload_assets/class_distribution.png)\n"
    
    return stats


def _generate_quality_analysis(audit_results: Dict[str, Any], assets_dir: Path) -> str:
    """Generate quality analysis section."""
    image_sanity = audit_results.get("image_sanity", {})
    label_sanity = audit_results.get("label_sanity", {})
    duplicates = audit_results.get("duplicates", [])
    
    quality = f"""## Quality Analysis

### Image Quality Issues
- **Unreadable Images:** {len(image_sanity.get('unreadable', []))}
- **Extreme Aspect Ratios:** {len(image_sanity.get('extreme_aspect_ratio', []))}
- **Tiny Dimensions:** {len(image_sanity.get('tiny_dims', []))}
- **Corrupt EXIF:** {len(image_sanity.get('corrupt_exif', []))}

### Label Quality Issues
- **Invalid Coordinates:** {len(label_sanity.get('invalid_coords', []))}
- **Out of Bounds:** {len(label_sanity.get('out_of_bounds', []))}
- **Unknown Classes:** {len(label_sanity.get('unknown_classes', []))}
- **Empty Labels:** {len(label_sanity.get('empty_labels', []))}

### Duplicates
- **Near-Duplicates Found:** {len(duplicates)}
"""
    
    # Generate quality charts
    _generate_quality_charts(image_sanity, label_sanity, assets_dir)
    
    if duplicates:
        quality += f"\n![Quality Issues](preupload_assets/quality_issues.png)\n"
    
    return quality


def _generate_pii_analysis(pii_results: Dict[str, Any], assets_dir: Path) -> str:
    """Generate PII analysis section."""
    summary = pii_results.get("summary", {})
    findings = {k: v for k, v in pii_results.items() if k != "summary"}
    
    pii_section = f"""## PII Analysis

### Summary
- **Files Scanned:** {summary.get('total_files_scanned', 0):,}
- **Total Findings:** {summary.get('total_findings', 0):,}
- **Files with PII:** {summary.get('files_with_pii', 0):,}

### Findings by Type
"""
    
    # Add findings for each type
    for pii_type, findings_list in findings.items():
        if findings_list and isinstance(findings_list, list):
            pii_section += f"\n#### {pii_type.replace('_', ' ').title()}\n"
            pii_section += f"- **Count:** {len(findings_list)}\n"
            
            # Add first 3 examples
            for finding in findings_list[:3]:
                pii_section += f"- **File:** {finding.get('file', 'Unknown')}\n"
                pii_section += f"  - **Original:** {finding.get('original', 'N/A')}\n"
                pii_section += f"  - **Redacted:** {finding.get('redacted', 'N/A')}\n"
                pii_section += f"  - **Context:** {finding.get('context', 'N/A')[:100]}...\n"
            
            if len(findings_list) > 3:
                pii_section += f"- ... and {len(findings_list) - 3} more\n"
    
    return pii_section


def _generate_rtl_analysis(rtl_results: Dict[str, Any], assets_dir: Path) -> str:
    """Generate RTL/rotation analysis section."""
    summary = rtl_results.get("summary", {})
    rtl_analysis = rtl_results.get("rtl_analysis", {})
    rotation_analysis = rtl_results.get("rotation_analysis", {})
    
    rtl_section = f"""## RTL and Rotation Analysis

### Summary
- **RTL Pages:** {summary.get('rtl_percentage', 0):.1f}%
- **Skewed Pages:** {summary.get('skewed_percentage', 0):.1f}%
- **Needs Preprocessing:** {'Yes' if summary.get('needs_preprocessing', False) else 'No'}

### RTL Analysis
- **RTL Pages Found:** {len(rtl_analysis.get('rtl_pages', []))}
- **LTR Pages Found:** {len(rtl_analysis.get('ltr_pages', []))}
- **Mixed Pages Found:** {len(rtl_analysis.get('mixed_pages', []))}

### Rotation Analysis
- **Skewed Pages:** {len(rotation_analysis.get('skewed_pages', []))}
- **Skew Threshold:** {rotation_analysis.get('skew_threshold', 1.5)}°
- **Average Angle:** {np.mean(rotation_analysis.get('rotation_angles', [0])):.1f}°

### Recommendations
"""
    
    recommendations = rotation_analysis.get("recommendations", [])
    for rec in recommendations:
        rtl_section += f"- {rec}\n"
    
    return rtl_section


def _generate_license_analysis(license_results: Dict[str, Any], assets_dir: Path) -> str:
    """Generate license analysis section."""
    summary = license_results.get("summary", {})
    compliance = license_results.get("compliance_status", {})
    
    license_section = f"""## License Analysis

### Summary
- **Total Folders:** {summary.get('total_folders', 0):,}
- **Licensed Folders:** {summary.get('licensed_folders', 0):,}
- **Unknown License Folders:** {summary.get('unknown_license_folders', 0):,}

### Compliance Status
- **Has License:** {'✅ Yes' if compliance.get('has_license', False) else '❌ No'}
- **Has Attribution:** {'✅ Yes' if compliance.get('has_attribution', False) else '❌ No'}

### Warnings
"""
    
    warnings = compliance.get("warnings", [])
    if warnings:
        for warning in warnings:
            license_section += f"- ⚠️ {warning}\n"
    else:
        license_section += "- No warnings\n"
    
    license_section += "\n### Recommendations\n"
    recommendations = compliance.get("recommendations", [])
    if recommendations:
        for rec in recommendations:
            license_section += f"- 💡 {rec}\n"
    else:
        license_section += "- No recommendations\n"
    
    return license_section


def _generate_class_distribution(audit_results: Dict[str, Any], assets_dir: Path) -> str:
    """Generate class distribution section."""
    imbalance = audit_results.get("imbalance", {})
    class_counts = imbalance.get("class_counts", {})
    imbalanced_classes = imbalance.get("imbalanced_classes", [])
    
    if not class_counts:
        return "## Class Distribution\n\nNo class distribution data available.\n"
    
    # Generate class imbalance heatmap
    _generate_class_imbalance_heatmap(class_counts, imbalanced_classes, assets_dir)
    
    distribution = f"""## Class Distribution

### Class Counts
"""
    
    for class_name, count in class_counts.items():
        distribution += f"- **{class_name}:** {count:,}\n"
    
    if imbalanced_classes:
        distribution += "\n### Imbalanced Classes\n"
        distribution += "Classes with less than 5% of total boxes:\n"
        for imbalanced in imbalanced_classes:
            distribution += f"- **{imbalanced['class']}:** {imbalanced['count']} ({imbalanced['percentage']:.1f}%)\n"
    
    distribution += f"\n![Class Imbalance Heatmap](preupload_assets/class_imbalance_heatmap.png)\n"
    
    return distribution


def _generate_sample_montages(sample_images: List[str], assets_dir: Path) -> str:
    """Generate sample montages section."""
    montage_section = "## Sample Montages\n\n"
    
    # Create sample montages for each class
    montage_paths = []
    for i, img_path in enumerate(sample_images[:20]):  # Limit to 20 samples
        try:
            img = cv2.imread(img_path)
            if img is not None:
                # Resize to standard size
                img_resized = cv2.resize(img, (200, 200))
                montage_path = assets_dir / f"sample_{i:02d}.jpg"
                cv2.imwrite(str(montage_path), img_resized)
                montage_paths.append(montage_path)
        except Exception as e:
            print(f"[WARN] Could not process sample image {img_path}: {e}")
    
    if montage_paths:
        montage_section += "Sample images from the dataset:\n\n"
        for i, montage_path in enumerate(montage_paths):
            montage_section += f"![Sample {i+1}](preupload_assets/{montage_path.name})\n"
    else:
        montage_section += "No sample images available.\n"
    
    return montage_section


def _generate_recommendations(
    audit_results: Dict[str, Any],
    pii_results: Dict[str, Any],
    rtl_results: Dict[str, Any],
    license_results: Dict[str, Any]
) -> str:
    """Generate recommendations section."""
    recommendations = ["## Recommendations\n"]
    
    # Data quality recommendations
    image_issues = len(audit_results.get("image_sanity", {}).get("unreadable", []))
    label_issues = len(audit_results.get("label_sanity", {}).get("invalid_coords", []))
    duplicates = len(audit_results.get("duplicates", []))
    
    if image_issues > 0:
        recommendations.append(f"- 🔧 Fix {image_issues} unreadable images")
    
    if label_issues > 0:
        recommendations.append(f"- 🔧 Fix {label_issues} invalid label coordinates")
    
    if duplicates > 0:
        recommendations.append(f"- 🔧 Review {duplicates} duplicate images")
    
    # PII recommendations
    pii_findings = pii_results.get("summary", {}).get("total_findings", 0)
    if pii_findings > 0:
        recommendations.append(f"- 🔒 Review {pii_findings} PII findings for compliance")
    
    # RTL/Rotation recommendations
    rtl_percentage = rtl_results.get("summary", {}).get("rtl_percentage", 0)
    skewed_percentage = rtl_results.get("summary", {}).get("skewed_percentage", 0)
    
    if rtl_percentage > 20:
        recommendations.append(f"- 🔄 Implement RTL-specific preprocessing ({rtl_percentage:.1f}% RTL pages)")
    
    if skewed_percentage > 10:
        recommendations.append(f"- 🔄 Implement automatic deskewing ({skewed_percentage:.1f}% skewed pages)")
    
    # License recommendations
    has_license = license_results.get("compliance_status", {}).get("has_license", False)
    if not has_license:
        recommendations.append("- 📄 Add license file to dataset")
    
    # General recommendations
    recommendations.extend([
        "- 📊 Monitor class distribution during training",
        "- 🔍 Implement data validation pipeline",
        "- 📝 Document data sources and preprocessing steps"
    ])
    
    return "\n".join(recommendations)


def _generate_class_distribution_chart(class_counts: Dict[str, int], output_path: Path) -> None:
    """Generate class distribution bar chart."""
    plt.figure(figsize=(10, 6))
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    bars = plt.bar(classes, counts, color='skyblue', edgecolor='navy', alpha=0.7)
    plt.title('Class Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                f'{count:,}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def _generate_quality_charts(image_sanity: Dict, label_sanity: Dict, assets_dir: Path) -> None:
    """Generate quality analysis charts."""
    # Create quality issues chart
    issues = {
        'Unreadable Images': len(image_sanity.get('unreadable', [])),
        'Extreme Aspect Ratios': len(image_sanity.get('extreme_aspect_ratio', [])),
        'Tiny Dimensions': len(image_sanity.get('tiny_dims', [])),
        'Corrupt EXIF': len(image_sanity.get('corrupt_exif', [])),
        'Invalid Coordinates': len(label_sanity.get('invalid_coords', [])),
        'Out of Bounds': len(label_sanity.get('out_of_bounds', [])),
        'Unknown Classes': len(label_sanity.get('unknown_classes', [])),
        'Empty Labels': len(label_sanity.get('empty_labels', []))
    }
    
    # Filter out zero values
    issues = {k: v for k, v in issues.items() if v > 0}
    
    if issues:
        plt.figure(figsize=(12, 6))
        classes = list(issues.keys())
        counts = list(issues.values())
        
        bars = plt.bar(classes, counts, color='lightcoral', edgecolor='darkred', alpha=0.7)
        plt.title('Quality Issues Summary', fontsize=16, fontweight='bold')
        plt.xlabel('Issue Type', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                    f'{count}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(assets_dir / "quality_issues.png", dpi=300, bbox_inches='tight')
        plt.close()


def _generate_class_imbalance_heatmap(class_counts: Dict[str, int], imbalanced_classes: List[Dict], assets_dir: Path) -> None:
    """Generate class imbalance heatmap."""
    if not class_counts:
        return
    
    # Create imbalance matrix
    total_boxes = sum(class_counts.values())
    imbalance_data = []
    
    for class_name, count in class_counts.items():
        percentage = (count / total_boxes * 100) if total_boxes > 0 else 0
        is_imbalanced = percentage < 5.0
        imbalance_data.append([percentage, 1 if is_imbalanced else 0])
    
    # Create heatmap
    plt.figure(figsize=(8, 6))
    classes = list(class_counts.keys())
    data = np.array(imbalance_data).T
    
    sns.heatmap(data, 
                xticklabels=classes, 
                yticklabels=['Percentage', 'Imbalanced (<5%)'],
                annot=True, 
                fmt='.1f',
                cmap='RdYlBu_r',
                cbar_kws={'label': 'Value'})
    
    plt.title('Class Imbalance Analysis', fontsize=16, fontweight='bold')
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Metric', fontsize=12)
    plt.tight_layout()
    plt.savefig(assets_dir / "class_imbalance_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
