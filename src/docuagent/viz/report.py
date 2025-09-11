"""Report generation utilities for evaluation results."""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime


def generate_evaluation_report(metrics: Dict[str, Any], output_dir: str, 
                             title: str = "Document Processing Evaluation Report") -> str:
    """Generate comprehensive evaluation report.
    
    Args:
        metrics: Dictionary containing all evaluation metrics
        output_dir: Output directory for report
        title: Report title
        
    Returns:
        Path to generated report
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    report_file = output_path / "evaluation_report.md"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        # Header
        f.write(f"# {title}\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Table of contents
        f.write("## Table of Contents\n\n")
        f.write("- [Dataset Overview](#dataset-overview)\n")
        f.write("- [Layout Detection Metrics](#layout-detection-metrics)\n")
        f.write("- [Text Processing Metrics](#text-processing-metrics)\n")
        f.write("- [Language ID Metrics](#language-id-metrics)\n")
        f.write("- [Description Quality Metrics](#description-quality-metrics)\n")
        f.write("- [Error Analysis](#error-analysis)\n")
        f.write("- [Summary](#summary)\n\n")
        
        # Dataset overview
        f.write("## Dataset Overview\n\n")
        if 'dataset' in metrics:
            dataset = metrics['dataset']
            f.write(f"- **Total Images**: {dataset.get('total_images', 'N/A')}\n")
            f.write(f"- **Total Annotations**: {dataset.get('total_annotations', 'N/A')}\n")
            f.write(f"- **Classes**: {', '.join(dataset.get('classes', []))}\n\n")
        
        # Layout detection metrics
        if 'layout_detection' in metrics:
            f.write("## Layout Detection Metrics\n\n")
            layout = metrics['layout_detection']
            
            f.write("### Overall Performance\n\n")
            f.write(f"- **mAP@0.5**: {layout.get('mAP_0.5', 0):.3f}\n")
            f.write(f"- **mAP@0.5:0.95**: {layout.get('mAP_0.5_0.95', 0):.3f}\n")
            f.write(f"- **Precision**: {layout.get('precision', 0):.3f}\n")
            f.write(f"- **Recall**: {layout.get('recall', 0):.3f}\n")
            f.write(f"- **F1 Score**: {layout.get('f1', 0):.3f}\n\n")
            
            # Per-class metrics
            if 'per_class' in layout:
                f.write("### Per-Class Performance\n\n")
                f.write("| Class | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |\n")
                f.write("|-------|---------|--------------|-----------|--------|\n")
                
                for class_name, class_metrics in layout['per_class'].items():
                    f.write(f"| {class_name} | {class_metrics.get('mAP_0.5', 0):.3f} | "
                           f"{class_metrics.get('mAP_0.5_0.95', 0):.3f} | "
                           f"{class_metrics.get('precision', 0):.3f} | "
                           f"{class_metrics.get('recall', 0):.3f} |\n")
                f.write("\n")
        
        # Text processing metrics
        if 'text_processing' in metrics:
            f.write("## Text Processing Metrics\n\n")
            text = metrics['text_processing']
            
            f.write("### OCR Quality\n\n")
            f.write(f"- **Mean CER**: {text.get('mean_cer', 0):.3f}\n")
            f.write(f"- **Mean WER**: {text.get('mean_wer', 0):.3f}\n")
            f.write(f"- **Total Samples**: {text.get('total_samples', 0)}\n\n")
            
            # Per-class text metrics
            if 'per_class' in text:
                f.write("### Per-Class Text Quality\n\n")
                f.write("| Class | Mean CER | Mean WER | Count |\n")
                f.write("|-------|----------|----------|-------|\n")
                
                for class_name, class_metrics in text['per_class'].items():
                    f.write(f"| {class_name} | {class_metrics.get('mean_cer', 0):.3f} | "
                           f"{class_metrics.get('mean_wer', 0):.3f} | "
                           f"{class_metrics.get('count', 0)} |\n")
                f.write("\n")
        
        # Language ID metrics
        if 'language_id' in metrics:
            f.write("## Language ID Metrics\n\n")
            langid = metrics['language_id']
            
            f.write(f"- **Overall Accuracy**: {langid.get('accuracy', 0):.3f}\n")
            f.write(f"- **Total Samples**: {langid.get('total_samples', 0)}\n")
            f.write(f"- **Correct Predictions**: {langid.get('correct_predictions', 0)}\n\n")
            
            # Per-language metrics
            if 'per_language' in langid:
                f.write("### Per-Language Performance\n\n")
                f.write("| Language | Precision | Recall | F1 | Support |\n")
                f.write("|----------|-----------|--------|----|---------|\n")
                
                for lang, lang_metrics in langid['per_language'].items():
                    f.write(f"| {lang} | {lang_metrics.get('precision', 0):.3f} | "
                           f"{lang_metrics.get('recall', 0):.3f} | "
                           f"{lang_metrics.get('f1', 0):.3f} | "
                           f"{lang_metrics.get('support', 0)} |\n")
                f.write("\n")
        
        # Description quality metrics
        if 'description_quality' in metrics:
            f.write("## Description Quality Metrics\n\n")
            desc = metrics['description_quality']
            
            f.write("### BLEU Scores\n\n")
            f.write(f"- **Mean BLEU**: {desc.get('bleu_mean', 0):.3f}\n")
            f.write(f"- **Std BLEU**: {desc.get('bleu_std', 0):.3f}\n\n")
            
            f.write("### BERTScore\n\n")
            f.write(f"- **Mean Precision**: {desc.get('bertscore_precision_mean', 0):.3f}\n")
            f.write(f"- **Mean Recall**: {desc.get('bertscore_recall_mean', 0):.3f}\n")
            f.write(f"- **Mean F1**: {desc.get('bertscore_f1_mean', 0):.3f}\n\n")
            
            f.write(f"- **Total Samples**: {desc.get('total_samples', 0)}\n\n")
        
        # Error analysis
        if 'error_analysis' in metrics:
            f.write("## Error Analysis\n\n")
            errors = metrics['error_analysis']
            
            f.write(f"- **False Positives**: {errors.get('false_positives', 0)}\n")
            f.write(f"- **False Negatives**: {errors.get('false_negatives', 0)}\n")
            f.write(f"- **True Positives**: {errors.get('true_positives', 0)}\n")
            f.write(f"- **True Negatives**: {errors.get('true_negatives', 0)}\n\n")
            
            # Error distribution
            if 'error_by_class' in errors:
                f.write("### Error Distribution by Class\n\n")
                f.write("| Class | False Positives | False Negatives |\n")
                f.write("|-------|-----------------|-----------------|\n")
                
                for class_name, class_errors in errors['error_by_class'].items():
                    f.write(f"| {class_name} | {class_errors.get('fp', 0)} | "
                           f"{class_errors.get('fn', 0)} |\n")
                f.write("\n")
        
        # Summary
        f.write("## Summary\n\n")
        f.write("### Key Findings\n\n")
        
        # Layout detection summary
        if 'layout_detection' in metrics:
            layout = metrics['layout_detection']
            f.write(f"- **Layout Detection mAP@0.5**: {layout.get('mAP_0.5', 0):.3f}\n")
        
        # Text processing summary
        if 'text_processing' in metrics:
            text = metrics['text_processing']
            f.write(f"- **OCR Mean CER**: {text.get('mean_cer', 0):.3f}\n")
        
        # Language ID summary
        if 'language_id' in metrics:
            langid = metrics['language_id']
            f.write(f"- **Language ID Accuracy**: {langid.get('accuracy', 0):.3f}\n")
        
        # Description quality summary
        if 'description_quality' in metrics:
            desc = metrics['description_quality']
            f.write(f"- **Description BLEU**: {desc.get('bleu_mean', 0):.3f}\n")
        
        f.write("\n### Recommendations\n\n")
        f.write("1. **Model Improvement**: Consider fine-tuning on domain-specific data\n")
        f.write("2. **Data Quality**: Review and improve annotation quality\n")
        f.write("3. **Error Analysis**: Focus on classes with lowest performance\n")
        f.write("4. **Pipeline Optimization**: Adjust confidence thresholds based on error analysis\n\n")
        
        # Footer
        f.write("---\n")
        f.write("*Report generated by DocuAgent evaluation pipeline*\n")
    
    print(f"[INFO] Evaluation report saved to {report_file}")
    
    return str(report_file)


def generate_html_report(markdown_file: str, output_dir: str) -> str:
    """Convert Markdown report to HTML.
    
    Args:
        markdown_file: Path to Markdown report
        output_dir: Output directory for HTML report
        
    Returns:
        Path to HTML report
    """
    try:
        import markdown
        
        # Read Markdown file
        with open(markdown_file, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        # Convert to HTML
        html_content = markdown.markdown(md_content, extensions=['tables', 'toc'])
        
        # Create HTML template
        html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Document Processing Evaluation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        h1, h2, h3 {{ color: #333; }}
        .toc {{ background-color: #f9f9f9; padding: 20px; border-radius: 5px; }}
    </style>
</head>
<body>
{html_content}
</body>
</html>
"""
        
        # Save HTML file
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        html_file = output_path / "evaluation_report.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_template)
        
        print(f"[INFO] HTML report saved to {html_file}")
        
        return str(html_file)
        
    except ImportError:
        print("[WARN] markdown not available, skipping HTML generation")
        return markdown_file


def create_summary_dashboard(metrics: Dict[str, Any], output_dir: str) -> str:
    """Create a summary dashboard with key metrics.
    
    Args:
        metrics: Evaluation metrics
        output_dir: Output directory for dashboard
        
    Returns:
        Path to dashboard file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    dashboard_file = output_path / "summary_dashboard.json"
    
    # Extract key metrics
    dashboard = {
        'timestamp': datetime.now().isoformat(),
        'overview': {
            'layout_detection_mAP': metrics.get('layout_detection', {}).get('mAP_0.5', 0),
            'text_processing_cer': metrics.get('text_processing', {}).get('mean_cer', 0),
            'language_id_accuracy': metrics.get('language_id', {}).get('accuracy', 0),
            'description_bleu': metrics.get('description_quality', {}).get('bleu_mean', 0)
        },
        'status': {
            'layout_detection': 'good' if metrics.get('layout_detection', {}).get('mAP_0.5', 0) > 0.7 else 'needs_improvement',
            'text_processing': 'good' if metrics.get('text_processing', {}).get('mean_cer', 1) < 0.1 else 'needs_improvement',
            'language_id': 'good' if metrics.get('language_id', {}).get('accuracy', 0) > 0.8 else 'needs_improvement',
            'description_quality': 'good' if metrics.get('description_quality', {}).get('bleu_mean', 0) > 0.3 else 'needs_improvement'
        }
    }
    
    # Save dashboard
    with open(dashboard_file, 'w', encoding='utf-8') as f:
        json.dump(dashboard, f, indent=2, ensure_ascii=False)
    
    print(f"[INFO] Summary dashboard saved to {dashboard_file}")
    
    return str(dashboard_file)
