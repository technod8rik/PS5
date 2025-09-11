"""Language ID evaluation metrics and confusion matrix."""

import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

try:
    import fasttext
    FASTTEXT_AVAILABLE = True
except ImportError:
    FASTTEXT_AVAILABLE = False


def load_language_labels(csv_path: str) -> List[Tuple[str, str]]:
    """Load language labels from CSV file.
    
    Args:
        csv_path: Path to CSV file with columns 'text' and 'language'
        
    Returns:
        List of (text, language) tuples
    """
    labels = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get('text', '').strip()
            language = row.get('language', '').strip()
            if text and language:
                labels.append((text, language))
    
    return labels


def evaluate_language_id(csv_path: str, model_path: str, output_dir: str) -> Dict[str, Any]:
    """Evaluate language ID model performance.
    
    Args:
        csv_path: Path to CSV file with ground truth labels
        model_path: Path to fastText model
        output_dir: Output directory for results
        
    Returns:
        Evaluation metrics
    """
    if not FASTTEXT_AVAILABLE:
        raise ImportError("fasttext not available. Install with: pip install fasttext")
    
    # Load model
    model = fasttext.load_model(model_path)
    
    # Load ground truth labels
    labels = load_language_labels(csv_path)
    
    if not labels:
        raise ValueError("No labels found in CSV file")
    
    # Predict languages
    predictions = []
    confidences = []
    
    for text, true_lang in labels:
        # Get prediction
        pred = model.predict(text)
        if pred and len(pred) > 0:
            pred_lang = pred[0][0].replace('__label__', '')
            conf = pred[0][1] if len(pred[0]) > 1 else 1.0
        else:
            pred_lang = 'unknown'
            conf = 0.0
        
        predictions.append(pred_lang)
        confidences.append(conf)
    
    # Compute accuracy
    correct = sum(1 for pred, (_, true) in zip(predictions, labels) if pred == true)
    accuracy = correct / len(labels)
    
    # Compute confusion matrix
    confusion_matrix = compute_confusion_matrix(labels, predictions)
    
    # Compute per-language metrics
    per_language_metrics = compute_per_language_metrics(labels, predictions)
    
    # Compute confidence statistics
    confidence_stats = {
        'mean': float(np.mean(confidences)),
        'std': float(np.std(confidences)),
        'min': float(np.min(confidences)),
        'max': float(np.max(confidences))
    }
    
    # Combine metrics
    metrics = {
        'accuracy': accuracy,
        'total_samples': len(labels),
        'correct_predictions': correct,
        'confusion_matrix': confusion_matrix,
        'per_language': per_language_metrics,
        'confidence_stats': confidence_stats
    }
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results_file = output_path / "langid_metrics.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    # Save confusion matrix as CSV
    confusion_file = output_path / "confusion_matrix.csv"
    save_confusion_matrix_csv(confusion_matrix, confusion_file)
    
    # Generate confusion matrix plot
    plot_file = output_path / "confusion_matrix.png"
    plot_confusion_matrix(confusion_matrix, plot_file)
    
    print(f"[INFO] Language ID evaluation completed. Accuracy: {accuracy:.3f}")
    
    return metrics


def compute_confusion_matrix(labels: List[Tuple[str, str]], predictions: List[str]) -> Dict[str, Dict[str, int]]:
    """Compute confusion matrix.
    
    Args:
        labels: List of (text, true_language) tuples
        predictions: List of predicted languages
        
    Returns:
        Confusion matrix as nested dictionary
    """
    # Get all unique languages
    all_languages = set()
    for _, true_lang in labels:
        all_languages.add(true_lang)
    for pred_lang in predictions:
        all_languages.add(pred_lang)
    
    all_languages = sorted(list(all_languages))
    
    # Initialize confusion matrix
    confusion_matrix = {}
    for true_lang in all_languages:
        confusion_matrix[true_lang] = {}
        for pred_lang in all_languages:
            confusion_matrix[true_lang][pred_lang] = 0
    
    # Fill confusion matrix
    for (_, true_lang), pred_lang in zip(labels, predictions):
        confusion_matrix[true_lang][pred_lang] += 1
    
    return confusion_matrix


def compute_per_language_metrics(labels: List[Tuple[str, str]], predictions: List[str]) -> Dict[str, Dict[str, float]]:
    """Compute per-language metrics (precision, recall, F1).
    
    Args:
        labels: List of (text, true_language) tuples
        predictions: List of predicted languages
        
    Returns:
        Per-language metrics
    """
    # Get all unique languages
    all_languages = set()
    for _, true_lang in labels:
        all_languages.add(true_lang)
    for pred_lang in predictions:
        all_languages.add(pred_lang)
    
    all_languages = sorted(list(all_languages))
    
    # Compute metrics for each language
    per_language_metrics = {}
    
    for language in all_languages:
        # True positives: correctly predicted as this language
        tp = sum(1 for (_, true_lang), pred_lang in zip(labels, predictions) 
                if true_lang == language and pred_lang == language)
        
        # False positives: incorrectly predicted as this language
        fp = sum(1 for (_, true_lang), pred_lang in zip(labels, predictions) 
                if true_lang != language and pred_lang == language)
        
        # False negatives: incorrectly predicted as other languages
        fn = sum(1 for (_, true_lang), pred_lang in zip(labels, predictions) 
                if true_lang == language and pred_lang != language)
        
        # True negatives: correctly predicted as other languages
        tn = sum(1 for (_, true_lang), pred_lang in zip(labels, predictions) 
                if true_lang != language and pred_lang != language)
        
        # Compute metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Support (number of true samples)
        support = sum(1 for _, true_lang in labels if true_lang == language)
        
        per_language_metrics[language] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn
        }
    
    return per_language_metrics


def save_confusion_matrix_csv(confusion_matrix: Dict[str, Dict[str, int]], output_file: Path) -> None:
    """Save confusion matrix as CSV file.
    
    Args:
        confusion_matrix: Confusion matrix dictionary
        output_file: Output CSV file path
    """
    # Get all languages
    languages = sorted(confusion_matrix.keys())
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write header
        header = [''] + languages
        writer.writerow(header)
        
        # Write rows
        for true_lang in languages:
            row = [true_lang]
            for pred_lang in languages:
                row.append(confusion_matrix[true_lang][pred_lang])
            writer.writerow(row)


def plot_confusion_matrix(confusion_matrix: Dict[str, Dict[str, int]], output_file: Path) -> None:
    """Plot confusion matrix as heatmap.
    
    Args:
        confusion_matrix: Confusion matrix dictionary
        output_file: Output PNG file path
    """
    # Get all languages
    languages = sorted(confusion_matrix.keys())
    
    # Create matrix array
    matrix = np.zeros((len(languages), len(languages)))
    for i, true_lang in enumerate(languages):
        for j, pred_lang in enumerate(languages):
            matrix[i, j] = confusion_matrix[true_lang][pred_lang]
    
    # Create plot
    plt.figure(figsize=(10, 8))
    plt.imshow(matrix, cmap='Blues', interpolation='nearest')
    plt.colorbar()
    
    # Set labels
    plt.xticks(range(len(languages)), languages, rotation=45)
    plt.yticks(range(len(languages)), languages)
    
    # Add text annotations
    for i in range(len(languages)):
        for j in range(len(languages)):
            plt.text(j, i, int(matrix[i, j]), ha='center', va='center')
    
    plt.xlabel('Predicted Language')
    plt.ylabel('True Language')
    plt.title('Language ID Confusion Matrix')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] Confusion matrix plot saved to {output_file}")


def generate_langid_report(metrics: Dict[str, Any], output_dir: str) -> str:
    """Generate language ID evaluation report.
    
    Args:
        metrics: Evaluation metrics
        output_dir: Output directory
        
    Returns:
        Path to report file
    """
    report_file = Path(output_dir) / "langid_report.md"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# Language ID Evaluation Report\n\n")
        
        # Overall metrics
        f.write("## Overall Metrics\n\n")
        f.write(f"- **Accuracy**: {metrics['accuracy']:.3f}\n")
        f.write(f"- **Total Samples**: {metrics['total_samples']}\n")
        f.write(f"- **Correct Predictions**: {metrics['correct_predictions']}\n\n")
        
        # Confidence statistics
        conf_stats = metrics['confidence_stats']
        f.write("## Confidence Statistics\n\n")
        f.write(f"- **Mean**: {conf_stats['mean']:.3f}\n")
        f.write(f"- **Std**: {conf_stats['std']:.3f}\n")
        f.write(f"- **Min**: {conf_stats['min']:.3f}\n")
        f.write(f"- **Max**: {conf_stats['max']:.3f}\n\n")
        
        # Per-language metrics
        f.write("## Per-Language Metrics\n\n")
        f.write("| Language | Precision | Recall | F1 | Support |\n")
        f.write("|----------|-----------|--------|----|---------|\n")
        
        for lang, lang_metrics in metrics['per_language'].items():
            f.write(f"| {lang} | {lang_metrics['precision']:.3f} | "
                   f"{lang_metrics['recall']:.3f} | {lang_metrics['f1']:.3f} | "
                   f"{lang_metrics['support']} |\n")
    
    print(f"[INFO] Language ID report saved to {report_file}")
    
    return str(report_file)
