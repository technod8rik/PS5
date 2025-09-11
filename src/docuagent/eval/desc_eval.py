"""Description evaluation metrics (BLEU, BERTScore) for VLM outputs."""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    from bert_score import score as bert_score
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False


def compute_bleu_score(reference: str, candidate: str) -> float:
    """Compute BLEU score between reference and candidate text.
    
    Args:
        reference: Reference text
        candidate: Candidate text
        
    Returns:
        BLEU score
    """
    if not NLTK_AVAILABLE:
        raise ImportError("nltk not available. Install with: pip install nltk")
    
    # Tokenize texts
    ref_tokens = reference.lower().split()
    cand_tokens = candidate.lower().split()
    
    if not ref_tokens or not cand_tokens:
        return 0.0
    
    # Use smoothing function to handle edge cases
    smoothing = SmoothingFunction().method1
    
    # Compute BLEU score
    bleu = sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smoothing)
    
    return float(bleu)


def compute_bertscore(references: List[str], candidates: List[str]) -> Tuple[List[float], List[float], List[float]]:
    """Compute BERTScore between references and candidates.
    
    Args:
        references: List of reference texts
        candidates: List of candidate texts
        
    Returns:
        Tuple of (precision, recall, f1) scores
    """
    if not BERTSCORE_AVAILABLE:
        raise ImportError("bert_score not available. Install with: pip install bert_score")
    
    # Compute BERTScore
    P, R, F1 = bert_score(candidates, references, lang="en", verbose=False)
    
    return P.tolist(), R.tolist(), F1.tolist()


def evaluate_descriptions(pred_json: str, ref_json: str, output_dir: str) -> Dict[str, Any]:
    """Evaluate description quality using BLEU and BERTScore.
    
    Args:
        pred_json: Path to predictions JSON file
        ref_json: Path to reference descriptions JSON file
        output_dir: Output directory for results
        
    Returns:
        Evaluation metrics
    """
    # Load predictions
    with open(pred_json, 'r', encoding='utf-8') as f:
        pred_data = json.load(f)
    
    # Load references
    with open(ref_json, 'r', encoding='utf-8') as f:
        ref_data = json.load(f)
    
    # Create reference mapping
    ref_map = {}
    if isinstance(ref_data, list):
        # List format
        for item in ref_data:
            key = item.get('image_name', item.get('id', ''))
            ref_map[key] = item.get('description', '')
    elif isinstance(ref_data, dict) and 'elements' in ref_data:
        # COCO format
        for element in ref_data['elements']:
            if element.get('class') in ['Table', 'Figure']:
                key = f"{element.get('page', 0)}_{element.get('bbox', [0,0,0,0])}"
                ref_map[key] = element.get('content', '')
    
    # Collect predictions and references
    predictions = []
    references = []
    valid_samples = []
    
    if isinstance(pred_data, list):
        # List format
        for item in pred_data:
            if item.get('class') in ['Table', 'Figure']:
                key = item.get('image_name', item.get('id', ''))
                pred_desc = item.get('content', '')
                ref_desc = ref_map.get(key, '')
                
                if pred_desc and ref_desc:
                    predictions.append(pred_desc)
                    references.append(ref_desc)
                    valid_samples.append(key)
    elif isinstance(pred_data, dict) and 'elements' in pred_data:
        # COCO format
        for element in pred_data['elements']:
            if element.get('class') in ['Table', 'Figure']:
                key = f"{element.get('page', 0)}_{element.get('bbox', [0,0,0,0])}"
                pred_desc = element.get('content', '')
                ref_desc = ref_map.get(key, '')
                
                if pred_desc and ref_desc:
                    predictions.append(pred_desc)
                    references.append(ref_desc)
                    valid_samples.append(key)
    
    if not predictions:
        print("[WARN] No valid prediction-reference pairs found")
        return {'error': 'No valid samples found'}
    
    print(f"[INFO] Evaluating {len(predictions)} description pairs")
    
    # Compute BLEU scores
    bleu_scores = []
    if NLTK_AVAILABLE:
        for ref, pred in zip(references, predictions):
            bleu = compute_bleu_score(ref, pred)
            bleu_scores.append(bleu)
    else:
        print("[WARN] NLTK not available, skipping BLEU computation")
        bleu_scores = [0.0] * len(predictions)
    
    # Compute BERTScore
    bert_precision = []
    bert_recall = []
    bert_f1 = []
    
    if BERTSCORE_AVAILABLE:
        try:
            P, R, F1 = compute_bertscore(references, predictions)
            bert_precision = P
            bert_recall = R
            bert_f1 = F1
        except Exception as e:
            print(f"[WARN] BERTScore computation failed: {e}")
            bert_precision = [0.0] * len(predictions)
            bert_recall = [0.0] * len(predictions)
            bert_f1 = [0.0] * len(predictions)
    else:
        print("[WARN] BERTScore not available, skipping BERTScore computation")
        bert_precision = [0.0] * len(predictions)
        bert_recall = [0.0] * len(predictions)
        bert_f1 = [0.0] * len(predictions)
    
    # Compute statistics
    metrics = {
        'total_samples': len(predictions),
        'bleu': {
            'mean': float(np.mean(bleu_scores)),
            'std': float(np.std(bleu_scores)),
            'min': float(np.min(bleu_scores)),
            'max': float(np.max(bleu_scores))
        },
        'bertscore': {
            'precision': {
                'mean': float(np.mean(bert_precision)),
                'std': float(np.std(bert_precision)),
                'min': float(np.min(bert_precision)),
                'max': float(np.max(bert_precision))
            },
            'recall': {
                'mean': float(np.mean(bert_recall)),
                'std': float(np.std(bert_recall)),
                'min': float(np.min(bert_recall)),
                'max': float(np.max(bert_recall))
            },
            'f1': {
                'mean': float(np.mean(bert_f1)),
                'std': float(np.std(bert_f1)),
                'min': float(np.min(bert_f1)),
                'max': float(np.max(bert_f1))
            }
        }
    }
    
    # Per-class analysis
    class_metrics = {}
    for class_name in ['Table', 'Figure']:
        class_indices = [i for i, sample in enumerate(valid_samples) 
                        if any(pred_data.get('elements', [{}])[j].get('class') == class_name 
                              for j in range(len(pred_data.get('elements', []))))]
        
        if class_indices:
            class_bleu = [bleu_scores[i] for i in class_indices]
            class_bert_f1 = [bert_f1[i] for i in class_indices]
            
            class_metrics[class_name] = {
                'count': len(class_indices),
                'bleu_mean': float(np.mean(class_bleu)),
                'bertscore_f1_mean': float(np.mean(class_bert_f1))
            }
    
    metrics['per_class'] = class_metrics
    
    # Save detailed results
    detailed_results = []
    for i, (pred, ref, bleu, bert_p, bert_r, bert_f) in enumerate(zip(
        predictions, references, bleu_scores, bert_precision, bert_recall, bert_f1)):
        detailed_results.append({
            'sample_id': valid_samples[i],
            'prediction': pred,
            'reference': ref,
            'bleu': bleu,
            'bertscore_precision': bert_p,
            'bertscore_recall': bert_r,
            'bertscore_f1': bert_f
        })
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    metrics_file = output_path / "desc_metrics.json"
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    # Save detailed results
    detailed_file = output_path / "desc_detailed.json"
    with open(detailed_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    
    print(f"[INFO] Description evaluation completed. Mean BLEU: {metrics['bleu']['mean']:.3f}, "
          f"Mean BERTScore F1: {metrics['bertscore']['f1']['mean']:.3f}")
    
    return metrics


def generate_desc_report(metrics: Dict[str, Any], output_dir: str) -> str:
    """Generate description evaluation report.
    
    Args:
        metrics: Evaluation metrics
        output_dir: Output directory
        
    Returns:
        Path to report file
    """
    report_file = Path(output_dir) / "desc_report.md"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# Description Evaluation Report\n\n")
        
        # Overall metrics
        f.write("## Overall Metrics\n\n")
        f.write(f"- **Total Samples**: {metrics['total_samples']}\n\n")
        
        # BLEU scores
        bleu = metrics['bleu']
        f.write("## BLEU Scores\n\n")
        f.write(f"- **Mean**: {bleu['mean']:.3f} ± {bleu['std']:.3f}\n")
        f.write(f"- **Min**: {bleu['min']:.3f}\n")
        f.write(f"- **Max**: {bleu['max']:.3f}\n\n")
        
        # BERTScore
        bert = metrics['bertscore']
        f.write("## BERTScore\n\n")
        f.write("### Precision\n")
        f.write(f"- **Mean**: {bert['precision']['mean']:.3f} ± {bert['precision']['std']:.3f}\n")
        f.write(f"- **Min**: {bert['precision']['min']:.3f}\n")
        f.write(f"- **Max**: {bert['precision']['max']:.3f}\n\n")
        
        f.write("### Recall\n")
        f.write(f"- **Mean**: {bert['recall']['mean']:.3f} ± {bert['recall']['std']:.3f}\n")
        f.write(f"- **Min**: {bert['recall']['min']:.3f}\n")
        f.write(f"- **Max**: {bert['recall']['max']:.3f}\n\n")
        
        f.write("### F1\n")
        f.write(f"- **Mean**: {bert['f1']['mean']:.3f} ± {bert['f1']['std']:.3f}\n")
        f.write(f"- **Min**: {bert['f1']['min']:.3f}\n")
        f.write(f"- **Max**: {bert['f1']['max']:.3f}\n\n")
        
        # Per-class metrics
        if 'per_class' in metrics:
            f.write("## Per-Class Metrics\n\n")
            f.write("| Class | Count | BLEU Mean | BERTScore F1 Mean |\n")
            f.write("|-------|-------|-----------|-------------------|\n")
            
            for class_name, class_metrics in metrics['per_class'].items():
                f.write(f"| {class_name} | {class_metrics['count']} | "
                       f"{class_metrics['bleu_mean']:.3f} | "
                       f"{class_metrics['bertscore_f1_mean']:.3f} |\n")
    
    print(f"[INFO] Description report saved to {report_file}")
    
    return str(report_file)
