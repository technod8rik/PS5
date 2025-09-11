"""Autopilot for active learning cycles - orchestrates mine → select → pseudo-label → retrain → eval."""

import json
import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import time
import subprocess
import sys

from .mine_errors import mine_errors
from .uncertainty import select_samples, UncertaintyStrategy
from .pseudo_labels import create_pseudo_labels, merge_with_existing_dataset


@dataclass
class CycleConfig:
    """Configuration for an active learning cycle."""
    cycle_id: int
    train_coco: str
    val_coco: str
    images_dir: str
    project_dir: str
    quota: int = 500
    pseudo_threshold: float = 0.6
    strategy: str = "margin"
    retrain_epochs: int = 20
    iou_threshold: float = 0.5


@dataclass
class CycleResults:
    """Results from an active learning cycle."""
    cycle_id: int
    start_time: str
    end_time: str
    duration_seconds: float
    
    # Error mining results
    error_cases: int
    false_positives: int
    false_negatives: int
    
    # Sample selection results
    selected_samples: int
    uncertainty_scores: List[float]
    
    # Pseudo-labeling results
    pseudo_labels: int
    pseudo_confidence_stats: Dict[str, float]
    
    # Training results
    training_metrics: Dict[str, float]
    
    # Evaluation results
    evaluation_metrics: Dict[str, float]
    metric_deltas: Dict[str, float]  # Change from previous cycle


def run_cycle(cfg: CycleConfig, previous_results: Optional[CycleResults] = None) -> CycleResults:
    """Run a single active learning cycle.
    
    Args:
        cfg: Cycle configuration
        previous_results: Results from previous cycle (for metric deltas)
        
    Returns:
        CycleResults object with all cycle information
    """
    start_time = time.time()
    start_str = time.strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"\n{'='*60}")
    print(f" ACTIVE LEARNING CYCLE {cfg.cycle_id}")
    print(f"{'='*60}")
    
    # Create cycle directory
    cycle_dir = Path(cfg.project_dir) / f"cycle_{cfg.cycle_id:02d}"
    cycle_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Mine errors from previous model predictions
    print(f"\n[STEP 1] Mining errors...")
    error_dir = cycle_dir / "errors"
    error_cases = mine_errors(
        cfg.val_coco, 
        f"{cfg.project_dir}/predictions_val.json",  # Assuming predictions exist
        cfg.images_dir, 
        str(error_dir),
        cfg.iou_threshold
    )
    
    fp_count = sum(1 for case in error_cases if case.type == "FP")
    fn_count = sum(1 for case in error_cases if case.type == "FN")
    
    print(f"[INFO] Mined {len(error_cases)} error cases ({fp_count} FPs, {fn_count} FNs)")
    
    # Step 2: Select samples for annotation
    print(f"\n[STEP 2] Selecting samples for annotation...")
    selection_dir = cycle_dir / "selection"
    selection_dir.mkdir(exist_ok=True)
    
    selected_samples = select_samples(
        f"{cfg.project_dir}/predictions_val.json",
        cfg.strategy,
        cfg.quota,
        str(selection_dir / "selected_samples.json")
    )
    
    uncertainty_scores = [s.score for s in selected_samples]
    print(f"[INFO] Selected {len(selected_samples)} samples for annotation")
    
    # Step 3: Generate pseudo-labels
    print(f"\n[STEP 3] Generating pseudo-labels...")
    pseudo_dir = cycle_dir / "pseudo_labels"
    
    pseudo_labels = create_pseudo_labels(
        f"{cfg.project_dir}/predictions_train.json",  # Assuming predictions exist
        cfg.images_dir,
        str(pseudo_dir),
        cfg.pseudo_threshold,
        f"cycle_{cfg.cycle_id}",
        cfg.cycle_id
    )
    
    pseudo_conf_stats = {
        'min': min(label.confidence for label in pseudo_labels) if pseudo_labels else 0,
        'max': max(label.confidence for label in pseudo_labels) if pseudo_labels else 0,
        'mean': sum(label.confidence for label in pseudo_labels) / len(pseudo_labels) if pseudo_labels else 0
    }
    
    print(f"[INFO] Generated {len(pseudo_labels)} pseudo-labels")
    
    # Step 4: Merge datasets
    print(f"\n[STEP 4] Merging datasets...")
    merged_dir = cycle_dir / "merged_dataset"
    
    # Create merged dataset with pseudo-labels
    merge_with_existing_dataset(
        str(pseudo_dir),
        cfg.train_coco.replace('.json', ''),  # Assume directory exists
        str(merged_dir)
    )
    
    print(f"[INFO] Merged dataset created at {merged_dir}")
    
    # Step 5: Retrain model
    print(f"\n[STEP 5] Retraining model...")
    training_dir = cycle_dir / "training"
    training_dir.mkdir(exist_ok=True)
    
    # Convert merged dataset to YOLO format if needed
    yolo_dir = training_dir / "yolo_dataset"
    yolo_dir.mkdir(exist_ok=True)
    
    # Run training (this would call the actual training function)
    training_metrics = run_training(
        str(merged_dir),
        str(yolo_dir),
        str(training_dir),
        cfg.retrain_epochs
    )
    
    print(f"[INFO] Training completed with metrics: {training_metrics}")
    
    # Step 6: Evaluate model
    print(f"\n[STEP 6] Evaluating model...")
    eval_dir = cycle_dir / "evaluation"
    eval_dir.mkdir(exist_ok=True)
    
    evaluation_metrics = run_evaluation(
        str(training_dir / "best.pt"),  # Assuming best weights saved
        cfg.val_coco,
        cfg.images_dir,
        str(eval_dir)
    )
    
    print(f"[INFO] Evaluation completed with metrics: {evaluation_metrics}")
    
    # Calculate metric deltas
    metric_deltas = {}
    if previous_results:
        for metric, current_value in evaluation_metrics.items():
            if metric in previous_results.evaluation_metrics:
                previous_value = previous_results.evaluation_metrics[metric]
                metric_deltas[metric] = current_value - previous_value
            else:
                metric_deltas[metric] = 0.0
    
    # Create results
    end_time = time.time()
    end_str = time.strftime("%Y-%m-%d %H:%M:%S")
    duration = end_time - start_time
    
    results = CycleResults(
        cycle_id=cfg.cycle_id,
        start_time=start_str,
        end_time=end_str,
        duration_seconds=duration,
        error_cases=len(error_cases),
        false_positives=fp_count,
        false_negatives=fn_count,
        selected_samples=len(selected_samples),
        uncertainty_scores=uncertainty_scores,
        pseudo_labels=len(pseudo_labels),
        pseudo_confidence_stats=pseudo_conf_stats,
        training_metrics=training_metrics,
        evaluation_metrics=evaluation_metrics,
        metric_deltas=metric_deltas
    )
    
    # Save results
    save_cycle_results(results, cycle_dir)
    
    print(f"\n[INFO] Cycle {cfg.cycle_id} completed in {duration:.1f} seconds")
    print(f"[INFO] Results saved to {cycle_dir}")
    
    return results


def run_training(
    dataset_dir: str,
    yolo_dir: str,
    output_dir: str,
    epochs: int
) -> Dict[str, float]:
    """Run model training.
    
    This is a placeholder - in practice, you'd call the actual training function.
    """
    print(f"[INFO] Training model for {epochs} epochs...")
    
    # Simulate training metrics
    metrics = {
        'loss': 0.5,
        'mAP': 0.75,
        'precision': 0.80,
        'recall': 0.70,
        'f1': 0.75
    }
    
    # In practice, you would:
    # 1. Convert dataset to YOLO format
    # 2. Call training function from docuagent.training.yolo_train
    # 3. Return actual metrics
    
    return metrics


def run_evaluation(
    model_path: str,
    val_coco: str,
    images_dir: str,
    output_dir: str
) -> Dict[str, float]:
    """Run model evaluation.
    
    This is a placeholder - in practice, you'd call the actual evaluation function.
    """
    print(f"[INFO] Evaluating model...")
    
    # Simulate evaluation metrics
    metrics = {
        'mAP': 0.78,
        'mAP_50': 0.85,
        'mAP_75': 0.72,
        'precision': 0.82,
        'recall': 0.73,
        'f1': 0.77
    }
    
    # In practice, you would:
    # 1. Call evaluation function from docuagent.eval.coco_eval
    # 2. Return actual metrics
    
    return metrics


def save_cycle_results(results: CycleResults, cycle_dir: Path) -> None:
    """Save cycle results to JSON file."""
    results_dict = {
        'cycle_id': results.cycle_id,
        'start_time': results.start_time,
        'end_time': results.end_time,
        'duration_seconds': results.duration_seconds,
        'error_cases': results.error_cases,
        'false_positives': results.false_positives,
        'false_negatives': results.false_negatives,
        'selected_samples': results.selected_samples,
        'uncertainty_scores': results.uncertainty_scores,
        'pseudo_labels': results.pseudo_labels,
        'pseudo_confidence_stats': results.pseudo_confidence_stats,
        'training_metrics': results.training_metrics,
        'evaluation_metrics': results.evaluation_metrics,
        'metric_deltas': results.metric_deltas
    }
    
    results_file = cycle_dir / "cycle_results.json"
    with open(results_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"[INFO] Cycle results saved to {results_file}")


def run_autopilot(
    train_coco: str,
    val_coco: str,
    images_dir: str,
    project_dir: str,
    cycles: int = 2,
    quota: int = 500,
    pseudo_thr: float = 0.6,
    strategy: str = "margin",
    retrain_epochs: int = 20
) -> List[CycleResults]:
    """Run multiple active learning cycles.
    
    Args:
        train_coco: Path to training COCO JSON
        val_coco: Path to validation COCO JSON
        images_dir: Directory containing images
        project_dir: Project directory for outputs
        cycles: Number of cycles to run
        quota: Number of samples to select per cycle
        pseudo_thr: Pseudo-label confidence threshold
        strategy: Uncertainty sampling strategy
        retrain_epochs: Number of epochs for retraining
        
    Returns:
        List of CycleResults for all cycles
    """
    print(f"\n{'='*60}")
    print(f" AUTOPILOT ACTIVE LEARNING")
    print(f"{'='*60}")
    print(f"Cycles: {cycles}")
    print(f"Quota per cycle: {quota}")
    print(f"Pseudo-label threshold: {pseudo_thr}")
    print(f"Strategy: {strategy}")
    print(f"Retrain epochs: {retrain_epochs}")
    
    # Create project directory
    project_path = Path(project_dir)
    project_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize results
    all_results = []
    previous_results = None
    
    # Run cycles
    for cycle_id in range(1, cycles + 1):
        cfg = CycleConfig(
            cycle_id=cycle_id,
            train_coco=train_coco,
            val_coco=val_coco,
            images_dir=images_dir,
            project_dir=project_dir,
            quota=quota,
            pseudo_threshold=pseudo_thr,
            strategy=strategy,
            retrain_epochs=retrain_epochs
        )
        
        results = run_cycle(cfg, previous_results)
        all_results.append(results)
        previous_results = results
        
        # Print cycle summary
        print(f"\n[SUMMARY] Cycle {cycle_id}:")
        print(f"  Error cases: {results.error_cases} ({results.false_positives} FP, {results.false_negatives} FN)")
        print(f"  Selected samples: {results.selected_samples}")
        print(f"  Pseudo-labels: {results.pseudo_labels}")
        print(f"  Training mAP: {results.training_metrics.get('mAP', 0):.3f}")
        print(f"  Evaluation mAP: {results.evaluation_metrics.get('mAP', 0):.3f}")
        
        if results.metric_deltas:
            print(f"  Metric deltas:")
            for metric, delta in results.metric_deltas.items():
                print(f"    {metric}: {delta:+.3f}")
    
    # Save overall results
    save_autopilot_results(all_results, project_path)
    
    print(f"\n[INFO] Autopilot completed {cycles} cycles")
    print(f"[INFO] All results saved to {project_path}")
    
    return all_results


def save_autopilot_results(results: List[CycleResults], project_dir: Path) -> None:
    """Save overall autopilot results."""
    summary = {
        'total_cycles': len(results),
        'total_duration': sum(r.duration_seconds for r in results),
        'cycles': [
            {
                'cycle_id': r.cycle_id,
                'duration_seconds': r.duration_seconds,
                'error_cases': r.error_cases,
                'selected_samples': r.selected_samples,
                'pseudo_labels': r.pseudo_labels,
                'evaluation_mAP': r.evaluation_metrics.get('mAP', 0),
                'metric_deltas': r.metric_deltas
            }
            for r in results
        ]
    }
    
    summary_file = project_dir / "autopilot_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"[INFO] Autopilot summary saved to {summary_file}")


def main():
    """Command line interface for autopilot."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run active learning autopilot")
    parser.add_argument("--coco", required=True, help="Training COCO JSON file")
    parser.add_argument("--val", required=True, help="Validation COCO JSON file")
    parser.add_argument("--images", required=True, help="Images directory")
    parser.add_argument("--cycles", type=int, default=2, help="Number of cycles")
    parser.add_argument("--quota", type=int, default=500, help="Samples per cycle")
    parser.add_argument("--pseudo-thr", type=float, default=0.6, help="Pseudo-label threshold")
    parser.add_argument("--strategy", choices=["margin", "entropy", "low_conf"], 
                       default="margin", help="Uncertainty strategy")
    parser.add_argument("--project", required=True, help="Project directory")
    parser.add_argument("--name", help="Project name")
    parser.add_argument("--retrain-epochs", type=int, default=20, help="Retrain epochs")
    
    args = parser.parse_args()
    
    results = run_autopilot(
        args.coco, args.val, args.images, args.project,
        args.cycles, args.quota, args.pseudo_thr, args.strategy, args.retrain_epochs
    )
    
    print(f"[INFO] Autopilot completed {len(results)} cycles")


if __name__ == "__main__":
    main()
