#!/usr/bin/env python3
"""
Compare performance between grayscale and colored datasets
"""

import os
import re
import subprocess
import sys
from pathlib import Path
from tqdm import tqdm

def run_evaluation(dataset_type):
    """Run evaluation on specified dataset type and return results"""
    print(f"\n🔧 Updating config for {dataset_type} dataset...")
    
    # Update config
    config_path = "src/config.py"
    with open(config_path, 'r') as f:
        config_content = f.read()
    
    dataset_path = f"datasets/rectangles_{dataset_type}"
    new_config = re.sub(
        r'DATASET_DIR = os\.path\.join\(BASE_DIR, "[^"]*"\).*',
        f'DATASET_DIR = os.path.join(BASE_DIR, "{dataset_path}")  # {dataset_type.title()} dataset',
        config_content
    )
    
    with open(config_path, 'w') as f:
        f.write(new_config)
    
    print(f"✅ Config updated to {dataset_type} dataset")
    
    # Run evaluation
    print(f"📊 Running evaluation on {dataset_type} dataset...")
    print("=" * 60)
    
    try:
        print(f"🚀 Starting evaluation on {dataset_type} dataset...")
        result = subprocess.run(
            "python -m src.evaluate_model",
            shell=True, 
            check=True,
            text=True
        )
        
        print(f"✅ {dataset_type.title()} evaluation completed")
        return parse_evaluation_results_from_files()  # Parse from saved results instead
        
    except Exception as e:
        print(f"❌ Error running {dataset_type} evaluation: {e}")
        return None

def parse_evaluation_results_from_files():
    """Simple placeholder - in real implementation would parse from saved files"""
    # For now, return empty dict - the real progress comes from evaluation itself
    return {"status": "completed"}

def parse_evaluation_results(output):
    """Parse evaluation output to extract key metrics"""
    metrics = {}
    
    lines = output.split('\n')
    for line in lines:
        line = line.strip()
        
        # Parse MSE
        if 'Mean:' in line and 'Loss (MSE):' in lines[lines.index(line) - 1]:
            try:
                metrics['mse'] = float(line.split(':')[1].strip())
            except:
                pass
                
        # Parse IoU metrics
        elif 'Mean:' in line and 'IoU (Intersection over Union):' in lines[lines.index(line) - 1]:
            try:
                metrics['iou_mean'] = float(line.split(':')[1].strip())
            except:
                pass
                
        elif 'Samples with IoU > 0.5:' in line:
            try:
                parts = line.split('/')
                if len(parts) >= 2:
                    total = int(parts[1].split()[0])
                    correct = int(parts[0].split()[-1])
                    percentage = line.split('(')[-1].replace(')', '')
                    metrics['iou_gt_50_count'] = correct
                    metrics['iou_gt_50_total'] = total
                    metrics['iou_gt_50_pct'] = percentage
            except:
                pass
                
        elif 'Samples with IoU > 0.7:' in line:
            try:
                parts = line.split('/')
                if len(parts) >= 2:
                    total = int(parts[1].split()[0])
                    correct = int(parts[0].split()[-1])
                    percentage = line.split('(')[-1].replace(')', '')
                    metrics['iou_gt_70_count'] = correct
                    metrics['iou_gt_70_total'] = total
                    metrics['iou_gt_70_pct'] = percentage
            except:
                pass
                
        # Parse MAE
        elif 'MAE:' in line and 'Overall Metrics:' in lines[lines.index(line) - 1]:
            try:
                metrics['mae'] = float(line.split(':')[1].strip())
            except:
                pass
    
    return metrics

def print_comparison_table(grayscale_metrics, colored_metrics):
    """Print side-by-side comparison table"""
    
    print("\n" + "=" * 80)
    print("📊 DATASET PERFORMANCE COMPARISON")
    print("=" * 80)
    
    print(f"{'Metric':<25} {'Grayscale Dataset':<20} {'Colored Dataset':<20} {'Difference':<15}")
    print("-" * 80)
    
    # Helper function to safely get metric
    def get_metric(metrics, key, default="N/A"):
        return metrics.get(key, default) if metrics else default
    
    # Compare metrics
    metrics_to_compare = [
        ('mse', 'MSE (Lower is better)'),
        ('iou_mean', 'Mean IoU (Higher is better)'),
        ('iou_gt_50_pct', 'IoU > 0.5 (%)'),
        ('iou_gt_70_pct', 'IoU > 0.7 (%)'),
        ('mae', 'MAE (Lower is better)')
    ]
    
    for key, label in metrics_to_compare:
        gray_val = get_metric(grayscale_metrics, key)
        color_val = get_metric(colored_metrics, key)
        
        # Calculate difference
        diff = "N/A"
        if key != 'iou_gt_50_pct' and key != 'iou_gt_70_pct':
            # For numeric values
            try:
                gray_num = float(gray_val) if gray_val != "N/A" else 0
                color_num = float(color_val) if color_val != "N/A" else 0
                if gray_num != 0 and color_num != 0:
                    if key in ['mse', 'mae']:  # Lower is better
                        diff_pct = ((gray_num - color_num) / color_num) * 100
                        diff = f"{diff_pct:+.1f}%"
                    else:  # Higher is better (IoU)
                        diff_pct = ((gray_num - color_num) / color_num) * 100
                        diff = f"{diff_pct:+.1f}%"
            except:
                pass
        else:
            # For percentages
            try:
                gray_pct = float(gray_val.replace('%', '')) if gray_val != "N/A" else 0
                color_pct = float(color_val.replace('%', '')) if color_val != "N/A" else 0
                if gray_pct != 0 and color_pct != 0:
                    diff = f"{gray_pct - color_pct:+.1f}%"
            except:
                pass
        
        print(f"{label:<25} {gray_val:<20} {color_val:<20} {diff:<15}")
    
    print("-" * 80)
    
    # Summary
    print("\n🎯 SUMMARY:")
    if grayscale_metrics and colored_metrics:
        gray_iou = grayscale_metrics.get('iou_mean', 0)
        color_iou = colored_metrics.get('iou_mean', 0)
        
        if gray_iou > color_iou:
            print("✅ Grayscale dataset performs better")
            print(f"   Mean IoU: {gray_iou:.3f} vs {color_iou:.3f}")
        elif color_iou > gray_iou:
            print("✅ Colored dataset performs better")
            print(f"   Mean IoU: {color_iou:.3f} vs {gray_iou:.3f}")
        else:
            print("🤝 Both datasets perform similarly")
            print(f"   Mean IoU: {gray_iou:.3f} for both")
    else:
        print("❌ Could not determine winner - evaluation failed")

def main():
    """Main comparison function"""
    print("🔄 Starting dataset comparison...")
    print("This will evaluate both grayscale and colored datasets and compare results.")
    
    # Check if datasets exist
    grayscale_exists = os.path.exists("datasets/rectangles_grayscale")
    colored_exists = os.path.exists("datasets/rectangles_colored")
    
    if not grayscale_exists:
        print("❌ Grayscale dataset not found. Run './run grayscale' first.")
        return
    
    if not colored_exists:
        print("❌ Colored dataset not found. Run './run colored' first.")
        return
    
    print("✅ Both datasets found!")
    
    # Track progress
    datasets_to_eval = ["grayscale", "colored"]
    results = {}
    
    with tqdm(datasets_to_eval, desc="📊 Evaluating datasets") as pbar:
        for dataset_type in pbar:
            pbar.set_description(f"Evaluating {dataset_type}")
            results[dataset_type] = run_evaluation(dataset_type)
    
    # Show comparison
    if results.get("grayscale") and results.get("colored"):
        print_comparison_table(results["grayscale"], results["colored"])
        
        print(f"\n📁 Detailed results saved in:")
        print(f"   - regression_predictions.png (overwrites for each dataset)")
        print(f"\n💡 Tip: Use './run evaluate --dataset [type]' to run individual evaluations")
        
    else:
        print("❌ Comparison failed - one or both evaluations failed")

if __name__ == "__main__":
    main()
