"""
Generate AUC plots from existing classification reports
Uses confusion matrix to approximate ROC curve and AUC score
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def parse_classification_report(filepath):
    """
    Parse classification report text file to extract:
    - Model name
    - Test size
    - Confusion matrix (inferred from classification report table)
    - Metrics per model
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract test size
    test_size_match = re.search(r'Test Size: (\d+)%', content)
    test_size = int(test_size_match.group(1)) if test_size_match else None
    
    # Extract feature selection method
    method_match = re.search(r'Feature Selection Method: ([^\n]+)', content)
    method = method_match.group(1).strip() if method_match else "Unknown"
    
    # Split by model sections (flexible separator length)
    model_sections = re.split(r'={20,}\nMODEL: ', content)
    
    results = []
    for section in model_sections[1:]:  # Skip header
        lines = section.split('\n')
        model_name = lines[0].strip()
        
        # Find Classification Report table
        # Look for lines starting with '0' and '1' (classes)
        recall_0 = None
        recall_1 = None
        support_0 = None
        support_1 = None
        
        for line in lines:
            parts = line.split()
            if len(parts) >= 5:
                if parts[0] == '0':
                    recall_0 = float(parts[2])
                    support_0 = int(parts[4])
                elif parts[0] == '1':
                    recall_1 = float(parts[2])
                    support_1 = int(parts[4])
        
        if recall_0 is not None and recall_1 is not None:
            # Calculate metrics
            tnr = recall_0  # Specificity
            tpr = recall_1  # Sensitivity / Recall
            fpr = 1 - tnr
            
            # Reconstruct confusion matrix (approximate)
            tn = int(tnr * support_0)
            fp = support_0 - tn
            tp = int(tpr * support_1)
            fn = support_1 - tp
            
            # Estimate AUC (simple approximation)
            auc_estimate = (tpr + tnr) / 2  # Balanced accuracy approach
            
            results.append({
                'model': model_name,
                'test_size': test_size,
                'method': method,
                'tn': tn,
                'fp': fp,
                'fn': fn,
                'tp': tp,
                'tpr': tpr,
                'fpr': fpr,
                'tnr': tnr,
                'auc': auc_estimate
            })
    
    return results


def plot_roc_curves(results, output_dir, test_size):
    """
    Plot ROC curves for all models (approximate from confusion matrix)
    """
    plt.figure(figsize=(10, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    for i, result in enumerate(results):
        model = result['model']
        fpr = result['fpr']
        tpr = result['tpr']
        auc = result['auc']
        
        # Create simple ROC curve: (0,0) -> (FPR, TPR) -> (1,1)
        fpr_points = [0, fpr, 1]
        tpr_points = [0, tpr, 1]
        
        plt.plot(fpr_points, tpr_points, marker='o', 
                label=f'{model} (AUC ≈ {auc:.3f})', 
                color=colors[i], linewidth=2, markersize=6)
    
    # Diagonal reference line (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.5)')
    
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.title(f'ROC Curves - Test Size {test_size}%\n(Approximate from Confusion Matrix)', 
              fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'roc_curve_test{test_size}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ ROC curve saved: {output_path}")


def plot_auc_comparison(results, output_dir, test_size):
    """
    Plot bar chart comparing AUC scores across models
    """
    models = [r['model'] for r in results]
    aucs = [r['auc'] for r in results]
    
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.array(aucs))
    bars = plt.barh(models, aucs, color=colors, edgecolor='black', linewidth=1)
    
    # Add value labels
    for i, (bar, auc) in enumerate(zip(bars, aucs)):
        plt.text(auc + 0.01, i, f'{auc:.3f}', 
                va='center', fontsize=10, fontweight='bold')
    
    plt.xlabel('AUC Score (Estimated)', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    plt.title(f'AUC Comparison - Test Size {test_size}%', 
              fontsize=14, fontweight='bold')
    plt.xlim([0, 1.1])
    plt.axvline(x=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Random')
    plt.grid(True, alpha=0.3, axis='x')
    plt.legend()
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'auc_comparison_test{test_size}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ AUC comparison saved: {output_path}")


def save_auc_summary(results, output_dir, test_size):
    """
    Save AUC scores to text file
    """
    output_path = os.path.join(output_dir, f'auc_summary_test{test_size}.txt')
    
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"AUC SUMMARY - Test Size {test_size}%\n")
        f.write("=" * 80 + "\n")
        f.write(f"Method: {results[0]['method']}\n")
        f.write(f"Note: AUC estimated from confusion matrix using (TPR + TNR) / 2\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"{'Model':<20} {'AUC':<10} {'TPR':<10} {'FPR':<10} {'TNR':<10}\n")
        f.write("-" * 80 + "\n")
        
        # Sort by AUC descending
        sorted_results = sorted(results, key=lambda x: x['auc'], reverse=True)
        
        for r in sorted_results:
            f.write(f"{r['model']:<20} {r['auc']:<10.4f} {r['tpr']:<10.4f} "
                   f"{r['fpr']:<10.4f} {r['tnr']:<10.4f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("Confusion Matrix Components:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Model':<20} {'TN':<8} {'FP':<8} {'FN':<8} {'TP':<8}\n")
        f.write("-" * 80 + "\n")
        
        for r in sorted_results:
            f.write(f"{r['model']:<20} {r['tn']:<8} {r['fp']:<8} "
                   f"{r['fn']:<8} {r['tp']:<8}\n")
    
    print(f"✓ AUC summary saved: {output_path}")


def process_result_directory(result_dir):
    """
    Process all classification reports in a result directory
    """
    result_path = Path(result_dir)
    
    if not result_path.exists():
        print(f"⚠ Directory not found: {result_dir}")
        return
    
    print(f"\n{'=' * 80}")
    print(f"Processing: {result_path.name}")
    print(f"{'=' * 80}")
    
    # Find all classification report files
    report_files = list(result_path.glob('classification_report_test*.txt'))
    
    if not report_files:
        print(f"⚠ No classification reports found in {result_dir}")
        return
    
    for report_file in sorted(report_files):
        print(f"\n📄 Parsing: {report_file.name}")
        results = parse_classification_report(report_file)
        
        if results:
            test_size = results[0]['test_size']
            
            # Generate plots
            plot_roc_curves(results, result_dir, test_size)
            plot_auc_comparison(results, result_dir, test_size)
            save_auc_summary(results, result_dir, test_size)
            
            print(f"\n📊 Summary for test_size={test_size}%:")
            for r in sorted(results, key=lambda x: x['auc'], reverse=True):
                print(f"  {r['model']:<20} AUC ≈ {r['auc']:.4f}")
        else:
            print(f"  ⚠ Could not parse results from {report_file.name}")


def main():
    """
    Main function to process all result directories
    """
    print("=" * 80)
    print("AUC PLOT GENERATOR FROM CLASSIFICATION REPORTS")
    print("=" * 80)
    print("Note: AUC is estimated from confusion matrix")
    print("Formula: AUC ≈ (TPR + TNR) / 2 = (Sensitivity + Specificity) / 2")
    print("This is less accurate than true ROC-AUC but sufficient for comparison")
    print("=" * 80)
    
    # Define specific result directories to process
    result_dirs = [
        'results_kruskal/run_20251208_133138'
    ]
    
    for result_dir in result_dirs:
        if not os.path.exists(result_dir):
            print(f"\n⚠ Skipping {result_dir} (not found)")
            continue
        
        process_result_directory(result_dir)
    
    print("\n" + "=" * 80)
    print("✅ ALL AUC PLOTS GENERATED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == '__main__':
    main()
