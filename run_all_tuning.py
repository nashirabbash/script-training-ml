"""
=============================================================================
AUTOMATED HYPERPARAMETER TUNING - ALL FEATURE SELECTION METHODS
=============================================================================
Script otomatis untuk menjalankan semua metode feature selection secara berurutan:
1. ANOVA (f_classif) - Blue theme
2. Chi-Square (chi2) - Green theme  
3. Kruskal-Wallis - Orange theme
4. ReliefF - Purple theme

Setiap metode akan:
- Test 6 classifiers (KNN, SVM, MLP, RandomForest, NaiveBayes, DecisionTree)
- Test 3 test sizes (20%, 25%, 30%)
- Generate comprehensive reports dan visualizations

Author: ML Pipeline
Date: 2025
=============================================================================
"""

import sys
import io
# Fix Windows encoding untuk Unicode characters
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import subprocess
import os
from datetime import datetime
import time

# =============================================================================
# KONFIGURASI
# =============================================================================

# Daftar semua tuning scripts yang akan dijalankan
TUNING_SCRIPTS = [
    {
        'name': 'ANOVA (f_classif)',
        'script': 'tuning_anova.py',
        'color': '🔵',
        'description': 'Univariate ANOVA F-test untuk fitur kontinyu'
    },
    {
        'name': 'Chi-Square (chi2)',
        'script': 'tuning_chi2.py',
        'color': '🟢',
        'description': 'Chi-Square test untuk fitur non-negatif'
    },
    {
        'name': 'Kruskal-Wallis',
        'script': 'tuning_kruskal.py',
        'color': '🟠',
        'description': 'Non-parametric test untuk distribusi non-normal'
    },
    {
        'name': 'ReliefF',
        'script': 'tuning_relieff.py',
        'color': '🟣',
        'description': 'Multivariate instance-based feature selection'
    }
]

# Python executable (sesuaikan dengan environment Anda)
PYTHON_EXEC = sys.executable  # Gunakan python yang sedang aktif


# =============================================================================
# FUNGSI UTILITAS
# =============================================================================

def print_header(text, char='='):
    """Print header dengan formatting"""
    length = 80
    print(f"\n{char * length}")
    print(f"{text.center(length)}")
    print(f"{char * length}\n")


def print_section(text):
    """Print section dengan formatting"""
    print(f"\n{'─' * 80}")
    print(f"  {text}")
    print(f"{'─' * 80}")


def format_time(seconds):
    """Format waktu dalam seconds ke format readable"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = seconds / 60
        return f"{mins:.1f} min"
    else:
        hours = seconds / 3600
        mins = (seconds % 3600) / 60
        return f"{int(hours)}h {int(mins)}m"


def check_file_exists(filename):
    """Check apakah file script ada"""
    if not os.path.exists(filename):
        print(f"❌ ERROR: File '{filename}' tidak ditemukan!")
        return False
    return True


def run_tuning_script(script_info):
    """Menjalankan satu tuning script"""
    script_name = script_info['script']
    display_name = script_info['name']
    color = script_info['color']
    description = script_info['description']
    
    print_section(f"{color} Running: {display_name}")
    print(f"Script: {script_name}")
    print(f"Description: {description}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if file exists
    if not check_file_exists(script_name):
        return False, 0
    
    # Run the script
    start_time = time.time()
    
    try:
        # Run dengan subprocess untuk capture output
        result = subprocess.run(
            [PYTHON_EXEC, script_name],
            capture_output=True,
            text=True,
            check=True
        )
        
        elapsed = time.time() - start_time
        
        # Print output
        if result.stdout:
            print("\n" + result.stdout)
        
        print(f"\n✅ {color} {display_name} - COMPLETED")
        print(f"Duration: {format_time(elapsed)}")
        
        return True, elapsed
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        
        print(f"\n❌ {color} {display_name} - FAILED")
        print(f"Duration: {format_time(elapsed)}")
        print(f"Error code: {e.returncode}")
        
        if e.stdout:
            print("\nStdout:")
            print(e.stdout)
        
        if e.stderr:
            print("\nStderr:")
            print(e.stderr)
        
        return False, elapsed
    
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n❌ {color} {display_name} - ERROR")
        print(f"Duration: {format_time(elapsed)}")
        print(f"Exception: {str(e)}")
        return False, elapsed


def print_summary(results):
    """Print summary hasil eksekusi semua scripts"""
    print_header("EXECUTION SUMMARY", '█')
    
    total_time = sum([r['duration'] for r in results])
    successful = sum([1 for r in results if r['success']])
    failed = len(results) - successful
    
    print(f"Total Scripts: {len(results)}")
    print(f"Successful:    {successful} ✅")
    print(f"Failed:        {failed} ❌")
    print(f"Total Time:    {format_time(total_time)}")
    
    print("\n" + "─" * 80)
    print(f"{'Method':<25} {'Status':<10} {'Duration':<15}")
    print("─" * 80)
    
    for r in results:
        status = "✅ SUCCESS" if r['success'] else "❌ FAILED"
        duration = format_time(r['duration'])
        print(f"{r['name']:<25} {status:<10} {duration:<15}")
    
    print("─" * 80)
    
    # Print overall result
    if failed == 0:
        print("\n🎉 ALL TUNING SCRIPTS COMPLETED SUCCESSFULLY!")
    else:
        print(f"\n⚠️  {failed} SCRIPT(S) FAILED - Please check errors above")
    
    print(f"\nTotal execution time: {format_time(total_time)}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function untuk menjalankan semua tuning scripts"""
    
    # Print intro
    print_header("AUTOMATED HYPERPARAMETER TUNING", '█')
    print("This script will run ALL feature selection methods sequentially:")
    for i, script in enumerate(TUNING_SCRIPTS, 1):
        print(f"  {i}. {script['color']} {script['name']:<20} - {script['description']}")
    
    print(f"\nPython executable: {PYTHON_EXEC}")
    print(f"Working directory: {os.getcwd()}")
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Confirm before running (optional - comment out if you want auto-run)
    print("\n" + "─" * 80)
    response = input("Press ENTER to start, or Ctrl+C to cancel... ")
    
    # Track results
    results = []
    overall_start = time.time()
    
    # Run each tuning script
    for i, script_info in enumerate(TUNING_SCRIPTS, 1):
        print_header(f"STEP {i}/{len(TUNING_SCRIPTS)}: {script_info['name']}", '=')
        
        success, duration = run_tuning_script(script_info)
        
        results.append({
            'name': script_info['name'],
            'script': script_info['script'],
            'color': script_info['color'],
            'success': success,
            'duration': duration
        })
        
        # Pause singkat antar script
        if i < len(TUNING_SCRIPTS):
            print("\n" + "─" * 80)
            print(f"Moving to next script in 3 seconds...")
            time.sleep(3)
    
    # Print final summary
    overall_time = time.time() - overall_start
    print_summary(results)
    
    # Return status
    return all([r['success'] for r in results])


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n❌ Execution interrupted by user (Ctrl+C)")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ CRITICAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
