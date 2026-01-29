#!/usr/bin/env python3
"""
Sequential notebook executor for notebooks 07-13.
Waits for user to finish 06, then runs remaining notebooks.
"""
import subprocess
import sys
import time
from datetime import datetime

NOTEBOOKS = [
    "07_train_efficientnetb0.ipynb",
    "08_train_densenet121.ipynb", 
    "09_evaluation.ipynb",
    "10_gradcam.ipynb",
    "11_comparison.ipynb",
    "12_robustness.ipynb",
    "13_final_report.ipynb"
]

def run_notebook(notebook_path):
    """Execute a Jupyter notebook using nbconvert."""
    print(f"\n{'='*80}")
    print(f"Starting: {notebook_path}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    cmd = [
        "jupyter", "nbconvert",
        "--to", "notebook",
        "--execute",
        "--inplace",
        "--ExecutePreprocessor.timeout=-1",
        notebook_path
    ]
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start_time
    
    if result.returncode == 0:
        print(f"\n✅ Completed: {notebook_path}")
        print(f"   Runtime: {elapsed/60:.1f} minutes")
        return True
    else:
        print(f"\n❌ Failed: {notebook_path}")
        print(f"   Error: {result.stderr}")
        return False

def main():
    print("="*80)
    print("SEQUENTIAL NOTEBOOK EXECUTION: 07-13")
    print("="*80)
    print(f"\nStarting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nNotebooks to run:")
    for i, nb in enumerate(NOTEBOOKS, 7):
        print(f"  {i}. {nb}")
    
    input("\n⏸️  Press ENTER when notebook 06 is complete and you're ready to continue...")
    
    print("\n🚀 Starting sequential execution...\n")
    
    results = {}
    total_start = time.time()
    
    for nb in NOTEBOOKS:
        nb_path = f"notebooks/{nb}"
        success = run_notebook(nb_path)
        results[nb] = success
        
        if not success:
            print(f"\n⚠️  Execution stopped due to failure in {nb}")
            break
    
    total_elapsed = time.time() - total_start
    
    # Summary
    print("\n" + "="*80)
    print("EXECUTION SUMMARY")
    print("="*80)
    print(f"\nTotal runtime: {total_elapsed/3600:.2f} hours")
    print(f"\nResults:")
    for nb, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {status} - {nb}")
    
    successful = sum(results.values())
    print(f"\nCompleted: {successful}/{len(results)} notebooks")
    
    if successful == len(NOTEBOOKS):
        print("\n🎉 All notebooks executed successfully!")
        print("\n📊 Check outputs/FINAL_REPORT.md for comprehensive results")
    
    return 0 if successful == len(NOTEBOOKS) else 1

if __name__ == "__main__":
    sys.exit(main())
