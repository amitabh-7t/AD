#!/usr/bin/env python3
"""
Auto-execute remaining notebooks after 06 completes.
Monitors notebook 06 completion and runs 07, 08, 06b, 07b, 08b, 09-13 automatically.
"""
import subprocess
import sys
import time
import os
from datetime import datetime
from pathlib import Path

# Notebooks to run in order after 06 completes
NOTEBOOKS = [
    "07_train_efficientnetb0.ipynb",
    "08_train_densenet121.ipynb",
    "06b_train_resnet50_attention.ipynb",
    "07b_train_efficientnetb0_attention.ipynb",
    "08b_train_densenet121_attention.ipynb",
    "09_evaluation.ipynb",
    "10_gradcam.ipynb",
    "11_comparison.ipynb",
    "12_robustness.ipynb",
    "13_final_report.ipynb"
]

CHECKPOINT_FILE = "outputs/models/resnet50_best.h5"

def log(msg):
    """Print with timestamp."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def wait_for_06_completion():
    """Wait for notebook 06 to complete by monitoring checkpoint file."""
    log("🔍 Monitoring notebook 06 (ResNet50) completion...")
    log(f"   Waiting for: {CHECKPOINT_FILE}")
    
    initial_mtime = None
    stable_count = 0
    
    while True:
        if os.path.exists(CHECKPOINT_FILE):
            current_mtime = os.path.getmtime(CHECKPOINT_FILE)
            
            if initial_mtime is None:
                initial_mtime = current_mtime
                log(f"✓ Checkpoint file found, monitoring for completion...")
            
            # Check if file hasn't been modified for 60 seconds (training complete)
            if current_mtime == initial_mtime:
                stable_count += 1
                if stable_count >= 12:  # 12 * 5sec = 60 seconds stable
                    log("✅ Notebook 06 appears complete (checkpoint stable for 60s)")
                    return True
            else:
                initial_mtime = current_mtime
                stable_count = 0
                log(f"   Still training... (checkpoint updated)")
        else:
            log(f"   Waiting for checkpoint to appear...")
        
        time.sleep(5)  # Check every 5 seconds

def run_notebook(notebook_path):
    """Execute a Jupyter notebook using nbconvert."""
    log(f"\n{'='*80}")
    log(f"▶️  Starting: {notebook_path}")
    log(f"{'='*80}\n")
    
    cmd = [
        "jupyter", "nbconvert",
        "--to", "notebook",
        "--execute",
        "--inplace",
        "--ExecutePreprocessor.timeout=-1",
        "--ExecutePreprocessor.kernel_name=python3",
        notebook_path
    ]
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        elapsed = time.time() - start_time
        
        log(f"\n✅ Completed: {notebook_path}")
        log(f"   Runtime: {elapsed/60:.1f} minutes")
        return True
        
    except subprocess.CalledProcessError as e:
        log(f"\n❌ Failed: {notebook_path}")
        log(f"   Error: {e.stderr[:500]}")
        return False

def main():
    log("="*80)
    log("🤖 AUTOMATED NOTEBOOK EXECUTION")
    log("="*80)
    log(f"\nThis script will:")
    log("  1. Wait for notebook 06 (ResNet50) to complete")
    log("  2. Automatically run notebooks 07, 08, 06b, 07b, 08b, 09-13")
    log(f"\nTotal notebooks to run: {len(NOTEBOOKS)}")
    log("\n" + "="*80 + "\n")
    
    # Wait for notebook 06
    wait_for_06_completion()
    
    log("\n🚀 Starting automatic execution of remaining notebooks...\n")
    time.sleep(10)  # Give 10 seconds buffer
    
    results = {}
    total_start = time.time()
    
    for notebook in NOTEBOOKS:
        nb_path = f"notebooks/{notebook}"
        
        # Check if notebook exists
        if not os.path.exists(nb_path):
            log(f"⚠️  Skipping {notebook} (file not found)")
            results[notebook] = "SKIPPED"
            continue
        
        success = run_notebook(nb_path)
        results[notebook] = "SUCCESS" if success else "FAILED"
        
        if not success:
            log(f"\n❌ Stopping execution due to failure in {notebook}")
            break
        
        # Brief pause between notebooks
        time.sleep(5)
    
    total_elapsed = time.time() - total_start
    
    # Summary
    log("\n" + "="*80)
    log("📊 EXECUTION SUMMARY")
    log("="*80)
    log(f"\nTotal runtime: {total_elapsed/3600:.2f} hours")
    log(f"\nResults:")
    
    for nb, status in results.items():
        icon = "✅" if status == "SUCCESS" else "❌" if status == "FAILED" else "⏭️"
        log(f"  {icon} {status:8} - {nb}")
    
    successful = sum(1 for s in results.values() if s == "SUCCESS")
    log(f"\nCompleted: {successful}/{len(NOTEBOOKS)} notebooks")
    
    if successful == len(NOTEBOOKS):
        log("\n🎉 All notebooks executed successfully!")
        log("\n📊 Check outputs/FINAL_REPORT.md for comprehensive results")
    
    return 0 if successful == len(NOTEBOOKS) else 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log("\n\n⚠️  Execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        log(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
