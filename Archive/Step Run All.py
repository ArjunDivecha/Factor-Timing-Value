#!/usr/bin/env python
"""
T2 Strategy Complete Pipeline Runner
====================================

This script executes all steps of the T2 strategy pipeline in sequence:

0. Create P2P Scores
1. Create T2 Master data file
2. Create normalized tidy data
3. Create benchmark returns
4. Generate Top 20 portfolios
5. Create monthly Top 20 returns
6. Calculate 60-month optimal portfolios
7. Perform T2 factor timing
8. Visualize factor weights
9. Write country weights
10. Calculate final portfolio returns

Each step is executed in order, with error handling to ensure the process
can be monitored and debugged if necessary.

Usage:
    python T2_Run_All.py

Output:
    Console logs showing progress and results of each step
    All output files from individual scripts
"""

import os
import sys
import time
import subprocess
import datetime

def run_script(script_path, description):
    """
    Run a Python script and handle its output and errors.
    
    Args:
        script_path: Path to the Python script to run
        description: Description of the step for logging
    
    Returns:
        True if successful, False otherwise
    """
    print("\n" + "="*80)
    print(f"RUNNING: {description}")
    print(f"SCRIPT: {script_path}")
    print(f"TIME: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    try:
        # Run the script and capture output
        process = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Stream output in real-time
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        # Get the return code
        return_code = process.poll()
        
        # Check if there were any errors
        if return_code != 0:
            stderr = process.stderr.read()
            error_message = f"\nFATAL ERROR: {description} failed with exit code {return_code}"
            if stderr:
                error_message += f"\nError details:\n{stderr}"
            sys.exit(error_message)
        
        print(f"\nSUCCESS: {description} completed successfully")
        return True
    
    except Exception as e:
        error_message = f"\nFATAL ERROR: Failed to run {script_path}\nError: {str(e)}"
        sys.exit(error_message)

def main():
    """
    Main function to run all scripts in sequence.
    """
    # Base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define all scripts to run in order
    scripts = [
        {
            "path": os.path.join(base_dir, "Step Zero Create P2P Scores.py"),
            "description": "Step 0: Create P2P Scores"
        },
        {
            "path": os.path.join(base_dir, "Step One Create T2Master.py"),
            "description": "Step 1: Create T2 Master data file"
        },
        {
            "path": os.path.join(base_dir, "Step Two Create Normalized Tidy.py"),
            "description": "Step 2: Create normalized tidy data"
        },
        {
            "path": os.path.join(base_dir, "Step Two Point Five Create Benchmark Rets.py"),
            "description": "Step 3: Create benchmark returns"
        },
        {
            "path": os.path.join(base_dir, "Step Three Top20 Portfolios.py"),
            "description": "Step 4: Generate Top 20 portfolios"
        },
        {
            "path": os.path.join(base_dir, "Step Four Create Monthly Top20 Returns.py"),
            "description": "Step 5: Create monthly Top 20 returns"
        },
        {
            "path": os.path.join(base_dir, "Step Five 60 Month Optimal Portfolios.py"),
            "description": "Step 6: Calculate 60-month optimal portfolios"
        },
        {
            "path": os.path.join(base_dir, "Step Six T2 Factor Timing Top3.py"),
            "description": "Step 7: Perform T2 factor timing"
        },
        {
            "path": os.path.join(base_dir, "Step Seven Visualize Factor Weights.py"),
            "description": "Step 8: Visualize factor weights"
        },
        {
            "path": os.path.join(base_dir, "Step Eight Write Country Weights.py"),
            "description": "Step 9: Write country weights"
        },
        {
            "path": os.path.join(base_dir, "Step Nine Calculate Portfolio Returns.py"),
            "description": "Step 10: Calculate final portfolio returns"
        }
    ]
    
    # Start time
    start_time = time.time()
    
    print("\n" + "*"*80)
    print("STARTING T2 STRATEGY COMPLETE PIPELINE")
    print(f"START TIME: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("*"*80 + "\n")
    
    # Run each script in sequence
    for script in enumerate(scripts):
        run_script(script[1]["path"], script[1]["description"])
    
    # End time and summary
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("\n" + "*"*80)
    print("T2 STRATEGY PIPELINE COMPLETE")
    print(f"END TIME: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"TOTAL TIME: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"STEPS COMPLETED: {len(scripts)} of {len(scripts)}")
    print("*"*80 + "\n")

if __name__ == "__main__":
    main()
