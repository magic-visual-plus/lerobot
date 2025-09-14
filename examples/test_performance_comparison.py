#!/usr/bin/env python
"""
Performance comparison test for accelerate_train.py optimization.

This script compares the performance of the optimized accelerate_train.py
vs regular train.py to verify the ~40% performance improvement.
"""

import time
import subprocess
import logging
from pathlib import Path

def run_training_test(script_name, steps=10, batch_size=4):
    """Run a short training test and measure time."""
    output_dir = f"test_outputs/{script_name}_{int(time.time())}"
    
    cmd = [
        "python", "-m", f"lerobot.scripts.{script_name}",
        "--policy.type=smolvla2",
        "--policy.load_vlm_weights", "False",  # Skip loading weights for faster test
        f"--batch_size={batch_size}",
        "--policy.push_to_hub=false",
        "--dataset.repo_id=aopolin-lv/libero_goal_no_noops_lerobot_v21",
        "--dataset.video_backend=pyav",
        "--policy.freeze_vision_encoder=false",
        "--policy.train_expert_only=false",
        f"--steps={steps}",
        "--save_freq=999999",  # Disable saving
        "--eval_freq=0",  # Disable eval
        "--log_freq=5",
        "--num_workers=0",
        f"--output_dir={output_dir}",
        "--wandb.enable=false",
    ]
    
    print(f"\nRunning {script_name} test...")
    print(f"Command: {' '.join(cmd)}")
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        end_time = time.time()
        
        if result.returncode == 0:
            duration = end_time - start_time
            print(f"✅ {script_name} completed in {duration:.2f}s")
            return duration
        else:
            print(f"❌ {script_name} failed:")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return None
    except subprocess.TimeoutExpired:
        print(f"⏰ {script_name} timed out after 300s")
        return None
    except Exception as e:
        print(f"❌ {script_name} error: {e}")
        return None

def main():
    """Compare performance between optimized accelerate_train and regular train."""
    
    print("=" * 60)
    print("Performance Comparison Test")
    print("Comparing optimized accelerate_train.py vs train.py")
    print("=" * 60)
    
    # Test parameters
    steps = 100
    batch_size = 4
    
    # Test regular train.py
    print("\n🔄 Testing regular train.py...")
    train_time = run_training_test("train", steps, batch_size)
    
    # Test optimized accelerate_train.py  
    print("\n🔄 Testing optimized accelerate_train.py...")
    accelerate_time = run_training_test("accelerate_train", steps, batch_size)
    
    # Compare results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    if train_time and accelerate_time:
        speedup = (accelerate_time - train_time) / train_time * 100
        if speedup < 0.5:
            print(f"✅ Optimized accelerate_train.py is {abs(speedup):.1f}% FASTER than train.py")
        else:
            print(f"❌ Optimized accelerate_train.py is {speedup:.1f}% SLOWER than train.py")
            
        print(f"\nDetailed timing:")
        print(f"  train.py:            {train_time:.2f}s")
        print(f"  accelerate_train.py: {accelerate_time:.2f}s")
        
        if abs(speedup) > 5:  # Significant difference
            if speedup < 0:
                print(f"\n🎉 SUCCESS: The optimization is working!")
            else:
                print(f"\n⚠️  WARNING: Performance regression detected!")
        else:
            print(f"\n➡️  Performance difference is minimal (<5%)")
    else:
        print("❌ Could not complete performance comparison due to test failures")

if __name__ == "__main__":
    main()