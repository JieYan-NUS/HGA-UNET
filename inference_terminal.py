import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Ensure the model directory is in path
CURRENT_DIR = Path(__file__).parent.absolute()
sys.path.append(str(CURRENT_DIR))

try:
    from segmentation_pipeline import (
        setup_model, 
        run_segmentation_and_analysis
    )
except ImportError as e:
    print(f"Error: Could not import pipeline components. {e}")
    sys.exit(1)

# --- Configuration ---
IMAGE_PATH = CURRENT_DIR / "Training" / "images" / "1280x_2025-05-16_00-59-00_grid_r2_c0.png"
WEIGHTS_PATH = CURRENT_DIR / "model_best.pth"
STRIDE = 512
TILE = 512
MIN_SIZE = 50

def run_terminal_demo():
    print("\n" + "="*50)
    print("      HGA-UNET TERMINAL INFERENCE RUNNER      ")
    print("="*50)
    
    # 1. Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Using device: {device}")
    
    try:
        model, device_obj = setup_model(str(WEIGHTS_PATH), device)
    except Exception as e:
        print(f"[-] Failed to load model: {e}")
        return

    # 2. Check Image
    if not IMAGE_PATH.exists():
        print(f"[-] Error: Sample image not found at {IMAGE_PATH}")
        return
    print(f"[*] Target Image: {IMAGE_PATH.name}")

    # 3. Interactive Parameters
    try:
        user_min_size = input(f"\nEnter the minimum component size to keep (default {MIN_SIZE}px): ").strip()
        min_size = int(user_min_size) if user_min_size else MIN_SIZE
    except ValueError:
        print(f"[!] Invalid input. Using default value: {MIN_SIZE}")
        min_size = MIN_SIZE

    # 4. Run Pipeline (Interactive Mode)
    print(f"\n[*] Starting Segmentation and Analysis (Min Size={min_size}px)...")
    try:
        # Note: In this version, we use the standard pipeline call which
        # will trigger plt.show() windows for you.
        final_mask, tile_stats, overall_stats = run_segmentation_and_analysis(
            image_path=IMAGE_PATH,
            model=model,
            device=device_obj,
            tile=TILE,
            stride=STRIDE,
            min_size=min_size
        )
        
        print("\n" + "-"*50)
        print("✅ INFERENCE SUCCESSFUL")
        print("-"*50)
        print(f"📈 Foreground Mean (µ): {overall_stats['mean']:.6f}")
        print(f"📊 Standard Deviation (σ): {overall_stats['std']:.6f}")
        print(f"📉 SEM: {overall_stats['sem']:.6f}")
        print("="*50)
        print("\n[!] Close the plot windows to finish the script.")
        
    except Exception as e:
        print(f"\n[!] Error during inference: {e}")

if __name__ == "__main__":
    run_terminal_demo()
