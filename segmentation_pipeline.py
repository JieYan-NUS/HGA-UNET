"""""""""""""""""""""
You need to prepare a folder containing the following files:1. Model.py
                                                            2. model_best.pth
                                                            3. segmentation_pipeline.py

The image file you want to analyze (e.g., my_image.tif)

Copy the code below into your main program (e.g., main_app.py). 
Change the MY_IMAGE_PATH and the MY_STRIDE value. The larger the MY_STRIDE value, the faster the speed; 
If the image is too dirty (noisy), change the MY_MIN_SIZE; try increasing this value (e.g., to 100 or 200) to find the best setting for noise removal.

Make sure you have the following dependencies installed:

pip install numpy pandas matplotlib pillow opencv-python scikit-image torch torchvision

(Note: Adjust the torch and torchvision installation commands if you are using a GPU/CUDA environment.)

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from segmentation_pipeline import setup_model, run_segmentation_and_analysis

MY_IMAGE_PATH = Path("/home/jovyan/TEST/10X_PBS_0.1pM.png") 
MY_TILE = 512          
MY_STRIDE = 512        
MY_MIN_SIZE = 10    
MODEL_WEIGHTS_PATH = "model_best.pth"
DEVICE = "cuda" # or 'cpu'

try:
    model, device = setup_model(MODEL_WEIGHTS_PATH, DEVICE)
except Exception as e:
    print(f"can not setup model: {e}")
    exit()

print(f"Analysis start: {MY_IMAGE_PATH}")
final_mask, tile_stats, overall_stats = run_segmentation_and_analysis(
    image_path=MY_IMAGE_PATH,
    model=model,
    device=device,
    tile=MY_TILE,            
    stride=MY_STRIDE,        
    min_size=MY_MIN_SIZE      
)
"""""""""""""""""""""""""""""""""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from PIL import Image
from scipy.ndimage import gaussian_filter1d
import torch
import torch.nn.functional as F
import cv2
from typing import Tuple

# --- Configuration ---
# Assuming Model.py is available or Model class definition is provided elsewhere
# For this script, we'll assume the MODEL class is defined or imported correctly.
# Since the definition of ConsensusAttnResUNetStudent is not in the prompt,
# I'll rely on the existing import statement.
try:
    from Model import ConsensusAttnResUNetStudent as MODEL
except ImportError:
    # Placeholder class if Model.py is not available for execution
    class MODEL(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.conv = torch.nn.Conv2d(1, 1, 3, padding=1)
        def forward(self, x):
            # Mock forward pass for structure completeness
            return self.conv(x) * 0.0 + 0.5 
    print("WARNING: MODEL class not found. Using a placeholder model.")


DEFAULT_TILE = 512
DEFAULT_STRIDE = 512
IMG_PATH = Path("./80x_2025-05-22_14-48-00.tif") 
DEFAULT_CROP_RATIO = 0.0
# CROP_PX = None # Can be used instead of CROP_RATIO
DEFAULT_MIN_SIZE = 50

# Pyplot configuration
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 12
})

# Determine device
device = "cuda" if torch.cuda.is_available() else "cpu"

def setup_model(model_path: str = "model_best.pth", device: str = "cuda"):
    device_ = torch.device(device if torch.cuda.is_available() else "cpu")
    model = MODEL(in_channels=1, n_filters=32, attn_window=(4, 4), attn_stride=(2, 2), attn_heads=2, dropout=0.0).to(device_)
    
    if Path(model_path).exists():
        state = torch.load(model_path, map_location=device_)
        model.load_state_dict(state, strict=False)
        model.eval()
        print(f"Model loaded from {model_path}.")
    else:
        print(f"Warning: Model checkpoint '{model_path}' not found. Using random weights.")
        model.eval()
        
    return model, device_

def run_segmentation_and_analysis(image_path: Path, model, device, 
                                  tile=DEFAULT_TILE,
                                  stride=DEFAULT_STRIDE,
                                  min_size=DEFAULT_MIN_SIZE,
                                  crop_ratio=DEFAULT_CROP_RATIO): 

    print("\n--- 1. Image Loading and Cropping (Visualization) ---")
    cropped, cropped_raw = visualize_crop_process(image_path, crop_ratio)
    
    if cropped is None:
        raise FileNotFoundError(f"Cannot load or process image at {image_path}")

    print("\n--- 2. Tiled Inference and Stitching ---")
    prob_map, tile_regions = stitch_probability_map_weighted(
        cropped, model, device, 
        tile=tile, 
        stride=stride
    )

    print("\n--- 3. Mask Building and Cleaning ---")
    _, _, mask_bin = build_final_mask_from_probmap(prob_map)
    final_mask = remove_small_components(mask_bin, min_size=min_size)
    print(f"Small components (<{min_size}px) removed.")

    print("\n--- 4. Image, Prediction, Mask Comparison (Visualization) ---")
    plot_image_pred_mask(cropped, prob_map, final_mask,
                         title_pred="FG Probability (1-Prob_BG)",
                         title_mask=f"Processed Mask (Min Size={min_size})")

    tile_stats = compute_tile_fg_ratio_from_mask(tile_regions, final_mask)
    overall_stats = compute_fg_ratio_stats(tile_stats)

    print("\n--- 5. Tile FG Ratio Analysis (Visualization) ---")
    plot_tile_ratio_heatmap_grid(tile_stats)
    plot_fg_ratio_boxplot(tile_stats)

    print("\n--- 6. Overall Foreground Ratio Statistics (per tile) ---")
    print(f"FG Mean (µ) = {overall_stats['mean']:.4f}")
    print(f"STD (σ)     = {overall_stats['std']:.4f}")
    print(f"SEM         = {overall_stats['sem']:.4f}")
    print(f"95% CI      = ±{overall_stats['ci95']:.4f}")

    return final_mask, tile_stats, overall_stats

def load_gray(path: Path) -> np.ndarray:
    """Loads an image as grayscale (L mode) and returns a float32 array."""
    im = Image.open(path).convert("L")
    arr = np.asarray(im, dtype=np.float32)
    return arr

def load_raw(path: Path) -> np.ndarray:
    """Loads an image in its original format and returns a numpy array."""
    im = Image.open(path)
    return np.array(im)

def compute_crop_bbox(H: int, W: int, crop_ratio: float = 0.05, crop_px: int = None):
    """Computes the coordinates for cropping an array."""
    if crop_px is not None:
        c = int(max(0, crop_px))
    else:
        crop_ratio = float(np.clip(crop_ratio, 0.0, 0.49))
        c = int(min(H, W) * crop_ratio)
    y0, y1 = c, max(c, H - c)
    x0, x1 = c, max(c, W - c)
    return y0, y1, x0, x1, c

def crop_array(arr: np.ndarray, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    """Crops a numpy array based on computed bounding box."""
    return arr[y0:y1, x0:x1]

def clean_and_sharpen(img01: np.ndarray,
                      denoise_h=5,
                      unsharp_strength=1.5,
                      unsharp_blur=3):
    """Applies non-local means denoising and unsharp masking (sharpening)."""
    img8 = np.clip(img01 * 255, 0, 255).astype(np.uint8)

    # Non-local means denoising
    clean = cv2.fastNlMeansDenoising(img8, None, h=denoise_h, templateWindowSize=7, searchWindowSize=21)

    # Unsharp mask
    blur = cv2.GaussianBlur(clean, (unsharp_blur, unsharp_blur), 0)
    # sharp = clean + unsharp_strength * (clean - blur)
    sharp = cv2.addWeighted(clean, 1 + unsharp_strength, blur, -unsharp_strength, 0)

    return sharp.astype(np.float32) / 255.0

def otsu_threshold(img01: np.ndarray):
    """Computes the Otsu threshold value for a grayscale image."""
    img8 = np.clip(img01 * 255, 0, 255).astype(np.uint8)
    thr_val, _ = cv2.threshold(img8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thr_val / 255.0

def tile_starts(L: int, tile: int, stride: int):
    """Calculates the starting indices for tiling/sliding window."""
    if L <= tile:
        return [0]
    starts = list(range(0, L - tile + 1, stride))
    last = L - tile
    if not starts or starts[-1] != last:
        starts.append(last)
    return starts

def hann2d(h, w):
    """Generates a 2D Hanning window (outer product of two 1D Hanning windows)."""
    wy = np.hanning(h)[:, None]
    wx = np.hanning(w)[None, :]
    w2 = wy * wx
    return (w2 + 1e-6).astype(np.float32)

@torch.no_grad()
def predict_tile_with_model_512(img_patch: np.ndarray, model: MODEL, device: str) -> np.ndarray:
    """Predicts the probability map for a single image patch (tile)."""
    h, w = img_patch.shape

    if h != DEFAULT_TILE or w != DEFAULT_TILE:
        # Pad if patch is smaller than TILE size
        pad_y, pad_x = DEFAULT_TILE - h, DEFAULT_TILE - w
        img_pad = np.pad(img_patch, ((0, pad_y), (0, pad_x)), mode="reflect")
    else:
        img_pad = img_patch

    img_pad = img_pad.astype(np.float32)
    if img_pad.max() > 1.0:
        img_pad = img_pad / 255.0

    # Prepare tensor: [1, 1, H, W]
    x = torch.from_numpy(img_pad).to(torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    # Inference
    logits = model(x)
    
    # Calculate foreground probability (prob_fg) and background probability (prob_bg)
    prob_fg = torch.sigmoid(logits)
    prob_bg = 1 - prob_fg

    # Extract the relevant part and convert to numpy [H, W]
    prob = prob_bg.squeeze(0).squeeze(0)[:h, :w].detach().cpu().numpy().astype(np.float32)
    return prob

def stitch_probability_map_weighted(img01: np.ndarray, model: MODEL, device: str,
                                    tile: int = DEFAULT_TILE, stride: int = DEFAULT_STRIDE):
    """Generates a probability map by tiling, prediction, and weighted averaging."""
    Hc, Wc = img01.shape

    prob_sum = np.zeros((Hc, Wc), dtype=np.float32)
    weight   = np.zeros((Hc, Wc), dtype=np.float32)

    ys = tile_starts(Hc, tile, stride)
    xs = tile_starts(Wc, tile, stride)

    tile_regions = []

    for row, y in enumerate(ys):
        for col, x in enumerate(xs):
            y2, x2 = min(y + tile, Hc), min(x + tile, Wc)
            patch = img01[y:y2, x:x2]

            prob_bg = predict_tile_with_model_512(patch, model, device)
            h, w = prob_bg.shape

            tile_regions.append({
                "row": row,
                "col": col,
                "y": y,
                "x": x,
                "h": h,
                "w": w,
            })

            win = hann2d(h, w)
            prob_sum[y:y2, x:x2] += prob_bg * win
            weight  [y:y2, x:x2] += win

    # Normalize by the accumulated weights
    weight[weight == 0] = 1.0
    prob_map = prob_sum / weight

    return prob_map, tile_regions

def build_final_mask_from_probmap(prob_map: np.ndarray):
    """Converts background probability map to foreground binary mask."""
    mask_base = 1.0 - prob_map  # Base mask (Foreground probability)
    mask_clean = clean_and_sharpen(mask_base)
    thr = otsu_threshold(mask_clean)
    mask_bin = (mask_clean >= thr).astype(np.float32)
    return mask_base, mask_clean, mask_bin

def remove_small_components(mask: np.ndarray, min_size: int = 50, connectivity: int = 8) -> np.ndarray:
    """Removes connected components (objects) smaller than min_size."""
    mask_bin = (mask > 0).astype(np.uint8)

    # Use cv2.connectedComponentsWithStats for size filtering
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_bin, connectivity, cv2.CV_32S
    )
    # stats: [x, y, w, h, area]
    cleaned = np.zeros_like(mask_bin)

    for label_id in range(1, num_labels): # Start from 1, as 0 is background
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area >= min_size:
            cleaned[labels == label_id] = 1

    return cleaned.astype(np.float32)

def compute_tile_fg_ratio_from_mask(tile_regions, mask_bin):
    """Calculates the foreground ratio for each tile region in the binary mask."""
    tile_stats = []
    for r in tile_regions:
        y, x, h, w = r["y"], r["x"], r["h"], r["w"]
        sub = mask_bin[y:y+h, x:x+w]
        ratio = float(sub.mean())

        tile_stats.append({
            "row": r["row"],
            "col": r["col"],
            "ratio_mask": ratio,
        })
    return tile_stats

def compute_fg_ratio_stats(tile_stats):
    """Calculates mean, std, sem, and 95% CI from tile foreground ratios."""
    fg_values = np.array([s["ratio_mask"] for s in tile_stats], dtype=np.float32)
    mean = float(fg_values.mean())
    # ddof=1 for sample standard deviation
    std  = float(fg_values.std(ddof=1))
    sem  = float(std / np.sqrt(len(fg_values)))
    # 1.96 for 95% CI (approx for large n)
    ci95 = float(1.96 * sem)
    return {
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci95": ci95
    }

# --- Plotting Functions ---

def visualize_crop_process(img_path, crop_ratio, crop_px=None):
    """Loads image, computes crop, and plots the original, crop frame, and cropped image."""
    try:
        img = load_gray(img_path)
        img_raw = load_raw(img_path)
        H, W = img.shape
    except Exception as e:
        print(f"Error during image loading or initial check: {e}")
        return None, None
    try:
        y0, y1, x0, x1, c = compute_crop_bbox(H, W, crop_ratio=crop_ratio, crop_px=crop_px)

        cropped = crop_array(img, y0, y1, x0, x1)
        cropped_raw = crop_array(img_raw, y0, y1, x0, x1)
    except Exception as e:
        print(f"Error during crop computation or execution: {e}")
        return None, None

    # Plotting: 
    fig, ax = plt.subplots(1, 3, figsize=(14, 6))

    ax[0].imshow(img_raw)
    ax[0].set_title("Input (Raw)")
    ax[0].axis("off")

    ax[1].imshow(img, cmap="gray")
    rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=2, edgecolor='r', facecolor='none')
    ax[1].add_patch(rect)
    ax[1].set_title(f"Crop Frame: y[{y0}:{y1}], x[{x0}:{x1}] ({c}px)")
    ax[1].axis("off")

    ax[2].imshow(cropped, cmap="gray")
    ax[2].set_title(f"Cropped ({cropped.shape[0]}×{cropped.shape[1]})")
    ax[2].axis("off")

    plt.tight_layout()
    plt.show()

    return cropped, cropped_raw

def plot_image_pred_mask(cropped, prob_map, mask_bin,
                         title_img="Image",
                         title_pred="Predict (1-Prob_BG)",
                         title_mask="Processed Mask"):
    """Plots the original cropped image, probability map, and final binary mask."""
    # Plotting: 
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(cropped, cmap="gray")
    plt.title(title_img, fontsize=12)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    # prob_map is probability of background (prob_bg), so 1 - prob_map is foreground prob
    plt.imshow(1.0 - prob_map, cmap="viridis") # Display foreground probability map
    plt.title(title_pred, fontsize=12)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(mask_bin, cmap="gray")
    plt.title(title_mask, fontsize=12)
    plt.axis("off")

    plt.tight_layout()
    plt.show()

def plot_tile_ratio_heatmap_grid(tile_stats, title="Tile FG ratio (mask-based)"):
    """Plots a heatmap grid of the foreground ratio for each tile."""
    max_row = max(s["row"] for s in tile_stats)
    max_col = max(s["col"] for s in tile_stats)
    n_rows = max_row + 1
    n_cols = max_col + 1

    grid = np.zeros((n_rows, n_cols), dtype=np.float32)
    for s in tile_stats:
        grid[s["row"], s["col"]] = s["ratio_mask"]

    # Plotting: 
    plt.figure(figsize=(n_cols * 1.5, n_rows * 1.5)) # Adjust figure size based on grid
    im = plt.imshow(grid, cmap="viridis", vmin=0.0, vmax=max(0.7, grid.max() * 1.05))
    plt.colorbar(im, label="FG ratio")

    # Label with ratio values
    for r in range(n_rows):
        for c in range(n_cols):
            val = grid[r, c]
            txt = f"{val:.2f}"
            text_color = "black"
            plt.text(c, r, txt, ha="center", va="center", color=text_color, fontsize=10, fontweight='bold')

    plt.title(title, fontsize=14)
    plt.xticks(range(n_cols), fontsize=12)
    plt.yticks(range(n_rows), fontsize=12)
    plt.xlabel("Tile Column (X)", fontsize=12)
    plt.ylabel("Tile Row (Y)", fontsize=12)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

def plot_fg_ratio_boxplot(tile_stats, title="FG Ratio Distribution (per tile)"):
    """Plots a box plot with jittered points of the tile foreground ratios."""
    fg_values = np.array([s["ratio_mask"] for s in tile_stats], dtype=np.float32)
    
    # Plotting: 
    plt.figure(figsize=(4, 5))
    plt.boxplot(
        fg_values,
        vert=True,
        patch_artist=True,
        labels=["FG ratio"],
        boxprops=dict(facecolor="#4C72B0", alpha=0.5),
        medianprops=dict(color="red", linewidth=2),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
        flierprops=dict(marker="o", markersize=4, markerfacecolor="gray", alpha=0.4),
    )

    # Add jittered scatter points for individual tile values
    x_jitter = 1 + 0.15 * (np.random.rand(len(fg_values)) - 0.5)

    plt.scatter(
        x_jitter,
        fg_values,
        color="black",
        s=20,
        alpha=0.6,
        edgecolors="none"
    )

    plt.ylabel("Foreground ratio (0–1)", fontsize=12)
    plt.title(title, fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0, max(0.8, fg_values.max() * 1.1)) # Adjust y-limit dynamically
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

# --- Main Execution ---

def main_script(img_path: Path, crop_ratio: float, tile: int, stride: int, min_size: int, device: str):
    """Main function to run the image processing pipeline."""
    
    # 1. Load Model
    print("Initializing model...")
    model = MODEL(in_channels=1, n_filters=32, attn_window=(4, 4), attn_stride=(2, 2), attn_heads=2, dropout=0.0).to(device)
    
    # Check if a model checkpoint exists
    model_path = Path("model_best.pth")
    if model_path.exists():
        state = torch.load(model_path, map_location=device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Model loaded from {model_path}. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    else:
        print(f"WARNING: Model checkpoint '{model_path}' not found. Using randomly initialized weights.")
        
    model.eval()
    print(f"Model ready on device: {device}!")

    # 2. Load and Crop Image (with visualization)
    print("\n--- 2. Image Loading and Cropping ---")
    cropped, _ = visualize_crop_process(img_path, crop_ratio)
    
    if cropped is None:
        print("Exiting due to image loading or cropping error.")
        return

    # 3. Tiling setup
    Hc, Wc = cropped.shape
    ys = tile_starts(Hc, tile, stride)
    xs = tile_starts(Wc, tile, stride)
    print(f"Image size: {Hc}x{Wc}")
    print(f"Tile size: {tile}x{tile}, Stride: {stride}")
    print(f"Total tiles: {len(ys) * len(xs)} (tiles_y={len(ys)}, tiles_x={len(xs)})")

    # 4. Tiled Inference and Stitching
    print("\n--- 4. Tiled Inference and Stitching ---")
    prob_map, tile_regions = stitch_probability_map_weighted(cropped, model, device, tile=tile, stride=stride)
    print("Probability map generated.")

    # 5. Mask Building and Cleaning
    print("\n--- 5. Mask Building and Cleaning ---")
    # mask_base (FG Prob), mask_clean (Sharpened FG Prob), mask_bin (Otsu Thresholded)
    mask_base, mask_clean, mask_bin = build_final_mask_from_probmap(prob_map)
    
    # Remove small components
    mask_bin_cleaned = remove_small_components(mask_bin, min_size=min_size)
    print(f"Small components (<{min_size}px) removed.")

    # 6. Visualize Results
    print("\n--- 6. Visualization ---")
    plot_image_pred_mask(cropped, prob_map, mask_bin_cleaned,
                         title_pred="FG Probability (1-Prob_BG)")

    # 7. Compute and Visualize Tile Statistics
    print("\n--- 7. Tile Foreground Ratio Analysis ---")
    tile_stats = compute_tile_fg_ratio_from_mask(tile_regions, mask_bin_cleaned)
    print(f"Computed FG ratio for {len(tile_stats)} tiles.")
    
    plot_tile_ratio_heatmap_grid(tile_stats)
    plot_fg_ratio_boxplot(tile_stats)

    # 8. Print Summary Statistics
    stats = compute_fg_ratio_stats(tile_stats)
    print("\n--- 8. Overall Foreground Ratio Statistics (per tile) ---")
    print(f"FG Mean (µ) = {stats['mean']:.4f}")
    print(f"STD (σ)     = {stats['std']:.4f}")
    print(f"SEM         = {stats['sem']:.4f}")
    print(f"95% CI      = ±{stats['ci95']:.4f}")


if __name__ == "__main__":
    # Check if the image path is valid before running the script
    if not IMG_PATH.exists():
        print(f"Error: Image file not found at {IMG_PATH}. Please provide a valid path or mock the image loading.")
        print("For demonstration purposes, please ensure the file exists or adjust IMG_PATH.")
        # You would typically mock the image loading for an environment without the file.
        # Since I cannot create or assume the existence of a file, the script will likely fail here
        # unless run in an environment where the path is valid.
    else:
        main_script(
            img_path=IMG_PATH,
            crop_ratio=DEFAULT_CROP_RATIO,
            tile=DEFAULT_TILE,
            stride=DEFAULT_STRIDE,
            min_size=DEFAULT_MIN_SIZE,
            device=device
        )