import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from pixel_transformations import binarize_avg
from morphological import opening, closing

IMAGE_PATHS = glob.glob("/Users/lila/Downloads/Flattened/*1.bmp")
RESULTS_DIR = "results"
OVERLAY_DIR = "results_overlay"

def get_center(mask: np.ndarray):
    m = mask[:, :, 0]
    proj_v = np.sum(m, axis=0)
    proj_h = np.sum(m, axis=1)

    if proj_v.sum() == 0 or proj_h.sum() == 0:
        raise ValueError("Mask is empty")

    idx_x = np.arange(len(proj_v))
    idx_y = np.arange(len(proj_h))
    cx = int(np.sum(idx_x * proj_v) / proj_v.sum())
    cy = int(np.sum(idx_y * proj_h) / proj_h.sum())
    
    return cx, cy

def get_radius(img_gray: np.ndarray, cx: int, cy: int, r_min: int, r_max: int, num_angles: int = 360, angle_mask=None):
    radii = np.arange(r_min, r_max)
    profile = np.zeros(len(radii))
    angles = np.linspace(0, 2 * np.pi, num_angles)
    
    if angle_mask is not None:
        angles = angles[angle_mask]
    
    for i, r in enumerate(radii):
        xs = np.clip((cx + r * np.cos(angles)).astype(int), 0, img_gray.shape[1] - 1)
        ys = np.clip((cy + r * np.sin(angles)).astype(int), 0, img_gray.shape[0] - 1)
        profile[i] = img_gray[ys, xs].mean()

    grad = np.abs(np.gradient(profile))
    return radii[np.argmax(grad)]

def draw_circle(image: np.ndarray, cx: int, cy: int, radius: int, color=[255, 0, 0]) -> np.ndarray:
    out = image.copy()
    out[cy, cx] = color
    for angle in range(360):
        x = int(cx + radius * np.cos(np.radians(angle)))
        y = int(cy + radius * np.sin(np.radians(angle)))
        if 0 <= x < out.shape[1] and 0 <= y < out.shape[0]:
            out[y, x] = color
    return out

def apply_iris_mask(img: np.ndarray, iris_2d: np.ndarray, pupil_2d: np.ndarray) -> np.ndarray:
    ring = np.clip(np.where(iris_2d == 255, 1, 0) - np.where(pupil_2d == 255, 1, 0), 0, 1)
    return (img * np.stack([ring] * 3, axis=-1)).astype(np.uint8)

def unwrap_iris(img: np.ndarray, cx: int, cy: int, r_inner: int, r_outer: int, num_angles: int = 360, num_radii: int = 64) -> np.ndarray:
    h, w = img.shape[:2]
    result = np.zeros((num_radii, num_angles, 3), dtype=np.uint8)
    angles = np.linspace(0, 2 * np.pi, num_angles)
    radii = np.linspace(r_inner, r_outer, num_radii)
    
    for i, r in enumerate(radii):
        xs = np.clip((cx + r * np.cos(angles)).astype(int), 0, w - 1)
        ys = np.clip((cy + r * np.sin(angles)).astype(int), 0, h - 1)
        result[i] = img[ys, xs]
    return result

def save_figure(arrays, titles, path, figsize_per=5):
    n = len(arrays)
    fig, axes = plt.subplots(1, n, figsize=(n * figsize_per, figsize_per), squeeze=False)
    for i, (arr, title) in enumerate(zip(arrays, titles)):
        axes[0, i].imshow(arr, cmap="gray" if arr.ndim == 2 else None)
        axes[0, i].set_title(title)
        axes[0, i].axis("off")
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)

def process_image(image_path: str) -> None:
    stem = os.path.splitext(os.path.basename(image_path))[0]
    img_dir = os.path.join(RESULTS_DIR, stem)
    os.makedirs(img_dir, exist_ok=True)

    print(f"[{stem}] Processing...")
    img = np.array(Image.open(image_path).convert("RGB"))
    gray = np.mean(img, axis=2)

    pupil_bin = binarize_avg(img, x=6.2, invert=True)
    iris_bin = binarize_avg(img, x=1.5, invert=True)
    save_figure([pupil_bin, iris_bin], ["Pupil Binary", "Iris Binary"], os.path.join(img_dir, "01_binary.png"))

    pupil_open = opening(pupil_bin, kernel_size=3, shape="circle")
    pupil_mask = closing(pupil_open, kernel_size=20, shape="circle")
    save_figure([pupil_bin, pupil_open, pupil_mask], ["Pupil raw", "Pupil opened", "Pupil closed"], os.path.join(img_dir, "02_pupil_morphology.png"))

    cx, cy = get_center(pupil_mask)
    
    short_side = min(img.shape[:2])
    angles_all = np.linspace(0, 2 * np.pi, 360)
    eyelid_mask = ~(((angles_all > np.radians(80)) & (angles_all < np.radians(100))) |
                    ((angles_all > np.radians(210)) & (angles_all < np.radians(330))))

    pupil_radius = int(get_radius(gray, cx, cy, int(0.05 * short_side), int(0.25 * short_side)))
    iris_radius = int(get_radius(gray, cx, cy, int(0.20 * short_side), int(0.55 * short_side), angle_mask=eyelid_mask))

    pupil_circle = draw_circle(pupil_mask, cx, cy, pupil_radius)
    save_figure([pupil_circle], ["Pupil circle"], os.path.join(img_dir, "03_pupil_circle.png"))

    iris_open = opening(iris_bin, kernel_size=35, shape="circle")
    iris_mask = closing(iris_open, kernel_size=20, shape="circle")
    save_figure([iris_bin, iris_open, iris_mask], ["Iris raw", "Iris opened", "Iris closed"], os.path.join(img_dir, "04_iris_morphology.png"))

    cx_iris, cy_iris = get_center(iris_mask)
    iris_circle = draw_circle(iris_mask, cx_iris, cy_iris, iris_radius)
    save_figure([iris_circle], ["Iris circle"], os.path.join(img_dir, "05_iris_circle.png"))

    overlay = draw_circle(img, cx, cy, pupil_radius)
    overlay = draw_circle(overlay, cx, cy, iris_radius)
    save_figure([overlay], ["Detected circles"], os.path.join(OVERLAY_DIR, f"{stem}_overlay.png"))

    iris_region = apply_iris_mask(img, iris_mask[:, :, 0], pupil_mask[:, :, 0])
    outside_region = apply_iris_mask(img, 255 - iris_mask[:, :, 0], pupil_mask[:, :, 0])
    save_figure([iris_region, outside_region], ["Extracted iris", "Outside iris"], os.path.join(img_dir, "07_iris_region.png"))

    unwrapped = unwrap_iris(img, cx, cy, pupil_radius, iris_radius)
    save_figure([unwrapped], ["Unwrapped iris"], os.path.join(img_dir, "08_unwrapped.png"))

    strips = np.array_split(unwrapped[10:54, :, 0], 8, axis=0)
    save_figure(strips, [f"Strip {i+1}" for i in range(8)], os.path.join(img_dir, "09_strips.png"), figsize_per=3)

    strip_1d = [np.mean(strip, axis=0)[np.linspace(0, len(np.mean(strip, axis=0)) - 1, 128).astype(int)] for strip in strips]
    np.save(os.path.join(img_dir, "features.npy"), np.array(strip_1d))

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(OVERLAY_DIR, exist_ok=True)
    for path in IMAGE_PATHS:
        try:
            process_image(path)
        except Exception as exc:
            print(f"[ERROR] {path}: {exc}\n")