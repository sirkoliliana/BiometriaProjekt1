import numpy as np
import state
from models import Operation
from filters import custom_kernel, averaging_filter, gaussian_blur, sharpen_filter, sobel, roberts_cross, emboss
from pixel_transformations import (grey_scale, gamma_transform, log_transform, invert, 
                                   histogram_equalization, histogram_equalization_3_chanel,
                                   contrast_stretching, contrast_stretching_3_channel,
                                   binarize_simple, binarize_bernsen, add_images, subtract_images, mix_images)

# dispatch table - add new operations here
OPERATIONS = {
    "Blur":                          lambda img, p: averaging_filter(img, p.get("n", 3)),
    "Gauss":                         lambda img, p: gaussian_blur(img),
    "Sharpen":                       lambda img, p: sharpen_filter(img),
    "Sobel Edge Detection":          lambda img, p: sobel(img, p.get("degrees", 0)),
    "Robert's Cross":                lambda img, p: roberts_cross(img, p.get("orientation", "orthogonal")),
    "Custom Kernel":                 lambda img, p: custom_kernel(img, np.array(p.get("kernel"))),
    "Grayscale":                     lambda img, p: grey_scale(img),
    "Brightness":                    lambda img, p: log_transform(img),
    "Gamma":                         lambda img, p: gamma_transform(img, p.get("gamma", 1.0)),
    "Invert":                        lambda img, p: invert(img),
    "Binarize simple":               lambda img, p: binarize_simple(img, p.get("threshold", 128)),
    "Binarize Bernsen":              lambda img, p: binarize_bernsen(img, p.get("contrast", 15)),
    "Histogram Equalization (Gray)": lambda img, p: histogram_equalization(img),
    "Histogram Equalization (RGB)":  lambda img, p: histogram_equalization_3_chanel(img),
    "Contrast Stretching (Gray)":    lambda img, p: contrast_stretching(img),
    "Contrast Stretching (RGB)":     lambda img, p: contrast_stretching_3_channel(img),
    "Add Images":                    lambda img, p: subtract_images(img, p.get("image")) if p.get("subtract") else add_images(img, p.get("image")),
    "Blend Image":                   lambda img, p: mix_images(img, p.get("image"), p.get("alpha", 0.5)),
    "Emboss":                        lambda img, p: emboss(img, p.get("emboss", 1.0)),
}

def run_operation(img: np.ndarray, op: Operation) -> np.ndarray:
    fn = OPERATIONS.get(op.name)
    return fn(img, op.params) if fn else img

def apply_pipeline() -> np.ndarray:
    result = state.original_image.copy()
    for op in state.pipeline:
        result = run_operation(result, op)
    state.current_image = result
    return result