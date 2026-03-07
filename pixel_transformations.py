import numpy as np

# TODO: Implement all transformations in this file. Later import them in app.py and use them in apply_operation.

def monotone(img):
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    gray = np.dot(img.astype(np.float32), weights)  # (H, W)
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)  
        