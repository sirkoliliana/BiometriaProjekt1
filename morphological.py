import numpy as np

from pixel_transformations import grey_scale

# -------------------------------------------------------------------
# -------------------- Operacje Morfologiczne -----------------------
# -------------------------------------------------------------------

def erosion(img, kernel_size=3):
    mat = np.array(img).astype(float)
    h, w, c = mat.shape
    padding = kernel_size // 2

    res = np.zeros_like(mat)

    padded = np.pad(mat, ((padding, padding), (padding, padding), (0, 0)), mode='edge')

    for i in range(h):
        for j in range(w):
            window = padded[i:i+kernel_size, j:j+kernel_size, :]
            for k in range(c):
                # najważniejsze - min
                res[i, j, k] = np.min(window[:, :, k])

    return res.astype(np.uint8)

def dilation(img, kernel_size=3):
    mat = np.array(img).astype(float)
    h, w, c = mat.shape
    padding = kernel_size // 2

    res = np.zeros_like(mat)

    padded = np.pad(mat, ((padding, padding), (padding, padding), (0, 0)), mode='edge')

    for i in range(h):
        for j in range(w):
            window = padded[i:i+kernel_size, j:j+kernel_size, :]
            for k in range(c):
                # najważniejsze - max
                res[i, j, k] = np.max(window[:, :, k])

    return res.astype(np.uint8)


def opening(img, kernel_size=3):
    return dilation(erosion(img, kernel_size), kernel_size)

def closing(img, kernel_size=3):
    return erosion(dilation(img, kernel_size), kernel_size)

