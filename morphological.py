import numpy as np

from pixel_transformations import grey_scale

# -------------------------------------------------------------------
# -------------------- Operacje Morfologiczne -----------------------
# -------------------------------------------------------------------

# Tworzy wybrany kształt elementu strukturalnego
def get_structuring_element(size, shape="square"):
    k = size
    center = k // 2

    if shape == "square":
        return np.ones((k, k), dtype=np.uint8)

    elif shape == "cross":
        kernel = np.zeros((k, k), dtype=np.uint8)
        kernel[center, :] = 1
        kernel[:, center] = 1
        return kernel

    elif shape == "circle":
        kernel = np.zeros((k, k), dtype=np.uint8)
        for i in range(k):
            for j in range(k):
                if (i - center)**2 + (j - center)**2 <= center**2:
                    kernel[i, j] = 1
        return kernel

    else:
        return np.ones((k, k), dtype=np.uint8)


def erosion(img, kernel_size=3, shape="square"):
    kernel = get_structuring_element(kernel_size, shape)

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
                values = window[:, :, k][kernel == 1]
                res[i, j, k] = np.min(values)

    return res.astype(np.uint8)

def dilation(img, kernel_size=3, shape="square"):
    kernel = get_structuring_element(kernel_size, shape)

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
                values = window[:, :, k][kernel == 1]
                res[i, j, k] = np.max(values)

    return res.astype(np.uint8)


def opening(img, kernel_size=3, shape="square"):
    return dilation(erosion(img, kernel_size, shape), kernel_size, shape)

def closing(img, kernel_size=3, shape="square"):
    return erosion(dilation(img, kernel_size, shape), kernel_size, shape)

