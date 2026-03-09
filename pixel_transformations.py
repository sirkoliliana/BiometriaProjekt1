import numpy as np

# TODO: Implement all transformations in this file. Later import them in app.py and use them in apply_operation.

# Konwersja do odcieni szarości
# Uwzględnienie wag kolorów
def monotone(img):
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    gray = np.dot(img.astype(np.float32), weights)  # (H, W)
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)  

# Konwersja do odcieni szarości
# Tylko średnia artmetyczna
def grey_scale(img):
    mat = np.array(img).astype(float)
    mean_val = (mat[:, :, 0]+mat[:, :, 1]+mat[:, :, 2])/3

    res = np.zeros_like(mat)
    res[:, :, 0] = res[:, :, 1] = res[:, :, 2] = mean_val
    return res.astype('uint8')

# Korekta jasności i kontrastu - operacje nieliniowe
def gamma_transform(img, alpha):
    mat = np.array(img).astype(float)

    for i in range(3):
        max_val = mat[:, :, i].max()
        if max_val > 0:
            # Normalizacja do 0-1 potęga i powrót do 0-255
            mat[:, :, i] = 255 * ((mat[:, :, i] / max_val) ** alpha)
    return np.clip(mat, 0, 255).astype('uint8')

def log_transform(img):
    mat = np.array(img).astype(float)

    for i in range(3):
        max_val = mat[:, :, i].max()
        if max_val > 0:
            # log(1+x) żeby nie było log(0)
            mat[:, :, i] = 255 * (np.log(1 + mat[:, :, i]) / np.log(1 + max_val))
    return np.clip(mat, 0, 255).astype('uint8')

# Negatyw
def invert(img):
    mat = np.array(img).astype(float)

    res = 255 - mat
    return res.astype('uint8')
