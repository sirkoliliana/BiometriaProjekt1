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

# Binaryzacja - metoda lokalna Bernsena
# wyznaczanie t lokalnie na podstawie min max okienka
def binarize(img, window_size = 15, contrast_threshold=15):
    mat = np.array(img).astype(float)

    # Greyscale (grey przechowuje tylko jedną wartość a nie rgb)
    gray = (mat[:, :, 0].astype(float) + mat[:, :, 1].astype(float) + mat[:, :, 2].astype(float)) / 3
    gray = gray.astype(np.uint8)

    h, w = gray.shape[:2]
    # Tworzę macierz wynikową (płaską)
    res = np.zeros_like(gray)

    # Ramka żeby okno mogło wyjść
    padding = window_size // 2
    padded_gray = np.pad(gray, padding, mode='edge')

    # Petla po każdym pixelu
    for y in range(h):
        for x in range(w):
            # Obliczanie min, max i mid dla okna
            window = padded_gray[y : y + window_size, x : x + window_size]
            min_val = np.min(window)
            max_val = np.max(window)
            mid_val = (max_val+min_val)/2

            if (max_val - min_val) < contrast_threshold:
                # Jeśli bardzo mały kontrast dookoła ustalam kolor
                # na podstawie otoczenia
                res[y, x] = 255 if mid_val > 128 else 0
            else:
                # Binaryzacja
                res[y, x] = 255 if gray[y, x] > mid_val else 0

    return np.stack([res, res, res], axis=-1)






