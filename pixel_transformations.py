import numpy as np

# Konwersja do odcieni szarości
# Uwzględnienie wag kolorów
def monochrome(img):
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

# Binaryzacja - klasyczna metoda
def binarize_simple(img, t_val=128):
    # zamiana na grayscale
    if img.ndim == 3:
        gray = (img[:, :, 0].astype(float) + img[:, :, 1].astype(float) + img[:, :, 2].astype(float)) / 3
    else:
        gray = img.astype(float)

    # jeśli piksel > próg to 255, inaczej 0
    binary = np.where(gray > t_val, 255, 0).astype(np.uint8)

    # powrót do rgb
    return np.stack([binary, binary, binary], axis=-1)

# Binaryzacja - do projektu 2
def binarize_avg(img, x=1, invert=False):
    if img.ndim == 3:
        gray = (img[:,:,0].astype(float) + img[:,:,1].astype(float) + img[:,:,2].astype(float)) / 3
    else:
        gray = img.astype(float)

    P = np.mean(gray)
    P_x = P / x

    if invert:
        binary = np.where(gray <= P_x, 255, 0).astype(np.uint8)
    else:
        binary = np.where(gray > P_x, 255, 0).astype(np.uint8)

    return np.stack([binary, binary, binary], axis=-1)

# Binaryzacja - metoda lokalna Bernsena
# wyznaczanie t lokalnie na podstawie min max okienka
def binarize_bernsen(img, window_size = 15, contrast_threshold=15):
    mat = np.array(img).astype(float)

    # Greyscale (grey przechowuje tylko jedną wartość a nie rgb)
    gray = (mat[:, :, 0] + mat[:, :, 1] + mat[:, :, 2]) / 3
    gray = gray.astype(np.uint8)

    h, w = gray.shape
    # Tworzę macierz wynikową (płaską)
    res = np.zeros_like(gray)

    # Ramka żeby okno mogło wyjść
    padding = window_size // 2
    padded_gray = np.pad(gray, padding, mode='edge').astype(float)

    # Petla po każdym pixelu
    for y in range(h):
        for x in range(w):
            # Obliczanie min, max i mid dla okna
            window = padded_gray[y : y + window_size, x : x + window_size]
            min_val = np.min(window)
            max_val = np.max(window)
            mid_val = (max_val+min_val)/2

            # Metoda Bernsena
            if (max_val - min_val) < contrast_threshold:
                # Jeśli bardzo mały kontrast dookoła ustalam kolor
                # na podstawie otoczenia
                res[y, x] = 255 if mid_val > 128 else 0
            else:
                # Binaryzacja
                res[y, x] = 255 if gray[y, x] > mid_val else 0

    return np.stack([res, res, res], axis=-1)

def add_images(img1, img2):
    # clipped to range 0-255
    mat1 = np.array(img1).astype(float)
    mat2 = np.array(img2).astype(float)

    res = mat1 + mat2
    return np.clip(res, 0, 255).astype('uint8')

def subtract_images(img1, img2):
    # clipped to range 0-255
    mat1 = np.array(img1).astype(float)
    mat2 = np.array(img2).astype(float)

    res = mat1 - mat2
    return np.clip(res, 0, 255).astype('uint8')

def mix_images(img1, img2, alpha):
    mat1 = np.array(img1).astype(float)
    mat2 = np.array(img2).astype(float)

    res = alpha * mat1 + (1 - alpha) * mat2
    return np.clip(res, 0, 255).astype('uint8')

# OPERACJE NA HISTOGRAMACH

def equalize_array(array):
    # Histogram
    hist, _ = np.histogram(array.flatten(), bins=256, range=(0, 256))

    # Dystrybuanta
    cdf = hist.cumsum()
    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())

    # Mapowanie pikseli
    equalized_array = np.interp(array.flatten(), np.arange(256), cdf_normalized)
    equalized_array = equalized_array.reshape(array.shape).astype(np.uint8)

    return equalized_array

def histogram_equalization(img):
    mat = np.array(img)
    gray = np.mean(mat, axis=2).astype(np.uint8) 
    eq_gray = equalize_array(gray)
    return np.stack([eq_gray] * 3, axis=-1).astype(np.uint8)

def histogram_equalization_3_chanel(img):
    mat = np.array(img)
    eq_img = np.dstack([equalize_array(mat[:, :, i]) for i in range(3)])
    return eq_img.astype(np.uint8)

def stretch_array(array):
    min_val = array.min()
    max_val = array.max()
    if max_val > min_val:
        stretched = (array - min_val) * (255 / (max_val - min_val))
        return np.clip(stretched, 0, 255).astype(np.uint8)
    else:
        return array.astype(np.uint8)
    
def contrast_stretching(img):
    gray = np.mean(np.array(img), axis=2)
    stretched = stretch_array(gray)
    return np.stack([stretched] * 3, axis=-1)
    
def contrast_stretching_3_channel(img):
    mat = np.array(img)
    return np.dstack([stretch_array(mat[:, :, i]) for i in range(3)])