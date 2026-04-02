import numpy as np

from pixel_transformations import grey_scale

# Funkcja do użycia w innych, przejeżdza podaną maską po macierzy
# Zwraca kolorowy obraz
def apply_kernel(img, kernel):
    # przygotowanie danych
    mat = np.array(img).astype(float)
    h, w, c = mat.shape
    n = kernel.shape[0]

    weights_sum = np.sum(kernel)
    if weights_sum == 0:
        weights_sum = 1 # Dla filtrów typu Sobel nie dzielimy bo suma wag to 0

    padding = n // 2
    
    # wynikowa macierz
    res = np.zeros_like(mat)
    
    # Dodanie padding do macierzy
    padded_mat = np.pad(mat, ((padding, padding), (padding, padding), (0, 0)), mode='edge')
    
    # pętle po każdym pikselu
    for i in range(h):
        for j in range(w):
            # wycinamy okno [n x n x 3]
            window = padded_mat[i : i + n, j : j + n, :]
            
            # przetwanie każdego kanału osobno
            for k in range(c):
                # mnożymy wartości pikseli przez wagi maski i sumujemy
                # window[:, :, k] to fragment obrazu dla jednego koloru
                res[i, j, k] = np.sum(window[:, :, k] * kernel) / weights_sum
                
    return res   

# Jako oddzielna funkcja żeby przyciąć po nałożeniu maski
def custom_kernel(img, kernel):
    res = apply_kernel(img, kernel)
    return np.clip(res, 0, 255).astype(np.uint8)

# Filtr uśredniający z maską nxn (macierz samych jedynek)
def averaging_filter(img, n):
    kernel = np.ones((n, n))
    res = apply_kernel(img, kernel)
    return np.clip(res, 0, 255).astype(np.uint8)

# Filtr Gaussa 3x3
def gaussian_blur(img):
    kernel = np.array([
        [1, 4, 1],
        [4, 12, 4],
        [1, 4, 1]
    ])
    res = apply_kernel(img, kernel)
    return np.clip(res, 0, 255).astype(np.uint8)

# Wyostrzanie
def sharpen_filter(img):
    kernel = np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ])
    res = apply_kernel(img, kernel)
    return np.clip(res, 0, 255).astype(np.uint8)

# Operator Sobela (Pionowy)
def sobel(img, degrees):
    # słownik z kernelami dla różnych kątów
    kernels = {
        0:   np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),   # Pionowe
        45:  np.array([[ 0, 1, 2], [-1, 0, 1], [-2,-1, 0]]),   # Skos 45
        90:  np.array([[ 1, 2, 1], [ 0, 0, 0], [-1,-2,-1]]),   # Poziome
        135: np.array([[ 2, 1, 0], [ 1, 0,-1], [ 0,-1,-2]]),   # Skos 135
        180: np.array([[ 1, 0,-1], [ 2, 0,-2], [ 1, 0,-1]]),   # Odwrócone pionowe
        225: np.array([[ 0,-1,-2], [ 1, 0,-1], [ 2, 1, 0]]),   # Skos 225
        270: np.array([[-1,-2,-1], [ 0, 0, 0], [ 1, 2, 1]]),   # Odwrócone poziome
        315: np.array([[-2,-1, 0], [-1, 0, 1], [ 0, 1, 2]])    # Skos 315
    }

    kernel = kernels.get(degrees, kernels[0])
    
    res = apply_kernel(img, kernel)
    return np.clip(np.abs(res), 0, 255).astype(np.uint8)

def roberts_cross(img, orientation="orthogonal"):
    if not (np.array_equal(img[:, :, 0], img[:, :, 1])
            and np.array_equal(img[:, :, 1], img[:, :, 2])):
        img = grey_scale(img)

    # orthogonal masks
    mask_1 = np.array([[ 1, -1], [ 0,  0]])
    mask_2 = np.array([[ 1,  0], [-1,  0]])

    # diagonal masks
    if orientation == "diagonal":
        mask_1 = np.array([[ 1,  0], [ 0, -1]])
        mask_2 = np.array([[ 0,  1], [-1,  0]])

    res_1 = apply_kernel(img, mask_1)
    res_2 = apply_kernel(img, mask_2) 
    combined = np.abs(res_1) + np.abs(res_2)
    return np.clip(combined, 0, 255).astype(np.uint8)

# Emboss
def emboss(img, fade: float):
    kernel = np.array([
        [-2, -1,  0],
        [-1,  1 + fade,  1],
        [ 0,  1,  2]
    ], dtype=float)
    res = apply_kernel(img, kernel)
    return np.clip(res, 0, 255).astype(np.uint8)    
