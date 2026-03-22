import numpy as np


IMG_W, IMG_H = 400, 400

original_image: np.ndarray = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
current_image: np.ndarray  = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
pipeline: list             = []
added_images: list         = []
