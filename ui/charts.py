import dearpygui.dearpygui as dpg
import numpy as np
import state

def update_histogram(img: np.ndarray):
    gray = img.mean(axis=2) if img.ndim == 3 else img
    counts, bin_edges = np.histogram(gray.flatten(), bins=256, range=(0, 256))
    dpg.configure_item("hist_series", x=bin_edges[:-1].tolist(), y=counts.tolist())
    dpg.set_axis_limits("histogram_x_axis", 0, 255)   # ← add this
    dpg.set_axis_limits("histogram_y_axis", 0, int(counts.max() * 1.05))

def update_rgb_histogram(img: np.ndarray):
    if img.ndim != 3 or img.shape[2] < 3:
        return
    max_count = 0
    for i, color in enumerate(["red", "green", "blue"]):
        channel = img[:, :, i]
        counts, bin_edges = np.histogram(channel.flatten(), bins=256, range=(0, 256))
        dpg.configure_item(f"hist_series_{color}", x=bin_edges[:-1].tolist(), y=counts.tolist())
        max_count = max(max_count, int(counts.max()))
    dpg.set_axis_limits("histogram_rgb_x_axis", 0, 255)   # ← add this
    dpg.set_axis_limits("histogram_rgb_y_axis", 0, int(max_count * 1.05))

def update_projection(img: np.ndarray):
    img = img[:, :, 0]
    horizontal_proj = img.sum(axis=0)
    vertical_proj = img.sum(axis=1)

    dpg.set_value("horizontal_projection_series", [list(range(img.shape[1])), horizontal_proj.tolist()])
    dpg.set_value("vertical_projection_series", [list(range(img.shape[0])), vertical_proj.tolist()])

    dpg.set_axis_limits("horizontal_projection_y_axis", 0, int(horizontal_proj.max() * 1.05))
    dpg.set_axis_limits("vertical_projection_y_axis", 0, int(vertical_proj.max() * 1.05))

def numpy_to_dpg(img: np.ndarray) -> list:
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    h, w = img.shape[:2]
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    alpha = np.ones((h, w, 1), dtype=np.float32)
    return np.concatenate([img, alpha], axis=-1).flatten().tolist()


def update_pipeline_list():
    items = [f"{i+1}. {op.name}" for i, op in enumerate(state.pipeline)]
    dpg.configure_item("pipeline_listbox", items=items if items else ["(empty)"])

