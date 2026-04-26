from PIL import Image, ImageOps
import dearpygui.dearpygui as dpg
import numpy as np
from pipeline import apply_pipeline
import state
from models import Operation
from ui.charts import update_histogram, update_rgb_histogram, update_projection, numpy_to_dpg, update_pipeline_list


def _refresh_ui():
    result = apply_pipeline()
    dpg.set_value("image_texture", numpy_to_dpg(result))
    update_pipeline_list()
    update_histogram(result)
    update_rgb_histogram(result)
    update_projection(result)

    # show projection instead of histogram if image is binarized
    is_binarized = all(
        not ((result[:, :, i] != 255) & (result[:, :, i] != 0)).any()
        for i in range(3)
    )
    if is_binarized:
        dpg.hide_item("histogram_group")
        dpg.show_item("projection_group")
    else:
        dpg.hide_item("projection_group")
        dpg.show_item("histogram_group")


def open_callback(sender, app_data):
    path = app_data["file_path_name"]
    img = Image.open(path).convert("RGB")
    img = ImageOps.fit(img, (state.IMG_W, state.IMG_H), Image.Resampling.LANCZOS)
    state.original_image = np.array(img)
    state.current_image = state.original_image.copy()
    state.pipeline.clear()
    _refresh_ui()

def save_callback(sender, app_data):
    path = app_data["file_path_name"]
    if not path.endswith((".png", ".jpg", ".bmp")):
        path += ".png"
    Image.fromarray(state.current_image).save(path)

def cancel_callback(sender, app_data):
    pass


def on_filter_change(sender, app_data):
    dpg.hide_item("blur_options")
    dpg.hide_item("sobel_options")
    dpg.hide_item("custom_kernel_options")
    dpg.hide_item("roberts_cross_options")
    dpg.hide_item("emboss_options")

    if app_data == "Blur (Averaging)":
        dpg.show_item("blur_options")
    elif app_data == "Sobel Edge Detection":
        dpg.show_item("sobel_options")
    elif app_data == "Robert's Cross":
        dpg.show_item("roberts_cross_options")
    elif app_data == "Custom Kernel":
        dpg.show_item("custom_kernel_options")
    elif app_data == "Emboss":
        dpg.show_item("emboss_options")


def on_pt_change(sender, app_data):
    dpg.hide_item("threshold_options")
    dpg.hide_item("gamma_options")
    dpg.hide_item("binarize_avg_options")
    dpg.hide_item("binarize_options")
    dpg.hide_item("add_image_options")
    dpg.hide_item("blend_image_options")

    if app_data == "Gamma":
        dpg.show_item("gamma_options")
    elif app_data == "Binarize Bernsen":
        dpg.show_item("binarize_options")
    elif app_data == "Binarize Avg":
        dpg.show_item("binarize_avg_options")
    elif app_data == "Binarize simple":
        dpg.show_item("threshold_options")
    elif app_data == "Add Images":
        dpg.show_item("add_image_options")
    elif app_data == "Blend Image":
        dpg.show_item("blend_image_options")


# Operacje morfologiczne
def on_morph_change(sender, app_data):
    dpg.hide_item("morph_options")

    if app_data != "None":
        dpg.show_item("morph_options")


def add_filter():
    selected = dpg.get_value("filter_combo")
    if selected == "None":
        return

    if selected == "Blur (Averaging)":
        op = Operation(name="Blur", params={"n": dpg.get_value("blur_kernel")})

    elif selected == "Gaussian Blur":
        op = Operation(name="Gauss")

    elif selected == "Sharpen":
        op = Operation(name="Sharpen")

    elif selected == "Sobel Edge Detection":
        op = Operation(name="Sobel Edge Detection", params={"degrees": int(dpg.get_value("sobel_angle_combo"))})

    elif selected == "Robert's Cross":
        op = Operation(name="Robert's Cross", params={"orientation": dpg.get_value("roberts_cross_orientation")})

    elif selected == "Custom Kernel":
        rows = [[dpg.get_value(f"k{i}{j}") for j in range(5)] for i in range(5)]
        op = Operation(name="Custom Kernel", params={"kernel": np.array(rows).tolist()})

    elif selected == "Emboss":
        op = Operation(name="Emboss", params={"emboss": dpg.get_value("emboss_slider")})

    else:
        return

    state.pipeline.append(op)
    _refresh_ui()


def add_pixel_transform():
    selected = dpg.get_value("pt_combo")
    if selected == "None":
        return

    if selected == "Grayscale":
        op = Operation(name="Grayscale")

    elif selected == "Monochrome":
        op = Operation(name="Monochrome")

    elif selected == "Invert":
        op = Operation(name="Invert")

    elif selected == "Brightness":
        op = Operation(name="Brightness")

    elif selected == "Gamma":
        op = Operation(name="Gamma", params={"gamma": dpg.get_value("gamma_slider")})

    elif selected == "Binarize simple":
        op = Operation(name="Binarize simple", params={"threshold": dpg.get_value("threshold_slider")})

    elif selected == "Binarize Avg":
        op = Operation(name="Binarize Avg", params={"adjust": dpg.get_value("binarize_avg_slider")})

    elif selected == "Binarize Bernsen":
        op = Operation(name="Binarize Bernsen", params={"contrast": dpg.get_value("binarize_slider")})

    elif selected == "Histogram Equalization (Gray)":
        op = Operation(name="Histogram Equalization (Gray)")

    elif selected == "Histogram Equalization (RGB)":
        op = Operation(name="Histogram Equalization (RGB)")

    elif selected == "Contrast Stretching (Gray)":
        op = Operation(name="Contrast Stretching (Gray)")

    elif selected == "Contrast Stretching (RGB)":
        op = Operation(name="Contrast Stretching (RGB)")

    elif selected == "Add Images":
        if not state.added_images:
            return
        op = Operation(name="Add Images", params={
            "image": state.added_images[-1],
            "subtract": dpg.get_value("subtract_image_checkbox")
        })

    elif selected == "Blend Image":
        if not state.added_images:
            return
        op = Operation(name="Blend Image", params={
            "image": state.added_images[-1],
            "alpha": dpg.get_value("blend_alpha_slider")
        })

    else:
        return

    state.pipeline.append(op)
    _refresh_ui()

# Operacje morfologiczne
def add_morph_operation():
    selected = dpg.get_value("morph_combo")

    if selected == "None":
        return

    kernel_size = dpg.get_value("morph_kernel")
    shape = dpg.get_value("morph_shape")

    # wymuszenie nieparzystości
    if kernel_size % 2 == 0:
        kernel_size += 1

    op = Operation(
        name=selected,
        params={
            "kernel_size": kernel_size,
            "shape": shape
        }
    )

    state.pipeline.append(op)
    _refresh_ui()


def undo():
    if state.pipeline:
        state.pipeline.pop()
        _refresh_ui()


def reset_pipeline():
    state.pipeline.clear()
    state.current_image = state.original_image.copy()
    _refresh_ui()

  
