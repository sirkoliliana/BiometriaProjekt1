import dearpygui.dearpygui as dpg
import numpy as np
from PIL import Image, ImageOps
from pixel_transformations import monotone, grey_scale, gamma_transform, log_transform, invert, binarize


IMG_W, IMG_H = 400, 400

# original_image: never modified, always the loaded file
# current_image: result after applying the full pipeline
# pipeline: ordered list of {"name": str, "params": dict}
original_image: np.ndarray = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
current_image: np.ndarray  = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
pipeline: list             = []



#  IMAGE HELPERS

def numpy_to_dpg(img: np.ndarray) -> list:
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    h, w = img.shape[:2]
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    alpha = np.ones((h, w, 1), dtype=np.float32)
    return np.concatenate([img, alpha], axis=-1).flatten().tolist()


def update_histogram(img: np.ndarray):
    gray = img.mean(axis=2) if img.ndim == 3 else img
    counts, bin_edges = np.histogram(gray.flatten(), bins=256, range=(0, 255))
    dpg.configure_item(
        "hist_series",
        x=bin_edges[:-1].tolist(),
        y=counts.tolist()
    )

# TODO: add separate histograms for R,G,B channels and update them in update_histogram, instead of just one for the average 
def update_rgb_histogram(img: np.ndarray):
    if img.ndim != 3 or img.shape[2] < 3:
        return
    colors = ["red", "green", "blue"]
    for i, color in enumerate(colors):
        channel = img[:, :, i]
        counts, bin_edges = np.histogram(channel.flatten(), bins=256, range=(0, 255))
        dpg.configure_item(
            f"hist_series_{color}",
            x=bin_edges[:-1].tolist(),
            y=counts.tolist()
        )


def update_pipeline_list():
    """Refresh the pipeline listbox in the UI."""
    items = [f"{i+1}. {op['name']}" for i, op in enumerate(pipeline)]
    dpg.configure_item("pipeline_listbox", items=items if items else ["(empty)"])


#  PIPELINE EXECUTION

def run_operation(img: np.ndarray, op: dict) -> np.ndarray:
    """Apply a single operation to img and return the result."""
    name   = op["name"]
    params = op["params"]

    if name == "Monotone":
        return monotone(img)

    elif name == "Brightness":
        return log_transform(img)

    elif name == "Gamma":
        g_val = params.get("gamma", 1.0)    
        return gamma_transform(img, g_val)
    
    elif name == "Invert":
        return invert(img)

    elif name == "Grayscale":
        return grey_scale(img)

    elif name == "Binarize":
        c_val = params.get("contrast", 15)
        return binarize(img, window_size=15, contrast_threshold=c_val)

    return img


def apply_pipeline():
    """Re-run all pipeline operations from the original image."""
    global current_image
    result = original_image.copy()
    for op in pipeline:
        result = run_operation(result, op)
    current_image = result
    dpg.set_value("image_texture", numpy_to_dpg(current_image))
    update_histogram(current_image)
    update_rgb_histogram(current_image)


#  UI CALLBACKS

def on_filter_change(sender, app_data):
    dpg.hide_item("blur_options")
    dpg.hide_item("brightness_options")
    if app_data == "Blur":
        dpg.show_item("blur_options")
    elif app_data == "Brightness":
        dpg.show_item("brightness_options")


def on_pt_change(sender, app_data):
    dpg.hide_item("threshold_options")
    dpg.hide_item("gamma_options")
    dpg.hide_item("binarize_options")
    if app_data == "Threshold":
        dpg.show_item("threshold_options")
    elif app_data == "Gamma":
        dpg.show_item("gamma_options")
    elif app_data == "Binarize":
        dpg.show_item("binarize_options")


def add_filter():
    """Read current filter selection + params and push to pipeline."""
    selected = dpg.get_value("filter_combo")
    if selected == "None":
        return

    if selected == "Blur":
        op = {"name": "Blur", "params": {"kernel": dpg.get_value("blur_kernel")}}
    elif selected == "Brightness":
        op = {"name": "Brightness", "params": {"factor": dpg.get_value("brightness_slider")}}
    else:
        return

    pipeline.append(op)
    update_pipeline_list()
    apply_pipeline()


def add_pixel_transform():
    selected = dpg.get_value("pt_combo")
    if selected == "None":
        return

    if selected == "Monotone":
        op = {"name": "Monotone", "params": {}}
    elif selected == "Grayscale":
        op = {"name": "Grayscale", "params": {}}
    elif selected == "Threshold":
        op = {"name": "Threshold", "params": {"threshold": dpg.get_value("threshold_slider")}}
    elif selected == "Invert":
        op = {"name": "Invert", "params": {}}
    elif selected == "Brightness":
        op = {"name": "Brightness", "params": {"factor": dpg.get_value("brightness_slider")}}
    elif selected == "Gamma":
        op = {"name": "Gamma", "params": {"gamma": dpg.get_value("gamma_slider")}}
    elif selected == "Binarize":
        op = {"name": "Binarize", "params": {"contrast": dpg.get_value("binarize_slider")}}
    else:
        return

    pipeline.append(op)
    update_pipeline_list()
    apply_pipeline()


def undo():
    """Remove the last operation and re-run the pipeline."""
    if pipeline:
        pipeline.pop()
        update_pipeline_list()
        apply_pipeline()


def reset_pipeline():
    """Clear all operations and restore original image."""
    global current_image
    pipeline.clear()
    current_image = original_image.copy()
    update_pipeline_list()
    dpg.set_value("image_texture", numpy_to_dpg(current_image))
    update_histogram(current_image)
    update_rgb_histogram(current_image)


#  MAIN

def main():
    global original_image, current_image

    dpg.create_context()
    dpg.create_viewport(title="Biometrics App", width=1100, height=750)

    # File dialog callbacks
    def open_callback(sender, app_data):
        global original_image, current_image

        path = app_data["file_path_name"]
        img = Image.open(path).convert("RGB")
        img = Image.open(path).convert("RGB")
        img = ImageOps.fit(img, (IMG_W, IMG_H), Image.LANCZOS)  # ← replaces thumbnail + padding
        original_image = np.array(img)
        pipeline.clear()
        update_pipeline_list()
        current_image = original_image.copy()
        dpg.set_value("image_texture", numpy_to_dpg(current_image))
        update_histogram(current_image)
        update_rgb_histogram(current_image)

    def save_callback(sender, app_data):
        path = app_data["file_path_name"]
        if not path.endswith((".png", ".jpg", ".bmp")):
            path += ".png"
        Image.fromarray(current_image).save(path)
        print(f"Saved to {path}")

    def cancel_callback(sender, app_data):
        print("Cancelled.")

    # For image display
    with dpg.texture_registry():
        dpg.add_dynamic_texture(IMG_W, IMG_H, numpy_to_dpg(current_image), tag="image_texture")

    # File dialogs
    with dpg.file_dialog(show=False, callback=open_callback,
                         cancel_callback=cancel_callback,
                         tag="open_dialog_id", width=700, height=400):
        dpg.add_file_extension(".png")
        dpg.add_file_extension(".jpg")
        dpg.add_file_extension(".bmp")

    with dpg.file_dialog(show=False, callback=save_callback,
                         cancel_callback=cancel_callback,
                         tag="save_dialog_id", width=700, height=400):
        dpg.add_file_extension(".png")
        dpg.add_file_extension(".jpg")
        dpg.add_file_extension(".bmp")

    # Main window
    with dpg.window(tag="main_window"):

        # Toolbar
        with dpg.group(horizontal=True):
            dpg.add_button(label="Open File", callback=lambda: dpg.show_item("open_dialog_id"))
            dpg.add_button(label="Save", callback=lambda: dpg.show_item("save_dialog_id"))
            dpg.add_button(label="Undo", callback=undo)
            dpg.add_button(label="Reset", callback=reset_pipeline)

        dpg.add_separator()

        with dpg.table(header_row=False, borders_innerV=True):
            dpg.add_table_column(init_width_or_weight=420, width_fixed=True)
            dpg.add_table_column()

            with dpg.table_row():

                # Image preview
                with dpg.table_cell():
                    dpg.add_text("Preview")
                    dpg.add_image("image_texture")

                # controls + pipeline + histogram
                with dpg.table_cell():
                    with dpg.tab_bar():

                        # Pixel Transformations tab
                        # combo: drop donw menu of transformations
                        with dpg.tab(label="Pixel Transformations"):
                            dpg.add_combo(
                                # TODO: add more options
                                ["None", 
                                 "Monotone", 
                                 "Grayscale",
                                 "Brightness", 
                                 "Gamma", 
                                 "Threshold", 
                                 "Binarize",
                                 "Invert"],
                                label="Transform", default_value="None",
                                tag="pt_combo", callback=on_pt_change, width=200
                            )

                            # groups of options for specific transformations, only one shown at a time based on selection
                            with dpg.group(tag="threshold_options", show=False):
                                dpg.add_text("Threshold Options")
                                dpg.add_slider_int(label="Threshold", tag="threshold_slider",
                                                   min_value=0, max_value=255, default_value=128)
                                

                            with dpg.group(tag="gamma_options", show=False):
                                dpg.add_text("Gamma Options")
                                dpg.add_slider_float(label="Gamma", tag="gamma_slider",
                                                     min_value=0.1, max_value=5.0, default_value=1.0)
                                
                            # Opcje dla Bernsena (Binarize)
                            with dpg.group(tag="binarize_options", show=False):
                                dpg.add_text("Bernsen Method Settings")
                                dpg.add_slider_int(label="Contrast Sensitivity", tag="binarize_slider",
                                                   min_value=0, max_value=50, default_value=15)
                                
                            # TODO: add more options groups here for other transformations
                                
                            dpg.add_spacer(height=4)
                            dpg.add_button(label="Add to Pipeline", callback=add_pixel_transform)

                        # Filters tab
                        with dpg.tab(label="Filters"):
                            dpg.add_combo(
                                # TODO: add more options
                                ["None", "Blur"],   
                                label="Filter", default_value="None",
                                tag="filter_combo", callback=on_filter_change, width=200
                            )

                            # groups of options for specific transformations, only one shown at a time based on selection
                            with dpg.group(tag="blur_options", show=False):
                                dpg.add_text("Blur Options")
                                dpg.add_slider_int(label="Kernel Size", tag="blur_kernel",
                                                   min_value=1, max_value=21, default_value=3)
                            
                            # TODO: add more options for other filters here

                            dpg.add_spacer(height=4)
                            dpg.add_button(label="Add to Pipeline", callback=add_filter)

                    dpg.add_spacer(height=10)
                    dpg.add_separator()

                    # Pipeline display
                    dpg.add_text("Pipeline (applied top to bottom)")
                    dpg.add_listbox(["(empty)"], tag="pipeline_listbox",
                                    width=-1, num_items=4)
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="Undo Last", callback=undo)
                        dpg.add_button(label="Reset All", callback=reset_pipeline)

                    dpg.add_spacer(height=6)
                    dpg.add_separator()
                    dpg.add_spacer(height=6)

                    # Histogram
                    dpg.add_text("Gray-scale Histogram")
                    with dpg.plot(label="", height=200, width=-1, tag="hist_plot"):
                        dpg.add_plot_axis(dpg.mvXAxis, label="Pixel Value")
                        dpg.add_plot_axis(dpg.mvYAxis, label="Count", tag="histogram_y_axis")
                        dpg.add_bar_series([], [], tag="hist_series",
                                           parent="histogram_y_axis", weight=1.0)
                        
                    # Create themes first
                    with dpg.theme() as red_theme:
                        with dpg.theme_component(dpg.mvBarSeries):
                            dpg.add_theme_color(dpg.mvPlotCol_Fill, [255, 50, 50, 200], category=dpg.mvThemeCat_Plots)
                            dpg.add_theme_color(dpg.mvPlotCol_Line, [255, 50, 50, 255], category=dpg.mvThemeCat_Plots)

                    with dpg.theme() as green_theme:
                        with dpg.theme_component(dpg.mvBarSeries):
                            dpg.add_theme_color(dpg.mvPlotCol_Fill, [50, 255, 50, 200], category=dpg.mvThemeCat_Plots)
                            dpg.add_theme_color(dpg.mvPlotCol_Line, [50, 255, 50, 255], category=dpg.mvThemeCat_Plots)

                    with dpg.theme() as blue_theme:
                        with dpg.theme_component(dpg.mvBarSeries):
                            dpg.add_theme_color(dpg.mvPlotCol_Fill, [50, 100, 255, 200], category=dpg.mvThemeCat_Plots)
                            dpg.add_theme_color(dpg.mvPlotCol_Line, [50, 100, 255, 255], category=dpg.mvThemeCat_Plots)
                                        # Plot
                    dpg.add_text("RGB Histograms")
                    with dpg.plot(label="", height=200, width=-1, tag="hist_plot_rgb"):
                        dpg.add_plot_axis(dpg.mvXAxis, label="Pixel Value")
                        dpg.add_plot_axis(dpg.mvYAxis, label="Count", tag="histogram_rgb_y_axis")

                        dpg.add_bar_series([], [], tag="hist_series_red", parent="histogram_rgb_y_axis", weight=1.0)
                        dpg.add_bar_series([], [], tag="hist_series_green", parent="histogram_rgb_y_axis", weight=1.0)
                        dpg.add_bar_series([], [], tag="hist_series_blue", parent="histogram_rgb_y_axis", weight=1.0)

                    # Bind themes
                    dpg.bind_item_theme("hist_series_red", red_theme)
                    dpg.bind_item_theme("hist_series_green", green_theme)
                    dpg.bind_item_theme("hist_series_blue", blue_theme)


    update_histogram(current_image)
    update_rgb_histogram(current_image)

    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("main_window", True)
    dpg.start_dearpygui()
    dpg.destroy_context()


if __name__ == "__main__":
    main()
