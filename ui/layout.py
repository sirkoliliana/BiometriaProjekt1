from PIL import Image

import dearpygui.dearpygui as dpg
import numpy as np
import state
from ui.charts import numpy_to_dpg
from ui.callbacks import (add_filter, add_pixel_transform, undo, add_morph_operation,
                           reset_pipeline, open_callback, save_callback, 
                           cancel_callback, on_filter_change, on_pt_change, on_morph_change)


def build_texture_registry(initial_texture):
    with dpg.texture_registry():
        dpg.add_dynamic_texture(state.IMG_W, state.IMG_H, initial_texture, tag="image_texture")

def build_file_dialogs():
    img_filter = "Image Files (*.png *.jpg *.bmp){.png,.PNG,.jpg,.JPG,.jpeg,.JPEG,.bmp,.BMP}"

    with dpg.file_dialog(show=False, callback=open_callback,
                         cancel_callback=cancel_callback,
                         tag="open_dialog_id", width=700, height=400):
        dpg.add_file_extension(img_filter)

    with dpg.file_dialog(show=False, callback=save_callback,
                         cancel_callback=cancel_callback,
                         tag="save_dialog_id", width=700, height=400):
        dpg.add_file_extension(img_filter)

def build_toolbar():
    with dpg.group(horizontal=True):
        dpg.add_button(label="Open File", callback=lambda: dpg.show_item("open_dialog_id"))
        dpg.add_button(label="Save",      callback=lambda: dpg.show_item("save_dialog_id"))
        dpg.add_button(label="Undo",      callback=undo)
        dpg.add_button(label="Reset",     callback=reset_pipeline)

def build_pixel_transforms_tab():
    with dpg.tab(label="Pixel Transformations"):
        dpg.add_combo([
            "None",
            "Grayscale",
            "Monochrome",
            "Brightness", 
            "Gamma", 
            "Binarize simple", 
            "Binarize Bernsen",
            "Invert",
            "Add Images",
            "Blend Image",
            "Histogram Equalization (Gray)",
            "Histogram Equalization (RGB)",
            "Contrast Stretching (Gray)",
            "Contrast Stretching (RGB)"], tag="pt_combo", default_value="None", callback=on_pt_change, width=200)
        

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
            
        with dpg.group(tag="add_image_options", show=False):
            dpg.add_text("Add Image Options")
            dpg.add_button(label="Select Image to Add", callback=lambda: dpg.show_item("open_dialog_id_add"))
            with dpg.file_dialog(show=False, callback=lambda s, a: state.added_images.append(np.array(Image.open(a["file_path_name"]).convert("RGB").resize((state.IMG_W, state.IMG_H)))),
                                    cancel_callback=cancel_callback,
                                    tag="open_dialog_id_add", width=700, height=400):
                dpg.add_file_extension(".png")
                dpg.add_file_extension(".jpg")
                dpg.add_file_extension(".bmp")

            dpg.add_checkbox(label="Subtract", tag="subtract_image_checkbox")

        with dpg.group(tag="blend_image_options", show=False):
            dpg.add_text("Blend Image Options")
            dpg.add_button(label="Select Image to Blend", callback=lambda: dpg.show_item("open_dialog_id_blend"))
            with dpg.file_dialog(show=False, callback=lambda s, a: state.added_images.append(np.array(Image.open(a["file_path_name"]).convert("RGB").resize((state.IMG_W, state.IMG_H)))),
                                    cancel_callback=cancel_callback,
                                    tag="open_dialog_id_blend", width=700, height=400):
                dpg.add_file_extension(".png")
                dpg.add_file_extension(".jpg")
                dpg.add_file_extension(".bmp")

            dpg.add_slider_float(label="Alpha", tag="blend_alpha_slider",
                                    min_value=0.0, max_value=1.0, default_value=0.5)


            
        dpg.add_spacer(height=4)
        dpg.add_button(label="Add to Pipeline", callback=add_pixel_transform)
        dpg.add_spacer(height=10)
        dpg.add_separator()


def build_filters_tab():
    with dpg.tab(label="Filters"):
        dpg.add_combo(
            ["None", "Custom Kernel", "Blur (Averaging)", "Gaussian Blur", 
             "Sharpen", "Sobel Edge Detection", "Robert's Cross", "Emboss"],
            label="Filter",
            default_value="None",
            tag="filter_combo",
            callback=on_filter_change,
            width=200
        )

        with dpg.group(tag="blur_options", show=False):
            dpg.add_text("Blur Options")
            dpg.add_slider_int(label="Kernel Size", tag="blur_kernel",
                               min_value=1, max_value=21, default_value=3)

        with dpg.group(tag="sobel_options", show=False):
            dpg.add_text("Sobel Edge Detection")
            dpg.add_combo(
                items=[0, 45, 90, 135, 180, 225, 270, 315],
                label="Angle (degrees)",
                tag="sobel_angle_combo",
                default_value=0
            )

        with dpg.group(tag="roberts_cross_options", show=False):
            dpg.add_text("Robert's Cross")
            dpg.add_combo(
                items=["orthogonal", "diagonal"],
                label="Orientation",
                tag="roberts_cross_orientation",
                default_value="orthogonal"
            )

        with dpg.group(tag="custom_kernel_options", show=False):
            dpg.add_text("Enter 5x5 Kernel Matrix:")
            for i in range(5):
                with dpg.group(horizontal=True):
                    for j in range(5):
                        default = 1 if (i == 2 and j == 2) else 0  # tylko środek = 1
                        dpg.add_input_float(
                            tag=f"k{i}{j}",
                            width=60,
                            step=0,
                            default_value=default,
                            format="%.1f"
                        )

        with dpg.group(tag="emboss_options", show=False):
            dpg.add_text("Emboss fade Options")
            dpg.add_slider_float(label="Emboss", tag="emboss_slider",
                                        min_value=0.1, max_value=5.0, default_value=1.0)

        dpg.add_spacer(height=4)
        dpg.add_button(label="Add to Pipeline", callback=add_filter)
        dpg.add_spacer(height=10)
        dpg.add_separator()


# Operacje morfologiczne
def build_morphological_tab():
    with dpg.tab(label="Morphological Operations"):
        dpg.add_combo(
            ["None", "Erosion", "Dilation", "Opening", "Closing"],
            label="Operation",
            default_value="None",
            tag="morph_combo",
            callback=on_morph_change,
            width=200
        )

        with dpg.group(tag="morph_options", show=False):
            dpg.add_text("Morphological Options")

            dpg.add_slider_int(
                label="Kernel Size",
                tag="morph_kernel",
                min_value=1,
                max_value=21,
                default_value=3
            )

            dpg.add_combo(
                ["square", "cross", "circle"],
                label="Kernel Shape",
                tag="morph_shape",
                default_value="square",
                width=200
            )

        dpg.add_button(label="Add to Pipeline", callback=add_morph_operation)


def build_histogram_panel():
    with dpg.group(horizontal=False, tag="histogram_group", show=True):
        dpg.add_text("Gray-scale Histogram")
        with dpg.plot(label="", height=200, width=-1, tag="hist_plot"):
            dpg.add_plot_axis(dpg.mvXAxis, label="Pixel Value", tag="histogram_x_axis") 
            dpg.add_plot_axis(dpg.mvYAxis, label="Count", tag="histogram_y_axis")
            dpg.add_bar_series([], [], tag="hist_series", parent="histogram_y_axis", weight=1.0)
        
            
        # Create themes first
        with dpg.theme() as red_theme:
            with dpg.theme_component(dpg.mvBarSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Fill, [255, 50, 50, 100], category=dpg.mvThemeCat_Plots)
                dpg.add_theme_color(dpg.mvPlotCol_Line, [255, 50, 50, 100], category=dpg.mvThemeCat_Plots)

        with dpg.theme() as green_theme:
            with dpg.theme_component(dpg.mvBarSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Fill, [50, 255, 50, 100], category=dpg.mvThemeCat_Plots)
                dpg.add_theme_color(dpg.mvPlotCol_Line, [50, 255, 50, 100], category=dpg.mvThemeCat_Plots)

        with dpg.theme() as blue_theme:
            with dpg.theme_component(dpg.mvBarSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Fill, [50, 100, 255, 100], category=dpg.mvThemeCat_Plots)
                dpg.add_theme_color(dpg.mvPlotCol_Line, [50, 100, 255, 100], category=dpg.mvThemeCat_Plots)
        # Plot
        dpg.add_text("RGB Histograms")
        with dpg.plot(label="", height=200, width=-1, tag="hist_plot_rgb"):
            dpg.add_plot_axis(dpg.mvXAxis, label="Pixel Value", tag="histogram_rgb_x_axis") 
            dpg.add_plot_axis(dpg.mvYAxis, label="Count", tag="histogram_rgb_y_axis")

            dpg.add_bar_series([], [], tag="hist_series_red", parent="histogram_rgb_y_axis", weight=1.0)
            dpg.add_bar_series([], [], tag="hist_series_green", parent="histogram_rgb_y_axis", weight=1.0)
            dpg.add_bar_series([], [], tag="hist_series_blue", parent="histogram_rgb_y_axis", weight=1.0)

        dpg.bind_item_theme("hist_series_red", red_theme)
        dpg.bind_item_theme("hist_series_green", green_theme)
        dpg.bind_item_theme("hist_series_blue", blue_theme)


def build_projection_panel():
    with dpg.group(horizontal=False, tag="projection_group", show=False):
        dpg.add_text("Horizontal Projection")
        with dpg.plot(label="", height=150, width=-1, tag="horizontal_projection_plot"):
            dpg.add_plot_axis(dpg.mvXAxis, label="Column Index")
            dpg.add_plot_axis(dpg.mvYAxis, label="Sum of Pixel Values", tag="horizontal_projection_y_axis")
            dpg.add_bar_series([], [], tag="horizontal_projection_series", parent="horizontal_projection_y_axis")

        dpg.add_text("Vertical Projection")
        with dpg.plot(label="", height=150, width=-1, tag="vertical_projection_plot"):
            dpg.add_plot_axis(dpg.mvXAxis, label="Row Index")
            dpg.add_plot_axis(dpg.mvYAxis, label="Sum of Pixel Values", tag="vertical_projection_y_axis")
            dpg.add_bar_series([], [], tag="vertical_projection_series", parent="vertical_projection_y_axis")


def build_pipeline_display():
    dpg.add_text("Pipeline (applied top to bottom)")
    dpg.add_listbox(["(empty)"], tag="pipeline_listbox", width=-1, num_items=4)
    with dpg.group(horizontal=True):
        dpg.add_button(label="Undo Last", callback=undo)
        dpg.add_button(label="Reset All", callback=reset_pipeline)

def build_ui():
    build_texture_registry(numpy_to_dpg(state.current_image)) 
    build_file_dialogs()

    with dpg.window(tag="main_window"):
        build_toolbar()
        dpg.add_separator()

        with dpg.table(header_row=False, borders_innerV=True):
            dpg.add_table_column(init_width_or_weight=420, width_fixed=True)
            dpg.add_table_column()

            with dpg.table_row():
                with dpg.table_cell():   # left: image preview
                    dpg.add_text("Preview")
                    dpg.add_image("image_texture")
                    
                with dpg.table_cell():   # right: controls + pipeline + charts
                    with dpg.tab_bar():
                        build_pixel_transforms_tab()
                        build_filters_tab()
                        build_morphological_tab()

                    build_pipeline_display()
                    build_histogram_panel()
                    build_projection_panel()