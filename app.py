import dearpygui.dearpygui as dpg
from ui.layout import build_ui

def main():
    dpg.create_context()
    dpg.create_viewport(title="Biometrics App", width=1100, height=750)
    dpg.setup_dearpygui()      # ← move this before build_ui
    dpg.show_viewport()        # ← and this
    build_ui()                 # ← build UI after viewport is ready
    dpg.set_primary_window("main_window", True)
    dpg.start_dearpygui()
    dpg.destroy_context()

if __name__ == "__main__":
    main()