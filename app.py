import dearpygui.dearpygui as dpg
from ui.layout import build_ui

def main():
    dpg.create_context()
    dpg.create_viewport(title="Biometrics App", width=1100, height=750)
    dpg.setup_dearpygui()    
    dpg.show_viewport()      
    build_ui()               
    dpg.set_primary_window("main_window", True)
    dpg.start_dearpygui()
    dpg.destroy_context()

if __name__ == "__main__":
    main()