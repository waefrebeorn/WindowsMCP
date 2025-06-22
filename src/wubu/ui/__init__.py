# WuBu UI Package
#
# This package is responsible for the graphical user interface (GUI) of WuBu.
# It uses CustomTkinter as the primary UI framework.

# Key components:
# - wubu_ui.py: Contains the main application class (WubuApp) and UI layout.

# To run the UI (example, assuming main.py or similar calls it):
# from wubu.ui.wubu_ui import main as run_ui
# if __name__ == "__main__":
# run_ui()

__all__ = ["WubuApp", "main"]

from .wubu_ui import WubuApp, main
