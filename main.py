# WuBu Main Application Entry Point
import sys
import os
import signal # For handling Ctrl+C

# Ensure the 'src' directory is in the Python path
# This allows importing 'wubu' module components correctly
current_dir = os.path.dirname(os.path.abspath(__file__))
# If main.py is at project root, src_dir is 'current_dir/src'
# However, wubu packages expect project root to be in path for sibling 'desktop_tools'
# The WubuEngine already adds project_root to sys.path if needed for desktop_tools
# For clarity, ensure 'src' is discoverable if needed, or rely on PYTHONPATH.
# If 'src' is not automatically in path, this might be needed:
# src_path = os.path.join(current_dir, "src")
# if src_path not in sys.path:
#    sys.path.insert(0, src_path)
# But if main.py is in root, 'import src.wubu...' or 'from src.wubu...' should work if src is a package.
# Let's assume 'wubu' can be imported directly if main.py is in the same root as 'src' dir.
# For robustness, let's add project root to path for consistency with how engine handles desktop_tools.
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)


try:
    from src.wubu.core.engine import WuBuEngine
    from src.wubu.ui.wubu_ui import WubuApp
    from src.wubu.utils.resource_loader import load_config
    import customtkinter as ctk
except ImportError as e:
    print("ERROR: Failed to import WuBu components.")
    print(f"Ensure that WuBu is installed correctly, PYTHONPATH is set up, or run from project root.")
    print(f"Import error details: {e}")
    sys.exit(1)

# Global engine instance to allow signal handler to access it
_wubu_engine_instance = None

def signal_handler(sig, frame):
    print("\nCtrl+C received. Shutting down WuBu...")
    if _wubu_engine_instance:
        _wubu_engine_instance.shutdown() # Call engine's shutdown
    # Forcing exit if UI doesn't close gracefully, though engine.shutdown() should handle UI.
    sys.exit(0)

def run_wubu_application():
    global _wubu_engine_instance

    print("Starting WuBu Application...")

    # 1. Load Configuration
    # load_config() by default looks for "wubu_config.yaml" or "config.yaml"
    # It's important that this config file exists and is correctly formatted.
    # The resource_loader.py calculates project root based on its own location.
    # If main.py is in root, it should find 'wubu_config.yaml' in root.
    app_config = load_config() # Uses default "wubu_config.yaml"

    if not app_config:
        print("CRITICAL: WuBu configuration 'wubu_config.yaml' not found or failed to load.")
        print("Please ensure 'wubu_config.yaml' exists in the project root (same directory as this main.py).")
        print("You can copy and customize 'src/wubu/config_template.yaml'.")
        # Fallback to a very minimal config to allow UI to show an error, maybe.
        # Or just exit. For now, exit if no config.
        sys.exit(1)

    # Pass the detected project root to the engine config if needed by ContextProvider
    # The engine itself also calculates a project_root, but this ensures consistency.
    app_config['project_root_dir'] = current_dir # Main.py is in project root

    # 2. Initialize WuBu Core Engine
    try:
        _wubu_engine_instance = WuBuEngine(config=app_config)
    except Exception as e:
        print(f"CRITICAL: Failed to initialize WuBuEngine: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 3. Initialize UI (WubuApp)
    # Set appearance and theme before creating the app window
    ctk.set_appearance_mode(app_config.get('ui_theme', {}).get('appearance_mode', "System")) # Modes: "System", "Dark", "Light"
    ctk.set_default_color_theme(app_config.get('ui_theme', {}).get('color_theme', "blue"))  # Themes: "blue", "green", "dark-blue"

    try:
        wubu_app_ui = WubuApp(engine=_wubu_engine_instance)
    except Exception as e:
        print(f"CRITICAL: Failed to initialize WubuApp (UI): {e}")
        import traceback
        traceback.print_exc()
        # Try to shutdown engine before exiting if UI fails
        if _wubu_engine_instance:
            _wubu_engine_instance.shutdown()
        sys.exit(1)

    # 4. Link Engine and UI
    _wubu_engine_instance.set_ui(wubu_app_ui)

    # 5. Setup signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)

    # 6. Start the UI main loop
    print(f"{app_config.get('wubu_name', 'WuBu')} GUI starting. Close window or press Ctrl+C in console to exit.")
    try:
        wubu_app_ui.mainloop()
    except KeyboardInterrupt: # Should be caught by signal_handler, but as a fallback
        print("KeyboardInterrupt in mainloop. Shutting down...")
    finally:
        # Ensure shutdown is called if mainloop exits for any reason other than SIGINT
        if _wubu_engine_instance and not getattr(_wubu_engine_instance, '_shutting_down_flag', False): # Avoid double shutdown
             _wubu_engine_instance.shutdown()


if __name__ == "__main__":
    # This structure assumes that if you run `python main.py` from the project root,
    # the imports for `src.wubu.*` will work because `src` is a package in the root.
    # If `src` itself is intended to be the root for imports (e.g. `import wubu.core...`),
    # then main.py might live inside `src` or PYTHONPATH needs `src`.
    # Given `from ..utils` in engine, `src` is part of the package path.
    run_wubu_application()
