# Helper functions for loading resources like sound files, images, etc.
# This is a placeholder and needs actual implementation based on how resources are packaged and deployed.

import os
import sys
import yaml # Assuming YAML for config files, add to requirements if not already there

# Determine base path for resources, accounting for PyInstaller's temporary folder if bundled
def get_base_path():
    """Get the base path for resources, compatible with PyInstaller."""
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        # Running in a PyInstaller bundle
        return sys._MEIPASS
    else:
        # Running in a normal Python environment
        # This file is src/wubu/utils/resource_loader.py
        # Project root is three levels up from this file's directory.
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


BASE_RESOURCE_PATH = get_base_path() # This should be the project root

def load_config(config_filename="wubu_config.yaml"):
    """Loads a YAML configuration file from the project root."""
    config_path = os.path.join(BASE_RESOURCE_PATH, config_filename)
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found.")
        # Try looking in src/ as a fallback for dev, though root is preferred
        config_path_src = os.path.join(BASE_RESOURCE_PATH, "src", config_filename)
        try:
            with open(config_path_src, 'r') as f:
                print(f"Note: Found config in '{config_path_src}' as a fallback.")
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Error: Configuration file also not found in '{config_path_src}'.")
            return None
        except yaml.YAMLError as e:
            print(f"Error parsing YAML configuration file '{config_path_src}': {e}")
            return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML configuration file '{config_path}': {e}")
        return None

def get_resource_path(resource_type_relative_to_wubu_package, filename):
    """
    Constructs the full path to a resource file within the 'wubu' package.
    'resource_type_relative_to_wubu_package' is a sub-path within the 'src/wubu' directory.
    Example: get_resource_path('tts/glados_tts_models', 'model.onnx')
             -> /path/to/project/src/wubu/tts/glados_tts_models/model.onnx
    """
    # wubu_package_root_path is 'src/wubu/'
    wubu_package_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    full_path = os.path.join(wubu_package_root_path, resource_type_relative_to_wubu_package, filename)

    # For PyInstaller, resources might be bundled differently.
    # If running bundled, sys._MEIPASS is the temp folder. We might need to adjust.
    # However, if get_base_path() correctly points to _MEIPASS when frozen,
    # and if our package structure is preserved relative to that, this might still work
    # or require paths relative to sys._MEIPASS directly for packaged data.
    # The current `get_base_path` returns project root for non-frozen, _MEIPASS for frozen.
    # `wubu_package_root_path` is calculated relative to __file__, which in _MEIPASS
    # will be something like _MEIPASS/src/wubu/utils. So this should be fine.

    if not os.path.exists(full_path):
        # This warning can be noisy if files are optional or loaded by other means
        # print(f"Warning: Resource not found at expected primary path: '{full_path}'")
        # Fallback: try looking in a general 'assets' directory at the project root
        # This is less ideal for packaged components but can be a fallback.
        # Project root is BASE_RESOURCE_PATH
        assets_path = os.path.join(BASE_RESOURCE_PATH, "assets", resource_type_relative_to_wubu_package, filename)
        if os.path.exists(assets_path):
            # print(f"Note: Found resource in fallback assets path: '{assets_path}'")
            return assets_path
        # else: # No need for another warning here, just return the primary path
            # print(f"Warning: Resource also not found in fallback assets path: '{assets_path}'")
    return full_path # Return the primary expected path, let caller handle existence check if critical


def load_sound(filename, sound_category="general_sounds"):
    """
    Placeholder for loading a sound file.
    `sound_category` is a subdirectory like 'effects', 'ui_feedback', etc.
    Expected location: src/wubu/sounds/<sound_category>/<filename>
    """
    sound_path = get_resource_path(os.path.join("sounds", sound_category), filename)
    print(f"Attempting to load sound: {sound_path}")
    if not os.path.exists(sound_path):
         print(f"Warning: Sound file not found at {sound_path}, but returning path anyway.")
    return sound_path

def load_image(filename, image_category="ui_icons"):
    """
    Placeholder for loading an image file.
    `image_category` is a subdirectory like 'logos', 'backgrounds', etc.
    Expected location: src/wubu/images/<image_category>/<filename>
    """
    image_path = get_resource_path(os.path.join("images", image_category), filename)
    print(f"Attempting to load image: {image_path}")
    if not os.path.exists(image_path):
        print(f"Warning: Image file not found at {image_path}, but returning path anyway.")
    return image_path

if __name__ == '__main__':
    print(f"Project Root (calculated by get_base_path()): {get_base_path()}")

    # Create a dummy config at project root for testing if it doesn't exist
    # This is where load_config will primarily look for it.
    dummy_config_filename = "wubu_config.yaml"
    dummy_config_path_at_root = os.path.join(get_base_path(), dummy_config_filename)

    if not os.path.exists(dummy_config_path_at_root):
        os.makedirs(os.path.dirname(dummy_config_path_at_root), exist_ok=True) # Ensure dir exists if root is ""
        with open(dummy_config_path_at_root, 'w') as f:
            yaml.dump({"test_key": "test_value_at_root", "wubu_name": "WuBu Test System (Root Config)"}, f)
        print(f"Created dummy config for testing: {dummy_config_path_at_root}")

    config = load_config(dummy_config_filename) # Uses "wubu_config.yaml" by default
    if config:
        print(f"Loaded config: {config}")
        if config.get("wubu_name"):
            print(f"WuBu Name from config: {config['wubu_name']}")
    else:
        print("Failed to load config for __main__ test.")

    # Test resource path resolution
    # These paths will point within src/wubu/...
    # Example: src/wubu/tts/glados_tts_models/wubu_coqui.pth
    print(f"Path to a TTS model (example): {get_resource_path('tts/glados_tts_models', 'wubu_coqui.pth')}")

    # Example: src/wubu/sounds/effects/startup.wav
    print(f"Path to a sound (example): {load_sound('startup.wav', sound_category='effects')}")

    # Example: src/wubu/images/ui_icons/icon.png
    print(f"Path to an image (example): {load_image('icon.png', image_category='ui_icons')}")

    # Note: The dummy config file is not removed automatically to allow inspection after test run.
    # Consider removing it if this script were part of an automated test suite.
    # if os.path.exists(dummy_config_path_at_root):
    # os.remove(dummy_config_path_at_root)
    # print(f"Removed dummy config: {dummy_config_path_at_root}")
    pass
