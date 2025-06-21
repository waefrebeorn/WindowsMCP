import json
from pathlib import Path
import logging

# --- Rich Console (for printing notices during config load) ---
# We need a minimal way to print if rich isn't fully configured yet,
# or use logging.
# For simplicity, using print for initial config load messages if
# console isn't passed in.
# Alternatively, could pass the rich console object to load_or_create_config.
# For now, using standard print for critical config messages,
# assuming logger is set up later.
logger = logging.getLogger(__name__)  # Use logger for less intrusive messages

# --- Configuration Loading ---
CONFIG_FILE_NAME = "config.json"
# Assuming config_manager.py is in the same directory as main.py
# (project root). If it's in a subdirectory, this ROOT_DIR needs to
# point to the actual project root.
ROOT_DIR = Path(__file__).resolve().parent

DEFAULT_CONFIG = {
    "GEMINI_MODEL_NAME": "gemini-1.5-flash-latest",
    "GEMINI_API_KEY": None,  # Encouraging use of .env for this
    "HISTORY_FILE_PATH": str(Path.home() / ".roblox_agent_history"),
    "OLLAMA_API_URL": "http://localhost:11434",
    # Default model for Ollama, chosen for tool calling:
    "OLLAMA_DEFAULT_MODEL": "qwen2.5-coder:7b-instruct-q4_K_M",
    "LLM_PROVIDER": "gemini",  # Can be "gemini" or "ollama"
    "MOONDREAM_API_URL": "",  # API endpoint for Moondream v2
    "SCREENSHOT_SAVE_PATH": "",  # Path to save screenshots, if empty, don't save
    # Enable/disable PyAutoGUI's failsafe (corner mouse move):
    "PYAUTOGUI_FAILSAFE_ENABLED": True,
    # Default pause in seconds after each PyAutoGUI action:
    "PYAUTOGUI_PAUSE_PER_ACTION": 0.1,
    # Default duration for voice recording in seconds:
    "VOICE_RECORDING_DURATION": 5,
    # Default Whisper model (e.g., "tiny", "base", "small"):
    "WHISPER_MODEL_NAME": "base",
    "ENABLE_TTS": True,  # Enable or disable Text-to-Speech output
    "TESSERACT_CMD_PATH": None,  # Optional: Path to tesseract executable
    "OCR_CONFIDENCE_THRESHOLD": 40,  # Default OCR confidence threshold (0-100)
    "INPUT_AUDIO_DEVICE_NAME_OR_ID": None,  # Optional: Specify input audio device for recording
    "TTS_VOICE_ID": None,  # Optional: Specify TTS voice ID (system dependent)
    "TTS_RATE": 150,  # TTS speaking rate
    "TTS_VOLUME": 0.9,  # TTS volume (0.0 to 1.0)
}


def load_or_create_config(r_console=None) -> dict:  # Optionally pass rich console
    """Loads configuration from JSON file or creates it with default values."""
    config_path = ROOT_DIR / CONFIG_FILE_NAME
    current_config = {}

    def _print_panel(message: str, title: str, style: str = "default"):
        if r_console:
            from rich.panel import (
                Panel,
            )  # Local import to avoid circular dep if console_ui uses this

            r_console.print(Panel(message, title=title, border_style=style))
        else:
            print(f"[{title.upper()}] {message}")

    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                loaded_config = json.load(f)

            current_config = DEFAULT_CONFIG.copy()
            current_config.update(loaded_config)  # Loaded values override defaults

            # Check if any keys from DEFAULT_CONFIG were missing in loaded_config,
            # or if new keys in DEFAULT_CONFIG are not in current_config (after update)
            # This indicates that the file should be updated with new/missing defaults.
            config_needs_update = False
            for key, default_value in DEFAULT_CONFIG.items():
                if key not in loaded_config:
                    current_config[key] = default_value  # Ensure new defaults are added
                    config_needs_update = True

            if config_needs_update:
                try:
                    with open(config_path, "w") as f:
                        json.dump(current_config, f, indent=4)
                    msg = (
                        f"Configuration file '{CONFIG_FILE_NAME}' updated "
                        f"with new default values where necessary. Please review."
                    )
                    _print_panel(msg, "[yellow]Config Notice[/yellow]", style="yellow")
                except IOError as e:
                    msg = (
                        f"Error: Could not write updates to "
                        f"'{CONFIG_FILE_NAME}': {e}. Please check permissions."
                    )
                    _print_panel(msg, "[red]Config Error[/red]", style="red")

            logger.info(f"Configuration loaded from {CONFIG_FILE_NAME}")
            return current_config
        except json.JSONDecodeError:
            msg = (
                f"Error: Configuration file '{CONFIG_FILE_NAME}' is malformed. "
                f"Using default configuration. Please fix or delete it to regenerate."
            )
            _print_panel(msg, "[red]Config Error[/red]", style="red")
            return DEFAULT_CONFIG.copy()
        except IOError as e:
            msg = (
                f"Error: Could not read configuration file "
                f"'{CONFIG_FILE_NAME}': {e}. Using default configuration."
            )
            _print_panel(msg, "[red]Config Error[/red]", style="red")
            return DEFAULT_CONFIG.copy()
    else:  # Config file does not exist
        try:
            with open(config_path, "w") as f:
                json.dump(DEFAULT_CONFIG, f, indent=4)
            msg = (
                f"Default configuration file '{CONFIG_FILE_NAME}' created. "
                f"Please review it, especially for 'GEMINI_API_KEY' "
                f"(if using Gemini), 'OLLAMA_API_URL', "
                f"'OLLAMA_DEFAULT_MODEL' (if using Ollama)."
                # f"and 'RBX_MCP_SERVER_PATH'." # Old key, removed for brevity
            )
            _print_panel(msg, "[green]Config Notice[/green]", style="green")
            logger.info(f"Default configuration file {CONFIG_FILE_NAME} created.")
            return DEFAULT_CONFIG.copy()
        except IOError as e:
            msg = (
                f"Error: Could not create default configuration file "
                f"'{CONFIG_FILE_NAME}': {e}. Using default configuration. "
                f"Please check permissions."
            )
            _print_panel(msg, "[red]Config Error[/red]", style="red")
            return DEFAULT_CONFIG.copy()


config = load_or_create_config()

if __name__ == "__main__":
    # Example of how to use it, assuming rich console is available for this test
    from rich.console import Console as RichConsoleForTest

    test_console = RichConsoleForTest()

    print(f"ROOT_DIR in config_manager.py: {ROOT_DIR}")
    # Create a dummy config for testing
    dummy_config_path = ROOT_DIR / "test_config.json"
    if dummy_config_path.exists():
        dummy_config_path.unlink()

    # Test creation
    cfg = load_or_create_config(r_console=test_console)
    print("\nInitial config loaded/created:")
    print(json.dumps(cfg, indent=4))

    # Test update - simulate an old config file
    if dummy_config_path.exists():  # Should have been created as config.json
        dummy_config_path.unlink()  # Clean up if it was named differently

    # Create a config that's missing a key
    old_config_data = DEFAULT_CONFIG.copy()
    del old_config_data["HISTORY_FILE_PATH"]  # Remove a key
    old_config_data["GEMINI_MODEL_NAME"] = "old-model-name"  # Change a value

    config_path_for_test = ROOT_DIR / CONFIG_FILE_NAME
    with open(config_path_for_test, "w") as f:
        json.dump(old_config_data, f, indent=4)

    print(f"\nTesting update with modified '{CONFIG_FILE_NAME}':")
    cfg_updated = load_or_create_config(r_console=test_console)
    print("\nUpdated config:")
    print(json.dumps(cfg_updated, indent=4))

    assert (
        cfg_updated["HISTORY_FILE_PATH"] == DEFAULT_CONFIG["HISTORY_FILE_PATH"]
    )  # Should be restored
    assert cfg_updated["GEMINI_MODEL_NAME"] == "old-model-name"  # Should be preserved

    # Clean up the test config file
    if config_path_for_test.exists():
        config_path_for_test.unlink()
    print(f"\nTest completed. Cleaned up '{CONFIG_FILE_NAME}'.")
