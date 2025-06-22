import platform
import subprocess
import logging
import pyperclip # type: ignore
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# --- Clipboard Functions ---
def get_clipboard_text() -> Dict[str, Any]:
    """Gets text content from the system clipboard."""
    try:
        text = pyperclip.paste()
        logger.info("Retrieved text from clipboard.")
        return {"text": text, "message": "Clipboard content retrieved."}
    except pyperclip.PyperclipException as e:
        logger.error(f"Error getting clipboard text: {e}", exc_info=True)
        # This can happen if no copy/paste mechanism is found (e.g. headless server)
        return {"error": f"Could not access clipboard: {e}. Ensure a copy/paste mechanism (e.g., xclip or xsel on Linux) is installed."}
    except Exception as e:
        logger.error(f"Unexpected error getting clipboard text: {e}", exc_info=True)
        return {"error": f"An unexpected error occurred while getting clipboard text: {e}"}

def set_clipboard_text(text: str) -> Dict[str, Any]:
    """Sets text content to the system clipboard."""
    try:
        pyperclip.copy(text)
        logger.info("Set text to clipboard.")
        return {"message": "Text copied to clipboard."}
    except pyperclip.PyperclipException as e:
        logger.error(f"Error setting clipboard text: {e}", exc_info=True)
        return {"error": f"Could not access clipboard: {e}. Ensure a copy/paste mechanism (e.g., xclip or xsel on Linux) is installed."}
    except Exception as e:
        logger.error(f"Unexpected error setting clipboard text: {e}", exc_info=True)
        return {"error": f"An unexpected error occurred while setting clipboard text: {e}"}

# --- System Power Functions ---
def lock_windows_session() -> Dict[str, Any]:
    """Locks the Windows user session."""
    system = platform.system()
    if system == "Windows":
        try:
            import ctypes
            ctypes.windll.user32.LockWorkStation()
            logger.info("Windows session lock command issued.")
            return {"message": "Windows session lock command issued."}
        except Exception as e:
            logger.error(f"Error locking Windows session: {e}", exc_info=True)
            return {"error": f"Failed to lock Windows session: {e}"}
    else:
        logger.warning(f"Lock session is not implemented for {system}.")
        return {"error": f"Lock session functionality is not implemented for {system}."}

def shutdown_windows_system(mode: str = 'shutdown', force: bool = False, delay_seconds: int = 0) -> Dict[str, Any]:
    """
    Shuts down, restarts, or logs off the system.
    Primarily targets Windows, with basic support for Linux/macOS via 'shutdown' command.

    Args:
        mode: 'shutdown', 'restart', or 'logoff'. Defaults to 'shutdown'.
        force: If True, forces operations (e.g., closes apps without saving). Defaults to False.
        delay_seconds: Delay in seconds before the operation. Defaults to 0.
    """
    system = platform.system()
    logger.info(f"Initiating system power command: mode='{mode}', force={force}, delay={delay_seconds}s on {system}")

    if system == "Windows":
        # Example: shutdown /s /t 60 /f
        # /s = shutdown, /r = restart, /l = logoff, /f = force, /t xxx = time in seconds
        option = ""
        if mode == "shutdown": option = "/s"
        elif mode == "restart": option = "/r"
        elif mode == "logoff": option = "/l"
        else: return {"error": "Invalid mode. Choose 'shutdown', 'restart', or 'logoff'."}

        cmd = ["shutdown", option]
        if force: cmd.append("/f")
        if delay_seconds > 0: cmd.extend(["/t", str(delay_seconds)])

        try:
            subprocess.run(cmd, check=True, shell=False)
            logger.info(f"Windows system command '{' '.join(cmd)}' executed successfully.")
            return {"message": f"Windows system {mode} command executed. Delay: {delay_seconds}s, Force: {force}."}
        except subprocess.CalledProcessError as e:
            logger.error(f"Error executing Windows shutdown command: {e}", exc_info=True)
            return {"error": f"Failed to execute Windows {mode} command: {e}"}
        except FileNotFoundError:
            logger.error("Windows 'shutdown' command not found.")
            return {"error": "Windows 'shutdown' command not found."}
        except Exception as e:
            logger.error(f"Unexpected error during Windows system power operation: {e}", exc_info=True)
            return {"error": f"Unexpected error during Windows {mode}: {e}"}

    elif system in ["Linux", "Darwin"]: # macOS also uses 'shutdown'
        # Linux/macOS: shutdown -h now (halt/poweroff), shutdown -r now (reboot)
        # Logoff is more complex (desktop environment specific) - not implemented here.
        # Delay uses +minutes for shutdown command. Force is often implicit or via other signals.

        actual_mode_arg = ""
        if mode == "shutdown": actual_mode_arg = "-h" # Halt/Power off
        elif mode == "restart": actual_mode_arg = "-r" # Reboot
        else:
            logger.warning(f"Mode '{mode}' not directly supported for {system} via simple shutdown command. Logoff is DE-specific.")
            return {"error": f"Mode '{mode}' not directly supported for {system} via this tool."}

        delay_arg = "now"
        if delay_seconds > 0:
            delay_minutes = (delay_seconds + 59) // 60 # Convert seconds to minutes, rounding up
            delay_arg = f"+{delay_minutes}"

        cmd = ["sudo", "shutdown", actual_mode_arg, delay_arg]
        if force and system == "Linux": # Force for Linux might mean sending different signals or using -P for poweroff
            # The `shutdown` command itself handles forcing closure of apps, but `sudo` is key.
            # A true "force without any delay" might involve `shutdown -P now` on some Linux.
            # For simplicity, `sudo` is the main "force" mechanism here.
            logger.info("On Linux/Darwin, 'force' typically means running with sudo. Ensure sudo privileges if needed.")

        try:
            logger.info(f"Attempting to execute: {' '.join(cmd)}. This may require sudo privileges.")
            # Note: This will likely fail in environments where sudo requires a password prompt
            # or if the agent doesn't have sudo rights.
            process = subprocess.Popen(cmd, shell=False)
            # We don't wait for Popen as shutdown detaches.
            logger.info(f"{system} system command '{' '.join(cmd)}' initiated.")
            return {"message": f"{system} system {mode} command initiated. Delay: {delay_arg}. Sudo may be required."}
        except Exception as e:
            logger.error(f"Error executing {system} shutdown command: {e}", exc_info=True)
            return {"error": f"Failed to execute {system} {mode} command: {e}. Sudo privileges might be required or command might differ."}
    else:
        return {"error": f"System power controls not implemented for {system}."}


# --- Volume Control Functions ---
# These are OS-specific. pycaw for Windows. Others need different libs.
def get_system_volume() -> Dict[str, Any]:
    """Gets the current master system volume (0-100). Windows only for now."""
    system = platform.system()
    if system == "Windows":
        try:
            from comtypes import CLSCTX_ALL # type: ignore
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume # type: ignore

            devices = AudioUtilities.GetSpeakers()
            if not devices:
                return {"error": "No audio output device (speakers) found."}
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            volume_control = interface.QueryInterface(IAudioEndpointVolume) # type: ignore
            current_volume_scalar = volume_control.GetMasterVolumeLevelScalar()
            current_volume_percent = int(current_volume_scalar * 100)
            logger.info(f"Retrieved system volume: {current_volume_percent}%")
            return {"volume_percent": current_volume_percent, "message": "System volume retrieved."}
        except ImportError:
            logger.error("pycaw or comtypes not installed. Cannot get volume on Windows.")
            return {"error": "Volume control library (pycaw/comtypes) not installed for Windows."}
        except Exception as e:
            logger.error(f"Error getting system volume on Windows: {e}", exc_info=True)
            return {"error": f"Failed to get system volume on Windows: {e}"}
    else:
        logger.warning(f"Get system volume not implemented for {system}.")
        return {"error": f"Get system volume not implemented for {system}."}

def set_system_volume(level: int) -> Dict[str, Any]:
    """Sets the master system volume (0-100). Windows only for now."""
    system = platform.system()
    if not (0 <= level <= 100):
        return {"error": "Volume level must be between 0 and 100."}

    if system == "Windows":
        try:
            from comtypes import CLSCTX_ALL # type: ignore
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume # type: ignore

            devices = AudioUtilities.GetSpeakers()
            if not devices:
                return {"error": "No audio output device (speakers) found."}
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            volume_control = interface.QueryInterface(IAudioEndpointVolume) # type: ignore

            volume_scalar = float(level / 100.0)
            volume_control.SetMasterVolumeLevelScalar(volume_scalar, None)
            logger.info(f"Set system volume to {level}%.")
            return {"message": f"System volume set to {level}%."}
        except ImportError:
            logger.error("pycaw or comtypes not installed. Cannot set volume on Windows.")
            return {"error": "Volume control library (pycaw/comtypes) not installed for Windows."}
        except Exception as e:
            logger.error(f"Error setting system volume on Windows: {e}", exc_info=True)
            return {"error": f"Failed to set system volume on Windows: {e}"}
    else:
        logger.warning(f"Set system volume not implemented for {system}.")
        return {"error": f"Set system volume not implemented for {system}."}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
    logger.info("System Control Module Example")

    # --- Clipboard Tests ---
    print("\n--- Testing Clipboard ---")
    original_clipboard_content = get_clipboard_text().get("text", "")
    print(f"Original clipboard content (first 50 chars): '{original_clipboard_content[:50]}...'")

    test_text = "Hello from system_control.py test! " + __file__
    print(f"Setting clipboard to: '{test_text}'")
    set_result = set_clipboard_text(test_text)
    print(set_result)
    if "error" not in set_result:
        retrieved_text_result = get_clipboard_text()
        print(f"Retrieved from clipboard: {retrieved_text_result}")
        assert retrieved_text_result.get("text") == test_text, "Clipboard set/get mismatch!"
        # Restore original clipboard content if possible (pyperclip might hold it if it was short)
        # For robustness, it's better if the user manually manages clipboard state if this test alters it undesirably.
        # For now, we'll just log it. If original_clipboard_content was substantial, this won't restore it fully.
        # pyperclip.copy(original_clipboard_content) # Attempt to restore
        # print(f"Attempted to restore original clipboard content.")
    else:
        print("Skipping detailed clipboard get test due to set error.")


    # --- Volume Tests (Windows Only) ---
    print("\n--- Testing Volume Control (Windows Only) ---")
    if platform.system() == "Windows":
        try:
            initial_volume_result = get_system_volume()
            print(f"Initial volume: {initial_volume_result}")
            if "volume_percent" in initial_volume_result:
                initial_volume = initial_volume_result["volume_percent"]

                test_volume_level = 50 # Set to a known value
                if initial_volume == test_volume_level: # if already 50, try 40
                    test_volume_level = 40 if initial_volume > 10 else 60

                print(f"Setting volume to {test_volume_level}%...")
                set_vol_result = set_system_volume(test_volume_level)
                print(set_vol_result)
                if "error" not in set_vol_result:
                    time.sleep(0.5) # Give a moment for system to apply
                    current_vol_result = get_system_volume()
                    print(f"Volume after setting: {current_vol_result}")
                    assert current_vol_result.get("volume_percent") == test_volume_level, "Volume set/get mismatch!"

                    # Restore initial volume
                    print(f"Restoring volume to {initial_volume}%...")
                    set_system_volume(initial_volume)
                    time.sleep(0.5)
                    final_vol_check = get_system_volume().get("volume_percent")
                    print(f"Volume after restoration attempt: {final_vol_check}% (should be close to {initial_volume}%)")

            else:
                print("Could not get initial volume to perform full test.")
        except Exception as e:
            print(f"Could not run volume tests (pycaw might be missing or error): {e}")
    else:
        print("Skipping Windows-specific volume tests on this platform.")
        # Test that it returns the correct error for non-Windows
        vol_result_get = get_system_volume()
        assert "error" in vol_result_get and "not implemented" in vol_result_get["error"], "get_system_volume should error on non-Windows"
        vol_result_set = set_system_volume(50)
        assert "error" in vol_result_set and "not implemented" in vol_result_set["error"], "set_system_volume should error on non-Windows"


    # --- System Power Tests (Informational, not executed by default) ---
    # These are dangerous to automate fully.
    print("\n--- System Power Operations (Informational) ---")
    print("lock_windows_session(): Locks the Windows session. (Not automatically tested)")
    print("shutdown_windows_system(mode='shutdown', delay_seconds=60): Initiates shutdown. (Not automatically tested)")
    print("shutdown_windows_system(mode='restart'): Initiates restart. (Not automatically tested)")

    if platform.system() == "Windows":
        # lock_result = lock_windows_session() # Uncomment to test manually
        # print(f"Manual lock test result: {lock_result}")
        pass
    else:
        lock_result_other_os = lock_windows_session()
        assert "error" in lock_result_other_os and "not implemented" in lock_result_other_os["error"], "lock_windows_session should error on non-Windows"

        # Test shutdown command generation for Linux/Darwin (doesn't actually run it with sudo here)
        # shutdown_result_other_os = shutdown_windows_system(mode='shutdown')
        # print(f"Example shutdown command generation for Linux/Darwin: {shutdown_result_other_os}")
        # assert "error" not in shutdown_result_other_os or "sudo" in shutdown_result_other_os.get("message","").lower() , "shutdown for Linux/Darwin should mention sudo or succeed if sudo not needed"


    logger.info("System Control module example finished.")
    # Need to import time for sleep in volume test
    import time
