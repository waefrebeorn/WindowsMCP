import subprocess
import logging
import psutil # type: ignore
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

def start_application(application_path_or_name: str, arguments: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Starts an application.

    Args:
        application_path_or_name: The path to the executable or the name of the command.
        arguments: A list of string arguments to pass to the application.

    Returns:
        A dictionary with "pid" and "message" on success, or "error".
    """
    try:
        cmd = [application_path_or_name]
        if arguments:
            cmd.extend(arguments)

        # For Windows, using shell=True can sometimes be helpful for finding executables in PATH
        # or running shell commands directly, but it has security implications if the input is not trusted.
        # For now, let's try without shell=True for better security and cross-platform behavior.
        # If it's a GUI application, Popen is non-blocking.
        # For CLI apps where we might want to wait, use subprocess.run, but for "starting" an app, Popen is fine.
        process = subprocess.Popen(cmd, shell=False)
        logger.info(f"Attempted to start application '{application_path_or_name}' with args {arguments}. PID: {process.pid}")
        return {"pid": process.pid, "message": f"Application '{application_path_or_name}' started with PID {process.pid}."}
    except FileNotFoundError:
        logger.error(f"Application not found: {application_path_or_name}")
        return {"error": f"Application not found: {application_path_or_name}"}
    except PermissionError:
        logger.error(f"Permission denied to start application: {application_path_or_name}")
        return {"error": f"Permission denied to start application: {application_path_or_name}"}
    except Exception as e:
        logger.error(f"Error starting application '{application_path_or_name}': {e}", exc_info=True)
        return {"error": f"An unexpected error occurred while starting application: {e}"}

def get_running_processes() -> Dict[str, Any]:
    """
    Lists currently running processes.

    Returns:
        A dictionary with "processes" list on success, or "error".
        Each process in the list is a dictionary with "pid", "name", "username", "status", "cpu_percent", "memory_percent".
    """
    processes_info = []
    try:
        for proc in psutil.process_iter(['pid', 'name', 'username', 'status', 'cpu_percent', 'memory_percent']):
            try:
                processes_info.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                # Process might have terminated, or we might not have access
                pass
        logger.info(f"Retrieved {len(processes_info)} running processes.")
        return {"processes": processes_info}
    except Exception as e:
        logger.error(f"Error retrieving running processes: {e}", exc_info=True)
        return {"error": f"An unexpected error occurred while retrieving processes: {e}"}

def _find_processes_by_title(title_substring: str) -> List[psutil.Process]:
    """Helper to find processes whose main window title contains the substring (case-insensitive)."""
    # This is tricky as psutil doesn't directly give window titles.
    # This would typically require platform-specific window enumeration (like window_manager.py uses).
    # For a simpler initial approach for this module, we might have to rely on process name
    # or ask the user to provide a PID if they know it.
    # Let's make a note that closing by title effectively needs window_manager integration.
    # For now, this function will be a placeholder or rely on what psutil can offer.

    # A more robust way would be to iterate through windows using PyGetWindow/PyWinCtl (like in window_manager.py)
    # and then map window handles to PIDs.
    # This is a simplified placeholder.
    procs_found = []
    for p in psutil.process_iter(['pid', 'name']):
        try:
            # This is not ideal, as process name is not window title.
            # True window title matching requires OS-specific APIs.
            if title_substring.lower() in p.info['name'].lower():
                procs_found.append(p)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    if not procs_found:
        logger.info(f"No processes found with name containing '{title_substring}' for title-based closing.")
    return procs_found


def close_application_by_pid(pid: int, force: bool = False) -> Dict[str, Any]:
    """
    Closes an application by its Process ID (PID).

    Args:
        pid: The PID of the process to terminate.
        force: If True, forcefully terminate (kill) the process.
               If False, try to terminate gracefully. Defaults to False.

    Returns:
        A dictionary with "pid", "message" on success, or "error".
    """
    try:
        if not psutil.pid_exists(pid):
            return {"error": f"Process with PID {pid} not found."}

        process = psutil.Process(pid)
        process_name = process.name()

        if force:
            process.kill()
            logger.info(f"Forcefully killed process '{process_name}' (PID: {pid}).")
            return {"pid": pid, "message": f"Process '{process_name}' (PID: {pid}) forcefully killed."}
        else:
            process.terminate() # Sends SIGTERM (Unix) or TerminateProcess (Windows)
            logger.info(f"Attempted graceful termination of process '{process_name}' (PID: {pid}).")
            # Check if process is still running after a short delay
            try:
                process.wait(timeout=3) # Wait for 3 seconds for process to terminate
                logger.info(f"Process '{process_name}' (PID: {pid}) terminated gracefully.")
                return {"pid": pid, "message": f"Process '{process_name}' (PID: {pid}) gracefully terminated."}
            except psutil.TimeoutExpired:
                logger.warning(f"Process '{process_name}' (PID: {pid}) did not terminate gracefully after 3s. May require force close.")
                return {"pid": pid, "message": f"Graceful termination signal sent to '{process_name}' (PID: {pid}). It may still be running. Consider force close if needed.", "warning": "Process may still be running"}

    except psutil.NoSuchProcess:
        return {"error": f"Process with PID {pid} not found."}
    except psutil.AccessDenied:
        logger.error(f"Permission denied to terminate process with PID {pid}.")
        return {"error": f"Permission denied to terminate process with PID {pid}."}
    except Exception as e:
        logger.error(f"Error closing application with PID {pid}: {e}", exc_info=True)
        return {"error": f"An unexpected error occurred while closing application with PID {pid}: {e}"}

def close_application_by_title(window_title_substring: str, force: bool = False) -> Dict[str, Any]:
    """
    Closes an application by its window title (substring match, case-insensitive).
    This is a convenience function that attempts to find the PID via window title
    and then calls close_application_by_pid.
    Note: This relies on window_manager functionality to map title to PID accurately.

    Args:
        window_title_substring: A substring of the window title to search for.
        force: If True, forcefully terminate the process. Defaults to False.

    Returns:
        A dictionary with results of the close operation or an error if no matching window found.
    """
    from . import window_manager # Local import to use its capabilities

    try:
        # Use window_manager to find windows and their PIDs
        all_windows = window_manager.list_windows_with_details() # Assuming this returns list of dicts with 'title' and 'pid'

        matched_pids = set()
        for win_details in all_windows:
            if window_title_substring.lower() in win_details.get("title", "").lower():
                pid = win_details.get("pid")
                if pid is not None:
                    matched_pids.add(pid)

        if not matched_pids:
            return {"error": f"No open window found with title containing '{window_title_substring}'."}

        results = []
        closed_any = False
        for pid_to_close in list(matched_pids): # Iterate over a copy
            logger.info(f"Attempting to close PID {pid_to_close} found for title '{window_title_substring}'.")
            close_result = close_application_by_pid(pid_to_close, force)
            results.append(close_result)
            if "error" not in close_result:
                closed_any = True

        if len(results) == 1:
            return results[0]
        else:
            # Aggregate results if multiple PIDs were targeted
            if closed_any:
                 return {"message": f"Attempted to close {len(matched_pids)} processes matching title '{window_title_substring}'. Check individual results.", "details": results}
            else:
                 # All attempts resulted in errors
                 return {"error": f"Failed to close any processes for title '{window_title_substring}'.", "details": results}


    except ImportError:
        logger.error("window_manager module could not be imported for close_application_by_title.")
        return {"error": "Internal error: window_manager module unavailable."}
    except Exception as e:
        logger.error(f"Error in close_application_by_title for '{window_title_substring}': {e}", exc_info=True)
        return {"error": f"An unexpected error occurred while trying to close by title: {e}"}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
    logger.info("App Manager Module Example")

    # --- Test get_running_processes ---
    print("\n--- Testing get_running_processes ---")
    procs_result = get_running_processes()
    if "error" in procs_result:
        print(f"Error: {procs_result['error']}")
    elif "processes" in procs_result:
        print(f"Found {len(procs_result['processes'])} processes. First 5:")
        for p_info in procs_result['processes'][:5]:
            print(f"  PID: {p_info['pid']}, Name: {p_info['name']}, User: {p_info['username']}, Status: {p_info['status']}")

    # --- Test start_application ---
    # Note: Starting GUI apps in a headless CI environment might not be meaningful or possible.
    # This example attempts to start 'notepad' or 'gedit' which are common simple text editors.
    # On Windows, 'notepad' should work. On Linux, 'gedit' or 'xed' or 'kate' might be available.
    # Using a simple cross-platform CLI command that detaches:
    app_to_start = ""
    import platform
    if platform.system() == "Windows":
        app_to_start = "notepad.exe" # Simple GUI app
        # app_to_start = "timeout" # CLI app
        # app_args = ["5"]
    elif platform.system() == "Linux":
        # Try common simple text editors, or a simple command
        for editor in ["gedit", "xed", "kate", "mousepad", "leafpad"]:
            if subprocess.call(['which', editor], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
                app_to_start = editor
                break
        if not app_to_start: # Fallback to a simple command if no known editor found
            app_to_start = "sleep" # CLI app
            app_args = ["5"] # Sleep for 5 seconds
        else:
            app_args = []

    elif platform.system() == "Darwin": # macOS
        app_to_start = "TextEdit" # This needs to be launched via 'open -a TextEdit'
        # A direct path might be /System/Applications/TextEdit.app/Contents/MacOS/TextEdit
        # For simplicity, let's use a command that should exist
        app_to_start = "sleep"
        app_args = ["5"]

    test_pid = None
    if app_to_start:
        print(f"\n--- Testing start_application ({app_to_start}) ---")
        start_result = start_application(app_to_start, app_args if 'app_args' in locals() else None)
        print(start_result)
        if "pid" in start_result:
            test_pid = start_result["pid"]
            print(f"Application '{app_to_start}' started with PID: {test_pid}")

            # --- Test close_application_by_pid ---
            if test_pid:
                print(f"\n--- Testing close_application_by_pid (PID: {test_pid}) ---")
                # Wait a moment for the app to fully start, especially GUI apps
                import time
                time.sleep(2)

                close_pid_result = close_application_by_pid(test_pid, force=False) # Try graceful first
                print(f"Graceful close result: {close_pid_result}")

                # Check if it's still alive (psutil.pid_exists might be true for a moment for zombie)
                time.sleep(1) # Give it a sec to process termination
                if psutil.pid_exists(test_pid):
                    try:
                        p = psutil.Process(test_pid)
                        if p.status() != psutil.STATUS_ZOMBIE: # Only force kill if not already zombie/gone
                            print(f"Process {test_pid} still exists. Trying force close.")
                            close_pid_force_result = close_application_by_pid(test_pid, force=True)
                            print(f"Force close result: {close_pid_force_result}")
                        else:
                             print(f"Process {test_pid} is a zombie or already terminated.")
                    except psutil.NoSuchProcess:
                         print(f"Process {test_pid} no longer exists after graceful attempt.")
                else:
                    print(f"Process {test_pid} successfully terminated gracefully or was short-lived.")
        else:
            print(f"Could not start '{app_to_start}' to test closing.")
    else:
        print("\nSkipping start/close tests as no suitable application/command was identified for the current OS.")

    # --- Test close_application_by_title ---
    # This is harder to test reliably without knowing what windows are open.
    # We'll try to close a common one if the start_application worked and opened a known window.
    # For 'notepad.exe', the title is often 'Untitled - Notepad' or 'filename - Notepad'
    # For 'gedit', it's often 'Unsaved Document 1 - gedit'
    # This test is highly dependent on the environment.
    print(f"\n--- Testing close_application_by_title ---")
    # Example: Try to close a Notepad or Gedit window if one was opened by 'start_application'
    # This part is very heuristic and might not work reliably in all environments or if the app wasn't started.
    # A more robust test would involve starting an app with a known title, then closing it.
    # For now, this is a placeholder for manual testing or more complex setup.
    title_to_try_close = "Notepad" if platform.system() == "Windows" else "gedit" # A common part of title
    # if test_pid and app_to_start in ["notepad.exe", "gedit"]: # Only if we think we started a relevant app
    #     print(f"Attempting to close windows with title containing '{title_to_try_close}' (heuristic test)")
    #     # It might take a moment for the window title to register
    #     import time; time.sleep(3)
    #     close_title_result = close_application_by_title(title_to_try_close, force=False)
    #     print(f"Close by title result for '{title_to_try_close}': {close_title_result}")
    # else:
    #     print(f"Skipping close_by_title test for '{title_to_try_close}' as the target app might not have been started by this script or title is unknown.")
    print("Note: close_application_by_title is difficult to test automatically without a predictable window. Manual testing recommended or a more controlled setup.")

    logger.info("App Manager module example finished.")
