import tkinter as tk
from tkinter import scrolledtext
import threading
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def display_code_in_popup(code_content: str, language: str = 'python', window_title: str = "Code Viewer") -> Dict[str, Any]:
    """
    Displays the given code content in a new, simple, read-only Tkinter pop-up window.
    Runs the Tkinter mainloop in a separate thread to avoid blocking.

    Args:
        code_content: The string containing the code to display.
        language: The language of the code (currently not used for syntax highlighting).
        window_title: The title for the pop-up window.

    Returns:
        A dictionary indicating success or error.
    """

    def create_editor_window():
        try:
            root = tk.Tk()
            root.title(window_title)
            root.geometry("800x600") # Default size

            txt_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, state='normal', font=("Courier New", 10))
            txt_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

            txt_area.insert(tk.INSERT, code_content)
            txt_area.configure(state='disabled') # Make it read-only

            # Add a close button
            close_button = tk.Button(root, text="Close", command=root.destroy)
            close_button.pack(pady=5)

            root.mainloop()
            logger.info(f"Code viewer window for '{window_title}' closed.")
        except Exception as e:
            logger.error(f"Error in Tkinter thread for code editor: {e}", exc_info=True)
            # How to report this back to the main thread/caller if it's an issue?
            # For now, just logging.

    try:
        # Running Tkinter in a separate thread is often necessary when integrating
        # with other event loops (like asyncio used in main.py) or to prevent blocking.
        logger.info(f"Creating and showing code viewer window for '{window_title}' in a new thread.")
        thread = threading.Thread(target=create_editor_window, daemon=True)
        thread.start()

        # This function will return immediately after starting the thread.
        # The window will live until closed by the user or if the main app exits.
        return {"status": "success", "message": f"Code viewer window for '{window_title}' launched in a separate thread."}
    except Exception as e:
        logger.error(f"Failed to launch code viewer window thread: {e}", exc_info=True)
        return {"status": "error", "message": f"Failed to launch code viewer window: {e}"}

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

    sample_python_code = """
def hello_world():
    print("Hello, Tkinter Code Viewer!")

class MyClass:
    def __init__(self, name):
        self.name = name

    def greet(self):
        print(f"Greetings from {self.name}")

# Create an instance and call a method
if __name__ == '__main__':
    obj = MyClass("TestObject")
    obj.greet()
    hello_world()
    """

    print("Displaying sample Python code in a pop-up window...")
    result_py = display_code_in_popup(sample_python_code, language="python", window_title="Python Code Example")
    print(f"Result of displaying Python code: {result_py}")

    sample_text_content = "This is just some plain text.\nIt can span multiple lines.\n\tAnd include tabs and other characters like !@#$%^&*()."
    print("\nDisplaying sample plain text in a pop-up window...")
    result_txt = display_code_in_popup(sample_text_content, language="text", window_title="Plain Text Example")
    print(f"Result of displaying plain text: {result_txt}")

    print("\nNote: The pop-up windows are running in separate threads.")
    print("You may need to manually close them if they appeared.")
    print("The main script will exit, but the Tkinter threads might keep running if not daemonic or handled.")
    print("(The threads are set to daemonic, so they should exit when the main script finishes if it doesn't wait.)")

    # To keep the main script alive for a bit to see the windows if they are very quick to close
    # In a real application, the main event loop (e.g., asyncio in main.py) would keep running.
    # For this simple test, we might add a small sleep or an input() prompt if windows don't stay.
    # However, since the Tkinter mainloop is running in a daemon thread, this script will exit,
    # and the Tkinter windows will also close. This is acceptable for a non-blocking tool call.
    # For interactive testing, one might run this script and expect to close windows manually.

    # If you want to ensure the windows are seen before the main script exits in this test:
    # input("Press Enter to close this script (this may also close daemon Tkinter threads)...")
    # Or, if threads were not daemonic, they would need to be joined or explicitly managed.
    # Since they are daemon, they will be terminated when the main program exits.

    # If the main application (e.g. main.py using this tool) is long-running, the Tkinter windows will persist.
    # The `daemon=True` for the thread ensures that if the main app is killed or exits, these Tkinter threads won't prevent it.

    # A simple way to see them in testing this script directly:
    if result_py.get("status") == "success" or result_txt.get("status") == "success":
        print("\nTest windows launched. Manually close them to fully end this test script if they persist,")
        print("or they will auto-close when this main script finishes due to daemon threads.")
        # Forcing a wait to see the windows during test:
        # print("Waiting for 10 seconds to allow window interaction...")
        # import time
        # time.sleep(10) # Keep main thread alive to see daemon threads' windows
        # print("10 seconds up. Main script finishing.")

    # The important part for the tool is that display_code_in_popup returns quickly.
    # The window management is then up to Tkinter's mainloop in its own thread.

    # Example of how the main app might handle this (conceptual):
    # async def some_async_function_in_main_py():
    #     # ...
    #     tool_args = {"code_content": "...", "language": "python"}
    #     # This would be called via tool_dispatcher
    #     # result = await asyncio.to_thread(code_editor.display_code_in_popup, tool_args["code_content"], tool_args["language"])
    #     # print(result)
    #     # ...
    #     pass

    # No specific assertions here as GUI interaction is hard to auto-assert in this context.
    # Success is primarily that it runs without error and launches the thread.
    # Manual verification would be needed to confirm window appearance and content.

    # If any error occurred during launch:
    if result_py.get("status") == "error":
        raise RuntimeError(f"Failed to launch Python code viewer: {result_py.get('message')}")
    if result_txt.get("status") == "error":
        raise RuntimeError(f"Failed to launch plain text viewer: {result_txt.get('message')}")

    print("\nCode editor test script finished.")
