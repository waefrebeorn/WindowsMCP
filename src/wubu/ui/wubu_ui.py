# Main WuBu User Interface
# This is a placeholder for the actual UI implementation.
# It could be a text-based console UI, a web UI (using Flask/Streamlit), or a desktop GUI (PyQt/CustomTkinter).

import threading
import queue
import time

class WuBuUI:
    """
    WuBu User Interface base class / placeholder.
    Manages interaction with the user, displaying WuBu's speech,
    and potentially capturing user input (text or voice commands).
    """
    def __init__(self, wubu_core_engine):
        self.core_engine = wubu_core_engine  # Reference to the main WuBu core
        self.message_queue = queue.Queue()    # For receiving messages/updates from the core
        self.input_queue = queue.Queue()      # For sending user input to the core
        self.is_running = False
        self.ui_thread = None
        print("WuBuUI Initialized (Placeholder Console UI)")

    def start(self):
        """Starts the UI processing loop in a separate thread."""
        if self.is_running:
            print("WuBuUI is already running.")
            return

        self.is_running = True
        self.ui_thread = threading.Thread(target=self._ui_loop, daemon=True)
        self.ui_thread.start()
        print("WuBuUI thread started.")

    def stop(self):
        """Stops the UI processing loop."""
        if not self.is_running:
            print("WuBuUI is not running.")
            return

        self.is_running = False
        self.message_queue.put(("QUIT", None))
        if self.ui_thread and self.ui_thread.is_alive():
            self.ui_thread.join(timeout=2)
        print("WuBuUI stopped.")

    def display_message(self, message_type: str, content: any):
        """
        Sends a message to the UI to be displayed. Called by WuBu's core engine.
        :param message_type: Type of message (e.g., "TTS_OUTPUT", "STATUS_UPDATE", "ERROR")
        :param content: The actual message content.
        """
        if not self.is_running:
            print(f"UI_FALLBACK ({message_type}): {content}") # Fallback if UI not running
            return
        self.message_queue.put((message_type, content))

    def get_user_input(self) -> str | None:
        """
        Retrieves user input from the UI (non-blocking). Called by WuBu core.
        :return: User input string, or None if no new input.
        """
        try:
            return self.input_queue.get_nowait()
        except queue.Empty:
            return None

    def _process_message(self, message_type, content):
        """Internal method to handle messages from the queue and update UI (console for now)."""
        if message_type == "TTS_OUTPUT":
            print(f"WuBu Says: \"{content}\"")
        elif message_type == "STATUS_UPDATE":
            print(f"WuBu Status: {content}")
        elif message_type == "ERROR":
            print(f"WuBu Error: {content}")
        elif message_type == "USER_PROMPT":
            print(f"WuBu Asks: {content}")
        else:
            print(f"WuBu UI ({message_type}): {content}")

    def _ui_loop(self):
        """Main UI processing loop (placeholder console version)."""
        print("WuBu Console UI Loop started. (This is a passive display loop; input is pushed to input_queue externally).")
        while self.is_running:
            try:
                message_type, content = self.message_queue.get(timeout=0.1)
                if message_type == "QUIT":
                    print("WuBu UI loop received QUIT signal.")
                    break
                self._process_message(message_type, content)
            except queue.Empty:
                pass
            except Exception as e:
                print(f"Error in WuBu UI message processing: {e}")
            # This loop primarily processes messages from WuBu core.
            # User input is expected to be put onto self.input_queue by another mechanism
            # (e.g., a main CLI loop calling input() and then self.input_queue.put()).
        print("WuBu Console UI Loop ended.")

if __name__ == '__main__':
    print("Testing WuBuUI (Placeholder Console Version)")

    class MockWuBuCore:
        def __init__(self):
            self.ui = None
            print("MockWuBuCore initialized.")

        def process_command_from_ui(self, command):
            print(f"MockCore: Received command from UI: '{command}'")
            if command.lower() == "status":
                if self.ui: self.ui.display_message("STATUS_UPDATE", "All systems nominal for WuBu.")
            elif command.lower() == "speak":
                if self.ui: self.ui.display_message("TTS_OUTPUT", "I am WuBu, functioning within normal parameters.")
            elif command.lower().startswith("error"):
                if self.ui: self.ui.display_message("ERROR", "WuBu reports a simulated error.")
            else:
                if self.ui: self.ui.display_message("USER_PROMPT", f"WuBu's mock core did not fully understand '{command}'. Try 'status', 'speak', 'error'.")

    mock_core_instance = MockWuBuCore()
    ui_instance = WuBuUI(wubu_core_engine=mock_core_instance)
    mock_core_instance.ui = ui_instance

    ui_instance.start()

    ui_instance.display_message("STATUS_UPDATE", "WuBu Core systems starting up...")
    time.sleep(0.5)
    ui_instance.display_message("TTS_OUTPUT", "Welcome to the WuBu testing facility.")
    time.sleep(0.5)
    ui_instance.display_message("USER_PROMPT", "Awaiting your command for WuBu:")

    def mock_core_input_poller_loop(core_ref, ui_ref):
        print("\nMockCoreInputPoller: Starting to poll WuBuUI for input...")
        poll_count = 0
        max_polls = 15 # Poll for a few seconds worth of checks
        while ui_ref.is_running and poll_count < max_polls:
            user_cmd = ui_ref.get_user_input()
            if user_cmd:
                print(f"MockCoreInputPoller: Got input '{user_cmd}', sending to core.")
                core_ref.process_command_from_ui(user_cmd)
            time.sleep(0.2) # Poll every 200ms
            poll_count +=1
        print("MockCoreInputPoller: Stopped polling.")

    # Simulate user inputs being pushed to the queue (as if by a separate input handler)
    print("\nSimulating user inputs being pushed to WuBuUI's input_queue:")
    ui_instance.input_queue.put("status")
    time.sleep(0.1) # give a small gap
    ui_instance.input_queue.put("speak")
    time.sleep(0.1)
    ui_instance.input_queue.put("error test from main")
    time.sleep(0.1)
    ui_instance.input_queue.put("unknown command for wubu")

    input_poller_thread = threading.Thread(target=mock_core_input_poller_loop, args=(mock_core_instance, ui_instance), daemon=True)
    input_poller_thread.start()

    try:
        # Keep main test alive for a bit to see interactions
        # Waits for the input poller to likely finish its limited polls
        time.sleep(max_polls * 0.2 + 1) # Wait slightly longer than poller's max run time
    except KeyboardInterrupt:
        print("\nWuBuUI Test interrupted by user.")
    finally:
        print("Stopping WuBuUI from test...")
        ui_instance.stop()
        if input_poller_thread.is_alive():
            input_poller_thread.join(timeout=1)
        print("WuBuUI test finished.")
