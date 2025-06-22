# WuBu API Server (e.g., using Flask)
# This provides an HTTP interface to interact with WuBu.

from flask import Flask, request, jsonify
import os
import sys

# Adjust imports based on project structure
# Assuming this file is src/wubu/api/app.py
# And project root is parent of src/
# This allows running `python -m wubu.api.app` from project root
try:
    from wubu.core.engine import WuBuEngine
    from wubu.utils.resource_loader import load_config
    from wubu.ui.wubu_ui import WuBuUI # For a simple message sink if needed by engine
except ImportError:
    # Fallback for some execution contexts, adjust if necessary
    # This indicates a potential PYTHONPATH issue or how the script/module is run
    print("ERROR in wubu.api.app: Could not import WuBu modules. Ensure PYTHONPATH is set or run as a module.")
    # Add 'src' to path if running 'python src/wubu/api/app.py' from project root
    # This is a common workaround but proper packaging or PYTHONPATH is better.
    # current_dir = os.path.dirname(os.path.abspath(__file__)) # .../src/wubu/api
    # wubu_dir = os.path.dirname(current_dir) # .../src/wubu
    # src_dir = os.path.dirname(wubu_dir) # .../src
    # project_root = os.path.dirname(src_dir) # .../
    # if src_dir not in sys.path:
    #    sys.path.insert(0, src_dir)
    # if project_root not in sys.path: # If config is at root
    #    sys.path.insert(0, project_root)
    # from wubu.core.engine import WuBuEngine
    # from wubu.utils.resource_loader import load_config
    # from wubu.ui.wubu_ui import WuBuUI
    raise # Re-raise to make the problem visible

# --- Global WuBu Engine Instance ---
# For simplicity in this initial implementation.
# A Flask app factory with application context is better for complex apps.
wubu_engine_instance: WuBuEngine | None = None
api_ui_sink: WuBuUI | None = None # A simple UI sink for the engine if it needs one

def initialize_wubu_engine():
    """Initializes the WuBu engine if not already done."""
    global wubu_engine_instance, api_ui_sink
    if wubu_engine_instance is None:
        print("WuBu API: Initializing WuBu Engine for API routes...")
        # Config file is relative to project root. load_config handles finding it.
        config_data = load_config("wubu_config.yaml")
        if not config_data:
            # Try a default path if not found by load_config's primary logic
            # This could happen if CWD is not project root during `flask run`
            # For now, rely on load_config's existing search (root, then src/)
            print("WuBu API FATAL: Could not load wubu_config.yaml. WuBu Engine not started.")
            # In a real app, might raise an error or prevent Flask from starting properly.
            return False # Indicate failure

        try:
            wubu_engine_instance = WuBuEngine(config=config_data)
            # The engine might need a UI handler, even for a headless API, to process its messages.
            # This UI won't be "displayed" but can catch TTS_OUTPUT, STATUS_UPDATE, ERROR from engine.
            api_ui_sink = WuBuUI(wubu_core_engine=wubu_engine_instance) # Create UI
            wubu_engine_instance.set_ui(api_ui_sink) # Link to engine
            # api_ui_sink.start() # Start its message processing loop if it has one
            print("WuBu API: WuBu Engine initialized and linked with API UI sink.")
            return True
        except Exception as e:
            print(f"WuBu API FATAL: Error initializing WuBuEngine: {e}")
            # import traceback; traceback.print_exc()
            wubu_engine_instance = None # Ensure it's None on failure
            return False
    return True # Already initialized


def create_app(testing_mode=False):
    """Factory function to create the Flask application for WuBu API."""
    app = Flask(__name__)
    app.config['TESTING'] = testing_mode

    if not testing_mode: # Don't init engine during test collection / if tests mock it
        if not initialize_wubu_engine():
            print("WuBu API Warning: WuBu Engine failed to initialize. API might not function correctly.")
            # Depending on severity, could raise an error here to stop app creation.
            # For now, allow app to start but endpoints will likely fail if engine is None.

    @app.route('/health', methods=['GET'])
    def health_check():
        """Basic health check endpoint for the WuBu API."""
        engine_status = "Initialized" if wubu_engine_instance else "Not Initialized"
        return jsonify({
            "status": "WuBu API is running",
            "version": "0.1.0",
            "wubu_engine_status": engine_status
        }), 200

    @app.route('/command', methods=['POST'])
    def process_command_route():
        """
        Endpoint to send a text command to WuBu.
        Expects JSON: {"text_command": "your command here"}
        """
        global wubu_engine_instance
        if not wubu_engine_instance:
            return jsonify({"error": "WuBu Engine not initialized or failed to load. Cannot process command."}), 503 # Service Unavailable

        data = request.get_json()
        if not data or 'text_command' not in data:
            return jsonify({"error": "Missing 'text_command' in JSON payload"}), 400

        command = data['text_command']
        print(f"WuBu API: Received command via /command route: '{command}'")

        try:
            # WuBuEngine's process_text_command typically handles TTS internally.
            # The API's role is to trigger this and confirm receipt.
            # For long-running commands, async handling would be needed.
            wubu_engine_instance.process_text_command(command)

            # The actual spoken/processed response comes via WuBu's UI/TTS.
            # This API endpoint just confirms the command was passed to the engine.
            return jsonify({
                "status": "Command sent to WuBu for processing.",
                "command_received": command
            }), 200 # OK
        except Exception as e:
            print(f"WuBu API Error: Error processing command '{command}': {e}")
            # import traceback; traceback.print_exc()
            return jsonify({"error": "Failed to process command in WuBu", "details": str(e)}), 500

    # TODO for WuBu API:
    # - Add /speak endpoint: {"text_to_speak": "...", "voice_id": "..."} (would call wubu_engine_instance.speak())
    # - Add /status endpoint to get more detailed WuBu system status from engine.
    # - Consider authentication.
    # - Implement asynchronous task handling for commands that take time.
    # - Proper logging.

    return app

if __name__ == '__main__':
    # This allows running the Flask dev server directly using `python src/wubu/api/app.py`
    # Make sure PYTHONPATH includes the project root or `src/` for wubu.* imports.
    # Best run from project root as `python -m wubu.api.app`

    # Attempt to initialize engine here for standalone run, if not done by create_app yet
    # This is mainly for `python src/wubu/api/app.py` execution.
    # If using `flask run`, Flask might handle app creation differently.
    if not wubu_engine_instance:
        print("Running WuBu API standalone, attempting engine initialization...")
        if not initialize_wubu_engine():
            print("WuBu API FATAL: Standalone engine initialization failed. Exiting.")
            sys.exit(1) # Exit if engine can't start

    app = create_app()
    print("Starting WuBu Flask API server...")
    # Port 5600 for WuBu API (GLaDOS was 5500)
    app.run(host='0.0.0.0', port=5600, debug=True) # debug=True for development

    # Cleanup engine on exit if it was started
    if wubu_engine_instance:
        print("WuBu API shutting down, requesting engine shutdown...")
        wubu_engine_instance.shutdown()
        if api_ui_sink and hasattr(api_ui_sink, 'is_running') and api_ui_sink.is_running:
            api_ui_sink.stop()
