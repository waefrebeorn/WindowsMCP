# WuBu Core Engine
# This class orchestrates the main WuBu functionalities, integrating
# TTS, ASR (SpeechListener), LLM processing, and UI interactions.

import json # Added for tool call argument/result processing
from ..tts.tts_engine_manager import TTSEngineManager, ZONOS_ENGINE_ID # Import ZONOS_ENGINE_ID for use
from ..asr.speech_listener import SpeechListener # Import SpeechListener
from .llm_processor import LLMProcessor # Corrected import
from ..ui.wubu_ui import WubuApp # Actual class name for type hinting

# Adjust path if desktop_tools is a top-level package or structured differently
# Assuming desktop_tools is a sibling to the src/wubu package directory
import sys
import os
# Add project root to sys.path to allow finding desktop_tools if it's a top-level dir
# This assumes engine.py is in src/wubu/core
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from desktop_tools.tool_dispatcher import DesktopToolDispatcher, FunctionCall
    from desktop_tools.context_provider import ContextProvider
    from desktop_tools.desktop_tools_definitions import get_ollama_tools_json_schema
except ImportError:
    print("WuBuEngine: Could not import DesktopToolDispatcher or schema. Desktop tools will be unavailable.")
    print("Ensure 'desktop_tools' is in PYTHONPATH or installed correctly relative to 'wubu' module.")
    DesktopToolDispatcher = None
    get_ollama_tools_json_schema = None
    FunctionCall = None


class WuBuEngine:
    """
    The main WuBu Core Engine.
    Manages subsystems like TTS, (future ASR), LLM, and optional tools.
    Handles the main logic for processing commands and generating responses.
    """
    def __init__(self, config: dict):
        self.config = config
        self.wubu_name = config.get('wubu_name', "WuBu")
        print(f"Initializing {self.wubu_name} Core Engine...")

        self.ui: WubuApp | None = None  # UI handler will be set via set_ui()

        # 1. Initialize TTS Engine Manager
        try:
            self.tts_manager = TTSEngineManager(config=self.config.get('tts', {}))
            print(f"WuBu TTS Manager initialized. Default voice: {self.tts_manager.default_engine_id if self.tts_manager else 'N/A'}")
        except Exception as e:
            print(f"Error initializing WuBu TTS Manager: {e}. TTS might not function.")
            self.tts_manager = None

        # 2. Initialize LLM Processor
        try:
            self.llm_processor = LLMProcessor(config=self.config) # Pass full config (it expects 'llm' sub-dict)
            print("WuBu LLM Processor initialized.")
        except Exception as e:
            print(f"Error initializing WuBu LLM Processor: {e}. LLM interaction will fail.")
            self.llm_processor = None


        # 3. Initialize Tool Dispatcher
        self.tool_dispatcher = None
        self.tool_schema = None
        if DesktopToolDispatcher and get_ollama_tools_json_schema:
            if self.config.get('desktop_tools', {}).get('enabled', True): # Enable by default if key exists
                try:
                    self.tool_dispatcher = DesktopToolDispatcher()
                    # TODO: The schema might be provider-specific (Ollama vs OpenAI).
                    # For now, assume Ollama schema if Ollama is the provider.
                    if self.llm_processor and self.llm_processor.provider == 'ollama':
                        self.tool_schema = get_ollama_tools_json_schema()
                    # else: self.tool_schema = get_openai_tools_json_schema() # If OpenAI tools are different
                    print("WuBu Tool Dispatcher initialized and schema loaded.")
                except Exception as e:
                    print(f"Error initializing WuBu Tool Dispatcher or getting schema: {e}")
                    self.tool_dispatcher = None
                    self.tool_schema = None
            else:
                print("WuBu Desktop tools disabled by configuration.")
        else:
            print("WuBu DesktopToolDispatcher or schema function not available due to import error. Tools disabled.")

        self.is_processing = False
        self.conversation_history = [] # For LLM context
        self.asr_active_popup = None # To manage ASR status popup instance

        # 4. Initialize Speech Listener
        self.speech_listener = None
        # Ensure ASR config is a dictionary
        asr_config = self.config.get('asr', {})
        if not isinstance(asr_config, dict): asr_config = {}
        if asr_config.get('enabled', False): # Check if ASR is enabled in config
            try:
                self.speech_listener = SpeechListener(wubu_engine_core=self, config=asr_config)
                print("WuBu Speech Listener initialized.")
            except Exception as e:
                print(f"Error initializing WuBu Speech Listener: {e}. ASR will be unavailable.")
                self.speech_listener = None
        else:
            print("WuBu ASR (SpeechListener) disabled by configuration.")

        # 5. Initialize Context Provider
        self.context_provider = None
        # Ensure desktop_tools config is a dictionary
        desktop_tools_config = self.config.get('desktop_tools', {})
        if not isinstance(desktop_tools_config, dict): desktop_tools_config = {}
        if ContextProvider: # Check if class was imported
            project_root_for_context = self.config.get('project_root_dir', project_root) # project_root from path setup
            if desktop_tools_config.get('context_provider_enabled', True): # Default to true if desktop_tools enabled
                try:
                    self.context_provider = ContextProvider(project_root_str=str(project_root_for_context))
                    print(f"WuBu Context Provider initialized for project root: {project_root_for_context}")
                except Exception as e:
                    print(f"Error initializing WuBu Context Provider: {e}")
                    self.context_provider = None
            else:
                print("WuBu Context Provider disabled by configuration.")
        else:
            print("WuBu ContextProvider class not available due to import error.")


        print(f"{self.wubu_name} Core Engine initialized successfully.")

    def set_ui(self, ui_handler: WubuApp): # Actual class name
        self.ui = ui_handler
        print(f"UI Handler ({type(ui_handler).__name__}) set for {self.wubu_name} Engine.")
        if self.ui:
            self.ui.status_label.configure(text=f"{self.wubu_name} Ready.")


    def get_ui(self) -> WubuApp | None: # Actual class name
        return self.ui

    # --- ASR Control Methods ---
    def toggle_asr_listening(self):
        if not self.speech_listener:
            if self.ui: self.ui.display_message_popup("ASR Error", "Speech recognition is not available or not configured.", "error")
            return

        if self.speech_listener.is_listening():
            self.stop_asr_listening()
        else:
            self.start_asr_listening()

    def start_asr_listening(self):
        if self.speech_listener and not self.speech_listener.is_listening():
            print(f"{self.wubu_name} Engine: Starting ASR listening...")
            if self.ui:
                if self.asr_active_popup and self.asr_active_popup.winfo_exists():
                    self.ui.close_popup(self.asr_active_popup)
                self.asr_active_popup = self.ui.display_asr_popup("Listening...")
                self.ui.mic_button.configure(text="🛑")
            self.speech_listener.start_listening() # This is a blocking call in current SpeechListener if not threaded there.
                                                 # SpeechListener's start_listening should be non-blocking (start a thread).
        elif self.speech_listener and self.speech_listener.is_listening():
            print(f"{self.wubu_name} Engine: ASR already listening.")


    def stop_asr_listening(self):
        if self.speech_listener and self.speech_listener.is_listening():
            print(f"{self.wubu_name} Engine: Stopping ASR listening...")
            self.speech_listener.stop_listening() # This processes buffer and then sets flags.
            if self.ui:
                if self.asr_active_popup and self.asr_active_popup.winfo_exists():
                    # Update popup to show processing, then handle_asr_transcription will close or update further
                    self.ui.update_popup_text(self.asr_active_popup, "Processing speech...")
                else: # If no popup, maybe create one for "Processing..."
                    self.asr_active_popup = self.ui.display_asr_popup("Processing speech...")
                self.ui.mic_button.configure(text="🎤")
        elif self.speech_listener:
             print(f"{self.wubu_name} Engine: ASR not currently listening.")
             if self.ui: self.ui.mic_button.configure(text="🎤") # Ensure button is reset


    def is_asr_listening(self) -> bool:
        return self.speech_listener.is_listening() if self.speech_listener else False

    def handle_asr_transcription(self, transcribed_text: str):
        """Called by SpeechListener with the final transcription."""
        print(f"{self.wubu_name} Engine: ASR transcribed: '{transcribed_text}'")
        if self.ui:
            if self.asr_active_popup and self.asr_active_popup.winfo_exists():
                self.ui.close_popup(self.asr_active_popup)
                self.asr_active_popup = None

            self.ui.set_prompt_input_text(transcribed_text)
            self.ui.mic_button.configure(text="🎤") # Ensure mic button is reset
            # Optionally, can make it auto-send:
            # self.process_user_prompt(transcribed_text)


    # --- Context Provider Methods ---
    def update_editor_context(self, current_file_rel_path: str = None, cursor_pos: tuple = None, open_files_rel_paths: list = None):
        if self.context_provider:
            self.context_provider.update_editor_state(current_file_rel_path, cursor_pos, open_files_rel_paths)
            if self.ui: self.ui.status_label.configure(text="Editor context updated.") # Example feedback
        else:
            if self.ui: self.ui.status_label.configure(text="Context Provider unavailable.")


    # --- Main Processing Logic ---
    def speak(self, text: str, voice_id: str = None, engine_id: str = None): # Added engine_id
        """Convenience method to make WuBu speak using the TTS manager."""
        if self.tts_manager:
            print(f"{self.wubu_name} Engine Speaking: {text[:50]}...")
            self.tts_manager.speak(text, voice_id=voice_id, engine_id=engine_id)
        else:
            fallback_msg = f"TTS UNAVAILABLE ({self.wubu_name}): {text}"
            print(fallback_msg)
            if self.ui: self.ui.add_message_to_chat(self.wubu_name, fallback_msg) # Use add_message_to_chat for consistency


    def process_user_prompt(self, prompt_text: str): # Renamed from process_text_command
        if not prompt_text:
            if self.ui: self.ui.hide_thinking_indicator() # Ensure hidden if called with empty
            return

        print(f"\n{self.wubu_name} Engine: Processing user prompt: '{prompt_text}'")
        if self.is_processing:
            if self.ui: self.ui.display_message_popup("Busy", f"{self.wubu_name} is currently processing. Please wait.", "info")
            return

        if self.ui: self.ui.show_thinking_indicator()

        self.is_processing = True
        # Run the actual LLM processing in a separate thread
        import threading
        threading.Thread(target=self._threaded_process_prompt, args=(prompt_text,), daemon=True).start()

    def _threaded_process_prompt(self, prompt_text: str):
        """Helper method to run LLM processing in a thread."""
        final_response_to_speak = None
        try:
            # Gather context
            context_str_for_llm = ""
            if self.context_provider:
                # Max chars for snippets/full files can be configured in main config if needed
                # Defaulting to ContextProvider's internal defaults for now.
                context_data = self.context_provider.gather_context(prompt_text)

                # Build a string representation of the context for the LLM
                # This needs to be carefully formatted.
                if context_data.get("current_file"):
                    cf = context_data["current_file"]
                    context_str_for_llm += f"Current open file: {cf['path']}\n"
                    if cf.get("cursor_snippet_formatted"):
                        context_str_for_llm += f"Context around cursor (line {self.context_provider.cursor_position[0] if self.context_provider.cursor_position else 'N/A'}):\n{cf['cursor_snippet_formatted']}\n\n"
                    elif cf.get("content"): # Fallback to full content if no snippet (e.g. no cursor)
                         context_str_for_llm += f"Full content of {cf['path']}:\n{cf['content']}\n\n"

                # TODO: Add open files and @-referenced files to context_str_for_llm if needed by LLM strategy.
                # For now, focusing on cursor context.

            full_prompt_for_llm = prompt_text
            if context_str_for_llm:
                full_prompt_for_llm = f"{context_str_for_llm.strip()}\n\nUser query: {prompt_text}"

            self.conversation_history.append({"role": "user", "content": full_prompt_for_llm}) # Log augmented prompt

            llm_response_data = "Error: LLM Processor not available."
            if self.llm_processor:
                llm_response_data = self.llm_processor.generate_response(
                    prompt=full_prompt_for_llm, # Send the potentially augmented prompt
                    history=self.conversation_history[:-1], # History up to the last user message
                    tools=self.tool_schema if self.tool_dispatcher else None
                )
            else:
                print("Error: WuBu LLM Processor not initialized. Cannot generate response.")
                llm_response_data = f"I understood your query about '{prompt_text}', but my brain (LLM) is currently offline."

            # --- Handle LLM Response (text or tool calls) ---
            if isinstance(llm_response_data, dict) and llm_response_data.get("type") == "tool_calls":
                tool_calls_data = llm_response_data.get("data", [])
                print(f"{self.wubu_name} Engine: LLM requested tool calls: {tool_calls_data}")
                if self.ui: self.ui.add_message_to_chat(self.wubu_name, f"[Requesting to use tools: {', '.join([tc.get('function',{}).get('name','N/A') for tc in tool_calls_data])}]")

                tool_results = self._execute_tool_calls(tool_calls_data)

                self.conversation_history.append({"role": "assistant", "content": None, "tool_calls": tool_calls_data})
                for res in tool_results:
                    tool_result_content_str = json.dumps(res.get("response", {"status": "error", "detail": "no response from tool"}))
                    self.conversation_history.append({
                        "role": "tool", "tool_call_id": res.get("id"), "name": res.get("name"),
                        "content": tool_result_content_str
                    })

                if self.llm_processor:
                    final_response_text = self.llm_processor.generate_response(
                        prompt=None, history=self.conversation_history,
                        tools=self.tool_schema if self.tool_dispatcher else None
                    )
                else:
                    final_response_text = "Error processing tool results: LLM Processor unavailable."

                self.conversation_history.append({"role": "assistant", "content": final_response_text})
                if self.ui: self.ui.add_message_to_chat(self.wubu_name, final_response_text)
                final_response_to_speak = final_response_text

            else: # Standard text response
                final_response_text = str(llm_response_data)
                self.conversation_history.append({"role": "assistant", "content": final_response_text})
                if self.ui: self.ui.add_message_to_chat(self.wubu_name, final_response_text)
                final_response_to_speak = final_response_text

        except Exception as e:
            error_message = f"An error occurred in {self.wubu_name} while processing prompt: {e}"
            print(error_message)
            import traceback; traceback.print_exc()
            if self.ui: self.ui.add_message_to_chat("System Error", error_message)
            final_response_to_speak = f"I seem to have encountered an internal error. {self.wubu_name} apologizes."
        finally:
            self.is_processing = False
            if self.ui: self.ui.hide_thinking_indicator() # Hide thinking indicator

            if final_response_to_speak and self.config.get('tts',{}).get('speak_llm_responses', True):
                self.speak(final_response_to_speak) # Speak the final textual response

            if len(self.conversation_history) > self.config.get('llm_history_length', 20):
                self.conversation_history = self.conversation_history[-self.config.get('llm_history_length', 20):]

    def _execute_tool_calls(self, tool_calls_data: list) -> list:
        """Helper to execute tool calls and gather results."""
        tool_results = []
        if not self.tool_dispatcher or not FunctionCall:
            print("Warning: Tool calls requested, but no ToolDispatcher or FunctionCall type.")
            # Return error results for each requested call
            for tc_data in tool_calls_data:
                tool_results.append({
                    "id": tc_data.get('id', 'unknown_id'),
                    "name": tc_data.get('function', {}).get('name', 'unknown_tool'),
                    "response": {"error": "Tool system unavailable."}
                })
            return tool_results

        for tc_data in tool_calls_data:
            function_details = tc_data.get('function', {})
            func_name = function_details.get('name')
            func_args_str = function_details.get('arguments')
            call_id = tc_data.get('id')
            if not call_id: # Should not happen
                import uuid; call_id = uuid.uuid4().hex

            if not func_name or func_args_str is None:
                print(f"Warning: Malformed tool call from LLM: {tc_data}. Skipping.")
                tool_results.append({"id": call_id, "name": func_name or "unknown_tool", "response": {"error": "Malformed tool call."}})
                continue
            try:
                func_args = json.loads(func_args_str) if isinstance(func_args_str, str) else func_args_str
                if not isinstance(func_args, dict): raise ValueError("Args not dict.")

                function_call_obj = FunctionCall(id=call_id, name=func_name, args=func_args)
                tool_output = self.tool_dispatcher.execute_tool_call(function_call_obj)
                tool_results.append(tool_output)
            except Exception as tool_exc:
                print(f"Error executing/parsing tool '{func_name}': {tool_exc}")
                tool_results.append({"id": call_id, "name": func_name, "response": {"error": str(tool_exc)}})
        return tool_results


    def handle_llm_response_for_ui(self, response_text: str): # Might be deprecated if _threaded_process_prompt handles UI
        """Handles displaying LLM response in UI and speaking it."""
        if self.ui:
            self.ui.add_message_to_chat(self.wubu_name, response_text)

        # Speak the response
        if self.config.get('tts',{}).get('speak_llm_responses', True): # Check config if responses should be spoken
            self.speak(response_text)


    def process_voice_command(self, transcribed_text: str): # This name is fine, but it calls handle_asr_transcription now
        """Processes a command transcribed from voice input."""
        # This method is called by SpeechListener.
        # It should now just call handle_asr_transcription.
        self.handle_asr_transcription(transcribed_text)


    def shutdown(self):
        """Performs cleanup when WuBu is shutting down."""
        print(f"Shutting down {self.wubu_name} Core Engine...")
        # if self.tts_manager and hasattr(self.tts_manager, 'shutdown'): # TTSEngineManager doesn't have shutdown yet
        #     self.tts_manager.shutdown()
        if self.llm_processor and hasattr(self.llm_processor, 'cleanup'):
             self.llm_processor.cleanup()
        # if self.tool_dispatcher and hasattr(self.tool_dispatcher, 'cleanup'):
        #     self.tool_dispatcher.cleanup()
        if self.ui and hasattr(self.ui, 'is_running') and self.ui.is_running:
            print(f"Requesting {self.wubu_name} UI to stop...")
            self.ui.stop()
        print(f"{self.wubu_name} Core Engine shutdown sequence complete.")


if __name__ == '__main__':
    print("--- WuBu Core Engine Direct Test ---")

    dummy_config_for_wubu = {
        'wubu_name': "TestWuBu MkIII",
        'tts': { 'default_voice': "TestWuBuVoice",
                 'wubu_glados_style_voice': {'enabled': True, 'language': 'en'}, # Need this key for TTSEngineManager
                 'wubu_kokoro_voice': {'enabled': False} }, # Ensure one is enabled for manager to pick default
        'llm': { 'provider': 'mock_llm', 'mock_llm_settings': {'model': 'test-dummy-wubu'} },
        'desktop_tools': {'enabled': False}
    }

    class MockWuBuUI:
        def __init__(self, engine): self.engine = engine; self.is_running = False
        def display_message(self, type, content): print(f"MockWuBuUI ({type}): {content}")
        def start(self): self.is_running = True; print("MockWuBuUI started.")
        def stop(self): self.is_running = False; print("MockWuBuUI stopped.")

    print("Initializing WuBu engine with placeholder/mocked subsystems...")
    # This test will use the actual TTSEngineManager but with its voices mocked by tests.tts.test_tts_manager if run via that test file.
    # Here, it will try to load actual voice engines if their configs enable them.
    # The dummy_config enables 'wubu_glados_style_voice', so TTSEngineManager will try to load it.
    # Ensure the mock voice classes in test_tts_manager.py are robust if those tests are run.
    # For this direct run, it will use the actual WubuGLaDOSStyleVoice, which might try to load Coqui.
    # To simplify direct testing here, ensure TTS config has at least one mockable or simple engine.
    # The TTSEngineManager in __init__ will try to load.

    # For a truly isolated engine test here, you'd mock TTSEngineManager and LLMProcessor too.
    # from unittest.mock import MagicMock
    # TTSEngineManager = MagicMock() # Replace the import at top with this for isolated test

    engine = WuBuEngine(config=dummy_config_for_wubu)
    mock_ui_instance = MockWuBuUI(engine)
    engine.set_ui(mock_ui_instance)
    mock_ui_instance.start() # Start the mock UI

    print("\n--- Test: Process User Prompt (WuBu) ---")
    engine.process_user_prompt("Hello, WuBu. How are you today?")
    engine.process_user_prompt("What is the meaning of life, WuBu?")

    print("\n--- Test: Shutdown (WuBu) ---")
    engine.shutdown() # This will also stop the mock UI

    print("\n--- WuBu Core Engine Direct Test Finished ---")
