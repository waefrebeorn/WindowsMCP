# WuBu Core Engine
# This class orchestrates the main WuBu functionalities, integrating
# TTS, ASR (SpeechListener), LLM processing, and UI interactions.

import json # Added for tool call argument/result processing
from ..tts.tts_engine_manager import TTSEngineManager
from ..llm.llm_processor import LLMProcessor
from ..ui.wubu_ui import WuBuUI # For type hinting if needed
# Adjust path if desktop_tools is a top-level package or structured differently
try:
    from desktop_tools.tool_dispatcher import DesktopToolDispatcher, FunctionCall
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

        self.ui: WuBuUI | None = None  # UI handler will be set via set_ui()

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
        self.conversation_history = []

        print(f"{self.wubu_name} Core Engine initialized successfully.")

    def set_ui(self, ui_handler: WuBuUI):
        self.ui = ui_handler
        print(f"UI Handler ({type(ui_handler).__name__}) set for {self.wubu_name} Engine.")
        if self.ui:
            # Example: self.ui.display_message("STATUS_UPDATE", f"{self.wubu_name} Engine linked to UI.")
            pass

    def get_ui(self) -> WuBuUI | None:
        return self.ui

    def speak(self, text: str, voice_id: str = None):
        """Convenience method to make WuBu speak using the TTS manager."""
        if self.tts_manager:
            if self.ui: self.ui.display_message("TTS_OUTPUT", text)
            self.tts_manager.speak(text, voice_id=voice_id)
        else:
            fallback_msg = f"TTS UNAVAILABLE (WuBu): {text}"
            print(fallback_msg)
            if self.ui: self.ui.display_message("TTS_OUTPUT", fallback_msg)


    def process_text_command(self, command: str):
        if not command:
            return

        print(f"\n{self.wubu_name} Engine: Processing text command: '{command}'")
        if self.is_processing:
            self.speak("I am currently processing another request. Please wait a moment, WuBu is thinking.")
            return

        self.is_processing = True
        try:
            self.conversation_history.append({"role": "user", "content": command})

            llm_response = "Error: LLM Processor not available." # Default if LLM processor fails
            if self.llm_processor:
                llm_response = self.llm_processor.generate_response(
                    prompt=command,
                    history=self.conversation_history,
                    tools=self.tool_schema if self.tool_dispatcher else None # Pass tool schema
                )
            else:
                print("Error: WuBu LLM Processor not initialized. Cannot generate response.")
                llm_response = f"I heard you say '{command}', but my brain (LLM) is currently offline. Please check WuBu's configuration."

            # Check if LLM response is a tool call request
            if isinstance(llm_response, dict) and llm_response.get("type") == "tool_calls":
                tool_calls_data = llm_response.get("data", [])
                print(f"{self.wubu_name} Engine: LLM requested tool calls: {tool_calls_data}")

                tool_results = []
                if self.tool_dispatcher and FunctionCall: # Ensure dispatcher and FunctionCall type are available
                    for tc_data in tool_calls_data:
                        # Adapt tc_data to FunctionCall object if necessary.
                        # Ollama's tool_call format: {'function': {'name': 'tool_name', 'arguments': {'arg': 'val'}}}
                        # DesktopToolDispatcher.execute_tool_call expects a FunctionCall object or similar dict.
                        function_details = tc_data.get('function', {})
                        func_name = function_details.get('name')
                        func_args_str = function_details.get('arguments') # Usually a string from Ollama

                        if not func_name or func_args_str is None:
                            print(f"Warning: Malformed tool call from LLM: {tc_data}. Skipping.")
                            tool_results.append({
                                "tool_call_id": tc_data.get('id', 'unknown_id'), # Ollama might provide an ID for the call
                                "name": func_name or "unknown_tool",
                                "response": {"error": "Malformed tool call received from LLM."}
                            })
                            continue

                        try:
                            func_args = json.loads(func_args_str) if isinstance(func_args_str, str) else func_args_str
                            if not isinstance(func_args, dict):
                                raise ValueError("Tool arguments are not a valid dictionary after parsing.")

                            # Assuming DesktopToolDispatcher's execute_tool_call can take a dict
                            # or we construct a FunctionCall object.
                            # Let's assume it can take a dict like: {'name': name, 'arguments': args_dict, 'id': call_id}
                            # The `id` is important for Ollama to map results back.
                            call_id = tc_data.get('id') # Ollama provides this in its tool_calls structure
                            if not call_id: # Should not happen if LLM follows schema with IDs
                                import uuid
                                call_id = uuid.uuid4().hex
                                print(f"Warning: Tool call from LLM for '{func_name}' did not have an ID. Generated: {call_id}")

                            function_call_obj = FunctionCall(id=call_id, name=func_name, args=func_args)

                            # TODO: DesktopToolDispatcher.execute_tool_call might need to be async
                            # For now, assuming it's synchronous.
                            tool_output = self.tool_dispatcher.execute_tool_call(function_call_obj)
                            # tool_output is expected to be like {'id': call_id, 'name': func_name, 'response': actual_tool_result_dict}
                            tool_results.append(tool_output)
                        except json.JSONDecodeError as je:
                            print(f"Error decoding JSON arguments for tool '{func_name}': {func_args_str}. Error: {je}")
                            tool_results.append({"tool_call_id": tc_data.get('id'), "name": func_name, "response": {"error": f"Invalid JSON arguments: {je}"}})
                        except Exception as tool_exc:
                            print(f"Error executing tool '{func_name}': {tool_exc}")
                            # import traceback; traceback.print_exc()
                            tool_results.append({"tool_call_id": tc_data.get('id'), "name": func_name, "response": {"error": str(tool_exc)}})
                else:
                    print("Warning: Tool calls requested by LLM, but no ToolDispatcher or FunctionCall type available.")
                    # Fallback: treat as text or error
                    final_response_text = "I wanted to use a tool, but my tool system is currently unavailable."
                    self.conversation_history.append({"role": "assistant", "content": final_response_text})
                    self.speak(final_response_text)
                    self.is_processing = False
                    return


                # Send tool results back to LLM
                # The history should now include the assistant's prior message (with tool_calls)
                # and then the tool role messages with results.
                self.conversation_history.append({"role": "assistant", "content": None, "tool_calls": tool_calls_data}) # Record LLM's request for tools

                for res in tool_results:
                    # Ollama expects tool role messages to have 'tool_call_id' and 'content' (result as string)
                    # The `response` field from `execute_tool_call` might be a dict.
                    # It needs to be JSON stringified for Ollama's 'content' field for the tool role.
                    tool_result_content_str = json.dumps(res.get("response", {"status": "error", "detail": "no response from tool"}))
                    self.conversation_history.append({
                        "role": "tool",
                        "tool_call_id": res.get("id"), # Use the ID from the tool_output
                        "name": res.get("name"), # Not strictly needed by Ollama in history but good for logs
                        "content": tool_result_content_str
                    })

                if self.llm_processor:
                    final_response_text = self.llm_processor.generate_response(
                        prompt=None, # No new user prompt, just processing tool results
                        history=self.conversation_history,
                        tools=self.tool_schema if self.tool_dispatcher else None # Resend schema in case of iterative tool use
                    )
                    # TODO: LLM might try to call tools AGAIN. Implement max_tool_call_iterations.
                else: # Should not happen if initial check passed
                    final_response_text = "Error processing tool results: LLM Processor unavailable."

                self.conversation_history.append({"role": "assistant", "content": final_response_text})
                self.speak(final_response_text)

            else: # Standard text response from LLM
                final_response_text = str(llm_response) # Ensure it's a string
                self.conversation_history.append({"role": "assistant", "content": final_response_text})
                self.speak(final_response_text)

        except Exception as e:
            error_message = f"An error occurred in {self.wubu_name} while processing command: {e}"
            print(error_message)
            # import traceback; traceback.print_exc() # Uncomment for detailed debugging
            if self.ui: self.ui.display_message("ERROR", error_message)
            self.speak(f"I seem to have encountered an internal error. {self.wubu_name} apologizes. Please try something else.")
        finally:
            self.is_processing = False
            if len(self.conversation_history) > 20: # Limit history size
                self.conversation_history = self.conversation_history[-20:]


    def process_voice_command(self, transcribed_text: str):
        """Processes a command transcribed from voice input."""
        print(f"{self.wubu_name} Engine: Received voice command (transcribed): '{transcribed_text}'")
        # Optional: self.speak("Understood.", voice_id="feedback_voice_short_confirm")
        self.process_text_command(transcribed_text)


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

    print("\n--- Test: Process Text Command (WuBu) ---")
    engine.process_text_command("Hello, WuBu. How are you today?")
    engine.process_text_command("What is the meaning of life, WuBu?")

    print("\n--- Test: Shutdown (WuBu) ---")
    engine.shutdown() # This will also stop the mock UI

    print("\n--- WuBu Core Engine Direct Test Finished ---")
