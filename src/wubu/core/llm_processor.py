# WuBu LLM (Large Language Model) Processor
# Interacts with LLMs (Ollama, OpenAI, etc.) to get responses.

import json
try:
    import ollama
except ImportError:
    ollama = None # Ollama is optional, handle if not installed

# try:
# from openai import OpenAI # For OpenAI models - uncomment if used
# except ImportError:
# OpenAI = None


class LLMProcessor:
    """
    Handles interactions with Large Language Models for WuBu.
    Manages client initialization, prompt formatting, and response parsing.
    """
    def __init__(self, config: dict):
        self.config = config.get('llm', {}) # Expects the 'llm' sub-config
        self.wubu_name = config.get('wubu_name', "WuBu") # Get WuBu's name from main config
        self.provider = self.config.get('provider', 'ollama').lower()
        self.client = None

        self._initialize_llm_client()
        print(f"WuBu LLMProcessor initialized with provider: {self.provider}")

    def _initialize_llm_client(self):
        """Initializes the specific LLM client."""
        print(f"WuBu LLMProcessor: Initializing LLM client for provider: {self.provider}...")
        if self.provider == 'ollama':
            self._init_ollama()
        elif self.provider == 'openai':
            # self._init_openai() # TODO: Implement if OpenAI is to be supported
            print("Placeholder: WuBu OpenAI client would be initialized here.")
            self.client = "mock_openai_client_for_wubu" # Placeholder
        else:
            message = f"Unsupported LLM provider for WuBu: {self.provider}"
            print(f"Error: {message}")
            raise ValueError(message)

        if self.client:
            print(f"WuBu LLM client for {self.provider} initialized.")
        else:
            print(f"Warning: WuBu LLM client for {self.provider} FAILED to initialize.")


    def _init_ollama(self):
        if not ollama:
            print("Error: Python 'ollama' library not installed. WuBu cannot use Ollama provider.")
            print("Please install it: pip install ollama")
            self.client = None
            return

        ollama_settings = self.config.get('ollama_settings', {})
        host = ollama_settings.get('host', 'http://localhost:11434')
        request_timeout = ollama_settings.get('request_timeout', 60.0) # Default timeout
        try:
            self.client = ollama.Client(host=host, timeout=request_timeout)
            # Test connection by listing local models (optional, but good check)
            print(f"WuBu Ollama client attempting to connect to host: {host}...")
            models = self.client.list() # This will raise if server not reachable
            print(f"WuBu Ollama client initialized. Host: {host}. Found models: {len(models.get('models', []))}")
            if not models.get('models'):
                print("Warning: No models found in local Ollama instance. Pull a model (e.g., 'ollama pull phi').")
        except Exception as e:
            print(f"Error initializing WuBu Ollama client (Host: {host}): {e}")
            print("Ensure Ollama server is running and accessible.")
            self.client = None


    # def _init_openai(self): # TODO: Implement if needed
    #     # Ensure 'openai' is in requirements.txt if this is uncommented
    #     # from openai import OpenAI
    #     # import os # Ensure os is imported at the top of the file
    #
    #     if not OpenAI: # Assuming OpenAI would be imported if uncommented
    #         print("Error: Python 'openai' library not available/imported. WuBu cannot use OpenAI provider.")
    #         self.client = None
    #         return
    #
    #     openai_settings = self.config.get('openai_settings', {})
    #     # Prioritize environment variable, then config file, then error.
    #     api_key = os.environ.get("OPENAI_API_KEY")
    #     if not api_key:
    #         api_key_from_config = openai_settings.get('api_key')
    #         if api_key_from_config and api_key_from_config != "YOUR_OPENAI_API_KEY_HERE": # Avoid using placeholder
    #             api_key = api_key_from_config
    #             print("WuBu LLMProcessor: Using OpenAI API key from wubu_config.yaml.")
    #         else:
    #             print("Error: OpenAI API key not found in environment variable OPENAI_API_KEY or wubu_config.yaml.")
    #             print("Please set the OPENAI_API_KEY environment variable or add the key to wubu_config.yaml (less secure).")
    #             self.client = None
    #             return
    #     else:
    #         print("WuBu LLMProcessor: Using OpenAI API key from environment variable OPENAI_API_KEY.")
    #
    #     try:
    #         self.client = OpenAI(api_key=api_key)
    #         # self.client.models.list() # Optional: Test connectivity
    #         print("WuBu OpenAI client initialized.")
    #     except Exception as e:
    #         print(f"Error initializing WuBu OpenAI client: {e}")
    #         self.client = None


    def generate_response(self, prompt: str, history: list = None, temperature: float = None, max_tokens: int = None, tools: list = None) -> str | dict: # Can return dict if tool call
        if not self.client:
            error_msg = f"LLM client for provider '{self.provider}' not initialized. WuBu cannot generate response."
            print(f"Error: {error_msg}")
            return f"I am {self.wubu_name}, but my connection to the LLM ({self.provider}) is currently offline."

        print(f"WuBu LLMProcessor: Generating response for prompt (first 50 chars): '{prompt[:50]}...' using {self.provider}")

        if self.provider == 'ollama':
            return self._generate_ollama_response(prompt, history, temperature, max_tokens)
        elif self.provider == 'openai':
            # return self._generate_openai_response(prompt, history, temperature, max_tokens) # TODO
            return f"WuBu OpenAI provider (placeholder) received: '{prompt}'"
        else:
            return f"LLM provider '{self.provider}' response generation not implemented for WuBu."


    def _generate_ollama_response(self, prompt, history, temperature, max_tokens, tools=None):
        if not isinstance(self.client, ollama.Client): # Check if client is correct type
             return "Error: Ollama client is not correctly initialized for WuBu."

        ollama_config = self.config.get('ollama_settings', {})
        model = ollama_config.get('model', 'phi:latest')

        messages = []
        # System prompt could be added here if needed for WuBu's personality
        # system_message_content = f"You are {self.wubu_name}, a helpful AI assistant."
        # messages.append({'role': 'system', 'content': system_message_content})
        if history:
            messages.extend(history)
        messages.append({'role': 'user', 'content': prompt})

        options = {} # Ollama specific options
        if temperature is not None: options['temperature'] = temperature
        else: options['temperature'] = ollama_config.get('temperature', 0.7) # Default from config or general default

        if max_tokens is not None: options['num_predict'] = max_tokens # Ollama uses num_predict
        else: options['num_predict'] = ollama_config.get('num_predict', 256) # Default prediction length

        try:
            print(f"Sending to Ollama (model: {model}): {messages[-1]['content'][:100]}...")
            response = self.client.chat(
                model=model,
                messages=messages,
                tools=tools if tools else None, # Pass tools to Ollama
                options=options if options else None
            )

            # Ollama response structure:
            # response = {
            #   'model': 'phi:latest', 'created_at': '...', 'message': {
            #     'role': 'assistant', 'content': '',
            #     'tool_calls': [{'function': {'name': 'tool_name', 'arguments': {'arg': 'val'}}}] # If tool called
            #   }, ...
            # }
            # Or if no tool call: response['message']['content'] has the text.

            message = response.get('message', {})
            if message.get('tool_calls'):
                print(f"WuBu LLMProcessor: Ollama returned tool_calls: {message['tool_calls']}")
                # Return the raw tool_calls structure for the Engine to process
                # The engine will then format it for DesktopToolDispatcher
                return {"type": "tool_calls", "data": message['tool_calls']}

            llm_text_response = message.get('content', '')
            return llm_text_response.strip() # Return text if no tool calls

        except Exception as e:
            print(f"Error during WuBu Ollama API call: {e}")
            return f"{self.wubu_name} encountered an issue communicating with the Ollama LLM."


    # def _generate_openai_response(self, prompt, history, temperature, max_tokens, tools=None): # TODO
    #     # ... Implementation for OpenAI ...
    #     pass

    # This _parse_for_tool_calls is now less relevant if Ollama returns structured tool_calls.
    # Kept for now if direct text parsing is ever needed as a fallback.
    def _parse_for_tool_calls(self, llm_response_text: str) -> str | dict:
        # Example: If LLM returns JSON in text for tool calls (less ideal than structured output)
        # try:
        #     potential_json = json.loads(llm_response_text)
        #     if isinstance(potential_json, dict) and potential_json.get("type") == "tool_calls":
        #         return potential_json
        # except json.JSONDecodeError:
        #     pass
        return llm_response_text # Assume it's plain text if not parsed as a tool call structure


    def cleanup(self):
        print(f"Cleaning up WuBu LLMProcessor for provider: {self.provider}...")
        self.client = None
        print("WuBu LLMProcessor cleanup complete.")


if __name__ == '__main__':
    print("--- WuBu LLMProcessor Direct Test ---")

    # Requires Ollama running with a model like 'phi' (e.g. `ollama pull phi`)
    dummy_wubu_config = {
        'wubu_name': "TestWuBu LLM Proc",
        'llm': {
            'provider': 'ollama',
            'ollama_settings': {'model': 'phi:latest', 'host': 'http://localhost:11434', 'temperature': 0.5}
        }
    }
    # dummy_wubu_config_openai = { # Example for OpenAI
    #     'wubu_name': "TestWuBu LLM OpenAI",
    #     'llm': {
    #         'provider': 'openai',
    #         'openai_settings': {'api_key': 'YOUR_OPENAI_KEY', 'model': 'gpt-3.5-turbo'}
    #     }
    # }

    print("\nTesting WuBu LLMProcessor with Ollama...")
    try:
        processor = LLMProcessor(config=dummy_wubu_config)
        if processor.client:
            test_prompt = "Who are you?"
            print(f"Test Prompt for WuBu (Ollama): {test_prompt}")
            response = processor.generate_response(test_prompt)
            print(f"WuBu (Ollama) response: {response}")

            test_prompt_hist = "What was my previous question?"
            history = [
                {"role": "user", "content": "Who are you?"},
                {"role": "assistant", "content": response} # Use previous response in history
            ]
            print(f"Test Prompt with History for WuBu (Ollama): {test_prompt_hist}")
            response_hist = processor.generate_response(test_prompt_hist, history=history)
            print(f"WuBu (Ollama) response with history: {response_hist}")
        else:
            print("WuBu Ollama client not initialized, skipping generation test.")
        processor.cleanup()
    except Exception as e:
        print(f"Error during WuBu Ollama LLMProcessor test: {e}")
        print("Ensure Ollama is running and the model (e.g., 'phi') is pulled.")

    print("\n--- WuBu LLMProcessor Direct Test Finished ---")
