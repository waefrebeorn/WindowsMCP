# WuBu Core Package

# This package contains the central logic for WuBu, including:
# - The main WuBuEngine.
# - LLM (Large Language Model) interaction and processing.
# - Speech listening and command parsing (ASR module handles actual listening).
# - TTS synthesis coordination (actual synthesis by TTS engines via TTSEngineManager).

# Re-export key classes for easier access from outside the core package.
# from .engine import WuBuEngine
# from .llm_processor import LLMProcessor # TODO: Create and import this class
# from .asr_integration_handler import ASRIntegrationHandler # Example if ASR needs a specific handler here

# print("WuBu Core package initialized.")

# TODO for WuBu Core:
# - Define and implement `LLMProcessor` for handling prompts, context, and LLM API calls.
# - Finalize how ASR (`SpeechListener` in `wubu.asr`) integrates with `WuBuEngine`.
# - Implement robust conversation history management within `WuBuEngine` or `LLMProcessor`.
# - Design and implement a plugin/tool system for extending WuBu's capabilities,
#   potentially integrating with `desktop_tools`.
