# Manages multiple TTS engines and provides a unified interface to them.
# Allows WuBu to switch between voices/engines dynamically based on configuration or context.

from .base_tts_engine import BaseTTSEngine, TTSPlaybackSpeed
# Import specific engine implementations that will be managed
from .glados_voice import WubuGLaDOSStyleVoice
from .kokoro_voice import WubuKokoroVoice

# Could also import other future engines:
# from .piper_tts_engine import PiperTTSEngine (if created)
# from .elevenlabs_tts_engine import ElevenLabsTTSEngine (if created)

class TTSEngineManager:
    """
    Manages and provides access to various TTS engines for WuBu.
    WuBu core would interact with this manager rather than individual engines.
    """
    def __init__(self, config=None):
        """
        Initializes the TTS Engine Manager.
        :param config: The WuBu configuration object/dict, which should contain
                       TTS-specific settings, including which engines to load,
                       their individual configs, and default voice choices.
        """
        self.config = config if config is not None else {}
        self.engines = {}  # Stores instances of loaded TTS engines, keyed by their VOICE_ID
        self.default_engine_id = None # VOICE_ID of the engine to use by default
        self._load_configured_engines()

    def _load_configured_engines(self):
        """
        Loads TTS engines based on the provided configuration.
        The configuration should specify which engines to enable and their parameters.
        """
        tts_config = self.config.get('tts', {})
        # Default to WubuGLaDOSStyleVoice if available, else Kokoro, else first loaded
        default_voice_preference = tts_config.get('default_voice', WubuGLaDOSStyleVoice.VOICE_ID)

        # --- Load WuBu GLaDOS-Style Voice ---
        # Config structure example:
        # tts:
        #   wubu_glados_style_voice: # Key matches a section name
        #     enabled: true
        #     language: 'en'
        #     model_subdir: 'glados_tts_models' # Name of subdir in src/wubu/tts/
        #     speaker_wav_filename: 'glados_reference.wav' # Filename in model_subdir
        #     use_gpu: true
        glados_style_config_section_name = 'wubu_glados_style_voice' # Key in YAML/dict config
        glados_style_config = tts_config.get(glados_style_config_section_name, {})
        if glados_style_config.get('enabled', True): # Enable by default
            try:
                print("Loading WubuGLaDOSStyleVoice engine...")
                engine_instance = WubuGLaDOSStyleVoice(
                    language=glados_style_config.get('language', 'en'),
                    model_subdir_override=glados_style_config.get('model_subdir'), # Passed to constructor
                    speaker_wav_filename_override=glados_style_config.get('speaker_wav_filename'),
                    config=glados_style_config,
                    use_gpu=glados_style_config.get('use_gpu', True)
                )
                if engine_instance.tts_engine:
                    self.engines[WubuGLaDOSStyleVoice.VOICE_ID] = engine_instance
                    print(f"WubuGLaDOSStyleVoice engine loaded successfully with ID: {WubuGLaDOSStyleVoice.VOICE_ID}")
                else:
                    print(f"Failed to initialize the underlying TTS for WubuGLaDOSStyleVoice.")
            except Exception as e:
                print(f"Error loading WubuGLaDOSStyleVoice engine: {e}")

        # --- Load WuBu Kokoro Voice ---
        # Config structure example:
        # tts:
        #   wubu_kokoro_voice:
        #     enabled: true
        #     engine_type: 'coqui' # or 'piper' (if implemented in WubuKokoroVoice)
        #     language: 'en'
        #     coqui_model_name: 'tts_models/en/ljspeech/tacotron2-DDC' # If Coqui
        #     use_gpu: true # If Coqui
        kokoro_config_section_name = 'wubu_kokoro_voice'
        kokoro_config = tts_config.get(kokoro_config_section_name, {})
        if kokoro_config.get('enabled', True): # Enable by default
            try:
                print("Loading WubuKokoroVoice engine...")
                engine_instance = WubuKokoroVoice(
                    language=kokoro_config.get('language', 'en'),
                    engine_type=kokoro_config.get('engine_type', WubuKokoroVoice.ENGINE_TYPE_COQUI),
                    model_name_or_path_override=kokoro_config.get('coqui_model_name'), # Assuming Coqui for now
                    config=kokoro_config,
                    use_gpu=kokoro_config.get('use_gpu', True)
                )
                if engine_instance.tts_engine:
                    self.engines[WubuKokoroVoice.VOICE_ID] = engine_instance
                    print(f"WubuKokoroVoice engine loaded successfully with ID: {WubuKokoroVoice.VOICE_ID}")
                else:
                    print(f"Failed to initialize the underlying TTS for WubuKokoroVoice.")
            except Exception as e:
                print(f"Error loading WubuKokoroVoice engine: {e}")

        # Set default engine
        if default_voice_preference in self.engines:
            self.default_engine_id = default_voice_preference
        elif WubuGLaDOSStyleVoice.VOICE_ID in self.engines: # Fallback 1
             self.default_engine_id = WubuGLaDOSStyleVoice.VOICE_ID
        elif WubuKokoroVoice.VOICE_ID in self.engines: # Fallback 2
            self.default_engine_id = WubuKokoroVoice.VOICE_ID
        elif self.engines: # Fallback 3 (any loaded engine)
            self.default_engine_id = list(self.engines.keys())[0]

        if self.default_engine_id:
            print(f"TTSManager: Default voice set to '{self.default_engine_id}'")
            if default_voice_preference not in self.engines and default_voice_preference is not None :
                 print(f"TTSManager: Note - Preferred default voice '{default_voice_preference}' was not available or failed to load.")
        else:
            print("TTSManager: No TTS engines loaded or no default could be set.")


    def get_engine(self, voice_id: str = None) -> BaseTTSEngine | None:
        target_id = voice_id or self.default_engine_id
        if target_id and target_id in self.engines:
            return self.engines[target_id]

        if voice_id:
            print(f"Warning: TTS engine/voice ID '{voice_id}' not found or not loaded in manager.")
        elif not self.default_engine_id: # No specific voice_id, and no default engine set
             print(f"Warning: No default TTS engine set and no specific voice_id provided.")
        elif not self.engines: # Should be caught by no default_engine_id too
            print(f"Warning: No TTS engines available in manager.")
        return None

    def speak(self, text: str, voice_id: str = None, speed: TTSPlaybackSpeed = TTSPlaybackSpeed.NORMAL, **kwargs):
        engine = self.get_engine(voice_id)
        if engine:
            # Use the engine's own default_voice if voice_id for speak() was None and resolved to default_engine_id
            # Or if a specific voice_id was given to speak() that matches the engine's ID
            effective_voice_id_for_engine = engine.default_voice
            if voice_id and voice_id == engine.default_voice : # Ensure we pass the engine's known ID
                 effective_voice_id_for_engine = voice_id

            engine.speak(text, voice_id=effective_voice_id_for_engine, speed=speed, **kwargs)
        else:
            print(f"TTS Manager: Cannot speak. Engine for voice '{voice_id or 'default'}' not available. Text: {text}")

    def synthesize_to_file(self, text: str, output_filename: str, voice_id: str = None, speed: TTSPlaybackSpeed = TTSPlaybackSpeed.NORMAL, **kwargs) -> bool:
        engine = self.get_engine(voice_id)
        if engine:
            effective_voice_id_for_engine = engine.default_voice
            if voice_id and voice_id == engine.default_voice :
                 effective_voice_id_for_engine = voice_id
            return engine.synthesize_to_file(text, output_filename, voice_id=effective_voice_id_for_engine, speed=speed, **kwargs)
        print(f"TTS Manager: Cannot synthesize to file. Engine for voice '{voice_id or 'default'}' not available.")
        return False

    def synthesize_to_bytes(self, text: str, voice_id: str = None, speed: TTSPlaybackSpeed = TTSPlaybackSpeed.NORMAL, **kwargs) -> bytes | None:
        engine = self.get_engine(voice_id)
        if engine:
            effective_voice_id_for_engine = engine.default_voice
            if voice_id and voice_id == engine.default_voice :
                 effective_voice_id_for_engine = voice_id
            return engine.synthesize_to_bytes(text, voice_id=effective_voice_id_for_engine, speed=speed, **kwargs)
        print(f"TTS Manager: Cannot synthesize to bytes. Engine for voice '{voice_id or 'default'}' not available.")
        return None

    def get_available_voices(self) -> list:
        all_voices = []
        for engine_id, engine_instance in self.engines.items():
            voices_from_engine = engine_instance.get_available_voices()
            for voice_info in voices_from_engine:
                if isinstance(voice_info, dict):
                    all_voices.append(voice_info)
        return all_voices

if __name__ == '__main__':
    print("Testing TTSEngineManager for WuBu...")

    # For __main__ testing, we need constants from the voice modules.
    # This structure assumes that when running this file directly, Python can find sibling modules.
    # `python -m src.wubu.tts.tts_engine_manager` from project root might be needed.
    from .glados_voice import DEFAULT_MODEL_SUBDIR as WUBU_GLADOS_DEFAULT_MODEL_SUBDIR, \
                            DEFAULT_SPEAKER_WAV_FILENAME as WUBU_GLADOS_DEFAULT_SPEAKER_WAV, \
                            WubuGLaDOSStyleVoice
    from .kokoro_voice import DEFAULT_KOKORO_COQUI_MODEL_NAME as WUBU_KOKORO_DEFAULT_MODEL_NAME, \
                            WubuKokoroVoice

    dummy_manager_config = {
        'tts': {
            'default_voice': WubuGLaDOSStyleVoice.VOICE_ID,
            'wubu_glados_style_voice': {
                'enabled': True, 'language': 'en',
                'model_subdir': WUBU_GLADOS_DEFAULT_MODEL_SUBDIR,
                'speaker_wav_filename': WUBU_GLADOS_DEFAULT_SPEAKER_WAV,
                'use_gpu': False
            },
            'wubu_kokoro_voice': {
                'enabled': True, 'language': 'en',
                'engine_type': WubuKokoroVoice.ENGINE_TYPE_COQUI,
                'coqui_model_name': WUBU_KOKORO_DEFAULT_MODEL_NAME,
                'use_gpu': False
            }
        }
    }

    manager = TTSEngineManager(config=dummy_manager_config)

    if not manager.engines:
        print("No engines loaded. Check configs and individual engine logs (especially dummy file creation).")
    else:
        print(f"\nAvailable voices in manager: {manager.get_available_voices()}")
        print(f"Default engine ID in manager: {manager.default_engine_id}")

        print("\n--- Testing speech with default voice (WuBu GLaDOS-Style) ---")
        manager.speak("Hello, I am WuBu, speaking with my default GLaDOS-style voice.")

        print(f"\n--- Testing speech with specific voice: {WubuKokoroVoice.VOICE_ID} ---")
        manager.speak("Hello, this is Kokoro from WuBu, with a standard voice.", voice_id=WubuKokoroVoice.VOICE_ID)

        manager.synthesize_to_file(
            "WuBu here. This is a file synthesis test using my GLaDOS-style voice.",
            "manager_wubu_glados_style_test.wav",
            voice_id=WubuGLaDOSStyleVoice.VOICE_ID
        )
        manager.synthesize_to_file(
            "This is WuBu's Kokoro voice, testing file synthesis.",
            "manager_wubu_kokoro_test.wav",
            voice_id=WubuKokoroVoice.VOICE_ID
        )
        print("\nFile synthesis tests complete (check .wav files, may fail if dummy models used by voices).")
        print("\n--- Testing with a non-existent voice ID ---")
        manager.speak("This message should indicate voice not found.", voice_id="NonExistentVoice99")
    print("\nTTSEngineManager test finished.")
