# Manages multiple TTS engines and provides a unified interface to them.
# Allows WuBu to switch between voices/engines dynamically based on configuration or context.

from .base_tts_engine import BaseTTSEngine, TTSPlaybackSpeed
# Import specific engine implementations that will be managed
from .glados_voice import WubuGLaDOSStyleVoice
from .kokoro_voice import WubuKokoroVoice
from .zonos_voice import ZonosVoice # Import ZonosVoice

# Define a conceptual ID for the Zonos engine itself within the manager
ZONOS_ENGINE_ID = "zonos_engine_cloning_service"


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

        # --- Load Zonos Voice Engine ---
        # Config structure example:
        # tts:
        #   zonos_voice_engine: # Key matches a section name
        #     enabled: true
        #     language: 'en' # Default language for Zonos if not specified per call
        #     zonos_model_name: "Zyphra/Zonos-v0.1-transformer" # or hybrid
        #     device: "cuda" # "cpu" or "cuda"
        #     default_reference_audio_path: "path/to/default_speaker.wav" # Optional: for a default cloned voice
        zonos_config_section_name = 'zonos_voice_engine' # Key in YAML/dict config
        zonos_config = tts_config.get(zonos_config_section_name, {})
        if zonos_config.get('enabled', False): # Disabled by default unless explicitly enabled
            try:
                print("Loading ZonosVoice engine...")
                # ZonosVoice uses its 'config' param for model_name, device, etc.
                # 'default_voice' for ZonosVoice is the path to a reference audio file.
                default_zonos_ref_path = zonos_config.get('default_reference_audio_path')

                engine_instance = ZonosVoice(
                    language=zonos_config.get('language', 'en'),
                    default_voice=default_zonos_ref_path, # This is the path to reference audio
                    config=zonos_config # Pass the whole zonos_voice_engine section
                )
                # For Docker-based Zonos, check if Docker is OK instead of direct model loading.
                if hasattr(engine_instance, 'is_docker_ok') and engine_instance.is_docker_ok:
                    self.engines[ZONOS_ENGINE_ID] = engine_instance
                    print(f"ZonosVoice (Docker) engine appears ready with manager ID: {ZONOS_ENGINE_ID}")
                    if default_zonos_ref_path:
                        print(f"ZonosVoice default reference audio (host path) set to: {default_zonos_ref_path}")
                elif not (hasattr(engine_instance, 'is_docker_ok')):
                    # This case would be if ZonosVoice was reverted to non-Docker and we forgot to update this check
                    print(f"WARNING: ZonosVoice instance does not have 'is_docker_ok' attribute. Assuming old non-Docker version and attempting to check 'zonos_model'.")
                    if hasattr(engine_instance, 'zonos_model') and engine_instance.zonos_model:
                         self.engines[ZONOS_ENGINE_ID] = engine_instance
                         print(f"ZonosVoice (non-Docker, fallback check) engine loaded with model.")
                    else:
                        print(f"Failed to initialize ZonosVoice engine (fallback check). Docker not okay or model not loaded.")
                else: # is_docker_ok is False
                    print(f"ZonosVoice (Docker) engine not loaded: Docker check failed. Please ensure Docker is installed, running, and accessible.")
            except Exception as e:
                print(f"Error loading ZonosVoice (Docker) engine: {e}")
                import traceback
                traceback.print_exc()


        # Set default engine
        if default_voice_preference in self.engines:
            self.default_engine_id = default_voice_preference
        elif ZONOS_ENGINE_ID in self.engines and tts_config.get('default_voice') == ZONOS_ENGINE_ID: # If Zonos is preferred
            self.default_engine_id = ZONOS_ENGINE_ID
        elif WubuGLaDOSStyleVoice.VOICE_ID in self.engines: # Fallback 1
             self.default_engine_id = WubuGLaDOSStyleVoice.VOICE_ID
        elif WubuKokoroVoice.VOICE_ID in self.engines: # Fallback 2
            self.default_engine_id = WubuKokoroVoice.VOICE_ID
        elif ZONOS_ENGINE_ID in self.engines: # Fallback for Zonos if other specific ones not chosen/available
            self.default_engine_id = ZONOS_ENGINE_ID
        elif self.engines: # Fallback (any loaded engine)
            self.default_engine_id = list(self.engines.keys())[0]

        if self.default_engine_id:
            print(f"TTSManager: Default voice/engine set to '{self.default_engine_id}'")
            if default_voice_preference not in self.engines and default_voice_preference is not None:
                 print(f"TTSManager: Note - Preferred default voice '{default_voice_preference}' was not available or failed to load.")
        else:
            print("TTSManager: No TTS engines loaded or no default could be set.")


    def get_engine(self, engine_id: str = None) -> BaseTTSEngine | None:
        """
        Retrieves a specific TTS engine instance by its registered ID.
        If engine_id is None, returns the default engine.
        """
        target_id = engine_id or self.default_engine_id
        if target_id and target_id in self.engines:
            return self.engines[target_id]

        if engine_id: # Only print warning if a specific engine_id was requested and not found
            print(f"Warning: TTS engine ID '{engine_id}' not found or not loaded in manager.")
        elif not self.default_engine_id:
             print(f"Warning: No default TTS engine set and no specific engine_id provided.")
        elif not self.engines:
            print(f"Warning: No TTS engines available in manager.")
        return None

    def speak(self, text: str, voice_id: str = None, engine_id: str = None, speed: TTSPlaybackSpeed = TTSPlaybackSpeed.NORMAL, **kwargs):
        """
        Synthesizes and speaks text using a specified engine and voice.
        :param text: Text to speak.
        :param voice_id: Specific voice parameter for the engine (e.g., path for Zonos, internal ID for others).
        :param engine_id: ID of the engine to use (e.g., ZONOS_ENGINE_ID). Uses default if None.
        :param speed: Playback speed.
        :param kwargs: Additional engine-specific parameters.
        """
        engine = self.get_engine(engine_id)
        if engine:
            # The 'voice_id' here is passed directly to the engine's speak method.
            # The engine itself is responsible for interpreting this voice_id.
            # For Zonos, it's a path. For others, it might be an internal name.
            # If voice_id is None, the engine will use its own default_voice.
            engine.speak(text, voice_id=voice_id, speed=speed, **kwargs)
        else:
            target_engine_desc = engine_id or "default"
            print(f"TTS Manager: Cannot speak. Engine '{target_engine_desc}' not available. Text: {text}")

    def synthesize_to_file(self, text: str, output_filename: str, voice_id: str = None, engine_id: str = None, speed: TTSPlaybackSpeed = TTSPlaybackSpeed.NORMAL, **kwargs) -> bool:
        """
        Synthesizes text to a file using a specified engine and voice.
        """
        engine = self.get_engine(engine_id)
        if engine:
            return engine.synthesize_to_file(text, output_filename, voice_id=voice_id, speed=speed, **kwargs)

        target_engine_desc = engine_id or "default"
        print(f"TTS Manager: Cannot synthesize to file. Engine '{target_engine_desc}' not available.")
        return False

    def synthesize_to_bytes(self, text: str, voice_id: str = None, engine_id: str = None, speed: TTSPlaybackSpeed = TTSPlaybackSpeed.NORMAL, **kwargs) -> bytes | None:
        """
        Synthesizes text to bytes using a specified engine and voice.
        """
        engine = self.get_engine(engine_id)
        if engine:
            return engine.synthesize_to_bytes(text, voice_id=voice_id, speed=speed, **kwargs)

        target_engine_desc = engine_id or "default"
        print(f"TTS Manager: Cannot synthesize to bytes. Engine '{target_engine_desc}' not available.")
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
    from .glados_voice import WubuGLaDOSStyleVoice # Only import class for VOICE_ID
    from .kokoro_voice import WubuKokoroVoice # Only import class for VOICE_ID
    # ZonosVoice class is already imported; ZONOS_ENGINE_ID is defined in this file.

    # Dummy paths for testing (these won't actually work unless files exist and models are downloaded)
    # For real testing of individual engines, use their own __main__ blocks or dedicated test files.
    WUBU_GLADOS_DEFAULT_MODEL_SUBDIR = "glados_tts_models"
    WUBU_GLADOS_DEFAULT_SPEAKER_WAV = "glados_reference.wav" # Needs to exist in subdir
    WUBU_KOKORO_DEFAULT_MODEL_NAME = "tts_models/en/ljspeech/tacotron2-DDC" # Coqui default

    dummy_manager_config = {
        'tts': {
            'default_voice': ZONOS_ENGINE_ID, # Make Zonos default for this test if enabled
            'wubu_glados_style_voice': {
                'enabled': True, 'language': 'en',
                'model_subdir': WUBU_GLADOS_DEFAULT_MODEL_SUBDIR,
                'speaker_wav_filename': WUBU_GLADOS_DEFAULT_SPEAKER_WAV,
                'use_gpu': False
            },
            'wubu_kokoro_voice': {
                'enabled': True, 'language': 'en',
                'engine_type': WubuKokoroVoice.ENGINE_TYPE_COQUI, # Ensure this matches class const
                'coqui_model_name': WUBU_KOKORO_DEFAULT_MODEL_NAME,
                'use_gpu': False
            },
            'zonos_voice_engine': { # Add Zonos config
                'enabled': True, # Set to True to test Zonos loading
                'language': 'en',
                'zonos_model_name': "Zyphra/Zonos-v0.1-transformer", # Default from ZonosVoice
                'device': "cpu", # Use CPU for this test to avoid GPU dependency here
                'default_reference_audio_path': "path/to/your/test_speaker.wav" # IMPORTANT: User must change this
            }
        }
    }
    # Advise user about the dummy path for Zonos testing
    print("INFO: The TTSEngineManager test config includes Zonos with a dummy 'default_reference_audio_path'.")
    print(f"      To fully test Zonos TTS via this script, update '{dummy_manager_config['tts']['zonos_voice_engine']['default_reference_audio_path']}'")
    print(f"      in tts_engine_manager.py's __main__ block to a valid .wav file path and ensure eSpeak NG is installed.")
    print(f"      Also ensure you have run 'pip install zonos'.\n")


    manager = TTSEngineManager(config=dummy_manager_config)

    if not manager.engines:
        print("No engines loaded. Check configs and individual engine logs (especially dummy file creation).")
    else:
        print(f"\nAvailable voices in manager: {manager.get_available_voices()}") # This will be empty for Zonos as it's dynamic
        print(f"Default engine ID in manager: {manager.default_engine_id}")

        # Test default engine (which might be Zonos if enabled and no other default is forced)
        if manager.default_engine_id:
            print(f"\n--- Testing speech with default engine: {manager.default_engine_id} ---")
            if manager.default_engine_id == ZONOS_ENGINE_ID:
                print(f"(This will use Zonos with its configured default_reference_audio_path: {dummy_manager_config['tts']['zonos_voice_engine']['default_reference_audio_path']})")
                manager.speak("Hello from WuBu, testing the default Zonos engine.", engine_id=ZONOS_ENGINE_ID) # Use engine_id
            elif manager.default_engine_id == WubuGLaDOSStyleVoice.VOICE_ID :
                manager.speak("Hello, I am WuBu, speaking with my default GLaDOS-style voice.", engine_id=WubuGLaDOSStyleVoice.VOICE_ID)
            elif manager.default_engine_id == WubuKokoroVoice.VOICE_ID:
                 manager.speak("Hello, this is Kokoro from WuBu, with a standard voice.", engine_id=WubuKokoroVoice.VOICE_ID)


        if WubuGLaDOSStyleVoice.VOICE_ID in manager.engines:
            print(f"\n--- Testing speech with specific engine: {WubuGLaDOSStyleVoice.VOICE_ID} ---")
            manager.speak("Hello, I am WuBu, speaking with GLaDOS-style voice.", engine_id=WubuGLaDOSStyleVoice.VOICE_ID)
            manager.synthesize_to_file(
                "WuBu here. This is a file synthesis test using GLaDOS-style voice.",
                "manager_wubu_glados_style_test.wav", # Output in current dir
                engine_id=WubuGLaDOSStyleVoice.VOICE_ID
            )

        if WubuKokoroVoice.VOICE_ID in manager.engines:
            print(f"\n--- Testing speech with specific engine: {WubuKokoroVoice.VOICE_ID} ---")
            manager.speak("Hello, this is Kokoro from WuBu, with a standard voice.", engine_id=WubuKokoroVoice.VOICE_ID)
            manager.synthesize_to_file(
                "This is WuBu's Kokoro voice, testing file synthesis.",
                "manager_wubu_kokoro_test.wav", # Output in current dir
                engine_id=WubuKokoroVoice.VOICE_ID
            )

        if ZONOS_ENGINE_ID in manager.engines:
            print(f"\n--- Testing speech with specific engine: {ZONOS_ENGINE_ID} ---")
            print(f"(This requires a valid reference audio path to be passed as 'voice_id' or configured as default in ZonosVoice)")
            # To test Zonos properly here, you'd need a real audio file path for voice_id:
            # manager.speak("Testing Zonos voice cloning.", engine_id=ZONOS_ENGINE_ID, voice_id="actual/path/to/your/speaker.wav")
            # For now, it will use the dummy default_reference_audio_path from config if called without voice_id
            manager.speak("Testing Zonos default speaker if configured.", engine_id=ZONOS_ENGINE_ID)


        print("\nFile synthesis tests complete (check .wav files). Zonos tests depend on valid setup.")
        print("\n--- Testing with a non-existent engine ID ---")
        manager.speak("This message should indicate engine not found.", engine_id="NonExistentEngine99")
    print("\nTTSEngineManager test finished.")
