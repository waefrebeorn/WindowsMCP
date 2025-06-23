# Manages multiple TTS engines and provides a unified interface to them.
# Allows WuBu to switch between voices/engines dynamically based on configuration or context.

from .base_tts_engine import BaseTTSEngine, TTSPlaybackSpeed
# Import specific engine implementations that will be managed
# from .glados_voice import WubuGLaDOSStyleVoice # Removed
# from .kokoro_voice import WubuKokoroVoice # Removed
from .zonos_voice import ZonosVoice # Import ZonosVoice

# Define a conceptual ID for the Zonos engine itself within the manager
ZONOS_ENGINE_ID = "zonos_engine_cloning_service"


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
        default_voice_preference = tts_config.get('default_voice', ZONOS_ENGINE_ID) # Default to Zonos

        # --- Load Zonos Voice Engine ---
        zonos_config_section_name = 'zonos_voice_engine' # Key in YAML/dict config
        zonos_config = tts_config.get(zonos_config_section_name, {})
        if zonos_config.get('enabled', True): # Enabled by default if section exists, or if tts.default_voice points to it
            try:
                print("Loading ZonosVoice engine...")
                default_zonos_ref_path = zonos_config.get('default_reference_audio_path')

                engine_instance = ZonosVoice(
                    language=zonos_config.get('language', 'en'),
                    default_voice=default_zonos_ref_path,
                    config=zonos_config
                )
                if hasattr(engine_instance, 'is_docker_ok') and engine_instance.is_docker_ok:
                    self.engines[ZONOS_ENGINE_ID] = engine_instance
                    print(f"ZonosVoice (Docker) engine appears ready with manager ID: {ZONOS_ENGINE_ID}")
                    if default_zonos_ref_path:
                        print(f"ZonosVoice default reference audio (host path) set to: {default_zonos_ref_path}")
                else:
                    print(f"ZonosVoice (Docker) engine not loaded: Docker check failed or attribute missing. Please ensure Docker is installed, running, and accessible.")
            except Exception as e:
                print(f"Error loading ZonosVoice (Docker) engine: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("ZonosVoice engine explicitly disabled in configuration.")


        # Set default engine
        if default_voice_preference == ZONOS_ENGINE_ID and ZONOS_ENGINE_ID in self.engines:
            self.default_engine_id = ZONOS_ENGINE_ID
        elif self.engines: # Fallback to the first loaded engine if any (should be Zonos if it loaded)
             self.default_engine_id = list(self.engines.keys())[0]


        if self.default_engine_id:
            print(f"TTSManager: Default voice/engine set to '{self.default_engine_id}'")
            if default_voice_preference != self.default_engine_id and default_voice_preference is not None :
                 print(f"TTSManager: Note - Preferred default voice '{default_voice_preference}' was not available or failed to load. Using '{self.default_engine_id}'.")
        else:
            print("TTSManager: No TTS engines loaded or no default could be set. Zonos may have failed to initialize.")


    def get_engine(self, engine_id: str = None) -> BaseTTSEngine | None:
        """
        Retrieves a specific TTS engine instance by its registered ID.
        If engine_id is None, returns the default engine.
        """
        target_id = engine_id or self.default_engine_id
        if target_id and target_id in self.engines:
            return self.engines[target_id]

        if engine_id:
            print(f"Warning: TTS engine ID '{engine_id}' not found or not loaded in manager.")
        elif not self.default_engine_id :
             print(f"Warning: No default TTS engine set and no specific engine_id provided.")
        elif not self.engines:
            print(f"Warning: No TTS engines available in manager.")
        return None

    def speak(self, text: str, voice_id: str = None, engine_id: str = None, speed: TTSPlaybackSpeed = TTSPlaybackSpeed.NORMAL, **kwargs):
        """
        Synthesizes and speaks text using a specified engine and voice.
        """
        engine = self.get_engine(engine_id)
        if engine:
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
        # For Zonos, "available voices" are dynamic (based on reference audio).
        # This method might return an empty list or conceptual info if Zonos is the only engine.
        all_voices = []
        for engine_id, engine_instance in self.engines.items():
            voices_from_engine = engine_instance.get_available_voices() # Zonos returns []
            for voice_info in voices_from_engine:
                if isinstance(voice_info, dict): # Should not happen for Zonos
                    all_voices.append(voice_info)
        if ZONOS_ENGINE_ID in self.engines and not all_voices:
            all_voices.append({
                "name": "Zonos Voice Cloning (Dynamic)",
                "engine_id": ZONOS_ENGINE_ID,
                "description": "Provide reference audio via 'voice_id' path for cloning."
            })
        return all_voices

if __name__ == '__main__':
    print("Testing TTSEngineManager for WuBu (Zonos Focused)...")

    # ZonosVoice class is already imported; ZONOS_ENGINE_ID is defined in this file.
    dummy_manager_config = {
        'tts': {
            'default_voice': ZONOS_ENGINE_ID,
            'zonos_voice_engine': {
                'enabled': True,
                'language': 'en',
                'zonos_docker_image': "wubu_zonos_image", # Ensure this matches your local build
                'zonos_model_name_in_container': "Zyphra/Zonos-v0.1-transformer",
                'device_in_container': "cpu",
                'default_reference_audio_path': "" # Set to a valid .wav path for testing default cloning
            }
        }
    }
    if not dummy_manager_config['tts']['zonos_voice_engine']['default_reference_audio_path']:
        print("INFO: For __main__ test, 'default_reference_audio_path' for Zonos is empty.")
        print("      Synthesis with default voice will likely use Zonos's internal fallback (if any) or fail if it requires a ref.")
        print("      To test cloning, provide a valid .wav path for 'default_reference_audio_path'.")


    manager = TTSEngineManager(config=dummy_manager_config)

    if not manager.engines:
        print("No engines loaded. Zonos failed to initialize. Check Docker and ZonosVoice logs.")
    else:
        print(f"\nAvailable voices in manager: {manager.get_available_voices()}")
        print(f"Default engine ID in manager: {manager.default_engine_id}")

        if manager.default_engine_id == ZONOS_ENGINE_ID:
            print(f"\n--- Testing speech with default engine: {manager.default_engine_id} ---")
            default_ref = dummy_manager_config['tts']['zonos_voice_engine']['default_reference_audio_path']
            if default_ref:
                 print(f"(This will use Zonos with its configured default_reference_audio_path: {default_ref})")
            else:
                print("(This will use Zonos without a specific reference audio, relying on its internal default behavior.)")
            manager.speak("Hello from WuBu, testing the default Zonos engine.", engine_id=ZONOS_ENGINE_ID)

            # Test with a specific (dummy) voice_id if you have a test file
            # manager.speak("Testing Zonos voice cloning with a specific file.", engine_id=ZONOS_ENGINE_ID, voice_id="path/to/your/test_speaker.wav")

        print("\n--- Testing with a non-existent engine ID ---")
        manager.speak("This message should indicate engine not found.", engine_id="NonExistentEngine99")
    print("\nTTSEngineManager (Zonos Focused) test finished.")
