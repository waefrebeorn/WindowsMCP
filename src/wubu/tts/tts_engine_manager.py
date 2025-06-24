# Manages multiple TTS engines and provides a unified interface to them.
# Allows WuBu to switch between voices/engines dynamically based on configuration or context.

from .base_tts_engine import BaseTTSEngine, TTSPlaybackSpeed
# Import specific engine implementations that will be managed
# from .glados_voice import WubuGLaDOSStyleVoice # Example of other engines
# from .kokoro_voice import WubuKokoroVoice # Example of other engines
# from .zonos_voice import ZonosVoice # OLD Docker-based Zonos
from .zonos_local_voice import ZonosLocalVoice # NEW Local Zonos

# Define conceptual IDs for the Zonos engines
ZONOS_LOCAL_ENGINE_ID = "zonos_engine_local_cloning"


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
        # Default preference can now be the local Zonos
        default_voice_preference = tts_config.get('default_voice', ZONOS_LOCAL_ENGINE_ID)

        # --- Load Local Zonos Voice Engine ---
        zonos_local_config_section_name = 'zonos_local_engine' # New config section name
        zonos_local_config = tts_config.get(zonos_local_config_section_name, {})
        # Enable local Zonos by default if its section exists, or if it's the default_voice preference
        enable_local_zonos = zonos_local_config.get('enabled', default_voice_preference == ZONOS_LOCAL_ENGINE_ID or zonos_local_config_section_name in tts_config)

        if enable_local_zonos:
            try:
                print("Loading ZonosLocalVoice engine...")
                # default_reference_audio_path is now directly used by ZonosLocalVoice via its config
                engine_instance_local = ZonosLocalVoice(
                    language=zonos_local_config.get('language', 'en'),
                    default_voice=zonos_local_config.get('default_reference_audio_path'), # Passed to init
                    config=zonos_local_config # Pass the whole section
                )
                # Check if model loaded successfully (zonos_model is not None)
                if engine_instance_local.zonos_model is not None:
                    self.engines[ZONOS_LOCAL_ENGINE_ID] = engine_instance_local
                    print(f"ZonosLocalVoice engine loaded successfully with manager ID: {ZONOS_LOCAL_ENGINE_ID}")
                    if engine_instance_local.default_voice:
                         print(f"ZonosLocalVoice default reference audio set to: {engine_instance_local.default_voice}")
                else:
                    print(f"ZonosLocalVoice engine not loaded: Zonos model failed to initialize within the engine.")
            except Exception as e:
                print(f"Error loading ZonosLocalVoice engine: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("ZonosLocalVoice engine explicitly disabled or not configured.")

        # --- Load OLD Docker Zonos Voice Engine (Commented out as per plan to replace) ---
        # zonos_docker_config_section_name = 'zonos_voice_engine' # Old config section for Docker
        # zonos_docker_config = tts_config.get(zonos_docker_config_section_name, {})
        # enable_docker_zonos = zonos_docker_config.get('enabled', False) # Disabled by default now

        # if enable_docker_zonos:
        #     try:
        #         print("Loading ZonosVoice (Docker) engine...")
        #         default_zonos_docker_ref_path = zonos_docker_config.get('default_reference_audio_path')
        #         from .zonos_voice import ZonosVoice # Import here to avoid issues if file is deleted
        #         engine_instance_docker = ZonosVoice(
        #             language=zonos_docker_config.get('language', 'en'),
        #             default_voice=default_zonos_docker_ref_path,
        #             config=zonos_docker_config
        #         )
        #         if hasattr(engine_instance_docker, 'is_docker_ok') and engine_instance_docker.is_docker_ok:
        #             self.engines[ZONOS_DOCKER_ENGINE_ID] = engine_instance_docker
        #             print(f"ZonosVoice (Docker) engine loaded with manager ID: {ZONOS_DOCKER_ENGINE_ID}")
        #         else:
        #             print(f"ZonosVoice (Docker) engine not loaded: Docker check failed.")
        #     except Exception as e:
        #         print(f"Error loading ZonosVoice (Docker) engine: {e}")
        # else:
        #     print("ZonosVoice (Docker) engine disabled in configuration.")


        # Set default engine based on preference and availability
        if default_voice_preference in self.engines:
            self.default_engine_id = default_voice_preference
        elif self.engines: # Fallback to the first loaded available engine
             self.default_engine_id = list(self.engines.keys())[0]
        else:
            self.default_engine_id = None


        if self.default_engine_id:
            print(f"TTSManager: Default voice/engine set to '{self.default_engine_id}'")
            if default_voice_preference != self.default_engine_id and default_voice_preference is not None :
                 print(f"TTSManager: Note - Preferred default voice '{default_voice_preference}' was not available or failed to load. Using '{self.default_engine_id}'.")
        else:
            print("TTSManager: No TTS engines loaded or no default could be set. WuBu TTS might not function.")


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
            voices_from_engine = engine_instance.get_available_voices()
            # voices_from_engine for ZonosLocalVoice returns a list with one conceptual dict.
            # For other potential engines, it might return a list of actual voice dicts.
            all_voices.extend(voices_from_engine)

        # The ZonosLocalVoice already adds its conceptual voice entry.
        # No need for the specific check that was here before using an undefined ZONOS_ENGINE_ID.
        # If other engines are added, their get_available_voices() should also return a list of dicts.
        return all_voices

if __name__ == '__main__':
    print("Testing TTSEngineManager for WuBu (Local Zonos Focused)...")

    # Configuration for the new ZonosLocalVoice
    dummy_manager_config_local_zonos = {
        'tts': {
            'default_voice': ZONOS_LOCAL_ENGINE_ID, # Prefer local Zonos
            'zonos_local_engine': { # New configuration section for local Zonos
                'enabled': True,
                'language': 'en',
                'model_id': "Zyphra/Zonos-v0.1-transformer", # Or a local path if needed by Zonos.from_pretrained
                'device': "cpu", # "cuda" if available and desired
                'default_reference_audio_path': "", # Path to a default .wav for cloning
                # Add other Zonos-specific params here if ZonosLocalVoice uses them from config
                # e.g. 'cfg_scale': 2.0, 'sampling_params': {'min_p':0.1}
            },
            # 'zonos_voice_engine': { # Old Docker config - can be removed or disabled
            #     'enabled': False,
            # }
        }
    }
    if not dummy_manager_config_local_zonos['tts']['zonos_local_engine']['default_reference_audio_path']:
        print("INFO: For __main__ test, 'default_reference_audio_path' for Local Zonos is empty.")
        print("      Synthesis with default voice will rely on Zonos model's behavior for speaker=None.")

    manager = TTSEngineManager(config=dummy_manager_config_local_zonos)

    if not manager.engines:
        print("No engines loaded. Local Zonos may have failed to initialize. Check logs.")
    else:
        print(f"\nAvailable voices in manager: {manager.get_available_voices()}")
        print(f"Default engine ID in manager: {manager.default_engine_id}")

        if manager.default_engine_id == ZONOS_LOCAL_ENGINE_ID:
            print(f"\n--- Testing speech with default engine: {manager.default_engine_id} ---")
            default_ref_local = dummy_manager_config_local_zonos['tts']['zonos_local_engine']['default_reference_audio_path']
            if default_ref_local:
                 print(f"(This will use Local Zonos with its configured default_reference_audio_path: {default_ref_local})")
            else:
                print("(This will use Local Zonos without a specific reference audio.)")

            # This speak call would actually try to run the model.
            # Ensure necessary model files for "Zyphra/Zonos-v0.1-transformer" are downloadable by Zonos.from_pretrained,
            # or that the model_id points to a valid local Zonos model setup.
            # Also, espeak-ng needs to be functional.
            manager.speak("Hello from WuBu, testing the local Zonos engine.", engine_id=ZONOS_LOCAL_ENGINE_ID)

            # Example for testing with a specific speaker reference file:
            # Create a dummy wav file if you don't have one.
            # dummy_speaker_path = "dummy_speaker_ref.wav"
            # try:
            #   import soundfile as sf
            #   import numpy as np
            #   sf.write(dummy_speaker_path, np.random.randn(16000).astype(np.float32), 16000)
            #   print(f"Attempting synthesis with specific speaker ref: {dummy_speaker_path}")
            #   manager.speak("Testing voice cloning with a local file.", engine_id=ZONOS_LOCAL_ENGINE_ID, voice_id=dummy_speaker_path)
            # except Exception as e_speak_test:
            #   print(f"Could not run specific speaker test: {e_speak_test}")


        print("\n--- Testing with a non-existent engine ID ---")
        manager.speak("This message should indicate engine not found.", engine_id="NonExistentEngine99")
    print("\nTTSEngineManager (Local Zonos Focused) test finished.")
