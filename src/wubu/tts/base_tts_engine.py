# Base class for Text-to-Speech (TTS) engines.
# This defines a common interface for different TTS implementations.

from abc import ABC, abstractmethod
import os
from enum import Enum

class TTSPlaybackSpeed(Enum):
    VERY_SLOW = 0.6
    SLOW = 0.8
    NORMAL = 1.0
    FAST = 1.2
    VERY_FAST = 1.5

class BaseTTSEngine(ABC):
    """
    Abstract base class for TTS engines.
    Provides a common interface for synthesizing speech from text.
    """

    def __init__(self, language='en', default_voice=None, config=None):
        """
        Initializes the TTS engine.
        :param language: Default language for synthesis.
        :param default_voice: Identifier for the default voice to use (engine-specific).
        :param config: Dictionary containing engine-specific configurations.
                       Typically loaded from the main WuBu config.
        """
        self.language = language
        self.default_voice = default_voice
        self.config = config if config is not None else {}
        self.available_voices = self.load_available_voices()

    @abstractmethod
    def synthesize_to_file(self, text: str, output_filename: str, voice_id: str = None, speed: TTSPlaybackSpeed = TTSPlaybackSpeed.NORMAL, **kwargs) -> bool:
        """
        Synthesizes speech from the given text and saves it to a file.

        :param text: The text to synthesize.
        :param output_filename: The path to the file where the audio will be saved.
        :param voice_id: (Optional) The specific voice to use for this synthesis.
                         If None, uses the engine's default or a globally configured voice.
        :param speed: (Optional) Playback speed. Engine must support this.
        :param kwargs: Additional engine-specific parameters.
        :return: True if synthesis was successful and file was saved, False otherwise.
        """
        pass

    @abstractmethod
    def synthesize_to_bytes(self, text: str, voice_id: str = None, speed: TTSPlaybackSpeed = TTSPlaybackSpeed.NORMAL, **kwargs) -> bytes | None:
        """
        Synthesizes speech from the given text and returns it as raw audio bytes.
        The format of the bytes (e.g., WAV, MP3, sample rate, channels) is engine-dependent
        but should be documented by the implementing class.

        :param text: The text to synthesize.
        :param voice_id: (Optional) The specific voice to use.
        :param speed: (Optional) Playback speed.
        :param kwargs: Additional engine-specific parameters.
        :return: Audio data as bytes if successful, None otherwise.
        """
        pass

    def play_synthesized_bytes(self, audio_bytes: bytes, speed: TTSPlaybackSpeed = TTSPlaybackSpeed.NORMAL):
        """
        Plays audio bytes directly. Requires an audio playback library.
        This is a convenience method; actual playback might be handled by a separate AudioPlayer module.

        :param audio_bytes: The audio data (bytes) to play.
        :param speed: (Optional) Playback speed. Note: Applying speed here might require
                      audio processing if the bytes are already at a fixed rate. Some engines
                      apply speed during synthesis.
        """
        try:
            import sounddevice as sd
            import soundfile as sf
            import io
            # import numpy as np # Not strictly needed for basic playback via soundfile

            data, samplerate = sf.read(io.BytesIO(audio_bytes))

            if speed != TTSPlaybackSpeed.NORMAL:
                # Basic playback does not robustly support post-synthesis speed changes without pitch distortion.
                # TTS engines should ideally apply speed during synthesis.
                print(f"Playback speed {speed.name} requested, but basic playback plays at normal speed. For speed changes, ensure TTS engine applies it during synthesis.")
                # If you had librosa for time_stretch:
                # import librosa
                # data_stretched = librosa.effects.time_stretch(data.T if data.ndim > 1 else data, rate=1.0/speed.value)
                # if data.ndim > 1: data = data_stretched.T
                # else: data = data_stretched

            sd.play(data, samplerate)
            sd.wait() # Wait until playback is finished

        except ImportError:
            print("Error: sounddevice or soundfile library not found. Cannot play audio bytes.")
            print("Please install them: pip install sounddevice soundfile")
        except Exception as e:
            print(f"Error during playback of synthesized bytes: {e}")


    def speak(self, text: str, voice_id: str = None, speed: TTSPlaybackSpeed = TTSPlaybackSpeed.NORMAL, **kwargs):
        """
        Synthesizes speech from the given text and plays it directly.
        This is a convenience method.

        :param text: The text to synthesize and speak.
        :param voice_id: (Optional) Specific voice to use.
        :param speed: (Optional) Playback speed.
        :param kwargs: Additional engine-specific parameters.
        """
        print(f"WuBu TTS Speak: '{text}' (Voice: {voice_id or self.default_voice}, Speed: {speed.name})")
        audio_bytes = self.synthesize_to_bytes(text, voice_id=voice_id, speed=speed, **kwargs)
        if audio_bytes:
            # Pass speed to play_synthesized_bytes if it can handle it,
            # otherwise speed should have been applied during synthesize_to_bytes.
            self.play_synthesized_bytes(audio_bytes, speed=speed)
        else:
            print(f"WuBu TTS Speak: Failed to synthesize audio for text: '{text}'")

    @abstractmethod
    def load_available_voices(self) -> list:
        """
        Loads and returns a list of available voices for this engine.
        Each item in the list could be a string (voice ID) or a more complex object/dict
        with voice metadata (name, language, gender, etc.).
        This should be implemented by the concrete TTS engine class.
        """
        pass

    def get_available_voices(self) -> list:
        """Returns the list of available voices."""
        return self.available_voices

    def is_voice_available(self, voice_id: str) -> bool: # Corrected typo here
        """Checks if a specific voice ID is available in this engine."""
        if not self.available_voices:
            return False

        if not self.available_voices: # Should be unreachable if first check passes, but good for safety
             return False
        if not isinstance(self.available_voices, list) or not self.available_voices: # Ensure it's a non-empty list
            return False

        first_item = self.available_voices[0]
        if isinstance(first_item, str):
            return voice_id in self.available_voices
        elif isinstance(first_item, dict) and 'id' in first_item:
             return any(v.get('id') == voice_id for v in self.available_voices)
        elif hasattr(first_item, 'id'): # Assuming voice object has an 'id' attribute
            try: # Protect against non-string IDs or comparison errors
                return any(str(v.id) == str(voice_id) for v in self.available_voices)
            except Exception:
                return False # Comparison failed
        # Add more checks if voice list items have a different structure
        print(f"Warning: Could not determine structure of available_voices to check for voice_id: {voice_id}")
        return False

    def set_default_voice(self, voice_id: str):
        """Sets the default voice for the engine, if available."""
        if self.is_voice_available(voice_id):
            self.default_voice = voice_id
            print(f"Default voice set to: {voice_id}")
            return True
        else:
            print(f"Error: Voice ID '{voice_id}' not available for this engine. Available: {self.available_voices}")
            return False

    def _get_model_path(self, model_filename: str, model_subdir: str) -> str | None:
        """
        Helper to get the path to a model file using the resource loader.
        model_subdir is relative to 'src/wubu/tts/', e.g., "glados_tts_models".

        :param model_filename: Name of the model file.
        :param model_subdir: Subdirectory under 'src/wubu/tts/' where models are stored.
        :return: Full path to the model file, or None if not found.
        """
        try:
            # Try relative import first (if utils is a sibling package part)
            from ..utils.resource_loader import get_resource_path
        except ImportError:
            # Fallback for direct execution or different structures (less ideal)
            try:
                # This assumes a specific project structure if run directly.
                # It's better if this module is always part of the wubu package.
                # For robustness, this path adjustment might be needed if __file__ is tricky.
                import sys
                # current_dir = os.path.dirname(__file__) # src/wubu/tts
                # wubu_dir = os.path.dirname(current_dir) # src/wubu
                # src_dir = os.path.dirname(wubu_dir) # src
                # project_root_dir = os.path.dirname(src_dir) # project root
                # if project_root_dir not in sys.path:
                #    sys.path.insert(0, project_root_dir)
                # This is getting complicated; relative import should just work if structure is sound.
                # If not, it implies a problem with how Python sees the packages.
                # For now, assume the relative import will work if the package structure is correct.
                # If it fails, this path method is a bit of a hack.
                print("Warning: Could not perform relative import of resource_loader. Attempting sys.path modification (less ideal).")
                # This dynamic sys.path modification is generally discouraged.
                # It's better to ensure yourPYTHONPATH or project structure allows direct relative imports.
                # Example: if 'src' is a source root, then from wubu.utils.resource_loader import ...
                # This code is inside src/wubu/tts, so ..utils should refer to src/wubu/utils
                raise # Re-raise the import error to make it clear this is an issue.

            except ImportError as e:
                print(f"Critical Error: Unable to import resource_loader from wubu.utils. Pathing issue? {e}")
                return None

        # Construct the path string that get_resource_path expects for TTS models
        # e.g., "tts/glados_tts_models"
        resource_type_path = os.path.join("tts", model_subdir)

        model_full_path = get_resource_path(resource_type_path, model_filename)

        if os.path.exists(model_full_path):
            return model_full_path
        else:
            print(f"Warning: Model file not found via resource loader at: {model_full_path}")
            return None

    def __str__(self):
        return f"{self.__class__.__name__} (Language: {self.language}, Default Voice: {self.default_voice})"
