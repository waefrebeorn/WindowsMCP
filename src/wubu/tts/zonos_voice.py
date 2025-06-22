# WuBu Zonos TTS Engine
# Implements Text-to-Speech using Zyphra's Zonos model.

import os
import subprocess
import tempfile
import io

try:
    import torch
    import torchaudio
    from zonos.model import Zonos
    from zonos.conditioning import make_cond_dict
except ImportError:
    # This will be caught by the TTSEngineManager or at instantiation
    # and should prevent the engine from being used if Zonos is not installed.
    print("WARNING: Zonos library or its dependencies (torch, torchaudio) not found. ZonosVoice engine will not be available.")
    Zonos = None # Ensure Zonos is None if import fails
    torch = None
    torchaudio = None
    make_cond_dict = None

from .base_tts_engine import BaseTTSEngine, TTSPlaybackSpeed

# Mapping WuBu language codes to Zonos language codes
# Zonos supports: "en-us", "ja-jp", "zh-cn", "fr-fr", "de-de"
LANGUAGE_MAP = {
    "en": "en-us",
    "en-us": "en-us",
    "en-gb": "en-us", # Zonos might not distinguish UK/US well, map to general English
    "ja": "ja-jp",
    "ja-jp": "ja-jp",
    "zh": "zh-cn",
    "zh-cn": "zh-cn",
    "fr": "fr-fr",
    "fr-fr": "fr-fr",
    "de": "de-de",
    "de-de": "de-de",
    # Add other mappings as needed
}

DEFAULT_ZONOS_MODEL = "Zyphra/Zonos-v0.1-transformer"

class ZonosVoice(BaseTTSEngine):
    """
    TTS Engine using Zyphra Zonos.
    Requires 'zonos', 'torch', 'torchaudio', and 'espeak-ng' (system dependency).
    """
    def __init__(self, language='en', default_voice=None, config=None):
        super().__init__(language, default_voice, config)
        self.zonos_model = None
        self.device = "cpu"
        self.speaker_embeddings_cache = {} # Cache for loaded speaker embeddings

        if not Zonos or not torch or not torchaudio:
            print("ERROR: ZonosVoice cannot initialize because Zonos library or PyTorch/Torchaudio is not installed.")
            return

        if not self._check_espeak():
            print("ERROR: eSpeak NG not found or not working. Zonos TTS requires eSpeak NG for phonemization.")
            # WuBu might still run but Zonos TTS will fail.
            # Consider raising an error or having a 'disabled' state.
            return

        self._initialize_model()

    def _check_espeak(self) -> bool:
        """Checks if espeak-ng is available and callable."""
        try:
            result = subprocess.run(['espeak-ng', '--version'], capture_output=True, text=True, check=False)
            if result.returncode == 0 and "eSpeak NG" in result.stdout:
                print("ZonosVoice: eSpeak NG found and working.")
                return True
            else:
                print(f"ZonosVoice: eSpeak NG check failed. Return code: {result.returncode}, Output: {result.stdout.strip()} {result.stderr.strip()}")
                return False
        except FileNotFoundError:
            print("ZonosVoice: eSpeak NG command not found. Ensure it's installed and in system PATH.")
            return False
        except Exception as e:
            print(f"ZonosVoice: Error checking eSpeak NG: {e}")
            return False

    def _initialize_model(self):
        """Loads the Zonos model."""
        model_name = self.config.get('zonos_model_name', DEFAULT_ZONOS_MODEL)

        # Determine device
        config_device = self.config.get('device', 'cpu').lower()
        if config_device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
            print(f"ZonosVoice: Using CUDA device for Zonos model.")
        elif config_device == "cuda" and not torch.cuda.is_available():
            print(f"ZonosVoice: CUDA requested but not available. Falling back to CPU for Zonos model.")
            self.device = "cpu"
        else:
            self.device = "cpu"
            print(f"ZonosVoice: Using CPU device for Zonos model.")

        try:
            print(f"ZonosVoice: Loading Zonos model '{model_name}' on device '{self.device}'...")
            self.zonos_model = Zonos.from_pretrained(model_name, device=self.device)
            print(f"ZonosVoice: Zonos model '{model_name}' loaded successfully.")
        except Exception as e:
            print(f"ERROR: Failed to load Zonos model '{model_name}': {e}")
            self.zonos_model = None # Ensure model is None if loading fails

    def _get_speaker_embedding(self, voice_id: str = None):
        """
        Loads or retrieves a speaker embedding.
        'voice_id' is expected to be a path to a reference audio file.
        If None, a default/generic speaker might be attempted or error.
        """
        if voice_id is None:
            # TODO: How to handle no voice_id? Zonos might have a generic speaker
            # or we might need a default reference audio. For now, assume it's an error
            # or Zonos `make_cond_dict` handles speaker=None.
            # From Zonos docs, `speaker` is an optional arg to make_cond_dict.
            print("ZonosVoice: No voice_id provided, will attempt synthesis without specific speaker embedding.")
            return None

        if not os.path.exists(voice_id):
            print(f"ERROR: ZonosVoice - Reference audio file not found: {voice_id}")
            return None

        if voice_id in self.speaker_embeddings_cache:
            return self.speaker_embeddings_cache[voice_id]

        try:
            print(f"ZonosVoice: Creating speaker embedding from {voice_id}...")
            wav, sampling_rate = torchaudio.load(voice_id)
            # Ensure wav is on the correct device for the model
            wav = wav.to(self.device)
            speaker_embedding = self.zonos_model.make_speaker_embedding(wav, sampling_rate)
            self.speaker_embeddings_cache[voice_id] = speaker_embedding
            print(f"ZonosVoice: Speaker embedding created and cached for {voice_id}.")
            return speaker_embedding
        except Exception as e:
            print(f"ERROR: ZonosVoice - Failed to create speaker embedding from {voice_id}: {e}")
            return None

    def _map_language(self, lang_code: str) -> str:
        return LANGUAGE_MAP.get(lang_code.lower(), LANGUAGE_MAP.get(lang_code.split('-')[0], "en-us"))

    def _map_speed_to_zonos_rate(self, speed: TTSPlaybackSpeed) -> float:
        # Zonos `make_cond_dict` takes a `rate` parameter. Default is 1.0.
        # Lower is slower, higher is faster. This mapping is an example.
        speed_map = {
            TTSPlaybackSpeed.VERY_SLOW: 0.7,
            TTSPlaybackSpeed.SLOW: 0.85,
            TTSPlaybackSpeed.NORMAL: 1.0,
            TTSPlaybackSpeed.FAST: 1.15,
            TTSPlaybackSpeed.VERY_FAST: 1.3,
        }
        return speed_map.get(speed, 1.0)

    def synthesize_to_bytes(self, text: str, voice_id: str = None, speed: TTSPlaybackSpeed = TTSPlaybackSpeed.NORMAL, **kwargs) -> bytes | None:
        if not self.zonos_model:
            print("ERROR: ZonosVoice - Zonos model not loaded.")
            return None

        speaker_embedding = self._get_speaker_embedding(voice_id or self.default_voice)
        # If speaker_embedding is None after trying default_voice, make_cond_dict will use its default.

        mapped_language = self._map_language(self.language)
        zonos_rate = self._map_speed_to_zonos_rate(speed)

        # Additional parameters for Zonos from kwargs (pitch, emotion, etc.)
        # Example: cond_dict can take 'pitch', 'energy', 'emotion_embedding', 'quality_embedding'
        # These would need to be passed via kwargs and match Zonos's expected names/formats.
        # For now, focus on text, speaker, language, rate.
        # Emotions: happiness, fear, sadness, anger - Zonos might take these as string names or embeddings.
        # The `make_cond_dict` function in Zonos has parameters like:
        # text, speaker, language, rate, pitch, energy, max_frequency, quality_embedding, emotion_embedding

        z_kwargs = {'rate': zonos_rate}
        # TODO: Map other kwargs like 'pitch', 'emotion' if provided.
        # Example: if 'emotion' in kwargs: z_kwargs['emotion_name'] = kwargs['emotion'] (assuming Zonos takes name)

        try:
            print(f"ZonosVoice: Preparing conditioning for text: '{text[:50]}...' (Lang: {mapped_language}, Rate: {zonos_rate})")
            cond_dict = make_cond_dict(
                text=text,
                speaker=speaker_embedding,
                language=mapped_language,
                **z_kwargs # Includes rate, and potentially other mapped params
            )
            conditioning = self.zonos_model.prepare_conditioning(cond_dict)

            print("ZonosVoice: Generating audio codes...")
            codes = self.zonos_model.generate(conditioning)

            print("ZonosVoice: Decoding audio codes...")
            # .cpu() is important as torchaudio.save expects CPU tensor
            wav_tensor = self.zonos_model.autoencoder.decode(codes).cpu()

            # Zonos outputs at 44.1kHz.
            # The wav_tensor is likely [batch_size, num_channels, num_samples]
            # For single synthesis, batch_size is 1. Assuming mono or first channel if stereo.
            # If stereo, wav_tensor[0] would be [2, num_samples].
            # Torchaudio save handles this.

            buffer = io.BytesIO()
            torchaudio.save(buffer, wav_tensor[0], self.zonos_model.autoencoder.sampling_rate, format="wav")
            return buffer.getvalue()

        except Exception as e:
            print(f"ERROR: ZonosVoice - Failed to synthesize audio: {e}")
            import traceback
            traceback.print_exc()
            return None

    def synthesize_to_file(self, text: str, output_filename: str, voice_id: str = None, speed: TTSPlaybackSpeed = TTSPlaybackSpeed.NORMAL, **kwargs) -> bool:
        audio_bytes = self.synthesize_to_bytes(text, voice_id, speed, **kwargs)
        if audio_bytes:
            try:
                with open(output_filename, 'wb') as f:
                    f.write(audio_bytes)
                print(f"ZonosVoice: Audio successfully saved to {output_filename}")
                return True
            except Exception as e:
                print(f"ERROR: ZonosVoice - Failed to write audio to file {output_filename}: {e}")
                return False
        return False

    def load_available_voices(self) -> list:
        """
        Zonos voices are primarily generated via speaker embeddings from audio files.
        This could list paths to pre-saved .pt speaker embedding files or known reference audio.
        For now, returns an empty list, implying voice_id must be a path to reference audio.
        """
        # Example: if we save speaker embeddings:
        # saved_embeddings_dir = self.config.get("zonos_speaker_embeddings_dir", "wubu_speaker_embeddings")
        # if os.path.exists(saved_embeddings_dir):
        #     return [{"id": f, "name": os.path.splitext(f)[0], "path": os.path.join(saved_embeddings_dir, f)}
        #             for f in os.listdir(saved_embeddings_dir) if f.endswith(".pt")]
        return []

    def set_default_voice(self, voice_id: str):
        """
        Sets the default voice for Zonos.
        `voice_id` should be a path to a reference audio file for cloning,
        or a key to a pre-cached/managed speaker embedding.
        """
        # Here, 'voice_id' is expected to be a path to a reference audio file.
        # We don't check its availability in a static list, but we can check if the file exists.
        if os.path.exists(voice_id):
            self.default_voice = voice_id
            print(f"ZonosVoice: Default voice (reference audio path) set to: {voice_id}")
            # Optionally, pre-load and cache the embedding
            # self._get_speaker_embedding(voice_id)
            return True
        else:
            # If it's not a file path, it might be a conceptual ID for a built-in Zonos voice (if any)
            # or a pre-saved embedding ID. For now, assume file path.
            # If self.load_available_voices() returned actual items, we'd use super().is_voice_available here.
            print(f"Warning: ZonosVoice - Default voice path '{voice_id}' does not exist. It might be a conceptual ID.")
            # Allow setting it anyway if it's not meant to be a direct file path for all cases.
            self.default_voice = voice_id
            return True # Or False if strict path checking is desired.

# Example usage (for testing if run directly, though BaseTTSEngine handles some of this)
if __name__ == '__main__':
    print("--- ZonosVoice Direct Test ---")
    if not Zonos:
        print("Zonos library not available, skipping test.")
    else:
        # Dummy config
        test_config = {
            "zonos_model_name": DEFAULT_ZONOS_MODEL, # or "Zyphra/Zonos-v0.1-hybrid"
            "device": "cpu", # "cuda" if available
        }

        # Create a dummy reference audio file for testing speaker embedding
        # This requires soundfile to be installed for sf.write
        try:
            import soundfile as sf
            import numpy as np
            dummy_samplerate = 44100 # Zonos native
            dummy_duration = 3 # seconds
            dummy_freq = 440 # A4
            t = np.linspace(0, dummy_duration, int(dummy_samplerate * dummy_duration), False)
            dummy_audio_data = 0.5 * np.sin(2 * np.pi * dummy_freq * t)
            # Ensure it's float32 for soundfile
            dummy_audio_data = dummy_audio_data.astype(np.float32)

            temp_dir = tempfile.gettempdir()
            dummy_ref_audio_path = os.path.join(temp_dir, "zonos_dummy_ref.wav")
            sf.write(dummy_ref_audio_path, dummy_audio_data, dummy_samplerate)
            print(f"Created dummy reference audio: {dummy_ref_audio_path}")

            zonos_tts = ZonosVoice(config=test_config, language="en")

            if zonos_tts.zonos_model: # Check if model loaded
                text_to_speak = "Hello from WuBu using Zonos! This is a test of the emergency broadcast system."

                output_wav_file = os.path.join(temp_dir, "zonos_test_output.wav")

                print(f"\nAttempting to synthesize with specific speaker: {dummy_ref_audio_path}")
                success = zonos_tts.synthesize_to_file(text_to_speak, output_wav_file, voice_id=dummy_ref_audio_path, speed=TTSPlaybackSpeed.NORMAL)

                if success:
                    print(f"Successfully synthesized to {output_wav_file}")
                    # zonos_tts.play_synthesized_bytes(open(output_wav_file, 'rb').read()) # Test playback
                else:
                    print(f"Failed to synthesize with specific speaker.")

                print(f"\nAttempting to synthesize with default (no specific speaker embedding):")
                # This will test if Zonos handles speaker=None in make_cond_dict gracefully
                output_wav_file_no_speaker = os.path.join(temp_dir, "zonos_test_output_no_speaker.wav")
                success_no_speaker = zonos_tts.synthesize_to_file(text_to_speak, output_wav_file_no_speaker, speed=TTSPlaybackSpeed.FAST)
                if success_no_speaker:
                    print(f"Successfully synthesized (no speaker) to {output_wav_file_no_speaker}")
                else:
                    print(f"Failed to synthesize (no speaker).")

            else:
                print("Zonos model not loaded, skipping synthesis test.")

            # Clean up dummy file
            if os.path.exists(dummy_ref_audio_path):
                os.remove(dummy_ref_audio_path)
                print(f"Cleaned up dummy reference audio: {dummy_ref_audio_path}")

        except ImportError:
            print("Skipping ZonosVoice direct test: soundfile or numpy not installed for dummy audio creation.")
        except Exception as e:
            print(f"Error during ZonosVoice direct test: {e}")
            import traceback
            traceback.print_exc()

    print("--- ZonosVoice Direct Test Finished ---")
