# WuBu Zonos Local TTS Engine
# Implements Text-to-Speech using the locally integrated Zonos model.

import os
import torch
import torchaudio
import hashlib # For caching speaker embeddings
import platform # For cache path
import shutil # For managing cache if needed (e.g. clearing)

from .base_tts_engine import BaseTTSEngine, TTSPlaybackSpeed

# Attempt to import the local Zonos library components
try:
    print("[DEBUG_ZLV] Attempting to import Zonos from src.zonos_local_lib.model...")
    from src.zonos_local_lib.model import Zonos
    print("[DEBUG_ZLV] Successfully imported Zonos.")

    print("[DEBUG_ZLV] Attempting to import make_cond_dict from src.zonos_local_lib.conditioning...")
    from src.zonos_local_lib.conditioning import make_cond_dict #, supported_language_codes (not directly used by engine)
    print("[DEBUG_ZLV] Successfully imported make_cond_dict.")

    print("[DEBUG_ZLV] Attempting to import DEFAULT_DEVICE from src.zonos_local_lib.utils...")
    from src.zonos_local_lib.utils import DEFAULT_DEVICE as ZONOS_DEFAULT_DEVICE
    print("[DEBUG_ZLV] Successfully imported DEFAULT_DEVICE.")
except ImportError as e:
    print(f"ERROR: ZonosLocalVoice - Failed to import local Zonos library: {e}. Ensure zonos_local_lib is correctly placed and importable.")
    # This is a critical error for this engine.
    # We can either raise it or define a dummy Zonos class to allow manager to load without crashing.
    # For now, let it fail at import time if Zonos is not found.
    raise

# Mapping WuBu language codes to Zonos language codes (consistent with Gradio and original Zonos)
# Zonos supports: "en-us", "ja-jp", "zh-cn", "fr-fr", "de-de" (example, actual from Zonos lib)
# We'll use the supported_language_codes from zonos_local_lib.conditioning if needed for validation,
# but make_cond_dict handles the language string directly.
LANGUAGE_MAP_ZONOS = {
    "en": "en-us", "en-us": "en-us", "en-gb": "en-us", # Default English to en-us
    "ja": "ja-jp", "ja-jp": "ja-jp",
    "zh": "zh-cn", "zh-cn": "zh-cn", # Assuming Zonos uses 'zh-cn'
    "fr": "fr-fr", "fr-fr": "fr-fr",
    "de": "de-de", "de-de": "de-de",
    # Add other mappings if Zonos supports more and WuBu uses different codes
}

class ZonosLocalVoice(BaseTTSEngine):
    """
    TTS Engine using the Zonos model running locally.
    """
    def __init__(self, language='en', default_voice=None, config=None):
        super().__init__(language, default_voice, config) # self.config is set here

        self.model_id = self.config.get('model_id', "Zyphra/Zonos-v0.1-transformer") # Default model
        self.device_str = self.config.get('device', ZONOS_DEFAULT_DEVICE) # "cuda", "cpu"
        self.target_device = torch.device(self.device_str)

        self.zonos_model: Zonos | None = None
        self.speaker_embeddings_cache = {} # Simple in-memory cache for speaker embeddings: {path: tensor}
        self.cache_dir = self._get_embedding_cache_dir() # On-disk cache directory

        self._load_model()

    def _load_model(self):
        try:
            print(f"ZonosLocalVoice: Loading Zonos model '{self.model_id}' onto device '{self.target_device}'...")
            # from_pretrained will download from HF hub if not found locally by that name.
            # It uses the ZonosConfig from the downloaded/local model files.
            self.zonos_model = Zonos.from_pretrained(self.model_id, device=str(self.target_device))
            self.zonos_model.eval() # Ensure it's in eval mode
            # self.zonos_model.requires_grad_(False) # from_pretrained in local Zonos should handle this
            print(f"ZonosLocalVoice: Zonos model '{self.model_id}' loaded successfully.")
        except Exception as e:
            print(f"ERROR: ZonosLocalVoice - Failed to load Zonos model '{self.model_id}': {e}")
            import traceback
            traceback.print_exc()
            self.zonos_model = None # Ensure model is None if loading fails

    def _map_language(self, lang_code: str) -> str:
        """Maps WuBu language codes to Zonos-compatible language codes."""
        main_code = lang_code.split('-')[0]
        return LANGUAGE_MAP_ZONOS.get(lang_code.lower(), LANGUAGE_MAP_ZONOS.get(main_code, "en-us"))

    def _map_speed_to_zonos_rate(self, speed: TTSPlaybackSpeed) -> float:
        """Maps TTSPlaybackSpeed to a rate multiplier for Zonos."""
        # Zonos's make_cond_dict has a `speaking_rate` parameter which seems to be a multiplier.
        # Default in make_cond_dict is 1.0.
        speed_map = {
            TTSPlaybackSpeed.VERY_SLOW: 0.7,
            TTSPlaybackSpeed.SLOW: 0.85,
            TTSPlaybackSpeed.NORMAL: 1.0,
            TTSPlaybackSpeed.FAST: 1.15,
            TTSPlaybackSpeed.VERY_FAST: 1.3,
        }
        return speed_map.get(speed, 1.0)

    def _get_embedding_cache_dir(self) -> str:
        app_data_dir = None
        if platform.system() == "Windows":
            app_data_dir = os.getenv("LOCALAPPDATA")
        elif platform.system() == "Darwin":
            app_data_dir = os.path.expanduser("~/Library/Application Support")
        else:
            app_data_dir = os.getenv("XDG_DATA_HOME", os.path.expanduser("~/.local/share"))

        cache_root = os.path.join(app_data_dir or os.path.expanduser("~"), "WuBu", "zonos_local_embeddings")
        os.makedirs(cache_root, exist_ok=True)
        return cache_root

    def _get_cached_embedding_path(self, reference_audio_path: str) -> str:
        abs_ref_path = os.path.abspath(reference_audio_path)
        hasher = hashlib.sha256(abs_ref_path.encode('utf-8'))
        filename = f"embedding_{hasher.hexdigest()[:16]}.pt"
        return os.path.join(self.cache_dir, filename)

    def _get_speaker_embedding(self, reference_audio_path: str) -> torch.Tensor | None:
        if not self.zonos_model: return None
        if not os.path.exists(reference_audio_path):
            print(f"WARNING: ZonosLocalVoice - Speaker reference audio not found: {reference_audio_path}")
            return None

        # Check in-memory cache first
        if reference_audio_path in self.speaker_embeddings_cache:
            return self.speaker_embeddings_cache[reference_audio_path]

        # Check on-disk cache
        cached_embedding_file_path = self._get_cached_embedding_path(reference_audio_path)
        if os.path.exists(cached_embedding_file_path):
            try:
                print(f"ZonosLocalVoice: Loading speaker embedding from disk cache: {cached_embedding_file_path}")
                embedding = torch.load(cached_embedding_file_path, map_location=self.target_device)
                self.speaker_embeddings_cache[reference_audio_path] = embedding # Store in memory
                return embedding
            except Exception as e:
                print(f"WARNING: ZonosLocalVoice - Failed to load embedding from disk cache '{cached_embedding_file_path}': {e}. Will regenerate.")

        # Generate new embedding
        try:
            print(f"ZonosLocalVoice: Generating new speaker embedding for: {reference_audio_path}")
            wav, sr = torchaudio.load(reference_audio_path)
            # make_speaker_embedding expects wav on its internal device, returns on model's device
            embedding = self.zonos_model.make_speaker_embedding(wav, sr) # Returns on self.target_device

            # Save to disk cache (on CPU to avoid device mismatch if cache is moved/used elsewhere)
            torch.save(embedding.cpu(), cached_embedding_file_path)
            print(f"ZonosLocalVoice: Speaker embedding saved to disk cache: {cached_embedding_file_path}")

            self.speaker_embeddings_cache[reference_audio_path] = embedding # Store in memory
            return embedding
        except Exception as e:
            print(f"ERROR: ZonosLocalVoice - Failed to create speaker embedding from '{reference_audio_path}': {e}")
            import traceback
            traceback.print_exc()
            return None


    def synthesize_to_bytes(self, text: str, voice_id: str = None, speed: TTSPlaybackSpeed = TTSPlaybackSpeed.NORMAL, **kwargs) -> bytes | None:
        if not self.zonos_model:
            print("ERROR: ZonosLocalVoice - Zonos model not loaded. Cannot synthesize.")
            return None

        speaker_ref_path = voice_id or self.default_voice
        speaker_embedding = None
        if speaker_ref_path:
            speaker_embedding = self._get_speaker_embedding(speaker_ref_path)
            if speaker_embedding is None:
                print(f"WARNING: ZonosLocalVoice - Could not get/create speaker embedding for {speaker_ref_path}. Proceeding without it.")

        # Parameters for make_cond_dict (defaults can be taken from Zonos's make_cond_dict)
        # These could also come from self.config or kwargs
        zonos_lang = self._map_language(self.language) # Use current engine language
        zonos_rate = self._map_speed_to_zonos_rate(speed)

        # TODO: Expose more Zonos conditioning params via self.config or kwargs
        # (emotion, fmax, pitch_std, vqscore_8, dnsmos_ovrl, unconditional_keys etc.)
        # For now, using defaults from make_cond_dict for these.
        # Example: unconditional_keys = self.config.get('unconditional_keys', {"vqscore_8", "dnsmos_ovrl"})

        unconditional_keys_cfg = self.config.get('unconditional_keys', ["emotion"]) # Match Gradio default

        # Default values from Zonos's make_cond_dict if not provided in engine config or kwargs
        cond_params = {
            'text': text,
            'language': zonos_lang,
            'speaker': speaker_embedding, # Can be None
            'speaking_rate': zonos_rate,
            'emotion': kwargs.get('emotion', [0.3077, 0.0256, 0.0256, 0.0256, 0.0256, 0.0256, 0.2564, 0.3077]),
            'fmax': kwargs.get('fmax', 22050.0),
            'pitch_std': kwargs.get('pitch_std', 20.0),
            'vqscore_8': kwargs.get('vqscore_8', [0.78]*8),
            'dnsmos_ovrl': kwargs.get('dnsmos_ovrl', 4.0),
            'speaker_noised': kwargs.get('speaker_noised', False),
            'unconditional_keys': unconditional_keys_cfg,
            'device': self.target_device,
        }

        try:
            cond_dict = make_cond_dict(**cond_params)
            conditioning = self.zonos_model.prepare_conditioning(cond_dict) # CFG handled inside if uncond_dict is passed by make_cond_dict logic

            # TODO: Handle audio_prefix_codes similar to Gradio if needed
            # audio_prefix_codes = None
            # if 'prefix_audio_path' in kwargs and kwargs['prefix_audio_path']:
            #     # ... load and preprocess prefix audio ...
            #     pass

            # Generation parameters (can also be from config/kwargs)
            gen_params = {
                'max_new_tokens': self.config.get('max_new_tokens', 86 * 30),
                'cfg_scale': float(self.config.get('cfg_scale', 2.0)),
                'sampling_params': self.config.get('sampling_params', dict(min_p=0.1)),
                # 'batch_size': 1, # Inferred by Zonos model
                'progress_bar': self.config.get('progress_bar', False), # Usually False for backend
            }

            print(f"ZonosLocalVoice: Generating audio for text: \"{text[:50]}...\"")
            codes = self.zonos_model.generate(
                prefix_conditioning=conditioning,
                # audio_prefix_codes=audio_prefix_codes, # Add if supporting prefix audio
                **gen_params
            )

            wav_out_tensor = self.zonos_model.autoencoder.decode(codes).cpu().detach()

            # Ensure single channel for output if it's stereo (some models might output stereo)
            if wav_out_tensor.dim() == 2 and wav_out_tensor.size(0) > 1: # [Channels, Samples]
                 wav_out_tensor = wav_out_tensor[0:1, :] # Take first channel
            elif wav_out_tensor.dim() == 3 and wav_out_tensor.size(1) > 1: # [Batch, Channels, Samples]
                 wav_out_tensor = wav_out_tensor[:, 0:1, :]


            # Save to a temporary in-memory buffer to get bytes
            buffer = io.BytesIO()
            torchaudio.save(buffer, wav_out_tensor.squeeze(0), self.zonos_model.autoencoder.sampling_rate, format="wav")
            audio_bytes = buffer.getvalue()

            print(f"ZonosLocalVoice: Synthesis successful.")
            return audio_bytes

        except Exception as e:
            print(f"ERROR: ZonosLocalVoice - Synthesis failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def synthesize_to_file(self, text: str, output_filename: str, voice_id: str = None, speed: TTSPlaybackSpeed = TTSPlaybackSpeed.NORMAL, **kwargs) -> bool:
        audio_bytes = self.synthesize_to_bytes(text, voice_id, speed, **kwargs)
        if audio_bytes:
            try:
                with open(output_filename, 'wb') as f:
                    f.write(audio_bytes)
                print(f"ZonosLocalVoice: Audio successfully saved to {output_filename}")
                return True
            except Exception as e:
                print(f"ERROR: ZonosLocalVoice - Failed to write audio to file {output_filename}: {e}")
                return False
        return False

    def load_available_voices(self) -> list:
        # For local Zonos, voices are dynamic based on reference audio files.
        # We can list files from a configured "speaker reference directory" if desired.
        # For now, returning a generic message like the Docker version.
        return [{
            "name": "Zonos Local Voice Cloning (Dynamic)",
            "engine_id": "zonos_engine_local_cloning", # This ID will be set in TTSEngineManager
            "description": "Provide reference audio path via 'voice_id' for cloning."
        }]

    def set_default_voice(self, voice_id: str):
        # voice_id is a path to a reference audio file.
        if os.path.exists(voice_id):
            self.default_voice = voice_id
            print(f"ZonosLocalVoice: Default voice (reference audio path) set to: {voice_id}")
            # Optionally pre-cache the embedding for the default voice
            # self._get_speaker_embedding(voice_id)
            return True
        else:
            print(f"WARNING: ZonosLocalVoice - Default voice path '{voice_id}' does not exist. Will not be used unless path becomes valid.")
            self.default_voice = voice_id # Store it, but it might fail at synthesis if still invalid
            return False

# Need to import io for BytesIO buffer in synthesize_to_bytes
import io
