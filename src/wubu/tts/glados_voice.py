# WuBu's GLaDOS-style specific TTS voice implementation.
# This will likely use Coqui TTS XTTSv2 or a similar high-quality, clonable voice model.

from .base_tts_engine import BaseTTSEngine, TTSPlaybackSpeed
import os
# import torch # Required for Coqui TTS - ensure listed in requirements.txt
# from TTS.api import TTS as CoquiTTS # Correct import for newer Coqui versions - ensure listed

# Model directory and speaker WAV are relative to this file's location (src/wubu/tts/)
# e.g. src/wubu/tts/glados_tts_models/
DEFAULT_MODEL_SUBDIR = "glados_tts_models" # Contains all model files (config.json, model.pth, vocab.json)
DEFAULT_SPEAKER_WAV_FILENAME = "glados_reference.wav" # High-quality reference audio in the model subdir

class WubuGLaDOSStyleVoice(BaseTTSEngine):
    """
    WuBu's GLaDOS-style TTS voice engine using Coqui TTS (XTTSv2).
    Assumes Coqui TTS is installed and model files are available in the specified model directory.
    """
    VOICE_ID = "WuBu-GLaDOS-Style"

    def __init__(self, language='en', model_subdir_override=None, speaker_wav_filename_override=None, config=None, use_gpu=True):
        super().__init__(language=language, default_voice=self.VOICE_ID, config=config)

        # Coqui XTTSv2 model name (can be a path to local files or a Coqui-recognized name)
        self.coqui_model_name_or_path = "tts_models/multilingual/multi-dataset/xtts_v2" # Default Coqui name
        self.use_gpu = use_gpu

        self.model_subdir_name = model_subdir_override or DEFAULT_MODEL_SUBDIR
        self.speaker_wav_filename = speaker_wav_filename_override or DEFAULT_SPEAKER_WAV_FILENAME

        # Resolve full paths
        # self.resolved_model_dir_path is the path to 'src/wubu/tts/<self.model_subdir_name>/'
        self.resolved_model_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), self.model_subdir_name))

        self.resolved_speaker_wav_path = None
        if self.speaker_wav_filename:
            temp_speaker_path = os.path.join(self.resolved_model_dir_path, self.speaker_wav_filename)
            if os.path.exists(temp_speaker_path):
                self.resolved_speaker_wav_path = temp_speaker_path
            else:
                print(f"Warning: WuBu GLaDOS-style speaker WAV '{self.speaker_wav_filename}' not found in '{self.resolved_model_dir_path}'. Voice may default or fail.")

        if not os.path.exists(self.resolved_model_dir_path) or not os.path.isdir(self.resolved_model_dir_path):
            print(f"ERROR: WuBu GLaDOS-style TTS model directory not found: {self.resolved_model_dir_path}")
            self.tts_engine = None
        else:
            print(f"WuBu GLaDOS-style TTS using model directory: {self.resolved_model_dir_path}")
            if self.resolved_speaker_wav_path:
                 print(f"WuBu GLaDOS-style TTS using speaker WAV: {self.resolved_speaker_wav_path}")
            else:
                print(f"WuBu GLaDOS-style TTS: No specific speaker WAV ('{self.speaker_wav_filename}') provided or found in model directory. Relying on model's default voice or cloning from text.")
            self.tts_engine = self._load_engine()

    def _load_engine(self):
        try:
            from TTS.api import TTS as CoquiTTS
            import torch

            if not torch.cuda.is_available() and self.use_gpu:
                print("Warning: CUDA not available for WubuGLaDOSStyleVoice, TTS will run on CPU (slower).")
                self.use_gpu = False # Force CPU if GPU not available

            # For local XTTSv2 models, model_path should be the directory.
            config_json_path = os.path.join(self.resolved_model_dir_path, "config.json")
            # vocab_json_path = os.path.join(self.resolved_model_dir_path, "vocab.json") # Or other vocab file like .pth

            if not os.path.exists(config_json_path):
                print(f"ERROR: config.json not found in {self.resolved_model_dir_path}. Cannot load WuBu GLaDOS-style voice.")
                # Fallback to try loading by Coqui model name if local path fails fundamentally
                print(f"Attempting to load Coqui model by name: {self.coqui_model_name_or_path}")
                engine = CoquiTTS(model_name=self.coqui_model_name_or_path, progress_bar=True, gpu=self.use_gpu)
                print("WuBu GLaDOS-style Coqui TTS engine loaded by name successfully.")
                # Check if a speaker wav is still needed/useful
                if self.resolved_speaker_wav_path and not engine.is_multi_speaker: # if model is single speaker, speaker_wav might not be used.
                    print(f"Note: Model loaded by name is single-speaker, provided speaker_wav '{self.resolved_speaker_wav_path}' might not be used.")
                elif not self.resolved_speaker_wav_path and engine.is_multi_speaker:
                    print(f"Warning: Model loaded by name is multi-speaker, but no speaker_wav provided. May use default speaker.")
                return engine

            print(f"Attempting to load Coqui XTTSv2 model from local directory: {self.resolved_model_dir_path}")
            engine = CoquiTTS(model_path=self.resolved_model_dir_path,
                              config_path=config_json_path,
                              # vocoder_path might also be needed if not in main model dir/config
                              progress_bar=True,
                              gpu=self.use_gpu)
            print("WuBu GLaDOS-style Coqui TTS engine loaded from local path successfully.")
            return engine
        except ImportError:
            print("Error: Coqui TTS (TTS) or PyTorch library not found. WubuGLaDOSStyleVoice will not work.")
            print("Please install them: pip install TTS torch sounddevice soundfile numpy") # Added soundfile, numpy
            return None
        except Exception as e:
            print(f"Error loading WuBu GLaDOS-style Coqui TTS engine: {e}")
            print("Ensure model files are downloaded/configured correctly for XTTSv2, or try default Coqui model name.")
            return None

    def synthesize_to_file(self, text: str, output_filename: str, voice_id: str = None, speed: TTSPlaybackSpeed = TTSPlaybackSpeed.NORMAL, **kwargs) -> bool:
        if not self.tts_engine:
            print("WuBu GLaDOS-style TTS engine not loaded. Cannot synthesize.")
            return False
        try:
            # XTTS specific parameters: speaker_wav, language, speed.
            # Ensure self.resolved_speaker_wav_path is valid if engine requires it.
            current_speaker_wav = self.resolved_speaker_wav_path
            if not current_speaker_wav and self.tts_engine.is_multi_speaker:
                 print(f"Warning: WubuGLaDOSStyleVoice (XTTS) is multi-speaker but no speaker_wav ('{self.speaker_wav_filename}') found for synthesis. Quality may be generic or use default speaker.")
                 # Use first available speaker if any, or None
                 if self.tts_engine.speakers and len(self.tts_engine.speakers) > 0:
                     current_speaker_wav = None # Let Coqui pick default or handle it
                     print(f"Using default/first speaker from model as no specific speaker_wav is set/found.")
                 else: # No specific speaker_wav, and model has no listed speakers (might be single-speaker fine-tune)
                     current_speaker_wav = None

            print(f"Synthesizing WuBu GLaDOS-style voice to file '{output_filename}': '{text}' (Speed: {speed.value})")
            self.tts_engine.tts_to_file(
                text=text,
                speaker_wav=current_speaker_wav,
                language=self.language,
                file_path=output_filename,
                speed=speed.value
            )
            print("Synthesis to file complete.")
            return True
        except Exception as e:
            print(f"Error during WuBu GLaDOS-style TTS synthesis to file: {e}")
            return False

    def synthesize_to_bytes(self, text: str, voice_id: str = None, speed: TTSPlaybackSpeed = TTSPlaybackSpeed.NORMAL, **kwargs) -> bytes | None:
        if not self.tts_engine:
            print("WuBu GLaDOS-style TTS engine not loaded. Cannot synthesize.")
            return None
        try:
            import soundfile as sf
            import io
            import numpy as np

            current_speaker_wav = self.resolved_speaker_wav_path
            if not current_speaker_wav and self.tts_engine.is_multi_speaker:
                 # Similar logic as in synthesize_to_file
                 if self.tts_engine.speakers and len(self.tts_engine.speakers) > 0:
                     current_speaker_wav = None
                 else:
                     current_speaker_wav = None


            print(f"Synthesizing WuBu GLaDOS-style voice to bytes: '{text}' (Speed: {speed.value})")
            wav_data_list = self.tts_engine.tts( # This returns List[float]
                text=text,
                speaker_wav=current_speaker_wav,
                language=self.language,
                speed=speed.value
            )

            wav_array = np.array(wav_data_list, dtype=np.float32)

            samplerate = getattr(getattr(self.tts_engine, 'synthesizer', None), 'output_sample_rate', 24000)
            if samplerate is None: samplerate = 24000 # Should not happen if engine loaded

            with io.BytesIO() as wav_io:
                sf.write(wav_io, wav_array, samplerate, format='WAV', subtype='PCM_16')
                audio_bytes = wav_io.getvalue()

            print("Synthesis to bytes complete.")
            return audio_bytes
        except Exception as e:
            print(f"Error during WuBu GLaDOS-style TTS synthesis to bytes: {e}")
            return None

    def load_available_voices(self) -> list:
        return [{"id": self.VOICE_ID, "name": "WuBu GLaDOS-Style (Coqui XTTSv2)", "language": self.language, "engine": "WubuGLaDOSStyleVoice"}]


if __name__ == '__main__':
    print("Testing WubuGLaDOSStyleVoice TTS Engine...")
    # Requires Coqui XTTSv2 model in src/wubu/tts/glados_tts_models/
    # and a reference speaker wav e.g. 'glados_reference.wav' in that same directory.

    dummy_model_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), DEFAULT_MODEL_SUBDIR))
    dummy_speaker_wav_path = os.path.join(dummy_model_dir_path, DEFAULT_SPEAKER_WAV_FILENAME)
    dummy_config_json_path = os.path.join(dummy_model_dir_path, "config.json")

    os.makedirs(dummy_model_dir_path, exist_ok=True)

    if not os.path.exists(dummy_config_json_path):
        with open(dummy_config_json_path, "w") as f:
            f.write('{"audio":{"sample_rate": 24000}, "model_args": {"num_chars": 256, "use_eos_bos": false}}') # Minimal valid-ish config
        print(f"Created dummy config.json for WubuGLaDOSStyleVoice test: {dummy_config_json_path}")

    if not os.path.exists(dummy_speaker_wav_path):
        try:
            import soundfile as sf
            import numpy as np
            sf.write(dummy_speaker_wav_path, np.zeros(24000, dtype=np.float32), 24000) # 1 sec silent wav
            print(f"Created dummy speaker wav for WubuGLaDOSStyleVoice test: {dummy_speaker_wav_path}")
        except Exception as e:
            print(f"Could not create dummy speaker.wav: {e}.")

    try:
        # use_gpu=False for CI/testing without GPU
        # Provide the filename for speaker_wav_filename_override
        wubu_tts = WubuGLaDOSStyleVoice(language='en', use_gpu=False, speaker_wav_filename_override=DEFAULT_SPEAKER_WAV_FILENAME)

        if wubu_tts.tts_engine:
            print(f"WuBu GLaDOS-style TTS voices available: {wubu_tts.get_available_voices()}")

            test_text = "Hello, this is a test of the WuBu GLaDOS-style voice. I am part of the WuBu system."
            output_file = "wubu_glados_style_test_output.wav"

            print(f"\nAttempting to synthesize to file: {output_file}")
            success_file = wubu_tts.synthesize_to_file(test_text, output_file, speed=TTSPlaybackSpeed.NORMAL)
            if success_file and os.path.exists(output_file):
                print(f"Successfully synthesized to {output_file}. Check this file for audio.")
            else:
                print(f"Failed to synthesize to file or file not found. This is expected if dummy models are used.")

            print(f"\nAttempting to synthesize to bytes and play directly (will likely fail with dummy models but tests structure):")
            wubu_tts.speak(f"This is WuBu, speaking directly. The year is currently {2024}.", speed=TTSPlaybackSpeed.FAST)

        else:
            print("WuBu GLaDOS-style TTS engine could not be initialized. Ensure Coqui TTS is installed and models are set up.")
            print("Try: pip install TTS torch sounddevice soundfile numpy")
            print(f"Model directory expected at: {wubu_tts.resolved_model_dir_path}")
            print(f"Speaker WAV expected at: {wubu_tts.resolved_speaker_wav_path if wubu_tts.resolved_speaker_wav_path else 'Not found/specified'}")

    except Exception as e:
        print(f"An error occurred during WubuGLaDOSStyleVoice testing: {e}")
        print("This might be due to missing dependencies or model setup issues.")
