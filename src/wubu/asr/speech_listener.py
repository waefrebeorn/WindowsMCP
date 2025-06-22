# WuBu Speech Listener for Automatic Speech Recognition (ASR)
# Captures audio input and transcribes it using an ASR engine (e.g., Vosk, Whisper).

import threading
import os # For path checks if loading local models
# import time
# import queue

# ASR Engine specific imports - Placeholders, uncomment and install when implementing
# For Vosk:
# import sounddevice as sd # Already in existing requirements.txt
# import json # For Vosk results
# from vosk import Model as VoskModel, KaldiRecognizer

# For Whisper (OpenAI):
# import whisper # Needs 'openai-whisper' in requirements
# import numpy as np # Needs 'numpy' in requirements

class SpeechListener:
    """
    Listens for speech via microphone and transcribes it using an ASR engine for WuBu.
    Sends transcribed text to the WuBu core engine.
    """
    def __init__(self, wubu_engine_core, config: dict):
        """
        Initializes the WuBu SpeechListener.
        :param wubu_engine_core: Instance of WuBuEngine to send transcriptions to.
        :param config: ASR configuration dictionary from wubu_config.yaml.
        """
        self.wubu_engine = wubu_engine_core
        self.config = config
        self.asr_engine_type = config.get('engine', 'vosk').lower()
        self.is_listening_flag = False
        self.audio_thread = None

        self._initialize_asr_engine()
        print(f"WuBu SpeechListener initialized with ASR engine: {self.asr_engine_type}")

    def _initialize_asr_engine(self):
        """Initializes the chosen ASR engine for WuBu based on configuration."""
        print(f"WuBu SpeechListener: Initializing ASR engine: {self.asr_engine_type}...")
        if self.asr_engine_type == 'vosk':
            # self._init_vosk() # TODO: Implement this
            print("Placeholder: WuBu Vosk engine would be initialized here.")
            self.asr_recognizer_placeholder = True
        elif self.asr_engine_type == 'whisper':
            # self._init_whisper() # TODO: Implement this
            print("Placeholder: WuBu Whisper engine would be initialized here.")
            self.whisper_model_placeholder = True
        else:
            message = f"Unsupported ASR engine type for WuBu: {self.asr_engine_type}"
            print(f"Error: {message}")
            if self.wubu_engine and self.wubu_engine.get_ui():
                 self.wubu_engine.get_ui().display_message("ERROR", message)
            raise ValueError(message)
        print(f"WuBu ASR engine ({self.asr_engine_type}) placeholder initialization complete.")

    # --- Vosk Specific Initialization (Example - TODO: Implement) ---
    # def _init_vosk(self):
    #     vosk_model_path = self.config.get('vosk_model_path') # e.g., "asr_models/vosk-model-small-en-us-0.15"
    #     sample_rate = self.config.get('vosk_sample_rate', 16000)
    #     device_index = self.config.get('audio_input_device_index')

    #     if not vosk_model_path or not os.path.exists(vosk_model_path):
    #         msg = f"Vosk model not found at '{vosk_model_path}'. WuBu ASR needs this."
    #         # Handle error: log, notify UI, raise
    #         raise FileNotFoundError(msg)

    #     self.vosk_model = VoskModel(vosk_model_path)
    #     self.asr_recognizer = KaldiRecognizer(self.vosk_model, sample_rate)
    #     self.sample_rate = sample_rate
    #     self.device_index = device_index
    #     self.block_size = int(sample_rate * 0.2) # 200ms chunks
    #     print(f"WuBu Vosk ASR initialized. Model: {vosk_model_path}, Rate: {sample_rate}, Device: {device_index or 'Default'}")

    # --- Whisper Specific Initialization (Example - TODO: Implement) ---
    # def _init_whisper(self):
    #     model_size = self.config.get('whisper_model_size', "base.en")
    #     device = self.config.get('whisper_device', "cpu")
    #     self.whisper_model = whisper.load_model(model_size, device=device)
    #     self.sample_rate = 16000 # Whisper standard
    #     self.device_index = self.config.get('audio_input_device_index')
    #     # Buffer for Whisper typically needed, as it processes longer segments
    #     # self.whisper_audio_buffer = []
    #     # self.whisper_buffer_duration_target = 5 # seconds
    #     print(f"WuBu Whisper ASR initialized. Model: {model_size}, Device: {device}")


    def start_listening(self):
        if self.is_listening_flag:
            print("WuBu SpeechListener is already listening.")
            return

        print("Starting WuBu SpeechListener...")
        self.is_listening_flag = True
        target_loop_method = None
        if self.asr_engine_type == 'vosk' and hasattr(self, '_vosk_listening_loop'):
            # target_loop_method = self._vosk_listening_loop
            print("Placeholder: WuBu Vosk listening loop would be assigned and started.")
        elif self.asr_engine_type == 'whisper' and hasattr(self, '_whisper_listening_loop'):
            # target_loop_method = self._whisper_listening_loop
            print("Placeholder: WuBu Whisper listening loop would be assigned and started.")

        if target_loop_method:
            # self.audio_thread = threading.Thread(target=target_loop_method, daemon=True)
            # self.audio_thread.start()
            # print("WuBu SpeechListener audio processing thread started.")
            pass # Keep as placeholder for now
        else:
            print(f"Error: No valid ASR listening loop for '{self.asr_engine_type}'. WuBu cannot start listening.")
            self.is_listening_flag = False


    def stop_listening(self):
        if not self.is_listening_flag:
            print("WuBu SpeechListener is not currently listening.")
            return

        print("Stopping WuBu SpeechListener...")
        self.is_listening_flag = False
        if self.audio_thread and self.audio_thread.is_alive():
            # The listening loop must check self.is_listening_flag and exit.
            # self.audio_thread.join(timeout=2)
            print("Placeholder: WuBu audio thread join would happen here.")
        print("WuBu SpeechListener stopped.")

    def is_listening(self):
        return self.is_listening_flag

    # --- Vosk Listening Loop (Example - TODO: Implement) ---
    # def _vosk_audio_callback(self, indata, frames, time, status):
    #     if status: print(f"Vosk audio status (WuBu): {status}")
    #     if self.is_listening_flag and self.asr_recognizer:
    #         if self.asr_recognizer.AcceptWaveform(bytes(indata)):
    #             result = json.loads(self.asr_recognizer.Result())
    #             if result.get('text'):
    #                 transcribed_text = result['text']
    #                 print(f"WuBu Vosk transcribed: '{transcribed_text}'")
    #                 if self.wubu_engine: self.wubu_engine.process_voice_command(transcribed_text)
    #         # else: # Partial result handling
    #         #     partial = json.loads(self.asr_recognizer.PartialResult()).get('partial')
    #         #     if partial and self.wubu_engine and self.wubu_engine.get_ui():
    #         #         self.wubu_engine.get_ui().display_message("ASR_PARTIAL", partial)

    # def _vosk_listening_loop(self):
    #     try:
    #         with sd.InputStream(samplerate=self.sample_rate, blocksize=self.block_size,
    #                             device=self.device_index, channels=1, dtype='int16',
    #                             callback=self._vosk_audio_callback):
    #             print("WuBu Vosk listening loop started.")
    #             while self.is_listening_flag: time.sleep(0.1)
    #     except Exception as e: # Catch specific exceptions like sounddevice errors
    #         # Handle error, log, notify UI
    #     finally:
    #         print("WuBu Vosk listening loop ended.")
    #         self.is_listening_flag = False

    # --- Whisper Listening Loop (Example - TODO: Implement) ---
    # def _whisper_listening_loop(self):
    #     # Simplified: record fixed chunks, transcribe. Real use needs silence detection.
    #     print("WuBu Whisper listening loop (simplified placeholder) started.")
    #     try:
    #         while self.is_listening_flag:
    #             # audio_chunk = sd.rec(...) # Record audio
    #             # sd.wait()
    #             # audio_np = audio_chunk.flatten().astype(np.float32)
    #             # result = self.whisper_model.transcribe(audio_np, language="en", fp16=torch.cuda.is_available())
    #             # transcribed_text = result['text']
    #             # Simulate this process:
    #             import time; time.sleep(self.config.get('whisper_buffer_duration', 5))
    #             if not self.is_listening_flag: break
    #             simulated_text = "this is a simulated WuBu whisper transcription"
    #             print(f"WuBu Whisper (simulated) transcribed: '{simulated_text}'")
    #             if self.wubu_engine: self.wubu_engine.process_voice_command(simulated_text)
    #     except Exception as e: # Catch specific exceptions
    #         # Handle error
    #     finally:
    #         print("WuBu Whisper listening loop ended.")
    #         self.is_listening_flag = False

if __name__ == '__main__':
    print("--- WuBu SpeechListener Direct Test (Placeholder) ---")

    class MockWuBuEngineForASR: # Minimal mock for testing SpeechListener instantiation
        def __init__(self): self.ui = Mock()
        def process_voice_command(self, text): print(f"MockWuBuEngine: WuBu heard: '{text}'")
        def get_ui(self): return self.ui # SpeechListener might use this

    mock_wubu_eng = MockWuBuEngineForASR()

    # Config for Vosk (dummy path, real test needs valid model)
    asr_conf_vosk = { 'engine': 'vosk', 'vosk_model_path': 'dummy_wubu_vosk_model_path' }
    print("\nTesting WuBu SpeechListener with Vosk (Placeholder Init)...")
    try:
        listener_v = SpeechListener(mock_wubu_eng, asr_conf_vosk)
        # listener_v.start_listening() # Would require actual _vosk_listening_loop
        # time.sleep(1)
        # listener_v.stop_listening()
    except Exception as e: print(f"Error in Vosk placeholder test: {e}")

    # Config for Whisper (uses default tiny.en model)
    asr_conf_whisper = { 'engine': 'whisper', 'whisper_model_size': 'tiny.en', 'whisper_device': 'cpu' }
    print("\nTesting WuBu SpeechListener with Whisper (Placeholder Init)...")
    try:
        listener_w = SpeechListener(mock_wubu_eng, asr_conf_whisper)
        # listener_w.start_listening() # Would require actual _whisper_listening_loop
        # time.sleep(1)
        # listener_w.stop_listening()
    except Exception as e: print(f"Error in Whisper placeholder test: {e}")

    print("\n--- WuBu SpeechListener Direct Test Finished ---")
