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
import whisper # Needs 'openai-whisper' in requirements
import numpy as np # Needs 'numpy' in requirements
import sounddevice as sd # Already in existing requirements.txt
import torch # For whisper, to check CUDA availability

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
            self._init_whisper()
        else:
            message = f"Unsupported ASR engine type for WuBu: {self.asr_engine_type}"
            print(f"Error: {message}")
            if self.wubu_engine and self.wubu_engine.get_ui():
                 self.wubu_engine.get_ui().display_message("ERROR", message)
            raise ValueError(message)
        # print(f"WuBu ASR engine ({self.asr_engine_type}) placeholder initialization complete.") # Removed, init methods will print

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

    # --- Whisper Specific Initialization ---
    def _init_whisper(self):
        model_size = self.config.get('whisper_model_size', "base.en") # e.g., tiny.en, base.en, small.en
        # Determine device: use CUDA if available and configured, else CPU
        config_device = self.config.get('whisper_device', "cpu").lower()
        if config_device == "cuda" and torch.cuda.is_available():
            self.whisper_device = "cuda"
        else:
            if config_device == "cuda" and not torch.cuda.is_available():
                print("Warning: WuBu Whisper configured for CUDA, but CUDA not available. Falling back to CPU.")
            self.whisper_device = "cpu"

        print(f"WuBu SpeechListener: Loading Whisper model '{model_size}' on device '{self.whisper_device}'...")
        try:
            self.whisper_model = whisper.load_model(model_size, device=self.whisper_device)
            self.sample_rate = 16000  # Whisper operates at 16kHz
            self.device_index = self.config.get('audio_input_device_index') # For sounddevice
            self.audio_buffer = [] # Buffer to store audio chunks
            # Define a target duration for audio segments to pass to Whisper (e.g., 5 seconds)
            self.whisper_segment_duration = self.config.get('whisper_segment_duration_seconds', 5)
            self.frames_per_segment = int(self.sample_rate * self.whisper_segment_duration)
            self.is_recording_for_segment = False # Flag to manage segment recording
            print(f"WuBu Whisper ASR initialized. Model: {model_size}, Device: {self.whisper_device}, Segment Duration: {self.whisper_segment_duration}s")
        except Exception as e:
            print(f"Error: Failed to load WuBu Whisper model '{model_size}' on device '{self.whisper_device}': {e}")
            # Potentially notify UI or raise a more specific error
            self.whisper_model = None # Ensure it's None if loading failed
            raise RuntimeError(f"Whisper model loading failed for WuBu: {e}")


    def start_listening(self):
        if not self.whisper_model and self.asr_engine_type == 'whisper':
            print("Error: WuBu Whisper model not loaded. Cannot start listening.")
            # Optionally, notify the UI
            if self.wubu_engine and self.wubu_engine.get_ui():
                self.wubu_engine.get_ui().display_message("ERROR", "ASR model failed to load.")
            return

        if self.is_listening_flag:
            print("WuBu SpeechListener is already listening.")
            return

        print("Starting WuBu SpeechListener...")
        self.is_listening_flag = True
        target_loop_method = None
        if self.asr_engine_type == 'vosk' and hasattr(self, '_vosk_listening_loop'):
            # target_loop_method = self._vosk_listening_loop
            print("Placeholder: WuBu Vosk listening loop would be assigned and started.")
        elif self.asr_engine_type == 'whisper': # Removed hasattr check, will implement directly
            target_loop_method = self._whisper_listening_loop
            # print("WuBu Whisper listening loop will be assigned and started.") # More accurate

        if target_loop_method:
            self.audio_buffer = [] # Ensure buffer is clean before starting
            self.is_recording_for_segment = True # Start recording immediately
            self.audio_thread = threading.Thread(target=target_loop_method, daemon=True)
            self.audio_thread.start()
            print("WuBu SpeechListener audio processing thread started.")
        else:
            print(f"Error: No valid ASR listening loop for '{self.asr_engine_type}'. WuBu cannot start listening.")
            self.is_listening_flag = False # Ensure this is set if loop doesn't start


    def stop_listening(self):
        if not self.is_listening_flag: # Check the primary flag. If not listening, nothing to stop.
            print("WuBu SpeechListener is not currently listening.")
            return

        print("Stopping WuBu SpeechListener...")

        if self.asr_engine_type == 'whisper':
            # For Whisper, first signal the audio accumulation to stop.
            # The audio thread will then process the buffer and exit its loop.
            if self.is_recording_for_segment:
                print("WuBu SpeechListener: Signaling Whisper recording to stop and process buffer.")
                self.is_recording_for_segment = False
            # self.is_listening_flag = False # Let the thread handle this after processing
        else:
            # For other ASR types (like Vosk if implemented similarly)
            self.is_listening_flag = False # Signal loop to stop directly

        if self.audio_thread and self.audio_thread.is_alive():
            print(f"WuBu SpeechListener: Waiting for '{self.asr_engine_type}' audio thread to complete...")
            # The thread's loop should see is_listening_flag / is_recording_for_segment as False
            # and then exit. The finally block in the loop will set self.is_listening_flag = False for whisper.
            self.audio_thread.join(timeout=5) # Wait for the thread to finish

            if self.audio_thread.is_alive():
                print("Warning: WuBu audio thread did not complete in time. Forcing main listening flag down.")
                self.is_listening_flag = False # Ensure it's down if thread is stuck
        else:
            # If thread wasn't running or already finished, ensure flags are down.
            self.is_listening_flag = False
            if self.asr_engine_type == 'whisper':
                self.is_recording_for_segment = False

        print("WuBu SpeechListener stopped.")

    def is_listening(self):
        # True if either the main listening process is active or if it's in the process of recording a segment for whisper
        return self.is_listening_flag or (self.asr_engine_type == 'whisper' and self.is_recording_for_segment)

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

    # --- Whisper Audio Handling & Listening Loop ---
    def _whisper_audio_callback(self, indata, frames, time, status):
        """Called by sounddevice for each new audio chunk."""
        if status:
            print(f"WuBu Whisper audio status: {status}") # Log or handle appropriately
        if self.is_recording_for_segment:
            self.audio_buffer.append(indata.copy())

    def _process_whisper_buffer(self):
        """Processes the accumulated audio buffer with Whisper."""
        if not self.audio_buffer:
            print("WuBu Whisper: No audio data in buffer to process.")
            return

        # Concatenate all parts of the audio buffer
        audio_data_np = np.concatenate(self.audio_buffer)
        self.audio_buffer = [] # Clear buffer after copying

        # Convert to float32, as Whisper expects
        # Sounddevice int16 is in range -32768 to 32767. Whisper needs -1.0 to 1.0.
        audio_float32 = audio_data_np.astype(np.float32) / 32768.0

        print(f"WuBu Whisper: Processing {len(audio_float32)/self.sample_rate:.2f}s of audio...")

        # Set language from config or default to 'en'
        language = self.config.get('whisper_language', 'en')
        fp16_enabled = (self.whisper_device == "cuda") # Use fp16 if on CUDA

        try:
            # Note: For long audio, Whisper's `transcribe` handles chunking internally.
            # However, very long continuous streams might benefit from pre-segmentation or VAD.
            result = self.whisper_model.transcribe(
                audio_float32.flatten(), # Ensure it's a 1D array
                language=language,
                fp16=fp16_enabled,
                # task="transcribe", # or "translate"
                # initial_prompt="User is speaking to WuBu." # Can help guide transcription
            )
            transcribed_text = result['text'].strip()
            print(f"WuBu Whisper transcribed: '{transcribed_text}'")

            if self.wubu_engine and transcribed_text:
                # This needs to be thread-safe if wubu_engine methods aren't,
                # or call it via a queue/event if wubu_engine runs in a different thread (e.g. UI thread)
                # For now, direct call, assuming wubu_engine can handle it or queues it.
                self.wubu_engine.process_voice_command(transcribed_text)
            elif not transcribed_text:
                print("WuBu Whisper: Transcription was empty.")

        except Exception as e:
            print(f"Error during WuBu Whisper transcription: {e}")
            # Handle error, log, notify UI (e.g., self.wubu_engine.get_ui().display_message("ERROR", ...))

    def _whisper_listening_loop(self):
        """Manages audio recording and triggers processing for Whisper."""
        print(f"WuBu Whisper listening loop started. Device: {self.device_index or 'Default'}, Sample Rate: {self.sample_rate}")

        # Calculate blocksize: e.g., 100ms chunks. Smaller means more frequent callbacks.
        blocksize = int(self.sample_rate * 0.1) # 100ms

        try:
            # `sd.InputStream` is a context manager, ensures stream is closed.
            with sd.InputStream(samplerate=self.sample_rate,
                                device=self.device_index,
                                channels=1,
                                dtype='int16', # Common format, convert to float32 for Whisper
                                blocksize=blocksize,
                                callback=self._whisper_audio_callback):

                while self.is_listening_flag and self.is_recording_for_segment:
                    # The callback appends to self.audio_buffer.
                    # In this simplified model, we record until stop_listening() is called.
                    # A more advanced version would check buffer length or use silence detection here.
                    sd.sleep(100) # Sleep for 100ms, then check flags again.

        except Exception as e:
            print(f"Error in WuBu Whisper listening loop: {e}")
            # Handle specific errors like sounddevice.PortAudioError etc.
            if self.wubu_engine and self.wubu_engine.get_ui():
                 self.wubu_engine.get_ui().display_message("ERROR", f"ASR Error: {e}")
        finally:
            print("WuBu Whisper listening loop ended.")
            self.is_recording_for_segment = False # Ensure this is reset
            # If there's remaining audio in the buffer when the loop exits (e.g., due to stop_listening), process it.
            if self.is_listening_flag: # Only process if stop_listening was called to end recording, not if error occurred
                print("WuBu Whisper: Processing remaining audio after loop stop.")
                self._process_whisper_buffer()
            self.is_listening_flag = False # Ensure main listening flag is also false

if __name__ == '__main__':
    print("--- WuBu SpeechListener Direct Test (Placeholder) ---")
    import time # for sleep in test

    # A more complete mock for testing UI interactions if needed
    class MockUI:
        def display_message(self, type, message):
            print(f"MockUI ({type}): {message}")
        def update_asr_status(self, message): # Example if UI shows live ASR status
            print(f"MockUI ASR Status: {message}")

    class MockWuBuEngineForASR: # Minimal mock for testing SpeechListener instantiation
        def __init__(self):
            self.ui = MockUI() # Use the more capable mock UI
        def process_voice_command(self, text):
            print(f"MockWuBuEngine: WuBu heard: '{text}'")
            # In a real app, this might then send text to LLM or UI input
            if self.ui:
                self.ui.display_message("USER_INPUT_ASR", text)
        def get_ui(self):
            return self.ui # SpeechListener might use this

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
