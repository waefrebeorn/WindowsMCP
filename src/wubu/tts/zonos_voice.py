# WuBu Zonos TTS Engine (Docker-based)
# Implements Text-to-Speech using Zyphra's Zonos model via a Docker container.

import os
import subprocess
import tempfile
import shutil
import platform  # For platform-specific path conversions if needed

from .base_tts_engine import BaseTTSEngine, TTSPlaybackSpeed

# Mapping WuBu language codes to Zonos language codes
# Zonos supports: "en-us", "ja-jp", "zh-cn", "fr-fr", "de-de"
LANGUAGE_MAP = {
    "en": "en-us",
    "en-us": "en-us",
    "en-gb": "en-us",
    "ja": "ja-jp",
    "ja-jp": "ja-jp",
    "zh": "zh-cn",
    "zh-cn": "zh-cn",
    "fr": "fr-fr",
    "fr-fr": "fr-fr",
    "de": "de-de",
    "de-de": "de-de",
}

# Configuration keys expected from wubu_config.yaml for zonos_voice_engine section
# These are examples, actual keys will be fetched from self.config
DEFAULT_ZONOS_DOCKER_IMAGE = "wubu_zonos_image" # Default image tag for the locally built image
DEFAULT_ZONOS_MODEL_INSIDE_CONTAINER = "Zyphra/Zonos-v0.1-transformer" # Model name for zonos_docker_entry.py
DEFAULT_DEVICE_INSIDE_CONTAINER = "cpu" # "cpu" or "cuda"

# Path to the docker entry script relative to this file's directory
# (src/wubu/tts/zonos_voice.py -> src/wubu/tts/docker_scripts/zonos_docker_entry.py)
DOCKER_ENTRY_SCRIPT_NAME = "zonos_docker_entry.py"
DOCKER_ENTRY_SCRIPT_PATH_RELATIVE = os.path.join("docker_scripts", DOCKER_ENTRY_SCRIPT_NAME)


class ZonosVoice(BaseTTSEngine):
    """
    TTS Engine using Zyphra Zonos via a Docker container.
    Requires Docker to be installed and running.
    The specified Zonos Docker image will be used.
    """
    def __init__(self, language='en', default_voice=None, config=None):
        super().__init__(language, default_voice, config) # self.config is set here
        self.is_docker_ok = False
        self.docker_image = self.config.get('zonos_docker_image', DEFAULT_ZONOS_DOCKER_IMAGE)

        # Model name to be used *inside* the container by zonos_docker_entry.py
        self.model_name_in_container = self.config.get('zonos_model_name_in_container', DEFAULT_ZONOS_MODEL_INSIDE_CONTAINER)
        # Device to be used *inside* the container
        self.device_in_container = self.config.get('device_in_container', DEFAULT_DEVICE_INSIDE_CONTAINER).lower()

        # Path to the zonos_docker_entry.py script on the host, to be mounted into the container.
        # It's assumed to be in a 'docker_scripts' subdirectory relative to this file.
        current_script_dir = os.path.dirname(__file__)
        self.host_docker_entry_script_path = os.path.abspath(os.path.join(current_script_dir, DOCKER_ENTRY_SCRIPT_PATH_RELATIVE))

        if not os.path.exists(self.host_docker_entry_script_path):
            print(f"ERROR: ZonosVoice - Docker entry script not found at {self.host_docker_entry_script_path}")
            # This is a critical setup error for this engine.

        self._check_docker()

    def _check_docker(self) -> bool:
        """Checks if Docker is available and callable."""
        try:
            result = subprocess.run(['docker', '--version'], capture_output=True, text=True, check=True)
            print(f"ZonosVoice: Docker found: {result.stdout.strip()}")
            self.is_docker_ok = True
            return True
        except FileNotFoundError:
            print("ERROR: ZonosVoice - Docker command not found. Please install Docker Desktop and ensure 'docker' is in your system PATH.")
            self.is_docker_ok = False
            return False
        except subprocess.CalledProcessError as e:
            print(f"ERROR: ZonosVoice - Docker command failed: {e.stderr}")
            self.is_docker_ok = False
            return False
        except Exception as e:
            print(f"ERROR: ZonosVoice - Error checking Docker: {e}")
            self.is_docker_ok = False
            return False

    def _map_language(self, lang_code: str) -> str:
        return LANGUAGE_MAP.get(lang_code.lower(), LANGUAGE_MAP.get(lang_code.split('-')[0], "en-us"))

    def _map_speed_to_zonos_rate(self, speed: TTSPlaybackSpeed) -> float:
        speed_map = {
            TTSPlaybackSpeed.VERY_SLOW: 0.7,
            TTSPlaybackSpeed.SLOW: 0.85,
            TTSPlaybackSpeed.NORMAL: 1.0,
            TTSPlaybackSpeed.FAST: 1.15,
            TTSPlaybackSpeed.VERY_FAST: 1.3,
        }
        return speed_map.get(speed, 1.0)

    def synthesize_to_bytes(self, text: str, voice_id: str = None, speed: TTSPlaybackSpeed = TTSPlaybackSpeed.NORMAL, **kwargs) -> bytes | None:
        if not self.is_docker_ok:
            print("ERROR: ZonosVoice - Docker is not available or not working. Cannot synthesize.")
            return None

        if not os.path.exists(self.host_docker_entry_script_path):
            print(f"ERROR: ZonosVoice - Docker entry script missing at {self.host_docker_entry_script_path}. Cannot synthesize.")
            return None

        # Determine speaker reference file on host
        host_speaker_ref_file = voice_id or self.default_voice
        if host_speaker_ref_file and not os.path.exists(host_speaker_ref_file):
            print(f"WARNING: ZonosVoice - Reference audio file not found on host: {host_speaker_ref_file}. Attempting synthesis without it.")
            host_speaker_ref_file = None # Will not mount if it doesn't exist

        tmp_dir = None
        try:
            # 1. Create a temporary directory on the host for I/O with Docker
            tmp_dir = tempfile.mkdtemp(prefix="wubu_zonos_")
            host_input_text_file = os.path.join(tmp_dir, "input.txt")
            host_output_wav_file = os.path.join(tmp_dir, "output.wav")

            with open(host_input_text_file, 'w', encoding='utf-8') as f:
                f.write(text)

            # 2. Define paths inside the container
            container_data_dir = "/data"
            container_scripts_dir = "/scripts"
            container_input_text_file = os.path.join(container_data_dir, "input.txt")
            container_output_wav_file = os.path.join(container_data_dir, "output.wav")
            container_docker_entry_script = os.path.join(container_scripts_dir, DOCKER_ENTRY_SCRIPT_NAME)

            container_speaker_ref_file = None
            if host_speaker_ref_file:
                # Mount the specific speaker file to a fixed name in /data or /assets inside container
                container_speaker_ref_file = os.path.join(container_data_dir, "speaker_ref.wav")

            # 3. Construct Docker command
            # Basic command
            docker_cmd = [
                "docker", "run", "--rm",
                "-v", f"{os.path.abspath(tmp_dir)}:{container_data_dir}",
                "-v", f"{os.path.abspath(self.host_docker_entry_script_path)}:{container_docker_entry_script}:ro",
            ]

            # Mount speaker reference file if provided and exists
            if host_speaker_ref_file:
                docker_cmd.extend(["-v", f"{os.path.abspath(host_speaker_ref_file)}:{container_speaker_ref_file}:ro"])

            # GPU access (if configured for container and host has it)
            # Note: self.device_in_container is what the script *inside* docker will use.
            # The --gpus flag is for the docker daemon.
            if self.device_in_container == "cuda":
                docker_cmd.append("--gpus=all") # Request all available GPUs

            # Docker image
            docker_cmd.append(self.docker_image)

            # Command to run inside the container (entrypoint script + args)
            docker_cmd.extend([
                "python", container_docker_entry_script,
                "--text-file", container_input_text_file,
                "--output-wav-file", container_output_wav_file,
                "--model-name", self.model_name_in_container,
                "--language", self._map_language(self.language), # Use instance language
                "--device", self.device_in_container, # Device for script inside container
                "--rate", str(self._map_speed_to_zonos_rate(speed)),
            ])
            if container_speaker_ref_file: # Only add if it was prepared
                docker_cmd.extend(["--speaker-ref-file", container_speaker_ref_file])

            print(f"ZonosVoice: Executing Docker command: {' '.join(docker_cmd)}")

            # 4. Run Docker command
            process = subprocess.run(docker_cmd, capture_output=True, text=True, check=False)

            if process.returncode != 0:
                print(f"ERROR: ZonosVoice - Docker execution failed (Return Code: {process.returncode}).")
                print(f"----- Docker STDOUT -----\n{process.stdout}")
                print(f"----- Docker STDERR -----\n{process.stderr}")
                return None

            # Check for SUCCESS message from zonos_docker_entry.py
            if "SUCCESS" not in process.stdout and "SUCCESS" not in process.stderr:
                print(f"ERROR: ZonosVoice - Docker script did not signal success or may have failed silently before 'SUCCESS'.")
                print(f"----- Docker STDOUT -----\n{process.stdout}")
                print(f"----- Docker STDERR -----\n{process.stderr}")
                return None # If SUCCESS not found, assume failure to produce valid output.
            else:
                print(f"ZonosVoice: Docker container executed successfully and signalled SUCCESS.")
                # Optional: print(f"Docker stdout: {process.stdout}") for debugging success cases too

            # 5. Read the output WAV file
            if os.path.exists(host_output_wav_file):
                with open(host_output_wav_file, 'rb') as f_wav:
                    audio_bytes = f_wav.read()
                print(f"ZonosVoice: Successfully read synthesized audio from {host_output_wav_file}")
                return audio_bytes
            else:
                print(f"ERROR: ZonosVoice - Output audio file '{host_output_wav_file}' not found after Docker execution.")
                print(f"----- Docker STDOUT -----\n{process.stdout}") # Show logs if file is missing
                print(f"----- Docker STDERR -----\n{process.stderr}")
                return None

        except Exception as e:
            print(f"ERROR: ZonosVoice - An error occurred during Docker-based synthesis: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            # 6. Cleanup temporary directory
            if tmp_dir and os.path.exists(tmp_dir):
                try:
                    shutil.rmtree(tmp_dir)
                    # print(f"ZonosVoice: Cleaned up temporary directory: {tmp_dir}")
                except Exception as e_clean:
                    print(f"ERROR: ZonosVoice - Failed to clean up temporary directory {tmp_dir}: {e_clean}")

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
        Zonos voices are dynamically generated via speaker embeddings from audio files.
        This engine does not offer a static list of pre-defined voices.
        The 'voice_id' parameter in synthesis methods should be a path to a reference audio file.
        """
        return []

    def set_default_voice(self, voice_id: str):
        """
        Sets the default voice for Zonos.
        `voice_id` should be a path to a reference audio file on the host system for cloning.
        """
        if os.path.exists(voice_id): # Check if the path exists on the host
            self.default_voice = voice_id
            print(f"ZonosVoice: Default voice (reference audio path) set to: {voice_id}")
            return True
        else:
            # If the path doesn't exist, it's likely an issue.
            # However, BaseTTSEngine.set_default_voice checks is_voice_available,
            # which for this engine will always be false. So, we handle it directly.
            print(f"WARNING: ZonosVoice - Default voice path '{voice_id}' does not exist on host. This might cause issues.")
            self.default_voice = voice_id # Set it anyway, errors will occur at synthesis time
            return False # Indicate that the voice path is not valid as per this check

# Example usage (for conceptual testing - cannot run Docker directly here)
if __name__ == '__main__':
    print("--- ZonosVoice (Docker-based) Conceptual Test ---")

    # Dummy config that would come from TTSEngineManager loading wubu_config.yaml
    test_config = {
        "zonos_docker_image": "zyphra/zonos-v0.1-transformer:latest", # Or your custom image
        "zonos_model_name_in_container": "Zyphra/Zonos-v0.1-transformer",
        "device_in_container": "cpu", # "cuda" if your Docker setup supports it
        # "default_reference_audio_path": "path/to/your/host_speaker_ref.wav" # This would be self.default_voice
    }

    # Path to a dummy reference audio file on the HOST for testing
    # Create a dummy .wav file for testing (requires soundfile and numpy)
    host_dummy_ref_audio_path = None
    temp_dir_for_dummy = tempfile.mkdtemp(prefix="wubu_zonos_test_dummy_")
    try:
        import soundfile as sf
        import numpy as np
        dummy_samplerate = 44100
        dummy_duration = 1
        dummy_freq = 440
        t = np.linspace(0, dummy_duration, int(dummy_samplerate * dummy_duration), False)
        dummy_audio_data = 0.5 * np.sin(2 * np.pi * dummy_freq * t)
        dummy_audio_data = dummy_audio_data.astype(np.float32)

        host_dummy_ref_audio_path = os.path.join(temp_dir_for_dummy, "host_dummy_ref.wav")
        sf.write(host_dummy_ref_audio_path, dummy_audio_data, dummy_samplerate)
        print(f"Created host dummy reference audio: {host_dummy_ref_audio_path}")
    except ImportError:
        print("Skipping dummy audio creation: soundfile or numpy not installed in test environment.")
    except Exception as e:
        print(f"Error creating dummy audio: {e}")

    # Instantiate ZonosVoice
    # default_voice (3rd arg) would be the 'default_reference_audio_path' from config
    zonos_tts = ZonosVoice(language="en", default_voice=host_dummy_ref_audio_path, config=test_config)

    if not zonos_tts.is_docker_ok:
        print("Docker is not OK. Cannot proceed with test synthesis.")
    elif not os.path.exists(zonos_tts.host_docker_entry_script_path):
        print(f"Docker entry script zonos_docker_entry.py not found at expected location: {zonos_tts.host_docker_entry_script_path}")
    else:
        text_to_speak = "Hello from WuBu using Zonos via Docker. This is a test."

        # Test with default voice (which is host_dummy_ref_audio_path if created)
        print(f"\nAttempting to synthesize with default voice (ref: {zonos_tts.default_voice})...")
        output_wav_file_default = os.path.join(temp_dir_for_dummy, "zonos_test_docker_output_default.wav")
        success_default = zonos_tts.synthesize_to_file(text_to_speak, output_wav_file_default, speed=TTSPlaybackSpeed.NORMAL)
        if success_default:
            print(f"Conceptual success: Synthesized to {output_wav_file_default} (using default voice)")
            # To actually play: zonos_tts.play_synthesized_bytes(open(output_wav_file_default, 'rb').read())
        else:
            print(f"Conceptual failure: Failed to synthesize with default voice.")

        # Test without any specific voice_id (if default_voice was also None or invalid)
        # This relies on zonos_docker_entry.py and Zonos model handling speaker=None
        original_default = zonos_tts.default_voice
        zonos_tts.default_voice = None # Temporarily remove default to test this case
        print(f"\nAttempting to synthesize without any specific speaker reference (speaker=None in container)...")
        output_wav_file_no_speaker = os.path.join(temp_dir_for_dummy, "zonos_test_docker_output_no_speaker.wav")
        success_no_speaker = zonos_tts.synthesize_to_file(text_to_speak, output_wav_file_no_speaker, speed=TTSPlaybackSpeed.FAST)
        if success_no_speaker:
            print(f"Conceptual success: Synthesized to {output_wav_file_no_speaker} (no specific speaker)")
        else:
            print(f"Conceptual failure: Failed to synthesize (no specific speaker).")
        zonos_tts.default_voice = original_default # Restore

    if os.path.exists(temp_dir_for_dummy):
        try:
            shutil.rmtree(temp_dir_for_dummy)
            print(f"Cleaned up test dummy directory: {temp_dir_for_dummy}")
        except Exception as e_clean:
            print(f"Error cleaning up test dummy directory {temp_dir_for_dummy}: {e_clean}")

    print("--- ZonosVoice (Docker-based) Conceptual Test Finished ---")
