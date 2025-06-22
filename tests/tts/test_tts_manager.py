# Tests for the WuBu TTSEngineManager.
# Verifies loading, switching, and routing of synthesis requests to mock engines.

import unittest
from unittest.mock import patch, MagicMock, Mock

# Define mock voice classes for testing the manager
# These simulate the actual voice classes from wubu.tts.glados_voice and wubu.tts.kokoro_voice
class MockWubuGLaDOSStyleVoice(MagicMock):
    VOICE_ID = "Testable-WuBu-GLaDOS-Style" # Unique ID for mock
    def __init__(self, language='en', model_subdir_override=None, speaker_wav_filename_override=None, config=None, use_gpu=True):
        super().__init__()
        self.tts_engine = Mock()
        self.default_voice = self.VOICE_ID
        self.language = language
        self.config = config
        print(f"MockWubuGLaDOSStyleVoice initialized with lang: {language}, config: {config.get('custom_mock_param', 'default') if config else 'N/A'}")

    def speak(self, text, voice_id=None, speed=None, **kwargs):
        print(f"MockWubuGLaDOSStyleVoice speaking: '{text}' with voice {voice_id or self.default_voice}, speed {speed}")

    def synthesize_to_file(self, text, output_filename, voice_id=None, speed=None, **kwargs):
        print(f"MockWubuGLaDOSStyleVoice synthesizing to file '{output_filename}': '{text}'")
        return True

    def synthesize_to_bytes(self, text, voice_id=None, speed=None, **kwargs):
        print(f"MockWubuGLaDOSStyleVoice synthesizing to bytes: '{text}'")
        return b"mock_wubu_glados_style_audio_bytes"

    def get_available_voices(self):
        return [{"id": self.VOICE_ID, "name": "Mock WuBu GLaDOS-Style Voice", "language": self.language}]


class MockWubuKokoroVoice(MagicMock):
    VOICE_ID = "Testable-WuBu-Kokoro" # Unique ID for mock
    ENGINE_TYPE_COQUI = "coqui"
    def __init__(self, language='en', engine_type=None, model_name_or_path_override=None, config=None, use_gpu=True):
        super().__init__()
        self.tts_engine = Mock()
        self.default_voice = self.VOICE_ID
        self.language = language
        self.config = config
        print(f"MockWubuKokoroVoice initialized with lang: {language}, config: {config.get('custom_mock_param', 'default') if config else 'N/A'}")

    def speak(self, text, voice_id=None, speed=None, **kwargs):
        print(f"MockWubuKokoroVoice speaking: '{text}' with voice {voice_id or self.default_voice}, speed {speed}")

    def synthesize_to_file(self, text, output_filename, voice_id=None, speed=None, **kwargs):
        print(f"MockWubuKokoroVoice synthesizing to file '{output_filename}': '{text}'")
        return True

    def synthesize_to_bytes(self, text, voice_id=None, speed=None, **kwargs):
        print(f"MockWubuKokoroVoice synthesizing to bytes: '{text}'")
        return b"mock_wubu_kokoro_audio_bytes"

    def get_available_voices(self):
        return [{"id": self.VOICE_ID, "name": "Mock WuBu Kokoro Voice", "language": self.language}]


# Patch the actual voice classes imported by TTSEngineManager with our mocks
# The paths must match where TTSEngineManager imports them from.
@patch('wubu.tts.tts_engine_manager.WubuGLaDOSStyleVoice', new=MockWubuGLaDOSStyleVoice)
@patch('wubu.tts.tts_engine_manager.WubuKokoroVoice', new=MockWubuKokoroVoice)
class TestTTSEngineManager(unittest.TestCase):

    TTSPlaybackSpeed = None # Will be set in setUp

    def setUp(self):
        from wubu.tts.tts_engine_manager import TTSEngineManager
        from wubu.tts.base_tts_engine import TTSPlaybackSpeed # For type hinting / enum values
        self.TTSEngineManager = TTSEngineManager
        self.TTSPlaybackSpeed = TTSPlaybackSpeed

        self.mock_config_both_enabled = {
            'tts': {
                'default_voice': MockWubuGLaDOSStyleVoice.VOICE_ID,
                'wubu_glados_style_voice': {'enabled': True, 'language': 'en-gb', 'custom_mock_param': 'glados_gb'},
                'wubu_kokoro_voice': {'enabled': True, 'language': 'en-us', 'custom_mock_param': 'kokoro_us'}
            }
        }
        self.manager_both = self.TTSEngineManager(config=self.mock_config_both_enabled)

        self.mock_config_glados_only = {
            'tts': {
                'default_voice': MockWubuGLaDOSStyleVoice.VOICE_ID,
                'wubu_glados_style_voice': {'enabled': True}, # Uses default lang 'en' from mock
                'wubu_kokoro_voice': {'enabled': False}
            }
        }
        self.manager_glados_only = self.TTSEngineManager(config=self.mock_config_glados_only)

        self.mock_config_kokoro_default = {
             'tts': {
                'default_voice': MockWubuKokoroVoice.VOICE_ID,
                'wubu_glados_style_voice': {'enabled': True},
                'wubu_kokoro_voice': {'enabled': True, 'language': 'ja', 'custom_mock_param': 'kokoro_jp'}
            }
        }
        self.manager_kokoro_default = self.TTSEngineManager(config=self.mock_config_kokoro_default)

        self.mock_config_no_engines = {'tts': {
            'wubu_glados_style_voice': {'enabled': False},
            'wubu_kokoro_voice': {'enabled': False}
        }}
        self.manager_none = self.TTSEngineManager(config=self.mock_config_no_engines)

    def test_initialization_and_engine_loading(self):
        self.assertIn(MockWubuGLaDOSStyleVoice.VOICE_ID, self.manager_both.engines)
        self.assertIn(MockWubuKokoroVoice.VOICE_ID, self.manager_both.engines)
        self.assertIsInstance(self.manager_both.engines[MockWubuGLaDOSStyleVoice.VOICE_ID], MockWubuGLaDOSStyleVoice)
        self.assertEqual(self.manager_both.engines[MockWubuGLaDOSStyleVoice.VOICE_ID].language, 'en-gb')
        self.assertEqual(self.manager_both.engines[MockWubuKokoroVoice.VOICE_ID].language, 'en-us')

        self.assertIn(MockWubuGLaDOSStyleVoice.VOICE_ID, self.manager_glados_only.engines)
        self.assertNotIn(MockWubuKokoroVoice.VOICE_ID, self.manager_glados_only.engines)
        self.assertEqual(len(self.manager_none.engines), 0)

    def test_default_engine_selection(self):
        self.assertEqual(self.manager_both.default_engine_id, MockWubuGLaDOSStyleVoice.VOICE_ID)
        engine_glados = self.manager_both.get_engine()
        self.assertIsInstance(engine_glados, MockWubuGLaDOSStyleVoice)

        self.assertEqual(self.manager_glados_only.default_engine_id, MockWubuGLaDOSStyleVoice.VOICE_ID)

        self.assertEqual(self.manager_kokoro_default.default_engine_id, MockWubuKokoroVoice.VOICE_ID)
        engine_kokoro = self.manager_kokoro_default.get_engine()
        self.assertIsInstance(engine_kokoro, MockWubuKokoroVoice)
        self.assertEqual(engine_kokoro.language, 'ja')

        self.assertIsNone(self.manager_none.default_engine_id)
        self.assertIsNone(self.manager_none.get_engine())

    def test_get_specific_engine(self):
        glados_engine = self.manager_both.get_engine(MockWubuGLaDOSStyleVoice.VOICE_ID)
        self.assertIsInstance(glados_engine, MockWubuGLaDOSStyleVoice)

        kokoro_engine = self.manager_both.get_engine(MockWubuKokoroVoice.VOICE_ID)
        self.assertIsInstance(kokoro_engine, MockWubuKokoroVoice)

        self.assertIsNone(self.manager_glados_only.get_engine(MockWubuKokoroVoice.VOICE_ID))
        self.assertIsNone(self.manager_both.get_engine("NonExistentWuBuVoice"))

    def test_speak_routes_to_correct_engine(self):
        self.manager_both.engines[MockWubuGLaDOSStyleVoice.VOICE_ID].reset_mock()
        self.manager_both.speak("Hello from WuBu default")
        self.manager_both.engines[MockWubuGLaDOSStyleVoice.VOICE_ID].speak.assert_called_once_with(
            "Hello from WuBu default", voice_id=MockWubuGLaDOSStyleVoice.VOICE_ID, speed=self.TTSPlaybackSpeed.NORMAL)

        self.manager_both.engines[MockWubuKokoroVoice.VOICE_ID].reset_mock()
        self.manager_both.speak("Hello from WuBu Kokoro", voice_id=MockWubuKokoroVoice.VOICE_ID, speed=self.TTSPlaybackSpeed.FAST)
        self.manager_both.engines[MockWubuKokoroVoice.VOICE_ID].speak.assert_called_once_with(
            "Hello from WuBu Kokoro", voice_id=MockWubuKokoroVoice.VOICE_ID, speed=self.TTSPlaybackSpeed.FAST)

    def test_synthesize_to_file_routes_correctly(self):
        self.manager_kokoro_default.engines[MockWubuKokoroVoice.VOICE_ID].reset_mock()
        self.manager_kokoro_default.synthesize_to_file("Test WuBu to file", "output.wav", voice_id=MockWubuKokoroVoice.VOICE__ID)
        self.manager_kokoro_default.engines[MockWubuKokoroVoice.VOICE_ID].synthesize_to_file.assert_called_once_with(
            "Test WuBu to file", "output.wav", voice_id=MockWubuKokoroVoice.VOICE_ID, speed=self.TTSPlaybackSpeed.NORMAL)

    def test_synthesize_to_bytes_routes_correctly(self):
        self.manager_both.engines[MockWubuGLaDOSStyleVoice.VOICE_ID].reset_mock()
        result_bytes = self.manager_both.synthesize_to_bytes("Test WuBu to bytes", voice_id=MockWubuGLaDOSStyleVoice.VOICE_ID)
        self.manager_both.engines[MockWubuGLaDOSStyleVoice.VOICE_ID].synthesize_to_bytes.assert_called_once_with(
            "Test WuBu to bytes", voice_id=MockWubuGLaDOSStyleVoice.VOICE_ID, speed=self.TTSPlaybackSpeed.NORMAL)
        self.assertEqual(result_bytes, b"mock_wubu_glados_style_audio_bytes")

    def test_get_available_voices_aggregation(self):
        voices = self.manager_both.get_available_voices()
        self.assertEqual(len(voices), 2)
        voice_ids = [v['id'] for v in voices]
        self.assertIn(MockWubuGLaDOSStyleVoice.VOICE_ID, voice_ids)
        self.assertIn(MockWubuKokoroVoice.VOICE_ID, voice_ids)

        voices_glados_only = self.manager_glados_only.get_available_voices()
        self.assertEqual(len(voices_glados_only), 1)
        self.assertEqual(voices_glados_only[0]['id'], MockWubuGLaDOSStyleVoice.VOICE_ID)

        voices_none = self.manager_none.get_available_voices()
        self.assertEqual(len(voices_none), 0)

if __name__ == '__main__':
    print("Running WuBu TTSEngineManager tests (using Mocks for voice engines)...")
    # To run these tests correctly, especially with patching:
    # python -m unittest tests.tts.test_tts_manager (from project root)
    # Or ensure 'src' is in PYTHONPATH if running directly for wubu.tts... imports to resolve.
    unittest.main()
