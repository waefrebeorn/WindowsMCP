import pyttsx3
import logging
import asyncio

logger = logging.getLogger(__name__)

_tts_engine = None
_tts_engine_initialized_successfully = False

# Default values if no configuration is passed (e.g. if used standalone)
DEFAULT_TTS_VOICE_ID = None
DEFAULT_TTS_RATE = 150
DEFAULT_TTS_VOLUME = 0.9

def _initialize_tts_engine(voice_id=None, rate=None, volume=None):
    """
    Initializes the TTS engine (pyttsx3) if not already done.
    Accepts optional parameters to override defaults.
    """
    global _tts_engine, _tts_engine_initialized_successfully
    if _tts_engine_initialized_successfully: # Already successfully initialized
        return _tts_engine
    if _tts_engine is not None and not _tts_engine_initialized_successfully: # Previous attempt failed
        logger.warning("TTS engine initialization previously failed. Not re-attempting in this call.")
        return None

    if _tts_engine is None:
        try:
            logger.info("Initializing pyttsx3 engine for voice_output.py...")
            _tts_engine = pyttsx3.init()

            # Use provided params or fall back to module defaults
            effective_voice_id = voice_id if voice_id is not None else DEFAULT_TTS_VOICE_ID
            effective_rate = rate if rate is not None else DEFAULT_TTS_RATE
            effective_volume = volume if volume is not None else DEFAULT_TTS_VOLUME

            if effective_voice_id:
                try:
                    available_voices = _tts_engine.getProperty("voices")
                    if any(v.id == effective_voice_id for v in available_voices):
                        _tts_engine.setProperty("voice", effective_voice_id)
                        logger.info(f"pyttsx3 voice set to ID: {effective_voice_id}")
                    else:
                        logger.warning(f"pyttsx3 voice ID '{effective_voice_id}' not found. Using system default.")
                except Exception as e_voice:
                    logger.error(f"Error setting pyttsx3 voice ID '{effective_voice_id}': {e_voice}")

            if effective_rate is not None:
                try:
                    _tts_engine.setProperty("rate", int(effective_rate))
                    logger.info(f"pyttsx3 rate set to: {effective_rate}")
                except ValueError: logger.warning(f"Invalid pyttsx3 rate '{effective_rate}'.")

            if effective_volume is not None:
                try:
                    vol = float(effective_volume)
                    if 0.0 <= vol <= 1.0: _tts_engine.setProperty("volume", vol); logger.info(f"pyttsx3 volume set to: {vol}")
                    else: logger.warning(f"pyttsx3 volume '{vol}' out of range (0.0-1.0).")
                except ValueError: logger.warning(f"Invalid pyttsx3 volume '{effective_volume}'.")

            _tts_engine_initialized_successfully = True
            logger.info("pyttsx3 engine initialized successfully for voice_output.py.")
            return _tts_engine
        except Exception as e:
            logger.error(f"Failed to initialize pyttsx3 engine: {e}", exc_info=True)
            _tts_engine = None # Ensure it's None on failure
            _tts_engine_initialized_successfully = False # Mark as failed
            return None
    return _tts_engine


def _speak_text_sync(text: str, voice_id=None, rate=None, volume=None):
    """Synchronous part of speaking text with pyttsx3."""
    # Initialize with potentially new parameters if provided, else uses existing/defaults
    engine = _initialize_tts_engine(voice_id=voice_id, rate=rate, volume=volume)
    if engine:
        try:
            engine.say(text)
            engine.runAndWait()
            logger.info(f'pyttsx3 spoke: "{text[:50]}..."')
        except Exception as e:
            logger.error(f"Error during pyttsx3 speech: {e}", exc_info=True)
    else:
        logger.warning("pyttsx3 engine not available. Cannot speak text via voice_output.py.")


async def speak_text_async(text: str, voice_id=None, rate=None, volume=None):
    """
    Speaks text using pyttsx3, running blocking calls in a separate thread.
    This function is part of the older system. WuBuEngine uses its own TTS manager.
    """
    if not text or not text.strip():
        logger.info("voice_output.speak_text_async called with empty text. Nothing to speak.")
        return

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _speak_text_sync, text, voice_id, rate, volume)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Desktop Tools Voice Output (pyttsx3) Example")

    async def main_example():
        # Test with default settings (or whatever _initialize_tts_engine picks up first time)
        await speak_text_async("Hello from pyttsx3, using default voice output settings.")

        # Example: Test with specific (potentially different) parameters if your system supports them
        # Note: Changing voice/rate/volume after first init might not always work as expected
        # with pyttsx3 depending on the underlying driver behavior.
        # It's often best to init with desired params.
        # await speak_text_async("Speaking faster now.", rate=200)

    asyncio.run(main_example())
    logger.info("Desktop Tools Voice output example finished.")
