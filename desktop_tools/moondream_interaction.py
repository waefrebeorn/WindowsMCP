import io
import os
import json
import logging
from PIL import Image

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
except ImportError:
    print(
        "Transformers or PyTorch not installed. Please install them: pip install transformers torch"
    )
    # Re-raise or handle as appropriate for your application's startup
    raise

# Configuration will be passed to MoondreamV2 instance if needed, or handled by ToolDispatcher.
# try:
#     from config_manager import config as global_config
# except ImportError:
#     global_config = {}

logger = logging.getLogger(__name__)


class MoondreamV2:
    """
    Encapsulates the Moondream V2 model interaction using the Hugging Face transformers library.
    """

    def __init__(self, model_id: str = "vikhyatk/moondream2", revision: str = "2025-06-21"):
        self.model_id = model_id
        self.revision = revision
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device} for MoondreamV2")

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                revision=self.revision,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32, # float16 for GPU, float32 for CPU
                # device_map="auto" # Let transformers handle device mapping if multiple GPUs or complex setups
                # Using explicit .to(self.device) instead of device_map for simplicity with single device focus
            ).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision)
            logger.info(f"MoondreamV2 model '{self.model_id}' (revision {self.revision}) loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load MoondreamV2 model '{self.model_id}': {e}", exc_info=True)
            # Depending on application requirements, you might want to re-raise or handle this
            # such that the application can continue without vision capabilities or exit gracefully.
            self.model = None
            self.tokenizer = None
            raise # Re-raise for now, as vision is critical for this module's purpose

    def _check_model_loaded(self):
        if not self.model or not self.tokenizer:
            # This state should ideally be prevented by __init__ raising an error.
            # If it occurs, it's a critical issue.
            raise RuntimeError("MoondreamV2 model is not loaded. Cannot perform operations.")

    def caption(self, image: Image.Image, length: str = "normal", stream: bool = False) -> dict:
        """
        Generates a caption for the given image.

        Args:
            image: A PIL Image object.
            length: "short" or "normal".
            stream: Whether to stream the output (not fully handled in this wrapper, returns full).

        Returns:
            A dictionary containing the caption or an error.
            Example: {"caption": "A description of the image."}
                     {"error": "Details of the error."}
        """
        self._check_model_loaded()
        if not isinstance(image, Image.Image):
            return {"error": "Invalid image_input type. Must be a PIL.Image.Image object."}

        try:
            # Moondream's caption method itself returns a dict with 'caption'
            # and handles streaming internally if stream=True (though this wrapper doesn't expose the stream directly)
            if stream:
                # The example shows iterating over model.caption(..., stream=True)['caption']
                # For a non-streaming wrapper, we'd collect it.
                # However, the model.caption itself returns the full dict.
                # If we wanted to stream out, this function's signature and usage would need to change.
                # For now, let's assume stream=False for simplicity of return type.
                logger.warning("Streaming for caption is requested but this wrapper returns the full result.")

            # The model.caption directly returns a dictionary like {'caption': 'text'}
            # For stream=True, model.caption returns {'caption': <generator object ...>}
            # Let's stick to the direct output of the model's method.
            caption_result = self.model.caption(image, length=length, stream=stream, tokenizer=self.tokenizer)

            if stream and 'caption' in caption_result and hasattr(caption_result['caption'], '__iter__'):
                # Collect from stream if it's a generator
                return {"caption": "".join([token for token in caption_result['caption']])}
            elif 'caption' in caption_result:
                 return {"caption": str(caption_result["caption"])} # Ensure it's a string
            else:
                return {"error": "Caption generation did not return expected 'caption' key."}

        except Exception as e:
            logger.error(f"Error during MoondreamV2 captioning: {e}", exc_info=True)
            return {"error": f"Captioning failed: {e}"}

    def query(self, image: Image.Image, question: str) -> dict:
        """
        Answers a question about the given image.

        Args:
            image: A PIL Image object.
            question: The question to ask about the image.

        Returns:
            A dictionary containing the answer or an error.
            Example: {"answer": "The answer to the question."}
                     {"error": "Details of the error."}
        """
        self._check_model_loaded()
        if not isinstance(image, Image.Image):
            return {"error": "Invalid image_input type. Must be a PIL.Image.Image object."}
        if not isinstance(question, str) or not question.strip():
            return {"error": "Question must be a non-empty string."}

        try:
            # The model.query directly returns a dictionary like {'answer': 'text'}
            query_result = self.model.query(image, question, tokenizer=self.tokenizer)
            if 'answer' in query_result:
                return {"answer": str(query_result["answer"])} # Ensure it's string
            else:
                return {"error": "Query processing did not return expected 'answer' key."}
        except Exception as e:
            logger.error(f"Error during MoondreamV2 query: {e}", exc_info=True)
            return {"error": f"Query failed: {e}"}

    def detect(self, image: Image.Image, object_name: str) -> dict:
        """
        Detects objects of a given name in the image.

        Args:
            image: A PIL Image object.
            object_name: The name of the object to detect (e.g., "face").

        Returns:
            A dictionary containing the detected objects or an error.
            Example: {"objects": [{"box": [x1, y1, x2, y2], "label": "face"}, ...]}
                     {"error": "Details of the error."}
        """
        self._check_model_loaded()
        # ... (implementation similar to query, using self.model.detect())
        # The example shows: objects = model.detect(image, "face")["objects"]
        try:
            detection_result = self.model.detect(image, object_name, tokenizer=self.tokenizer)
            if 'objects' in detection_result:
                return {"objects": detection_result["objects"]}
            else:
                return {"error": "Detection did not return expected 'objects' key."}
        except Exception as e:
            logger.error(f"Error during MoondreamV2 detect: {e}", exc_info=True)
            return {"error": f"Detection failed: {e}"}


    def point(self, image: Image.Image, object_name: str) -> dict:
        """
        Identifies points for a given object name in the image.

        Args:
            image: A PIL Image object.
            object_name: The name of the object to point to (e.g., "person").

        Returns:
            A dictionary containing the points or an error.
            Example: {"points": [[x1, y1], [x2, y2], ...]}
                     {"error": "Details of the error."}
        """
        self._check_model_loaded()
        # ... (implementation similar to query, using self.model.point())
        # The example shows: points = model.point(image, "person")["points"]
        try:
            pointing_result = self.model.point(image, object_name, tokenizer=self.tokenizer)
            if 'points' in pointing_result:
                return {"points": pointing_result["points"]}
            else:
                return {"error": "Pointing did not return expected 'points' key."}

        except Exception as e:
            logger.error(f"Error during MoondreamV2 point: {e}", exc_info=True)
            return {"error": f"Pointing failed: {e}"}

# --- Global Moondream Analyzer Instance (REMOVED) ---
# Initialization and management of MoondreamV2 instance will be handled by DesktopToolDispatcher
# based on configuration.

# The analyze_image_with_moondream function is also removed from here.
# DesktopToolDispatcher will call methods directly on its MoondreamV2 instance.
# If a standalone utility function is ever needed, it should be designed to accept a MoondreamV2 instance.


if __name__ == "__main__":
    # Configure basic logging for the example
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("Moondream Interaction Module Example (Transformers-based)")

    # To test MoondreamV2 class directly:
    try:
        logger.info("Attempting to initialize MoondreamV2 directly for testing...")
        # Example: Using default model_id and revision.
        # In real use, these could come from a config passed to DesktopToolDispatcher.
        test_analyzer = MoondreamV2(model_id="vikhyatk/moondream2", revision="2025-06-21") # Use the documented revision
        if not test_analyzer.model:
             logger.error("Moondream model within test_analyzer is not loaded. Example cannot run.")
             exit(1)
        logger.info("MoondreamV2 test_analyzer initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize MoondreamV2 for direct test: {e}", exc_info=True)
        logger.error("Ensure 'transformers' and 'torch' are installed and model is accessible.")
        exit(1)


    # 1. Create a dummy image for testing
    test_image_path = "test_image_moondream_transformers.png"
    try:
        from PIL import ImageDraw, ImageFont

        img = Image.new("RGB", (600, 150), color=(200, 200, 255))
        d = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 40)
        except IOError:
            font = ImageFont.load_default()
        d.text((10, 10), "Hello Moondream V2!", fill=(0, 0, 0), font=font)
        d.text((10, 70), "Transcribe this text.", fill=(50, 50, 50), font=font)
        img.save(test_image_path)
        logger.info(f"Created a dummy image '{test_image_path}' for testing.")

        # Test with image path for general query
        prompt1 = "What is written in the image?"
        logger.info(f"\nTesting query with image path: '{test_image_path}' and prompt: '{prompt1}'")
        result1 = analyze_image_with_moondream(test_image_path, prompt1)
        logger.info("Response from Moondream (path input, query):")
        logger.info(json.dumps(result1, indent=2))

        # Test with PIL Image object for OCR-like query
        pil_image = Image.open(test_image_path)
        prompt_ocr = "Transcribe the text in natural reading order."
        logger.info(f"\nTesting query with PIL Image object and OCR prompt: '{prompt_ocr}'")
        result_ocr = analyze_image_with_moondream(pil_image, prompt_ocr)
        logger.info("Response from Moondream (PIL input, OCR query):")
        logger.info(json.dumps(result_ocr, indent=2))
        if result_ocr.get("status") == "success":
            logger.info(f"Extracted text for OCR: {result_ocr['data']['text']}")


        # Test captioning
        logger.info(f"\nTesting captioning with PIL Image object:")
        result_caption = moondream_analyzer.caption(pil_image, length="normal")
        logger.info("Response from Moondream (PIL input, caption):")
        logger.info(json.dumps(result_caption, indent=2))
        if "caption" in result_caption:
            logger.info(f"Generated caption: {result_caption['caption']}")

        # Test detection (example)
        logger.info(f"\nTesting detection with PIL Image object (detect 'text'):")
        # Note: "text" might be a conceptual object. Moondream's detection capabilities
        # might be more geared towards common objects like "cat", "dog", "car", "figure", "table" etc.
        # For document layout, it supports "figure", "formula", "text" etc.
        result_detect = moondream_analyzer.detect(pil_image, "text") # Example: try to detect "text" regions
        logger.info("Response from Moondream (PIL input, detect 'text'):")
        logger.info(json.dumps(result_detect, indent=2))
        if "objects" in result_detect:
            logger.info(f"Detected {len(result_detect['objects'])} 'text' object(s).")


    except ImportError:
        logger.error("Pillow (PIL) is not fully available. Cannot create dummy image or test with PIL object.")
    except RuntimeError as re:
        logger.error(f"A runtime error occurred, possibly during model interaction: {re}", exc_info=True)
    except Exception as e:
        logger.error(f"An error occurred during the example run: {e}", exc_info=True)
        logger.error("Ensure you have 'transformers', 'torch', and 'Pillow' installed.")
        logger.error("If using GPU, ensure CUDA is set up correctly.")

    finally:
        # Clean up dummy image
        if os.path.exists(test_image_path):
            try:
                os.remove(test_image_path)
                logger.info(f"\nCleaned up dummy image '{test_image_path}'.")
            except Exception as e:
                logger.error(f"Error cleaning up dummy image: {e}")
