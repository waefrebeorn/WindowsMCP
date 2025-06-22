import pytesseract
from PIL import Image
from typing import List, TypedDict, Optional  # Removed Dict
import logging
import os  # For os.path.exists

logger = logging.getLogger(__name__)
# pandas is imported by pytesseract.image_to_data if output_type is DataFrame

# Attempt to import global config for TESSERACT_CMD_PATH
try:
    from config_manager import config as global_config
except ImportError:
    global_config = {}


class OCRData(TypedDict):
    level: int
    page_num: int
    block_num: int
    par_num: int
    line_num: int
    word_num: int
    left: int
    top: int
    width: int
    height: int
    conf: float  # Confidence, float because Tesseract often gives it as -1 or float
    text: str


def get_text_and_bounding_boxes(
    image: Image.Image, lang: Optional[str] = None
) -> List[OCRData]:
    """
    Extracts text and bounding box information from an image using Tesseract OCR.

    Args:
        image: A PIL.Image.Image object to process.
        lang: Optional. Language string for Tesseract (e.g., 'eng', 'fra').
              If None, Tesseract will use its default.

    Returns:
        A list of OCRData dictionaries, where each dictionary contains
        information about a recognized text segment, including its bounding box
        (left, top, width, height), text content, and confidence score.
        Returns an empty list if OCR fails or no text is found.
    """
    results: List[OCRData] = []
    try:
        # Check for Tesseract command path in config
        tesseract_cmd = global_config.get("TESSERACT_CMD_PATH")
        if tesseract_cmd:
            if os.path.exists(
                tesseract_cmd
            ):  # Better check if it's a file and executable
                pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
                logger.info(f"Using Tesseract executable from config: {tesseract_cmd}")
            else:
                logger.warning(
                    f"TESSERACT_CMD_PATH '{tesseract_cmd}' in config not found. Using default PATH."
                )

        # Get verbose data including boxes, confidences, line and page numbers
        # Output type can be DataFrame, string, bytes, dict. We'll use DataFrame.
        ocr_df = pytesseract.image_to_data(
            image, lang=lang, output_type=pytesseract.Output.DATAFRAME
        )

        # Filter out entries with low confidence or no text
        # Tesseract uses -1 for confidence when it's not applicable.
        # Use configured confidence threshold.
        conf_threshold_config = global_config.get(
            "OCR_CONFIDENCE_THRESHOLD", "0"
        )  # Get as string first
        try:
            conf_threshold = float(conf_threshold_config)
            # Tesseract confidence is usually -1 (for non-text) or 0-100 for text.
            # A threshold of 0 will filter out negative confidences.
            if not (0 <= conf_threshold <= 100):
                logger.warning(
                    f"OCR_CONFIDENCE_THRESHOLD '{conf_threshold_config}' is out of range (0-100). "
                    f"Using a threshold of 0."
                )
                conf_threshold = 0
        except ValueError:
            logger.warning(
                f"Invalid OCR_CONFIDENCE_THRESHOLD '{conf_threshold_config}'. Using a threshold of 0."
            )
            conf_threshold = 0

        logger.info(f"Using OCR confidence threshold for filtering: > {conf_threshold}")
        # Filter based on the threshold.
        ocr_df = ocr_df[ocr_df.conf > conf_threshold]

        ocr_df = ocr_df.dropna(subset=["text"])  # Remove rows where text is NaN
        ocr_df["text"] = (
            ocr_df["text"].astype(str).str.strip()
        )  # Ensure text is string and stripped
        ocr_df = ocr_df[
            ocr_df.text != ""
        ]  # Remove rows with empty string text after stripping

        for i, row in ocr_df.iterrows():
            results.append(
                OCRData(
                    level=int(row["level"]),
                    page_num=int(row["page_num"]),
                    block_num=int(row["block_num"]),
                    par_num=int(row["par_num"]),
                    line_num=int(row["line_num"]),
                    word_num=int(row["word_num"]),
                    left=int(row["left"]),
                    top=int(row["top"]),
                    width=int(row["width"]),
                    height=int(row["height"]),
                    conf=float(row["conf"]),
                    text=str(row["text"]),
                )
            )

        logger.info(
            f"OCR processed image, found {len(results)} text segments with positive confidence."
        )

    except pytesseract.TesseractNotFoundError:
        logger.error(
            "Tesseract is not installed or not in your PATH. "
            "Please install Tesseract OCR: https://tesseract-ocr.github.io/tessdoc/Installation.html"
        )
        # Optionally, re-raise or return a specific error code/message
        # For now, returns empty list, caller should check.
    except RuntimeError as e:  # Can be raised by tesseract for various reasons
        logger.error(f"Runtime error during Tesseract OCR processing: {e}")
    except Exception as e:
        logger.error(
            f"An unexpected error occurred in OCR processing: {e}", exc_info=True
        )

    return results


# --- Moondream v2 based OCR ---
try:
    from .moondream_interaction import analyze_image_with_moondream as analyze_with_moondream_v2
    from .moondream_interaction import moondream_analyzer as global_moondream_analyzer
    MOONDREAM_AVAILABLE = True
    if global_moondream_analyzer is None or global_moondream_analyzer.model is None:
        MOONDREAM_AVAILABLE = False
        logger.info("Moondream analyzer or model not loaded; Moondream OCR will not be available.")
except ImportError:
    logger.warning(
        "Could not import moondream_interaction. Moondream OCR will not be available."
    )
    MOONDREAM_AVAILABLE = False
    global_moondream_analyzer = None # Ensure it's defined for checks later

def get_text_with_moondream(image: Image.Image, prompt: Optional[str] = None) -> List[OCRData]:
    """
    Extracts text from an image using Moondream V2.
    Moondream provides full text transcription rather than detailed bounding boxes per word/line.
    This function returns a single OCRData item representing the full transcription.

    Args:
        image: A PIL.Image.Image object to process.
        prompt: Optional custom prompt for Moondream. If None, a default OCR prompt is used.

    Returns:
        A list containing a single OCRData dictionary with the transcribed text,
        or an empty list if OCR fails or no text is found. Bounding box information
        will be for the whole image, and confidence will be placeholder -1.0.
    """
    if not MOONDREAM_AVAILABLE:
        logger.warning("Moondream V2 is not available. Cannot perform OCR with Moondream.")
        return []

    ocr_prompt = prompt if prompt else "Transcribe the text in natural reading order."
    logger.info(f"Performing OCR with MoondreamV2 using prompt: '{ocr_prompt}'")

    try:
        # analyze_image_with_moondream expects image path or PIL.Image
        # It returns a dict like:
        # {"status": "success", "data": {"text": "...", "raw_response": ...}}
        # or {"error": "..."}
        result = analyze_with_moondream_v2(image, ocr_prompt)

        if result.get("status") == "success" and result.get("data") and "text" in result["data"]:
            full_text = result["data"]["text"]
            if full_text and full_text.strip():
                # Moondream doesn't give detailed bounding boxes or confidence scores like Tesseract.
                # We'll create a single OCRData entry for the whole image text.
                # Level 1 (page), block 1, par 1, line 1, word 1
                # Dimensions are for the entire image.
                width, height = image.size
                return [
                    OCRData(
                        level=1, # Page level
                        page_num=1,
                        block_num=1,
                        par_num=1,
                        line_num=1,
                        word_num=1,
                        left=0,
                        top=0,
                        width=width,
                        height=height,
                        conf=-1.0,  # Moondream doesn't provide per-word confidence
                        text=full_text.strip(),
                    )
                ]
            else:
                logger.info("Moondream OCR returned empty text.")
                return []
        else:
            error_message = result.get("error", "Unknown error from Moondream V2.")
            logger.error(f"Moondream V2 OCR failed: {error_message}")
            return []

    except Exception as e:
        logger.error(f"An unexpected error occurred during Moondream V2 OCR: {e}", exc_info=True)
        return []


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("OCR Service Module Example")

    # Create a dummy image with text
    dummy_image_created = False
    try:
        from PIL import ImageDraw, ImageFont  # Import for example only
        img = Image.new("RGB", (600, 200), color=(220, 220, 255))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", size=30)
        except IOError:
            font = ImageFont.load_default()

        draw.text((10, 10), "Hello Tesseract!", fill=(0, 0, 0), font=font)
        draw.text((30, 60), "This is line two for Tesseract.", fill=(50, 50, 50), font=font)
        draw.text((10, 110), "Moondream, can you read this?", fill=(0, 0, 100), font=font)
        draw.text((30, 150), "And this line for Moondream too.", fill=(0, 50, 100), font=font)
        dummy_image_created = True
        logger.info("Created a dummy image for testing.")
    except ImportError:
        logger.error("Pillow (PIL) for ImageDraw/ImageFont not available. Cannot create dummy image for full test.")
        img = None # Ensure img is None if creation failed
    except Exception as e:
        logger.error(f"Error creating dummy image: {e}")
        img = None

    if img:
        # --- Test Tesseract OCR ---
        logger.info("\n--- Testing Tesseract OCR ---")
        try:
            tesseract_ocr_results = get_text_and_bounding_boxes(img)
            if tesseract_ocr_results:
                logger.info(f"Tesseract found {len(tesseract_ocr_results)} text segments (words/phrases):")
                for item in tesseract_ocr_results:
                    if item["level"] == 5: # Word level
                        logger.info(
                            f"  Text: '{item['text']}', Conf: {item['conf']:.2f}, "
                            f"Box: (L:{item['left']}, T:{item['top']}, W:{item['width']}, H:{item['height']})"
                        )
            else:
                logger.warning("No text segments found by Tesseract OCR or an error occurred.")
                logger.warning("Ensure Tesseract OCR is correctly installed and accessible.")
        except pytesseract.TesseractNotFoundError:
            logger.error("Tesseract is not installed or not in your PATH. Tesseract test cannot run.")
        except Exception as e:
            logger.error(f"An error occurred during Tesseract test: {e}", exc_info=True)

        # --- Test Moondream v2 OCR ---
        logger.info("\n--- Testing Moondream v2 OCR ---")
        if MOONDREAM_AVAILABLE:
            try:
                moondream_ocr_results = get_text_with_moondream(img)
                if moondream_ocr_results:
                    logger.info(f"Moondream found {len(moondream_ocr_results)} text block(s):")
                    for item in moondream_ocr_results:
                        logger.info(f"  Full Text: '{item['text']}'")
                        logger.info(f"  (Note: Bounding box is for the whole image, conf is placeholder for Moondream)")
                else:
                    logger.warning("No text found by Moondream v2 OCR or an error occurred.")
            except Exception as e:
                logger.error(f"An error occurred during Moondream test: {e}", exc_info=True)
        else:
            logger.warning("Moondream v2 is not available (likely not installed or configured). Skipping Moondream OCR test.")
    else:
        logger.warning("Dummy image not created. Skipping OCR tests.")

    # General advice if things fail
    if not dummy_image_created:
         logger.info("To run full tests, ensure Pillow (for ImageDraw/Font) is installed.")
    logger.info("\nFor Tesseract: Ensure Tesseract OCR is installed and in PATH (or TESSERACT_CMD_PATH in config).")
    logger.info("For Moondream: Ensure 'transformers' and 'torch' are installed, and the model can be downloaded/loaded.")
    logger.info("If using GPU for Moondream, ensure CUDA is set up correctly.")
