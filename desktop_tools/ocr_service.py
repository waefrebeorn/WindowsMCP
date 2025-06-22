import pytesseract
from PIL import Image
from typing import List, TypedDict, Optional
import logging
import os

logger = logging.getLogger(__name__)

# Default OCR confidence threshold if not specified or configurable later via dispatcher
DEFAULT_OCR_CONFIDENCE_THRESHOLD = 0  # Filters out Tesseract's negative confidence values

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
    conf: float
    text: str

def get_text_and_bounding_boxes(
    image: Image.Image,
    lang: Optional[str] = None,
    tesseract_cmd: Optional[str] = None, # Optional path to Tesseract executable
    confidence_threshold: float = DEFAULT_OCR_CONFIDENCE_THRESHOLD
) -> List[OCRData]:
    """
    Extracts text and bounding box information from an image using Tesseract OCR.
    Args:
        image: A PIL.Image.Image object to process.
        lang: Optional. Language string for Tesseract (e.g., 'eng', 'fra').
        tesseract_cmd: Optional. Path to the Tesseract executable.
        confidence_threshold: Optional. Minimum confidence score (0-100) to include results.
    Returns:
        A list of OCRData dictionaries.
    """
    results: List[OCRData] = []
    try:
        if tesseract_cmd and os.path.exists(tesseract_cmd):
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
            logger.info(f"Using Tesseract executable from parameter: {tesseract_cmd}")

        ocr_df = pytesseract.image_to_data(
            image, lang=lang, output_type=pytesseract.Output.DATAFRAME
        )

        if not (0 <= confidence_threshold <= 100):
            logger.warning(
                f"OCR confidence_threshold '{confidence_threshold}' is out of range (0-100). "
                f"Using default: {DEFAULT_OCR_CONFIDENCE_THRESHOLD}."
            )
            conf_filter = float(DEFAULT_OCR_CONFIDENCE_THRESHOLD)
        else:
            conf_filter = float(confidence_threshold)

        logger.debug(f"Using OCR confidence threshold for filtering: > {conf_filter}")
        ocr_df = ocr_df[ocr_df.conf > conf_filter]
        ocr_df = ocr_df.dropna(subset=["text"])
        ocr_df["text"] = ocr_df["text"].astype(str).str.strip()
        ocr_df = ocr_df[ocr_df.text != ""]

        for i, row in ocr_df.iterrows():
            results.append(
                OCRData(
                    level=int(row["level"]), page_num=int(row["page_num"]),
                    block_num=int(row["block_num"]), par_num=int(row["par_num"]),
                    line_num=int(row["line_num"]), word_num=int(row["word_num"]),
                    left=int(row["left"]), top=int(row["top"]),
                    width=int(row["width"]), height=int(row["height"]),
                    conf=float(row["conf"]), text=str(row["text"]),
                )
            )
        logger.info(f"OCR processed image, found {len(results)} text segments meeting confidence > {conf_filter}.")

    except pytesseract.TesseractNotFoundError:
        logger.error("Tesseract is not installed or not in your PATH/configured correctly.")
    except RuntimeError as e:
        logger.error(f"Runtime error during Tesseract OCR processing: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred in OCR processing: {e}", exc_info=True)
    return results

# Moondream related OCR (`get_text_with_moondream`) is removed as DesktopToolDispatcher
# now handles Moondream interaction directly via its own MoondreamV2 instance
# when the 'analyze_image_with_vision_model' tool is called.

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("OCR Service Module Example (Tesseract only now)")

    dummy_image_created = False
    img = None
    try:
        from PIL import ImageDraw, ImageFont
        img = Image.new("RGB", (400, 100), color=(230, 230, 230))
        draw = ImageDraw.Draw(img)
        try: font = ImageFont.truetype("arial.ttf", size=24)
        except IOError: font = ImageFont.load_default()
        draw.text((10, 10), "Tesseract Test Text 123", fill=(0,0,0), font=font)
        draw.text((10, 50), "Another line for OCR.", fill=(30,30,30), font=font)
        dummy_image_created = True
        logger.info("Created a dummy image for Tesseract testing.")
    except ImportError: logger.error("Pillow (PIL) for ImageDraw/Font not available for dummy image.")
    except Exception as e: logger.error(f"Error creating dummy image: {e}")

    if img:
        logger.info("\n--- Testing Tesseract OCR ---")
        try:
            # Example: If Tesseract is not in PATH, you might pass the command path:
            # tesseract_executable_path = "C:/Program Files/Tesseract-OCR/tesseract.exe" # Windows example
            # results = get_text_and_bounding_boxes(img, tesseract_cmd=tesseract_executable_path)
            results = get_text_and_bounding_boxes(img, confidence_threshold=30) # Example with threshold
            if results:
                logger.info(f"Tesseract found {len(results)} text segments (words/phrases):")
                for item in results:
                    if item["level"] == 5: # Word level
                        logger.info(f"  Text: '{item['text']}', Conf: {item['conf']:.2f}, Box: (L:{item['left']}, T:{item['top']}, W:{item['width']}, H:{item['height']})")
            else:
                logger.warning("No text segments found by Tesseract OCR or an error occurred.")
        except pytesseract.TesseractNotFoundError:
             logger.error("Tesseract is not installed or not in your PATH. Tesseract test cannot run.")
        except Exception as e:
            logger.error(f"An error occurred during Tesseract test: {e}", exc_info=True)
    else:
        logger.warning("Dummy image not created. Skipping Tesseract OCR test.")

    logger.info("\nFor Tesseract: Ensure Tesseract OCR is installed and in PATH (or provide tesseract_cmd_path).")
