# Utilities for converting spoken text (ASR output) to a more canonical form,
# or for preparing text for TTS to ensure proper pronunciation of numbers, symbols, etc.

import re

# Basic number to words conversion (simple example, not exhaustive)
# More robust libraries like 'num2words' should be used for production.
_number_map = {
    '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
    '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine',
    '10': 'ten', '11': 'eleven', '12': 'twelve', '13': 'thirteen', '14': 'fourteen',
    '15': 'fifteen', '16': 'sixteen', '17': 'seventeen', '18': 'eighteen', '19': 'nineteen',
    '20': 'twenty', '30': 'thirty', '40': 'forty', '50': 'fifty',
    '60': 'sixty', '70': 'seventy', '80': 'eighty', '90': 'ninety'
}

def _num_to_words_simple(n_str):
    """Very basic number to words conversion for small integers."""
    if n_str in _number_map:
        return _number_map[n_str]
    # This is extremely limited, doesn't handle teens, hundreds, etc.
    # Placeholder for a real implementation or library.
    return n_str # Fallback to original number string

def convert_numbers_to_words(text):
    """
    Converts numerical digits in text to their word equivalents for TTS.
    Example: "Call me at 555-1234" -> "Call me at five five five one two three four" (simple digit-by-digit)
    A more advanced version would convert "Set timer for 10 minutes" -> "Set timer for ten minutes".
    """
    # This is a very naive implementation, processing digit by digit.
    # A production system would use a proper number parsing and conversion library.

    def replace_num(match):
        num_str = match.group(0)
        # Option 1: Convert whole number if possible (e.g. "10" to "ten")
        # return _num_to_words_simple(num_str)

        # Option 2: Spell out digits (e.g. "10" to "one zero" or "one oh")
        # This is often better for phone numbers, codes, etc.
        return ' '.join([_num_to_words_simple(digit) for digit in num_str])

    # Find sequences of digits
    return re.sub(r'\d+', replace_num, text)

def normalize_spoken_text(text):
    """
    Normalizes text from ASR (Automatic Speech Recognition).
    - Lowercase
    - Remove punctuation (or handle it appropriately)
    - Convert spoken numbers to digits (e.g., "set volume to five" -> "set volume to 5")
    - Handle common ASR misinterpretations or filler words (e.g., "uhm", "ah")

    This is highly dependent on the ASR engine and desired command structure.
    """
    text = text.lower()

    # Basic punctuation removal (can be customized)
    text = re.sub(r'[^\w\s\-\']', '', text) # Keep words, spaces, hyphens, apostrophes

    # TODO: Word-to-number conversion (e.g., "five" to "5")
    # This is complex and usually requires a dedicated library or extensive rules.
    # Example: text = text.replace(" five", " 5").replace(" ten ", " 10 ") # very naive

    # TODO: Remove filler words (customize list as needed)
    # filler_words = ['uh', 'um', 'uhm', 'ah', 'like', 'you know']
    # for word in filler_words:
    #     text = re.sub(r'\b' + word + r'\b', '', text)

    text = text.strip() # Remove leading/trailing whitespace
    text = re.sub(r'\s+', ' ', text) # Normalize multiple spaces to single space

    return text


if __name__ == '__main__':
    # Test number to words conversion
    print("--- Number to Words (for TTS) ---")
    test_tts_inputs = [
        "The code is 123.",
        "Call 911 in an emergency.",
        "My favorite number is 7.",
        "It's 20 degrees.",
        "Order 66 was executed."
    ]
    for t in test_tts_inputs:
        converted = convert_numbers_to_words(t)
        print(f"Original: '{t}' -> TTS Ready: '{converted}'")

    print("\n--- Spoken Text Normalization (from ASR) ---")
    test_asr_inputs = [
        "Set volume to five, please.",
        "Uhm, what time is it?",
        "  Open   the   DOOR!  ",
        "Search for WuBu project.", # Changed GLaDOS to WuBu
        "Is it 20 degrees or 25?"
    ]
    for t in test_asr_inputs:
        normalized = normalize_spoken_text(t)
        print(f"Original ASR: '{t}' -> Normalized: '{normalized}'")

    # Example of where word-to-number would be useful
    asr_with_word_num = "Set timer for twenty five minutes"
    print(f"ASR with word num: '{asr_with_word_num}' -> Normalized: '{normalize_spoken_text(asr_with_word_num)}' (needs word2num)")
