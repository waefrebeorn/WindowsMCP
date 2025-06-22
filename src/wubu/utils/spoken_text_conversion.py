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
    # Keep words, spaces, hyphens, apostrophes. Numbers also kept for word2num.
    text = re.sub(r'[^\w\s\-\'\d]', '', text)

    # --- Word-to-number conversion ---
    try:
        from word2number import w2n
        # w2n.word_to_num can convert phrases like "two point five" to 2.5
        # or "one hundred and twenty three" to 123.
        # It's best to apply this before removing filler words that might be part of a number phrase (e.g. "a hundred").
        # This is a simple pass; more complex scenarios might need sentence tokenization
        # and applying w2n to potential number phrases.

        # A simple approach: iterate through words and try to convert segments.
        # This is still naive. A more robust solution would use NLP to identify number phrases.
        # For now, let's try a regex-based replacement for simple number words if w2n is too broad.
        # The library w2n.word_to_num(text) tries to convert the first number phrase it finds.
        # This might not be ideal if text is "set timer for five minutes and alert in one hour".

        # A more targeted approach could be to replace known number words.
        # However, w2n is generally good. Let's try to use it by splitting and rejoining.
        # This is still not perfect for phrases like "two point five".
        words = text.split()
        processed_words = []
        i = 0
        while i < len(words):
            # Attempt to convert word by word or small phrases - this is tricky.
            # For simplicity in this step, we'll assume w2n can handle full sentences
            # or we'd need a more complex phrase detection.
            # Let's assume the user's speech for numbers is somewhat direct.
            # A common pattern: "number <unit>" e.g. "five minutes"
            # If w2n.word_to_num is applied to the whole string, it finds the first number.
            # This is usually what's needed for commands like "set volume to five".
            try:
                # Try to convert the whole text, this might be too greedy or only find the first.
                # A better way is to find number phrases.
                # For now, we'll stick to a simpler replacement strategy if a full one is too complex.
                # The original TODO was "five" to "5".

                # Let's use a simpler list for direct replacement before trying full w2n for robustness.
                simple_num_map_to_digit = {
                    'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
                    'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
                    'ten': '10'
                    # Add more common small numbers if needed
                }
                # Replace standalone number words first
                for word, digit in simple_num_map_to_digit.items():
                    text = re.sub(r'\b' + word + r'\b', digit, text)

                # After simple replacement, try w2n for more complex phrases if they remain
                # This part is tricky because w2n might convert parts of words if not careful.
                # Example: "winner" -> "win 1". So, apply carefully.
                # For now, the simple map above is safer than a general w2n pass on whole text.
                # If using w2n, it should be on identified number phrases.
                # print(f"Text after simple num replace: {text}")

            except Exception as e: # Broad exception for w2n issues
                # print(f"Word to number conversion issue (w2n): {e}")
                pass # Keep text as is if w2n fails

    except ImportError:
        print("Warning: 'word2number' library not found. Spoken numbers will not be converted to digits.")
        # Fallback: text = text.replace(" five", " 5").replace(" ten ", " 10 ") # very naive placeholder

    # --- Remove filler words ---
    filler_words = [
        'uh', 'um', 'uhm', 'ah', 'hmm', 'mhm',
        'like', 'you know', 'actually', 'basically', 'literally',
        'please', 'thank you', 'thanks' # Optional: remove politeness if commands are strict
    ]
    # Remove whole words/phrases
    for filler in filler_words:
        text = re.sub(r'\b' + re.escape(filler) + r'\b', '', text, flags=re.IGNORECASE)

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
