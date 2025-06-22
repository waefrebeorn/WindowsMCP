import webbrowser
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def open_url_in_default_browser(url: str) -> Dict[str, Any]:
    """
    Opens the given URL in the system's default web browser.

    Args:
        url: The URL to open (e.g., "http://www.google.com").
             It should be a complete URL including the scheme (http/https).

    Returns:
        A dictionary with "url" and "message" on success, or "error".
    """
    if not url or not (url.startswith("http://") or url.startswith("https://")):
        logger.error(f"Invalid URL provided: {url}. Must include http:// or https://")
        return {"error": "Invalid URL. It must start with http:// or https://."}

    try:
        # webbrowser.open_new_tab(url) # Opens in a new tab if possible, otherwise new window
        # For simplicity and broader compatibility, webbrowser.open() is often sufficient.
        # open_new_tab is generally preferred if available.
        success = webbrowser.open_new_tab(url)
        if success:
            logger.info(f"Successfully requested to open URL: {url}")
            return {"url": url, "message": f"Attempted to open URL '{url}' in the default browser."}
        else:
            # This 'else' case for webbrowser.open/open_new_tab returning False is rare
            # on desktop systems but indicates an issue finding/launching a browser.
            logger.warning(f"webbrowser.open_new_tab({url}) returned False. Browser might not have launched.")
            return {"url": url, "error": "Failed to open URL. The browser could not be launched or the action was blocked."}
    except Exception as e:
        logger.error(f"Error opening URL '{url}': {e}", exc_info=True)
        return {"error": f"An unexpected error occurred while opening URL: {e}"}

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
    logger.info("Web Interaction Module Example")

    test_url_valid = "https://www.google.com"
    test_url_invalid_scheme = "www.google.com"
    test_url_empty = ""

    print(f"\n--- Testing open_url_in_default_browser with valid URL: {test_url_valid} ---")
    # Note: This will actually try to open a browser if run in a desktop environment.
    # In a CI/headless environment, it might fail gracefully or do nothing.
    result_valid = open_url_in_default_browser(test_url_valid)
    print(result_valid)
    # We can't easily assert that a browser opened, so we check for no error.
    assert "error" not in result_valid, f"Valid URL test failed: {result_valid.get('error')}"


    print(f"\n--- Testing open_url_in_default_browser with invalid scheme: {test_url_invalid_scheme} ---")
    result_invalid_scheme = open_url_in_default_browser(test_url_invalid_scheme)
    print(result_invalid_scheme)
    assert "error" in result_invalid_scheme and "Invalid URL" in result_invalid_scheme["error"], "Invalid scheme test failed"

    print(f"\n--- Testing open_url_in_default_browser with empty URL: {test_url_empty} ---")
    result_empty = open_url_in_default_browser(test_url_empty)
    print(result_empty)
    assert "error" in result_empty and "Invalid URL" in result_empty["error"], "Empty URL test failed"

    logger.info("Web Interaction module example finished. If in a desktop environment, a browser tab to google.com might have opened.")
